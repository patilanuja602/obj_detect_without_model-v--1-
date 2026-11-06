#!/usr/bin/env python3
"""
eval_sequence.py -- BEFORE vs AFTER on identical video segments (logs from run_sequence.py).
Lane metrics (no ground truth needed; they measure stability / uncertainty / physical plausibility):
  non-LANE share, state changes per 100 frames, longest non-LANE run, corridor-width jumps (> 25 processing px between consecutive frames),
  boundary jitter (p95 of frame-to-frame |dx| at row 340), frames where a lane boundary passes through a reliable vehicle box (score >= 0.5, boxes
  from the BEFORE run so both lanes are judged against the same vehicles), and for the UHD segment the failure window frames 1093-1147 of the video.
Vehicle metrics on the DISPLAYED boxes:
  frames with duplicate boxes (IoU >= 0.6), class flips per 100 links (boxes linked between consecutive frames by IoU >= 0.5), blink events per 100
  boxes (a box vanishes for 1-2 frames and reappears), boxes per frame, first-appearance delay of AFTER relative to BEFORE, zone distribution (AFTER).
"""
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def runs(st, pred):
    best = c = 0
    for s in st:
        c = c + 1 if pred(s) else 0
        best = max(best, c)
    return best


def lane_metrics(R, k, ref_dets, window=None):
    st = [r["state"] for r in R]
    n = len(st)
    out = dict(frames=n, lane_pct=100.0 * st.count("LANE") / n, one_sided_pct=100.0 * st.count("ONE_SIDED") / n, none_pct=100.0 * st.count("NONE") / n,
               changes_per_100=100.0 * sum(a != b for a, b in zip(st, st[1:])) / n, longest_non_lane=runs(st, lambda s: s != "LANE"))
    w = np.array([r["R340"] - r["L340"] for r in R], float)
    dw = np.abs(np.diff(w))
    out["width_jumps_gt25px"] = int(np.nansum(dw > 25))
    out["width_jump_max_px"] = float(np.nanmax(dw)) if np.isfinite(dw).any() else 0.0
    for s in "LR":
        x = np.array([r[s + "340"] for r in R], float)
        d = np.abs(np.diff(x))
        d = d[np.isfinite(d)]
        out["jitter_p95_" + s] = float(np.percentile(d, 95)) if len(d) else float("nan")
    bad = 0
    for r, dets in zip(R, ref_dets):
        hit = False
        for b in dets:
            if b[4] < 0.5:
                continue
            x1, y1, x2, y2 = b[0] / k, b[1] / k, b[2] / k, b[3] / k
            for s in "LR":
                for y in (340, 300, 260):
                    x = r["%s%d" % (s, y)]
                    if np.isfinite(x) and y1 <= y <= y2 and x1 <= x <= x2:
                        hit = True
        bad += hit
    out["frames_boundary_on_vehicle"] = int(bad)
    if window:
        a, b = window
        sub = st[a:b + 1]
        out["window_frames"] = len(sub)
        out["window_non_lane"] = sum(s != "LANE" for s in sub)
        out["window_no_lane"] = sum(s == "NONE" for s in sub)
    return out


def boxes_of(R):
    return [[(t[1], t[2], t[3], t[4]) for t in r["tracks"]] for r in R]


def vehicle_metrics(R):
    B = boxes_of(R)
    nbox = sum(len(b) for b in B)
    dup = sum(any(iou(b[i][0], b[j][0]) >= 0.6 for i in range(len(b)) for j in range(i + 1, len(b))) for b in B)
    links = flips = 0
    for t in range(len(B) - 1):
        used = set()
        for a in B[t]:
            best, bj = 0.5, -1
            for j, b in enumerate(B[t + 1]):
                if j in used:
                    continue
                v = iou(a[0], b[0])
                if v >= best:
                    best, bj = v, j
            if bj >= 0:
                used.add(bj)
                links += 1
                flips += a[1] != B[t + 1][bj][1]
    blinks = 0
    for t in range(len(B) - 3):
        for a in B[t]:
            if any(iou(a[0], b[0]) >= 0.5 for b in B[t + 1]):
                continue
            for g in (2, 3):
                if any(iou(a[0], b[0]) >= 0.5 for b in B[t + g]):
                    blinks += 1
                    break
    return dict(boxes_per_frame=nbox / max(len(B), 1), frames_with_duplicates=int(dup), dup_pct=100.0 * dup / max(len(B), 1),
                class_flips_per_100_links=100.0 * flips / max(links, 1), links=links, blinks_per_100_boxes=100.0 * blinks / max(nbox, 1), zones=dict(Counter(t[4] for r in R for t in r["tracks"])))


def first_appearance_delay(Rb, Ra):
    Bb, Ba = boxes_of(Rb), boxes_of(Ra)
    delays, missing = [], 0
    for t in range(1, len(Bb) - 7):
        for a in Bb[t]:
            if any(iou(a[0], b[0]) >= 0.5 for b in Bb[t - 1]):
                continue                                              # not a first appearance
            found = None
            for g in range(-2, 7):
                if 0 <= t + g < len(Ba) and any(iou(a[0], b[0]) >= 0.4 for b in Ba[t + g]):
                    found = g
                    break
            if found is None:
                missing += 1
            else:
                delays.append(found)
    d = np.array(delays) if delays else np.array([0])
    return dict(n=len(delays), mean_delay_frames=float(d.mean()), pct_delayed_ge1=100.0 * float((d >= 1).mean()), max_delay=int(d.max()), not_shown_by_after=missing)


def fmt(x):
    return "%.2f" % x if isinstance(x, float) else str(x)


if __name__ == "__main__":
    res = {}
    for tag, window in (("uhd", (1093 - 720, 1147 - 720)), ("hd", None), ("nd20", None)):
        pb, pa = pickle.load(open(ROOT / "results" / ("seq_before_%s.pkl" % tag), "rb")), pickle.load(open(ROOT / "results" / ("seq_after_%s.pkl" % tag), "rb"))
        Rb, Ra, k = pb["records"], pa["records"], pb["k"]
        ref = [[(t[1][0], t[1][1], t[1][2], t[1][3], t[3]) for t in r["tracks"]] for r in Rb]            # BEFORE run's displayed boxes (score >= .5 used)
        ref = [[(b[0], b[1], b[2], b[3], b[4]) for b in fr] for fr in ref]
        res[tag] = dict(lane_before=lane_metrics(Rb, k, ref, window), lane_after=lane_metrics(Ra, k, ref, window),
                        veh_before=vehicle_metrics(Rb), veh_after=vehicle_metrics(Ra), delay=first_appearance_delay(Rb, Ra))
        print("\n================ %s  (%d frames) ================" % (tag.upper(), len(Rb)))
        print("%-34s %12s %12s" % ("LANE metric", "BEFORE", "AFTER"))
        for key in res[tag]["lane_before"]:
            print("%-34s %12s %12s" % (key, fmt(res[tag]["lane_before"][key]), fmt(res[tag]["lane_after"][key])))
        print("%-34s %12s %12s" % ("VEHICLE metric", "BEFORE", "AFTER"))
        for key in res[tag]["veh_before"]:
            if key != "zones":
                print("%-34s %12s %12s" % (key, fmt(res[tag]["veh_before"][key]), fmt(res[tag]["veh_after"][key])))
        print("zones AFTER:", res[tag]["veh_after"]["zones"])
        print("first-appearance delay of AFTER vs BEFORE:", res[tag]["delay"])
    json.dump(res, open(ROOT / "results" / "sequence_eval.json", "w"), indent=1, default=float)
