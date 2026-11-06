#!/usr/bin/env python3
"""
compare_baseline.py -- baseline system vs the integrated app on the SAME frames of the SAME video.

  python3 scripts/compare_baseline.py --video V.mp4 --start 3000 --frames 150 --out results/cmp_V_3000.json --review results/review
  python3 scripts/compare_baseline.py --video V.mp4 --frames 150 --check-identical     # proof that the lane output is unchanged without vehicles
  add  --ref-full N   to also run the full-frame 960x544 reference detector on every Nth frame (distant-vehicle recall check)

Measured (no ground-truth labels exist for these videos, so every number below is a PROXY, stated as such in the report):
  LANE      state shares (LANE / ONE_SIDED / NONE), mean |delta| of the bottom x of each boundary between consecutive frames (jitter),
            number of frames where the corridor width changes by > 12 % in one frame, share of frames whose state differs from baseline,
            displayed lane-overlay pixels inside vehicle bodies (what the viewer sees: lane drawn over cars),
  VEHICLE   boxes per frame, same-frame near-duplicate pairs (IoU >= 0.5), appear/disappear flicker (a chain of boxes that vanishes and
            re-appears within 5 frames), class changes within a chain, recall against a full-frame reference (far / mid / near),
  ZONES     how many displayed vehicles fall in CURRENT / LEFT_ADJ / RIGHT_ADJ / OUTSIDE / UNCERTAIN.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lvapp import config as cfgmod                         # noqa: E402
from lvapp.baseline_system import VD, BaselineEngine       # noqa: E402
from lvapp.lane_v7 import LP                               # noqa: E402
from lvapp.pipeline import Engine                          # noqa: E402
from lvapp import overlay                                  # noqa: E402


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def chains(frames_boxes, gate=0.3):
    """Neutral greedy IoU chaining of per-frame box lists [(x1,y1,x2,y2,cls)] -> flicker events and class changes (same for both systems)."""
    active = []                                          # dict(last=box, cls, last_frame, gaps)
    flick = cls_changes = n_chains = 0
    closed = []
    for fi, boxes in enumerate(frames_boxes):
        used = set()
        for ch in active:
            best, bj = 0.0, -1
            for j, b in enumerate(boxes):
                if j in used:
                    continue
                v = iou(ch["last"], b)
                if v > best:
                    best, bj = v, j
            if best >= gate:
                b = boxes[bj]
                if fi - ch["last_frame"] > 1:
                    ch["gaps"] += 1
                if b[4] != ch["cls"]:
                    ch["cls_changes"] += 1
                ch["last"], ch["cls"], ch["last_frame"] = b[:4], b[4], fi
                ch["n"] += 1
                used.add(bj)
        for j, b in enumerate(boxes):
            if j not in used:
                active.append(dict(last=b[:4], cls=b[4], last_frame=fi, gaps=0, cls_changes=0, n=1))
        still = []
        for ch in active:
            (still if fi - ch["last_frame"] <= 5 else closed).append(ch)
        active = still
    closed += active
    for ch in closed:
        if ch["n"] >= 3:
            n_chains += 1
            flick += ch["gaps"]
            cls_changes += ch["cls_changes"]
    return dict(chains=n_chains, flicker_gaps=flick, class_changes=cls_changes)


def body_rects(boxes_disp, W, H, inset=0.15, trim=0.04):
    out = []
    for x1, y1, x2, y2 in boxes_disp:
        bw, bh = x2 - x1, y2 - y1
        a, b = int(max(x1 + inset * bw, 0)), int(min(x2 - inset * bw, W))
        c, d = int(max(y1, 54)), int(min(y2 - trim * bh, H))
        if b > a and d > c:
            out.append((a, c, b, d))
    return out


def lane_only_render(eng, disp, res, is_v7, cfg):
    """What the viewer sees from the LANE overlay alone (no vehicle boxes): baseline draw, or draw + occlusion restore for v7."""
    d = disp.copy()
    patches = []
    lv = cfg["lane_vehicle"]
    if is_v7 and lv["restore_overlay"] and res.tracks:
        patches = overlay.snapshot_bodies(d, res.tracks, d.shape[1] / eng.src_w, lv["mask_inset_x"], lv["mask_bottom_trim"])
    eng.lane.draw(d, None, None, None)
    if patches:
        overlay.restore_bodies(d, patches)
    return d


def run_system(name, eng, video, start, frames, cfg, ow, review_dir, review_every, vid_tag):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    rec = dict(state=[], xl=[], xr=[], width=[], boxes=[], zones=[], overlay_px=[], overlay_px_tot=[], reviews={})
    ref_cache = {}
    for i in range(frames):
        ok, f = cap.read()
        if not ok:
            break
        if i == 0:
            eng.warmup(f)
        res = eng.process(f, i)
        lane = res.lane
        rec["state"].append(lane["state"])
        cr = lane.get("corridor")
        if lane["state"] == "LANE" and cr is not None:
            rec["xl"].append(float(cr["xl"][0]))
            rec["xr"].append(float(cr["xr"][0]))
        else:
            rec["xl"].append(None)
            rec["xr"].append(None)
        if name == "baseline":
            fb = [(float(b[0]), float(b[1]), float(b[2]), float(b[3]), int(b[5])) for b in res.boxes]
            sc = [float(b[4]) for b in res.boxes]
        else:
            fb = [(float(t.box[0]), float(t.box[1]), float(t.box[2]), float(t.box[3]), int(t.cls)) for t in res.tracks]
            sc = [float(t.score) for t in res.tracks]
            rec["zones"].extend(t.zone for t in res.tracks)
        rec["boxes"].append(fb)
        # displayed lane overlay inside vehicle bodies
        disp = cv2.resize(f, (ow, int(round(f.shape[0] * ow / f.shape[1]))), interpolation=cv2.INTER_LINEAR)
        s = ow / f.shape[1]
        rects = body_rects([(b[0] * s, b[1] * s, b[2] * s, b[3] * s) for b in fb], disp.shape[1], disp.shape[0])
        if rects:
            d = lane_only_render(eng, disp, res, name != "baseline", cfg)
            diff = np.any(d != disp, axis=2)
            ch = tot = 0
            for a, c, b, e in rects:
                ch += int(diff[c:e, a:b].sum())
                tot += (b - a) * (e - c)
            rec["overlay_px"].append(ch)
            rec["overlay_px_tot"].append(tot)
        if review_dir and i % review_every == 0:
            d = disp.copy()
            eng.draw(d, res, None)
            rec["reviews"][i] = d
    cap.release()
    return rec


def lane_metrics(rec):
    st = rec["state"]
    n = len(st)
    xl = [(i, v) for i, v in enumerate(rec["xl"]) if v is not None]
    jit = []
    wj = 0
    for i in range(1, n):
        if rec["xl"][i] is not None and rec["xl"][i - 1] is not None:
            jit.append((abs(rec["xl"][i] - rec["xl"][i - 1]) + abs(rec["xr"][i] - rec["xr"][i - 1])) / 2)
            w0, w1 = rec["xr"][i - 1] - rec["xl"][i - 1], rec["xr"][i] - rec["xl"][i]
            if w0 > 1 and abs(w1 - w0) / w0 > 0.12:
                wj += 1
    return dict(frames=n, LANE=st.count("LANE") / n, ONE_SIDED=st.count("ONE_SIDED") / n, NONE=st.count("NONE") / n,
                jitter_px_mean=float(np.mean(jit)) if jit else None, jitter_px_p95=float(np.percentile(jit, 95)) if jit else None,
                width_jumps=wj, n_lane_frames=len(xl))


def veh_metrics(rec):
    fb = rec["boxes"]
    dup = 0
    for boxes in fb:
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if iou(boxes[i], boxes[j]) >= 0.5:
                    dup += 1
    m = chains(fb)
    m.update(boxes_per_frame=float(np.mean([len(b) for b in fb])), duplicate_pairs=dup,
             overlay_px_in_vehicles=float(sum(rec["overlay_px"]) / max(sum(rec["overlay_px_tot"]), 1)) if rec["overlay_px_tot"] else None,
             frames_with_vehicles_overlay=len(rec["overlay_px"]))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--review", default=None, help="directory for side-by-side review images")
    ap.add_argument("--review-every", type=int, default=30)
    ap.add_argument("--ref-full", type=int, default=0, help="run the full-frame reference detector on every Nth frame")
    ap.add_argument("--check-identical", action="store_true")
    ap.add_argument("--det-threads", type=int, default=None)
    a = ap.parse_args()
    cfg = cfgmod.load(a.config, a.set)
    if a.det_threads:
        cfg["detector"]["threads"] = a.det_threads
    cv = cv2.VideoCapture(a.video)
    sw, sh = int(cv.get(3)), int(cv.get(4))
    cv.release()
    ow = cfg["pipeline"]["display_w"]
    tag = Path(a.video).stem

    if a.check_identical:
        # lane output of the app with NO vehicle information must be bit-identical to the unmodified baseline lane detector
        e0 = BaselineEngine(cfg, sw, sh, with_vehicles=False)
        e1 = Engine(cfg, sw, sh, mode="lane", with_vehicles=False)
        cap = cv2.VideoCapture(a.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)
        bad = 0
        for i in range(a.frames):
            ok, f = cap.read()
            if not ok:
                break
            r0 = e0.process(f, i)
            r1 = e1.process(f, i)
            l0, l1 = r0.lane, r1.lane
            same = l0["state"] == l1["state"]
            for k in ("L", "R"):
                c0 = None if l0[k] is None else np.asarray(l0[k]["coef"])
                c1 = None if l1[k] is None else np.asarray(l1[k]["coef"])
                same &= (c0 is None and c1 is None) or (c0 is not None and c1 is not None and np.array_equal(c0, c1))
            if l0["state"] == "LANE":
                same &= np.array_equal(l0["corridor"]["xl"], l1["corridor"]["xl"]) and np.array_equal(l0["corridor"]["xr"], l1["corridor"]["xr"])
            bad += (not same)
        print("lane output identical to baseline on %d frames: %s (mismatching frames: %d)" % (i + 1, "YES" if bad == 0 else "NO", bad))
        sys.exit(0 if bad == 0 else 1)

    results = {}
    recs = {}
    for name in ("baseline", "v7"):
        eng = BaselineEngine(cfg, sw, sh) if name == "baseline" else Engine(cfg, sw, sh, mode="seq")
        recs[name] = run_system(name, eng, a.video, a.start, a.frames, cfg, ow, a.review, a.review_every, tag)
        results[name] = dict(lane=lane_metrics(recs[name]), vehicle=veh_metrics(recs[name]))
        if name == "v7":
            from collections import Counter
            results[name]["zones"] = dict(Counter(recs[name]["zones"]))
            results[name]["tracker_stats"] = dict(eng.tracker.stats)
        eng.close()
    # lane differences baseline vs v7
    sb, s7 = recs["baseline"]["state"], recs["v7"]["state"]
    diff_state = sum(1 for x, y in zip(sb, s7) if x != y) / len(sb)
    dx = [abs(recs["baseline"]["xl"][i] - recs["v7"]["xl"][i]) + abs(recs["baseline"]["xr"][i] - recs["v7"]["xr"][i])
          for i in range(len(sb)) if recs["baseline"]["xl"][i] is not None and recs["v7"]["xl"][i] is not None]
    results["lane_diff"] = dict(state_differs=diff_state, bottom_x_abs_diff_mean=float(np.mean(dx) / 2) if dx else None, n_compared=len(dx))
    # distant-vehicle recall vs a full-frame reference detector (the reference is a model, not human labels)
    if a.ref_full:
        from lvapp.vehicle_detector import VehicleDetector
        ref = VehicleDetector("F", cfg["detector"]["threads"], conf_low=0.35)
        cap = cv2.VideoCapture(a.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)
        agg = {n: {b: [0, 0] for b in ("far", "mid", "near", "all")} for n in ("baseline", "v7")}
        for i in range(a.frames):
            ok, f = cap.read()
            if not ok:
                break
            if i % a.ref_full:
                continue
            d, _ = ref.detect(f)
            for k in range(len(d.score)):
                rb = d.boxes[k]
                hfrac = (rb[3] - rb[1]) / f.shape[0]
                bucket = "far" if hfrac < 0.06 else ("mid" if hfrac < 0.15 else "near")
                for n in ("baseline", "v7"):
                    hit = any(iou(rb, b) >= 0.5 for b in recs[n]["boxes"][i])
                    for bk in (bucket, "all"):
                        agg[n][bk][0] += int(hit)
                        agg[n][bk][1] += 1
        cap.release()
        results["recall_vs_full_frame_reference"] = {n: {b: (v[0] / v[1] if v[1] else None, v[1]) for b, v in agg[n].items()} for n in agg}
    results["per_frame"] = dict(state_baseline=recs["baseline"]["state"], state_v7=recs["v7"]["state"],
                                n_boxes_v7=[len(b) for b in recs["v7"]["boxes"]], n_boxes_baseline=[len(b) for b in recs["baseline"]["boxes"]])
    results["meta"] = dict(video=a.video, start=a.start, frames=a.frames)
    print(json.dumps(results, indent=1, default=lambda o: float(o)))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(a.out, "w"), indent=1, default=lambda o: float(o))
    if a.review:
        Path(a.review).mkdir(parents=True, exist_ok=True)
        for i, imb in recs["baseline"]["reviews"].items():
            im7 = recs["v7"]["reviews"].get(i)
            if im7 is None:
                continue
            sbs = np.hstack([cv2.resize(imb, (960, int(imb.shape[0] * 960 / imb.shape[1]))), cv2.resize(im7, (960, int(im7.shape[0] * 960 / im7.shape[1])))])
            cv2.putText(sbs, "BASELINE", (10, sbs.shape[0] - 12), 0, 0.6, (0, 255, 255), 2)
            cv2.putText(sbs, "V7", (970, sbs.shape[0] - 12), 0, 0.6, (0, 255, 255), 2)
            cv2.imwrite(str(Path(a.review) / ("%s_f%d.jpg" % (tag, a.start + i))), sbs, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
