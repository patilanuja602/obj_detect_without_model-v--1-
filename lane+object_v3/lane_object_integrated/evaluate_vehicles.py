#!/usr/bin/env python3
"""Evaluate vehicle_detector against hand-annotated frames (eval/gt_*.json).

Runs V6 + VehicleDetector SEQUENTIALLY over every frame up to the last annotated one (tracker/road model see normal video flow),
then scores only the annotated frames:  detection = state ACCEPTED or TRACKED (observed).  PREDICTED boxes are not observations
and are excluded from precision (reported separately).  Matching: greedy by IoU, threshold --iou (default 0.30, small objects).
Detections overlapping an 'ignore' box (IoU>=0.2 or centre inside) are dropped (neither TP nor FP).

  python3 evaluate_vehicles.py --video B.mp4 --gt eval/gt_B.json --exclude 0.68 0.82 1 1
"""
import argparse, json, sys, time
import cv2, numpy as np

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u > 0 else 0.0

def score_frame(dets, gt, ignore, thr):
    """dets: list of boxes. returns tp, fp, fn, matches"""
    dets = [d for d in dets if not any(iou(d, g) >= 0.2 or (g[0] <= (d[0]+d[2])/2 <= g[2] and g[1] <= (d[1]+d[3])/2 <= g[3]) for g in ignore)]
    pairs = sorted(((-iou(d, g), di, gi) for di, d in enumerate(dets) for gi, g in enumerate(gt) if iou(d, g) >= thr))
    ud, ug = set(), set()
    for _, di, gi in pairs:
        if di in ud or gi in ug: continue
        ud.add(di); ug.add(gi)
    return len(ud), len(dets) - len(ud), len(gt) - len(ug)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--v6", default=".", help="directory containing lane_path_v6.py")
    ap.add_argument("--profile", default="mountain"); ap.add_argument("--iou", type=float, default=0.30)
    ap.add_argument("--exclude", type=float, nargs=4, action="append", default=[], metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--policy", default="fallback")
    ap.add_argument("--cfg", nargs="*", default=[], metavar="KEY=VAL", help="override VCFG values, e.g. thr_accept=0.45")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--min-gt-width", type=float, default=0, help="GT boxes narrower than this (px) are moved to 'ignore' (size-stratified report)")
    a = ap.parse_args()
    sys.path.insert(0, a.v6)
    import lane_path_v6 as V
    from vehicle_detector import VehicleDetector
    G = json.load(open(a.gt)); frames = {int(k): v for k, v in G["frames"].items()}
    cap = cv2.VideoCapture(a.video); ok, f = cap.read()
    sh, sw = f.shape[:2]; pw = G.get("proc_width", 640); ph = int(round(sh * pw / sw))
    cfg = dict(V.CFG); cfg.update(V.PROFILES[a.profile])
    det = V.PathDetectorV6(pw, ph, cfg)
    over = {k: float(v) if v.replace('.', '', 1).replace('-', '', 1).isdigit() else v for k, v in (kv.split("=") for kv in a.cfg)}
    vd = VehicleDetector(pw, ph, getattr(det, "poly", None), dict(exclude=tuple(map(tuple, a.exclude)), unreliable_policy=a.policy, **over))
    last = max(frames); TP = FP = FN = PRED = 0; i = 0; rows = []
    while ok and i <= last:
        small = cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA)
        det.process(small); vf = vd.process(small, det.result)
        if i in frames:
            obs = [d.box for d in vf.detections if d.observed]
            if a.min_score is not None:
                obs = [d.box for d in vf.detections if d.observed and d.score >= a.min_score]
            gtb = [g for g in frames[i]["gt"] if g[2] - g[0] >= a.min_gt_width]
            ign = frames[i]["ignore"] + [g for g in frames[i]["gt"] if g[2] - g[0] < a.min_gt_width]
            tp, fp, fn = score_frame(obs, gtb, ign, a.iou)
            PRED += sum(1 for d in vf.detections if not d.observed)
            TP += tp; FP += fp; FN += fn; rows.append((i, vf.lane_state, tp, fp, fn))
        i += 1; ok, f = cap.read()
    if not a.quiet:
        print("frame  v6state   TP FP FN")
        for r in rows: print("%5d  %-9s %2d %2d %2d" % r)
    P = TP / max(TP + FP, 1); R = TP / max(TP + FN, 1)
    print("\nannotated frames: %d | GT vehicles: %d | TP %d FP %d FN %d | precision %.2f recall %.2f | FP/frame %.2f | predicted-only boxes %d (excluded)"
          % (len(rows), TP + FN, TP, FP, FN, P, R, FP / max(len(rows), 1), PRED))
    print("IoU threshold %.2f, detections = observed ACCEPTED/TRACKED" % a.iou)

if __name__ == "__main__":
    main()
