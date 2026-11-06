#!/usr/bin/env python3
"""
eval_vehicle_static.py -- detector-level comparison on the 36 Phase-2 evaluation frames (95 reference vehicles; reference boxes made by a much larger
YOLO model and reviewed by eye -- NOT human ground truth; recall carries about +/-5 points of noise).
Frames are cut from your two videos at fixed times (the same ones used in Phase 2).
  python eval/eval_vehicle_static.py --uhd videos/<uhd>.mp4 --hd videos/<hd>.mp4
Reports per variant: recall (all / far / mid / near), false positives per frame, cross-class duplicate pairs, boxes, AP50.  Operating point: score >= 0.25.
far < 32 px | mid 32-96 | near > 96  (box size normalised to a 1080-px-high frame).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lvapp.vehicle_detector import VehicleDetector              # noqa: E402

GT = json.load(open(ROOT / "eval" / "gt_phase2_reference.json"))


def extract(uhd, hd):
    fr = {}
    for video, prefix, t0, step, n in ((uhd, "uhd", 5.0, 10.2, 24), (hd, "hd", 3.0, 6.0, 12)):
        cap = cv2.VideoCapture(video)
        for i in range(n):
            t = t0 + step * i
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, f = cap.read()
            fr["%s_%03d.png" % (prefix, int(t))] = f
    return fr


def iou_vec(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / np.maximum((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter, 1e-9)


def bucket(b, H):
    s = np.sqrt(max(b[2] - b[0], 1) * max(b[3] - b[1], 1)) * 1080.0 / H
    return "far" if s < 32 else ("mid" if s < 96 else "near")


def match(dets, gt, ign, thr):
    d = dets[dets[:, 4] >= thr]
    d = d[np.argsort(-d[:, 4])]
    gtb = np.array([g[:4] for g in gt], float).reshape(-1, 4)
    ig = np.array(ign, float).reshape(-1, 4)
    used = np.zeros(len(gt), bool)
    flags = []
    for r in d:
        j = -1
        if len(gtb):
            ious = iou_vec(r[:4], gtb)
            ious[used] = 0
            k = int(ious.argmax())
            if ious[k] >= 0.5:
                j = k
        if j >= 0:
            used[j] = True
            flags.append(1)
        else:
            flags.append(-1 if (len(ig) and (iou_vec(r[:4], ig) >= 0.3).any()) else 0)
    return d, np.array(flags, int), used


def ap50(scores, tp, n_gt):
    if n_gt == 0 or not len(scores):
        return 0.0
    o = np.argsort(-np.array(scores))
    t = np.array(tp)[o]
    t = t[t >= 0]
    ctp, cfp = np.cumsum(t == 1), np.cumsum(t == 0)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1)
    mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[1], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


def dup_pairs(d):
    n = 0
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            if d[i][5] != d[j][5] and iou_vec(d[i][:4], d[j:j + 1, :4])[0] >= 0.6:
                n += 1
    return n


def evaluate(det, frames, source, op=0.25):
    cnt = {b: [0, 0] for b in ("all", "far", "mid", "near")}
    fp = dups = nbox = 0
    sc, tp = [], []
    ngt = 0
    for name, img in frames.items():
        H, W = img.shape[:2]
        if source == "display":
            ow = 1280
            inp = cv2.resize(img, (ow, int(round(H * ow / W))), interpolation=cv2.INTER_AREA)
            d = det.detect(inp)
            d[:, :4] *= W / ow
        else:
            d = det.detect(img)
        gt, ign = GT["gt"][name], GT["ignore"].get(name, [])
        dd, fl, used = match(d, gt, ign, op)
        fp += int((fl == 0).sum())
        nbox += len(dd)
        dups += dup_pairs(dd)
        for g, u in zip(gt, used):
            for key in (bucket(g, H), "all"):
                cnt[key][1] += 1
                cnt[key][0] += int(u)
        d2, f2, _ = match(d, gt, ign, 0.05)
        sc += d2[:, 4].tolist()
        tp += f2.tolist()
        ngt += len(gt)
    n = len(frames)
    r = {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in cnt.items()}
    return dict(recall=r["all"], far=r["far"], mid=r["mid"], near=r["near"], fp_per_frame=fp / n, dup_pairs=dups, boxes=nbox, ap50=ap50(sc, tp, ngt))


VARIANTS = {   # name: (source, detector kwargs)
    "OLD  class-aware NMS, conf .25, full-frame input": ("original", dict(conf=0.05, nms_mode="class_aware", nms_iou=0.5, resize="area", min_height_px=0, aspect_range=(0, 1000))),
    "NEW1 + class-agnostic merge (IoU .55)": ("original", dict(conf=0.05, nms_mode="agnostic_merge", nms_iou=0.55, resize="area", min_height_px=0, aspect_range=(0, 1000))),
    "NEW2 + size/aspect plausibility": ("original", dict(conf=0.05, nms_mode="agnostic_merge", nms_iou=0.55, resize="area", min_height_px=6, aspect_range=(0.4, 4.0))),
    "NEW3 + input from 1280 display frame (default)": ("display", dict(conf=0.05, nms_mode="agnostic_merge", nms_iou=0.55, resize="linear", min_height_px=6, aspect_range=(0.4, 4.0))),
    "OPT  NEW3 + reject non-vehicle-dominant (x1.5)": ("display", dict(conf=0.05, nms_mode="agnostic_merge", nms_iou=0.55, resize="linear", min_height_px=6, aspect_range=(0.4, 4.0), reject_nonvehicle_ratio=1.5)),
    "OPT  NEW3 with NMS IoU .50": ("display", dict(conf=0.05, nms_mode="agnostic_merge", nms_iou=0.50, resize="linear", min_height_px=6, aspect_range=(0.4, 4.0))),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--uhd", required=True)
    ap.add_argument("--hd", required=True)
    ap.add_argument("--model", default="models/yolo26n_960x352.onnx")
    ap.add_argument("--band", type=float, nargs=2, default=[0.35, 1.0])
    ap.add_argument("--out", default=str(ROOT / "results" / "vehicle_static_eval.json"))
    a = ap.parse_args()
    frames = extract(a.uhd, a.hd)
    res = {}
    print("%-50s %6s %6s %6s %6s %7s %5s %6s %6s" % ("variant (operating point score>=0.25)", "recall", "far", "mid", "near", "FP/frm", "dups", "boxes", "AP50"))
    for name, (src, kw) in VARIANTS.items():
        det = VehicleDetector(ROOT / a.model, band=a.band, threads=1, **kw)
        r = evaluate(det, frames, src)
        res[name] = r
        print("%-50s %6.2f %6.2f %6.2f %6.2f %7.2f %5d %6d %6.2f" % (name, r["recall"], r["far"], r["mid"], r["near"], r["fp_per_frame"], r["dup_pairs"], r["boxes"], r["ap50"]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
