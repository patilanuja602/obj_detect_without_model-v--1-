#!/usr/bin/env python3
"""
agreement_display_vs_original.py -- what does feeding the detector from the 1280-wide display frame (cheaper preprocessing) cost, compared with feeding it
the original full-resolution frame?  Same model, same settings; for every Nth frame of the videos: detections (score >= 0.25) from the original frame are the
reference; we count how many the display-fed detector also finds (IoU >= 0.5), by size, and how many extra boxes it adds.
  python eval/agreement_display_vs_original.py --video videos/a.mp4 --video videos/b.mp4 --step 12
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lvapp.vehicle_detector import VehicleDetector              # noqa: E402


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)


ap = argparse.ArgumentParser()
ap.add_argument("--video", action="append", required=True)
ap.add_argument("--step", type=int, default=12)
ap.add_argument("--max-frames", type=int, default=120)
ap.add_argument("--model", default="models/yolo26n_960x352.onnx")
a = ap.parse_args()
det = VehicleDetector(ROOT / a.model, band=(0.35, 1.0), threads=1, conf=0.25, nms_mode="agnostic_merge", nms_iou=0.55, resize="linear")
detA = VehicleDetector(ROOT / a.model, band=(0.35, 1.0), threads=1, conf=0.25, nms_mode="agnostic_merge", nms_iou=0.55, resize="area")
tot = {"far": [0, 0], "mid": [0, 0], "near": [0, 0]}
extra = nref = nd = 0
for v in a.video:
    cap = cv2.VideoCapture(v)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for fi in list(range(60, n - 5, a.step))[:a.max_frames]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, f = cap.read()
        if not ok:
            continue
        H, W = f.shape[:2]
        ref = detA.detect(f)                                          # original frame, area resize (the old path)
        disp = cv2.resize(f, (1280, int(round(H * 1280 / W))), interpolation=cv2.INTER_AREA)
        d = det.detect(disp)
        d[:, :4] *= W / 1280
        for r in ref:
            s = np.sqrt(max(r[2] - r[0], 1) * max(r[3] - r[1], 1)) * 1080 / H
            b = "far" if s < 32 else ("mid" if s < 96 else "near")
            tot[b][1] += 1
            tot[b][0] += any(iou(r, q) >= 0.5 for q in d)
        extra += sum(1 for q in d if not any(iou(q, r) >= 0.5 for r in ref))
        nref += len(ref)
        nd += len(d)
print("reference boxes (original-frame input): %d ; display-input boxes: %d ; display-input boxes with no counterpart: %d" % (nref, nd, extra))
for b, (m, t) in tot.items():
    print("  %-4s found by display-input: %d / %d  (%.0f%%)" % (b, m, t, 100.0 * m / max(t, 1)))
