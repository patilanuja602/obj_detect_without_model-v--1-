#!/usr/bin/env python3
"""
review_samples.py -- build montages for MANUAL verification (the numbers in eval_sequence.py cannot say whether a box is a real vehicle):
  missing : vehicles the BEFORE system first showed but the AFTER system never showed (is the tracker hiding real vehicles?)
  zones   : random AFTER tracks with their current / left / right / outside / uncertain tag (is the road association right?)
  python eval/review_samples.py --video clip.mp4 --tag uhd --start 24 --mode missing
"""
import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from eval_sequence import boxes_of, iou                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--start", type=float, required=True)
ap.add_argument("--mode", choices=["missing", "zones"], required=True)
ap.add_argument("--n", type=int, default=24)
a = ap.parse_args()
Rb = pickle.load(open(ROOT / "results" / ("seq_before_%s.pkl" % a.tag), "rb"))["records"]
Ra = pickle.load(open(ROOT / "results" / ("seq_after_%s.pkl" % a.tag), "rb"))["records"]
cap = cv2.VideoCapture(a.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
f0 = int(round(a.start * fps))
W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
k = W / 1280.0
items = []
if a.mode == "missing":
    Bb, Ba = boxes_of(Rb), boxes_of(Ra)
    for t in range(1, len(Bb) - 7):
        for b in Bb[t]:
            if any(iou(b[0], c[0]) >= 0.5 for c in Bb[t - 1]):
                continue
            if not any(0 <= t + g < len(Ba) and any(iou(b[0], c[0]) >= 0.4 for c in Ba[t + g]) for g in range(-2, 7)):
                items.append((t, b[0], "BEFORE %s %.2f" % ({1: "bic", 2: "car", 3: "mot", 5: "bus", 7: "trk"}.get(b[1], "?"), b[2])))
else:
    for t, r in enumerate(Ra):
        for tr in r["tracks"]:
            items.append((t, tr[1], "%s %.2f [%s]" % ({1: "bic", 2: "car", 3: "mot", 5: "bus", 7: "trk"}.get(tr[2], "?"), tr[3], tr[4].upper())))
rng = np.random.default_rng(1)
print("candidates:", len(items))
sel = [items[i] for i in rng.choice(len(items), size=min(a.n, len(items)), replace=False)] if items else []
tiles = []
for t, box, lab in sel:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0 + t)
    ok, f = cap.read()
    if not ok:
        continue
    x1, y1, x2, y2 = [v * k for v in box]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    sz = max(x2 - x1, y2 - y1) * 4 + 160
    xa, ya = int(max(cx - sz, 0)), int(max(cy - sz * 0.6, 0))
    xb, yb = int(min(cx + sz, f.shape[1])), int(min(cy + sz * 0.6, f.shape[0]))
    c = f[ya:yb, xa:xb].copy()
    cv2.rectangle(c, (int(x1 - xa), int(y1 - ya)), (int(x2 - xa), int(y2 - ya)), (0, 255, 255), 2)
    c = cv2.resize(c, (400, 240))
    cv2.putText(c, "%s f%d" % (lab, t), (4, 16), 0, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    tiles.append(c)
while len(tiles) % 4:
    tiles.append(np.zeros((240, 400, 3), np.uint8))
out = ROOT / "results" / ("review_%s_%s.jpg" % (a.mode, a.tag))
cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
print("wrote", out)
