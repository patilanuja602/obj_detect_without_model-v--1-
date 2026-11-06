#!/usr/bin/env python3
"""
overlap_stats.py -- how often does a visible vehicle actually overlap a lane boundary?  (decides whether the vehicle edge mask can matter)
Runs the app with the mask OFF (lane output == baseline) and counts frames in which a visible vehicle body (processing px) lies within
`--margin` px of a displayed boundary polyline, per segment, split by vehicle height (small = < mask_min_h_px, large = >=).
  python3 scripts/overlap_stats.py --video V.mp4 --start S --frames N
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lvapp import config                       # noqa: E402
from lvapp.pipeline import Engine              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--frames", type=int, default=150)
ap.add_argument("--margin", type=float, default=6.0)
ap.add_argument("--det-threads", type=int, default=1)
a = ap.parse_args()
cfg = config.load()
cfg["detector"]["threads"] = a.det_threads
cfg["lane_vehicle"]["mask_enabled"] = False
cap = cv2.VideoCapture(a.video)
cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)
ok, f = cap.read()
e = Engine(cfg, f.shape[1], f.shape[0], mode="seq")
e.warmup(f)
n = lane_frames = ov_small = ov_large = 0
min_h = cfg["lane_vehicle"]["mask_min_h_px"]
while ok and n < a.frames:
    r = e.process(f, n)
    lane = r.lane
    if lane["state"] == "LANE":
        lane_frames += 1
        cr = lane["corridor"]
        ys, xl, xr = np.asarray(cr["ys"]), np.asarray(cr["xl"]), np.asarray(cr["xr"])
        hit_s = hit_l = False
        for t in r.tracks:
            x1, y1, x2, y2 = (t.box * e.scale).tolist()
            sel = (ys >= y1) & (ys <= y2)
            if not sel.any():
                continue
            touch = np.any((xl[sel] >= x1 - a.margin) & (xl[sel] <= x2 + a.margin)) or np.any((xr[sel] >= x1 - a.margin) & (xr[sel] <= x2 + a.margin))
            if touch:
                if (y2 - y1) >= min_h:
                    hit_l = True
                else:
                    hit_s = True
        ov_small += hit_s
        ov_large += hit_l
    n += 1
    ok, f = cap.read()
print("%s @%d: frames=%d LANE=%d | frames with a boundary touching a vehicle body: large(h>=%d px)=%d  small=%d" % (Path(a.video).name, a.start, n, lane_frames, min_h, ov_large, ov_small))
