#!/usr/bin/env python3
"""
ablate_lane.py -- compare lane-side settings on one video segment (all variants share the detector, tracker and start frame; only the lane's use of vehicle
evidence changes).  Variants: mask off (= baseline lane behaviour) / mask with different margins / confidence / occlusion hold.
  python eval/ablate_lane.py --video clip.mp4 --start 33 --frames 285 --window 103 157
"""
import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from lvapp import config                                       # noqa: E402
from lvapp.pipeline import Pipeline                            # noqa: E402
from eval_sequence import lane_metrics                         # noqa: E402

VARIANTS = [
    ("lane mask OFF (baseline lane)", {"lane.mask_mode": "off"}),
    ("mask v1: all reliable vehicles, unprotected", {"lane.mask_min_height_px": 0, "lane.protect_tracked_px": 0}),
    ("mask near-field only (h>=15 px), unprotected", {"lane.mask_min_height_px": 15, "lane.protect_tracked_px": 0}),
    ("mask near-field only (h>=15 px) + protect 4 px", {"lane.mask_min_height_px": 15, "lane.protect_tracked_px": 4}),
    ("mask near-field only (h>=25 px) + protect 4 px", {"lane.mask_min_height_px": 25, "lane.protect_tracked_px": 4}),
]

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--start", type=float, default=33.0)
ap.add_argument("--frames", type=int, default=285)
ap.add_argument("--window", type=int, nargs=2, default=None)
ap.add_argument("--only", type=int, nargs="*", default=None, help="indices of VARIANTS to run")
a = ap.parse_args()
base = config.load()
ref = None
rows = []
for vi, (name, over) in enumerate(VARIANTS):
    if a.only is not None and vi not in a.only:
        continue
    cfg = config.Cfg(copy.deepcopy(dict(base)))
    for k, v in over.items():
        sec, key = k.split(".")
        cfg[sec][key] = v
    cfg.source.start_sec = a.start
    p = Pipeline(cfg, a.video, frames=a.frames, warmup=0, output="none", log=True)
    p.run()
    R = p.records
    if ref is None:
        ref = [[(t[1][0], t[1][1], t[1][2], t[1][3], t[3]) for t in r["tracks"]] for r in R]       # same vehicles for every variant
    m = lane_metrics(R, 1280 / 640.0, ref, tuple(a.window) if a.window else None)
    rows.append((name, m))
    print("done", name, flush=True)
print("\n%-50s %6s %9s %6s %8s %7s %9s %10s %12s" % ("variant", "LANE%", "1-sided%", "NONE%", "longest", "jumps", "bndry-on-veh", "win nonLANE", "win NO-LANE"))
for name, m in rows:
    print("%-50s %6.1f %9.1f %6.1f %8d %7d %9d %10s %12s" % (name, m["lane_pct"], m["one_sided_pct"], m["none_pct"], m["longest_non_lane"], m["width_jumps_gt25px"], m["frames_boundary_on_vehicle"],
                                                              m.get("window_non_lane", "-"), m.get("window_no_lane", "-")))
