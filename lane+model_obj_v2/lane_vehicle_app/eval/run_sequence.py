#!/usr/bin/env python3
"""
run_sequence.py -- run one preset over a video segment and save every frame's lane state, boundary positions, raw detections and displayed tracks.
  python eval/run_sequence.py --video clip.mp4 --preset baseline_reconstruction.json --start 24 --frames 640 --out results/seq_before_uhd.pkl
"""
import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lvapp import config                                       # noqa: E402
from lvapp.pipeline import Pipeline                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--preset", default=None)
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--frames", type=int, default=300)
ap.add_argument("--out", required=True)
ap.add_argument("--set", action="append", default=[])
a = ap.parse_args()
cfg = config.load(a.preset, a.set)
cfg.source.start_sec = a.start
p = Pipeline(cfg, a.video, frames=a.frames, warmup=0, output="none", log=True)
res = p.run()
pickle.dump(dict(records=p.records, result=res, k=cfg.pipeline.display_w / 640.0, start=a.start, preset=a.preset), open(a.out, "wb"))
print("saved", a.out, len(p.records), "frames; lane/det calls", res["lane_calls"], res["det_calls"])
