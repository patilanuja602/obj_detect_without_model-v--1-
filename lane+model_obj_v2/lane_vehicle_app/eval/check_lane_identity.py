#!/usr/bin/env python3
"""
check_lane_identity.py -- regression gate: the lane output of lvapp's LaneEngine must be BIT-IDENTICAL to the untouched V6 baseline whenever no vehicle
mask is set (mask_mode off, or a mask list that is empty).  Compares the per-frame lane coefficients (L, R) and the full per-frame lane state.
  python eval/check_lane_identity.py --video clip.mp4 [--video other.mp4] --frames 300 --start 20
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lvapp import config                                       # noqa: E402
from lvapp.lane_engine import LaneEngine, load_lane_module     # noqa: E402


def run(video, frames, start):
    cfg = config.load()
    lp = load_lane_module(cfg.lane.file)
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    ok, f = cap.read()
    sh, sw = f.shape[:2]
    # reference: the baseline classes exactly as the baseline main() builds them
    c = dict(lp.CFG)
    c.update(lp.PROFILES[cfg.lane.profile])
    pw = c["proc_w"]
    ph = int(round(sh * pw / sw))
    ref = lp.PathDetectorV6(pw, ph, c, cfg6={})
    rrs = lp._ProcResizer(ref, sw, sh, pw, ph, True)
    eng = LaneEngine(cfg, sw, sh)
    eng2 = LaneEngine(cfg, sw, sh)                              # engine with an EMPTY mask list set every frame
    bad = bad2 = n = 0
    while ok and n < frames:
        L0, R0 = ref.process(rrs(f))
        s0 = ref.result["lane"]
        L1, R1, s1 = eng.process(f)
        eng2.set_mask([])
        L2, R2, s2 = eng2.process(f)
        for (L, R, s) in ((L1, R1, s1),):
            bad += not (np.array_equal(L0, L) and np.array_equal(R0, R) and s0["state"] == s["state"] and np.array_equal(s0["pts_L"], s["pts_L"]) and np.array_equal(s0["pts_R"], s["pts_R"]))
        bad2 += not (np.array_equal(L0, L2) and np.array_equal(R0, R2) and s0["state"] == s2["state"])
        n += 1
        ok, f = cap.read()
    return n, bad, bad2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="append", required=True)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--start", type=float, default=20.0)
    a = ap.parse_args()
    allok = True
    for v in a.video:
        n, bad, bad2 = run(v, a.frames, a.start)
        print("%-50s frames %d | differing frames: engine %d, engine+empty-mask %d -> %s" % (Path(v).name[:50], n, bad, bad2, "IDENTICAL" if bad == 0 and bad2 == 0 else "DIFFERENT"))
        allok &= bad == 0 and bad2 == 0
    sys.exit(0 if allok else 1)
