#!/usr/bin/env python3
"""verify_integration.py -- proves the integration changed nothing that matters.

Reference  = System 1 behaviour: original V6 (baselines/lane_path_v6_original.py, full processing resize) + VehicleDetector,
             exactly the loop of run_vehicle_pipeline.run_stream.
Integrated = System 2 lane_path_v6.py (ROI-only resize widened to the vehicle band) + the same VehicleDetector.
Both are fed the same consecutive frames.  Per frame it checks (bit-exact):
  1. det.result (full lane result dict: lane, corridor, partial, mode, extent_y ...)  -> lane detection unchanged
  2. every vehicle detection: track id, box, contact, state, lane association, score, observed -> vehicle detection unchanged
  3. the rows the vehicle detector reads are identical in both processing frames (rows >= band_top*h) and the lane ROI too
Usage: python3 verify_integration.py VIDEO [--start S] [--frames N] [--exclude X0 Y0 X1 Y1]
"""
import argparse, importlib.util, sys
import cv2, numpy as np
sys.path.insert(0, ".")
from vehicle_detector import VehicleDetector
import run_integrated as RI


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def same(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.shape == b.shape and np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f" else np.array_equal(a, b)
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    return a == b


def sig(vf):
    return [(d.track_id, tuple(d.box), tuple(d.contact), d.state, d.lane, d.score, d.observed) for d in vf.detections]


ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("--start", type=float, default=0); ap.add_argument("--frames", type=int, default=300)
ap.add_argument("--exclude", type=float, nargs=4, action="append", default=[])
a = ap.parse_args()
ref_mod = load("lane_ref", "baselines/lane_path_v6_original.py")
new_mod = load("lane_new", "lane_path_v6.py")
cap = cv2.VideoCapture(a.video); cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
ok, f = cap.read(); sh, sw = f.shape[:2]
cfg = dict(ref_mod.CFG); cfg.update(ref_mod.PROFILES["mountain"]); pw = cfg["proc_w"]; ph = int(round(sh * pw / sw))
cfg2 = dict(new_mod.CFG); cfg2.update(new_mod.PROFILES["mountain"])
dref, dnew = ref_mod.PathDetectorV6(pw, ph, cfg), new_mod.PathDetectorV6(pw, ph, cfg2)
over = {"exclude": tuple(tuple(e) for e in a.exclude)}
vref, vnew = VehicleDetector(pw, ph, dref.poly, over), VehicleDetector(pw, ph, dnew.poly, over)
rs = RI.make_resizer(dnew, vnew, sw, sh, pw, ph, True); print("roi-resize fast path:", rs.fast, "rows", rs.roi if rs.fast else "-"); rs.self_check(f)
bad = dict(lane=0, veh=0, rows=0); n = ndet = 0; states = {}
while ok and n < a.frames:
    full = cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA)
    sm = rs(f)
    y0 = min(dnew.y0, vnew.yb)
    if not np.array_equal(full[y0:], sm[y0:]): bad["rows"] += 1
    dref.process(full); dnew.process(sm)
    if not same(dref.result, dnew.result): bad["lane"] += 1
    a1 = vref.process(full, dref.result); a2 = vnew.process(sm, dnew.result)
    if sig(a1) != sig(a2): bad["veh"] += 1
    ndet += len(a1.detections); st = dnew.result["lane"]["state"]; states[st] = states.get(st, 0) + 1
    n += 1; ok, f = cap.read()
print("frames %d | lane states %s | vehicle detections compared %d" % (n, states, ndet))
print("mismatching frames:", bad)
print("RESULT:", "PASS (bit-identical lane result and vehicle output)" if not any(bad.values()) else "FAIL")
sys.exit(1 if any(bad.values()) else 0)
