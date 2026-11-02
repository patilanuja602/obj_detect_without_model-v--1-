#!/usr/bin/env python3
"""Run lane (V6, unmodified) + a vehicle detector implementation over a video from frame 0 and dump every frame's
detections in ORIGINAL SOURCE PIXELS to JSON.  Used for identical-input before/after comparison.
  python3 evalkit/dump_detections.py VIDEO OUT.json --impl baseline|new [--end N] [--profile highway] [--exclude x0 y0 x1 y1]
"""
import argparse, importlib.util, json, os, sys, time
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import lane_path_v6 as V

def load_impl(name):
    path = os.path.join(ROOT, "baselines/vehicle_detector_original.py") if name == "baseline" else os.path.join(ROOT, "vehicle_detector.py")
    spec = importlib.util.spec_from_file_location("vd_" + name, path); m = importlib.util.module_from_spec(spec)
    sys.modules["vd_" + name] = m; spec.loader.exec_module(m); return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("out")
    ap.add_argument("--impl", default="new"); ap.add_argument("--end", type=int, default=10**9)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--profile", default="highway")
    ap.add_argument("--exclude", type=float, nargs=4, action="append", default=[])
    ap.add_argument("--cfg", nargs="*", default=[])
    a = ap.parse_args()
    M = load_impl(a.impl)
    cap = cv2.VideoCapture(a.video); ok, f = cap.read(); sh, sw = f.shape[:2]
    cfg = dict(V.CFG); cfg.update(V.PROFILES[a.profile]); pw = cfg["proc_w"]; ph = int(round(sh * pw / sw)); s = sw / pw
    det = V.PathDetectorV6(pw, ph, cfg)
    over = {"exclude": tuple(tuple(e) for e in a.exclude)}
    for kv in a.cfg:
        k, v = kv.split("=", 1); over[k] = float(v)
    vd = M.VehicleDetector(pw, ph, getattr(det, "poly", None), over)
    new = a.impl != "baseline"
    # same ROI-only resize as run_integrated (bit identical to full resize)
    rs = V._ProcResizer(det, sw, sh, pw, ph, enabled=True)
    if rs.fast:
        y0, y1, x0, x1 = rs.roi; rs.roi = (min(y0, vd.yb), ph, 0, pw)
    frames, i, tl, tv = [], 0, [], []
    while ok and i < a.end:
        if i >= a.start or True:
            small = rs(f)
            t0 = time.perf_counter(); det.process(small); t1 = time.perf_counter()
            vf = vd.process(small, det.result, full=f) if new else vd.process(small, det.result)
            t2 = time.perf_counter(); tl.append(t1 - t0); tv.append(t2 - t1)
            rec = dict(f=i, lane=det.result["lane"]["state"], dets=[])
            for d in vf.detections:
                b = [round(float(v) * s, 2) for v in d.box]
                if new and getattr(d, "refined", False):
                    b = [round(float(v), 2) for v in d.box_src]
                c = getattr(d, "contact_src", None) if new else None
                if c is None and not new: c = (d.contact[0] * s, d.contact[1] * s)
                rec["dets"].append(dict(id=d.track_id, st=d.state, box=b, obs=bool(d.observed), lane=d.lane, score=round(float(d.score), 3),
                    contact=None if c is None else [round(float(c[0]), 2), round(float(c[1]), 2)],
                    cq=getattr(d, "contact_quality", "BBOX_BOTTOM"), ev=round(float(getattr(d, "evidence", 0.0)), 3),
                    pt={k: round(float(v), 3) for k, v in (getattr(d, "parts", {}) or {}).items() if isinstance(v, (int, float))},
                    hits=getattr(d, "hits", 0),
                    cc=[[round(float(p[0]), 2), round(float(p[1]), 2)] for p in (getattr(d, "tyres_src", None) or [])]))
            frames.append(rec)
        ok, f = cap.read(); i += 1
    json.dump(dict(impl=a.impl, video=os.path.basename(a.video), scale=s, frames=frames,
                   ms_lane=float(np.mean(tl[30:]) * 1000), ms_veh=float(np.mean(tv[30:]) * 1000)), open(a.out, "w"))
    print("frames", len(frames), "lane ms %.2f veh ms %.2f" % (np.mean(tl[30:]) * 1000, np.mean(tv[30:]) * 1000))
main()
