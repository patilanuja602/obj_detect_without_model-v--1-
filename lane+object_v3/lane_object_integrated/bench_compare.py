#!/usr/bin/env python3
"""bench_compare.py -- ONE sequential loop, three variants, same frames, same hardware, interleaved rounds.

Loop per frame (identical for all variants; nothing skipped, nothing dropped):
  decode -> processing resize -> lane.process -> [vehicles.process] -> display resize (INTER_AREA) -> lane draw -> [vehicle draw] -> write (mp4v)
Variants
  S1  System 1 : baselines/lane_path_v6_original.py (stand-in for System 1's lane file, which was not in the zip: its detection is bit-identical
                 to System 2's, see verify_integration.py) + vehicle detector, FULL-frame processing resize
  S2  System 2 : lane_path_v6.py (p3), ROI-only resize, lane only (no object detection exists in System 2)
  INT Integrated: lane_path_v6.py + vehicle detector, ROI-only resize widened to the vehicle band
Statistics skip the first --warmup frames of every round.  FPS = 1 / (time of the whole iteration incl. decode and write).
Usage: python3 bench_compare.py VIDEO [--start S] [--frames N] [--rounds R] [--exclude X0 Y0 X1 Y1] [--csv out.csv]
"""
import argparse, importlib.util, os, sys, tempfile, time
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vehicle_detector import VehicleDetector
import run_integrated as RI
from run_vehicle_pipeline import draw_vehicles


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def run(variant, a, mods):
    mod = mods["orig" if variant == "S1" else "new"]
    cap = cv2.VideoCapture(a.video); cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
    ok, f = cap.read(); sh, sw = f.shape[:2]
    cfg = dict(mod.CFG); cfg.update(mod.PROFILES["mountain"]); pw = cfg["proc_w"]; ph = int(round(sh * pw / sw)); ow = 1280; oh = int(round(sh * ow / sw))
    det = mod.PathDetectorV6(pw, ph, cfg)
    vd = None if variant == "S2" else VehicleDetector(pw, ph, det.poly, {"exclude": tuple(tuple(e) for e in a.exclude)})
    if variant == "S1":
        rs = None
    else:
        rs = RI.make_resizer(det, vd, sw, sh, pw, ph, True); rs.self_check(f)
    tmp = tempfile.mktemp(suffix=".mp4"); vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 30, (ow, oh))
    keys = ["decode", "resize_proc", "lane", "vehicles", "disp_resize", "draw", "write", "frame"]
    T = {k: [] for k in keys}; n = 0
    while ok and n < a.frames:
        t0 = time.perf_counter()
        small = cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA) if rs is None else rs(f)
        t1 = time.perf_counter(); L, R = det.process(small); t2 = time.perf_counter()
        vf = vd.process(small, det.result) if vd else None; t3 = time.perf_counter()
        disp = cv2.resize(f, (ow, oh), interpolation=cv2.INTER_AREA); t4 = time.perf_counter()
        disp = det.draw(disp, L, R, None, False)
        if vf is not None: disp = draw_vehicles(disp, vf, pw, False, None, None)
        t5 = time.perf_counter(); vw.write(disp); t6 = time.perf_counter()
        ok, f = cap.read(); t7 = time.perf_counter()
        for k, v in zip(keys, [t7 - t6, t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t0]): T[k].append(v)
        n += 1
    vw.release(); cap.release(); os.remove(tmp)
    return {k: np.array(v[a.warmup:]) * 1000 for k, v in T.items()}


ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("--start", type=float, default=5); ap.add_argument("--frames", type=int, default=330)
ap.add_argument("--warmup", type=int, default=30); ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--exclude", type=float, nargs=4, action="append", default=[]); ap.add_argument("--csv", default=None)
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
mods = {"orig": load("lane_orig", os.path.join(here, "baselines/lane_path_v6_original.py")), "new": load("lane_new", os.path.join(here, "lane_path_v6.py"))}
names = {"S1": "System 1 (lane v6 + vehicles)", "S2": "System 2 (lane p3 only)", "INT": "Integrated (lane p3 + vehicles)"}
acc = {v: [] for v in names}
for r in range(a.rounds):
    for v in (["S1", "S2", "INT"] if r % 2 == 0 else ["INT", "S2", "S1"]):
        acc[v].append(run(v, a, mods)); print("round %d %s done" % (r + 1, v), file=sys.stderr)
rows = []
print("\nvideo %s  start %.0fs  %d frames/round (%d warm-up skipped)  rounds %d  cv2 threads %d  cpus %s" % (os.path.basename(a.video), a.start, a.frames, a.warmup, a.rounds, cv2.getNumThreads(), os.cpu_count()))
print("%-34s %8s %8s %8s %8s | %7s" % ("variant", "meanFPS", "medFPS", "minFPS", "p5 FPS", "ms/frame(mean)"))
for v, nm in names.items():
    fr = np.concatenate([x["frame"] for x in acc[v]])
    rr = [1000.0 / x["frame"].mean() for x in acc[v]]
    print("%-34s %8.1f %8.1f %8.1f %8.1f | %7.2f   (per-round mean FPS: %s)" % (nm, 1000 / fr.mean(), 1000 / np.median(fr), 1000 / fr.max(), 1000 / np.percentile(fr, 95), fr.mean(), ", ".join("%.1f" % q for q in rr)))
    rows.append((v, fr))
print("\nstage medians, ms/frame")
ks = ["decode", "resize_proc", "lane", "vehicles", "disp_resize", "draw", "write", "frame"]
print("%-34s " % "variant" + " ".join("%11s" % k for k in ks))
for v, nm in names.items():
    print("%-34s " % nm + " ".join("%11.2f" % np.median(np.concatenate([x[k] for x in acc[v]])) for k in ks))
if a.csv:
    with open(a.csv, "w") as fh:
        fh.write("variant,round,frame_ms\n")
        for v in names:
            for r, x in enumerate(acc[v]):
                for t in x["frame"]: fh.write("%s,%d,%.3f\n" % (v, r, t))
