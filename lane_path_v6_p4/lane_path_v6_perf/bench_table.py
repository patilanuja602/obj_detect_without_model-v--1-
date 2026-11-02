#!/usr/bin/env python3
"""bench_table.py -- reproducible A/B/C benchmark.  Runs bench_v6.py (coarse mode) for every variant, INTERLEAVED over several rounds
(A,B,C,A,B,C,...) so slow drifts of the machine hit every variant equally, then reports the median over rounds of each per-run
median / p95, plus the min..max of the per-round end-to-end FPS.  Same video, start, frames, warm-up, processing/display size for all.

  python3 bench_table.py VIDEO --variants lane_path_v6_before.py lane_path_v6_pre_opt.py lane_path_v6.py \
         --labels "A original V6" "B V6+clip (current)" "C optimised" --rounds 5 --start 20 --frames 300 --warmup 30 [--write]
"""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("--variants", nargs="+", required=True); ap.add_argument("--labels", nargs="+")
ap.add_argument("--rounds", type=int, default=5); ap.add_argument("--start", type=float, default=20)
ap.add_argument("--frames", type=int, default=300); ap.add_argument("--warmup", type=int, default=30)
ap.add_argument("--write", action="store_true", help="include cv2.VideoWriter(mp4v) writing, like the real CLI")
a = ap.parse_args()
labels = a.labels or a.variants
here = os.path.dirname(os.path.abspath(__file__))
runs = {v: [] for v in a.variants}
tmp = tempfile.mkdtemp()
for r in range(a.rounds):
    for v in a.variants:
        js = os.path.join(tmp, "r.json")
        cmd = [sys.executable, os.path.join(here, "bench_v6.py"), a.video, "--variant", v, "--mode", "coarse", "--start", str(a.start),
               "--frames", str(a.frames), "--warmup", str(a.warmup), "--json", js]
        if a.write: cmd += ["--write", os.path.join(tmp, "o.mp4")]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        runs[v].append(json.load(open(js)))
        print("round %d  %-28s e2e %.1f FPS" % (r + 1, os.path.basename(v), runs[v][-1]["e2e_fps"]), flush=True)
stages = ["decode", "resize_proc", "process", "resize_disp", "draw"] + (["write"] if a.write else []) + ["pipeline(no decode/write)", "total"]
print("\nmedian over %d interleaved rounds of the per-run median / p95 (ms per frame); %d frames per run after %d warm-up; write=%s\n"
      % (a.rounds, a.frames, a.warmup, a.write))
hdr = "%-26s" % "stage" + "".join("| %-21s" % l[:21] for l in labels)
print(hdr); print("-" * len(hdr))
for st in stages:
    row = "%-26s" % st
    for v in a.variants:
        med = np.median([r["res"][st][0] for r in runs[v]]); p95 = np.median([r["res"][st][1] for r in runs[v]])
        row += "| %7.2f / %-11.2f" % (med, p95)
    print(row)
row = "%-26s" % "END-TO-END FPS (median)"
for v in a.variants: row += "| %-21s" % ("%.1f  [%.1f..%.1f]" % (np.median([r["e2e_fps"] for r in runs[v]]), min(r["e2e_fps"] for r in runs[v]), max(r["e2e_fps"] for r in runs[v])))
print(row)
row = "%-26s" % "algorithm-only FPS"
for v in a.variants: row += "| %-21s" % ("%.1f" % (1000 / np.median([r["res"]["process"][0] for r in runs[v]])))
print(row)
