#!/usr/bin/env python3
"""
benchmark.py -- before/after timing with the frame-integrity proof.   Run it on the T480 for real numbers.

  python benchmark.py --video videos/clip.mp4                       # BEFORE (reconstruction of the old system) vs AFTER (default), 300 frames each
  python benchmark.py --video clip.mp4 --presets default.json --det-config B2 --threads-list 1 2 3 4
Each run: same video, same start, same warm-up; every frame goes through lane AND detector.  After the timed run the video is decoded again
sequentially and its frame fingerprints must equal the frames the pipeline processed (no dropped or reordered frames), and the lane and the
detector must have received the same frame (fingerprint + frame id).  FPS = timed frames / wall-clock seconds of the whole loop (decode wait, lane,
detector, tracking, drawing, output).  Writes results/benchmark_<name>.json and prints a table.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lvapp import config                                       # noqa: E402
from lvapp.pipeline import Pipeline, fp                        # noqa: E402
from run_app import DET, apply_det                             # noqa: E402


def verify_sequential(cfg, source, start, pipe):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    ow = int(cfg.pipeline.display_w)
    use_disp = cfg.vehicle.input_source == "display"
    ref = []
    for _ in range(len(pipe.fps_main)):
        ok, f = cap.read()
        if not ok:
            break
        ref.append(fp(f))
    return ref == pipe.fps_main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--presets", nargs="*", default=["baseline_reconstruction.json", "default.json"])
    ap.add_argument("--det-config", choices=list(DET), default=None)
    ap.add_argument("--threads-list", nargs="*", type=int, default=None, help="vehicle inference threads to test for every preset")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--start", type=float, default=20.0)
    ap.add_argument("--output", choices=["none", "file", "show"], default="none")
    ap.add_argument("--name", default="run")
    a = ap.parse_args()
    rows, allres = [], {}
    for preset in a.presets:
        for th in (a.threads_list or [None]):
            cfg = config.load(preset)
            if a.det_config:
                apply_det(cfg, a.det_config)
            if th:
                cfg.vehicle.threads = th
            cfg.source.start_sec = a.start
            pipe = Pipeline(cfg, a.video, frames=a.frames, warmup=a.warmup, output=a.output)
            r = pipe.run()
            r["sequential_decode_matches"] = verify_sequential(cfg, a.video, a.start, pipe)
            label = "%s thr=%s" % (preset.replace(".json", ""), cfg.vehicle.threads)
            allres[label] = r
            S = r["stages_ms"]
            rows.append((label, r["fps"]["end_to_end_wall"], S["total"]["median"], S["total"]["p95"], S["lane"]["median"], r["detector_ms"]["median"], r["cpu"]["process_cores_used"],
                         r["memory"]["rss_MB"], "%d/%d/%d" % (r["lane_calls"], r["det_calls"], r["frames_total"]), r["dropped_frames"],
                         "ok" if (r["lane_and_detector_same_frame"] and r["sequential_decode_matches"]) else "FAILED"))
            print("done", label, flush=True)
    print("\n%-40s %7s %9s %8s %8s %8s %6s %6s %12s %5s %9s" % ("config", "FPS", "median ms", "p95 ms", "lane ms", "det ms", "cores", "RSS MB", "lane/det/frm", "drop", "integrity"))
    for r in rows:
        print("%-40s %7.1f %9.1f %8.1f %8.1f %8.1f %6.2f %6.0f %12s %5d %9s" % r)
    out = config.ROOT / "results" / ("benchmark_%s.json" % a.name)
    json.dump(allres, open(out, "w"), indent=1, default=float)
    print("wrote", out)


if __name__ == "__main__":
    main()
