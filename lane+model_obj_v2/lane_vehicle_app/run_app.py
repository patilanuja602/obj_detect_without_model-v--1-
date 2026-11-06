#!/usr/bin/env python3
"""
run_app.py -- improved real-time lane + vehicle detection.

  python run_app.py --video path/to/video.mp4                 # on-screen window (q to quit)
  python run_app.py --video clip.mp4 --output file            # write results/output.mp4 instead (works headless)
  python run_app.py --source 0                                # camera 0
  python run_app.py --video clip.mp4 --autotune               # measure OpenVINO / OpenCV thread settings on THIS machine first, then run with the best
  python run_app.py --video clip.mp4 --preset baseline_reconstruction.json   # the old behaviour (for comparison)
  python run_app.py --video clip.mp4 --det-config B2          # faster / smaller detector input (A2 default, B2 = 768x288, F = full frame 960x544)
  python run_app.py --video clip.mp4 --set vehicle.conf=0.3 --set association.outside_policy=hide
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lvapp import config                                       # noqa: E402
from lvapp.pipeline import Pipeline                            # noqa: E402

DET = {"A2": ("models/yolo26n_960x352.onnx", [0.35, 1.0]), "B2": ("models/yolo26n_768x288.onnx", [0.35, 1.0]), "F": ("models/yolo26n_960x544.onnx", None)}


def apply_det(cfg, name):
    f, band = DET[name]
    cfg.vehicle.model_file = f
    cfg.vehicle.band = band


def autotune(cfg, source, frames=60, warmup=15):
    import copy
    rows = []
    for vt in (1, 2, 3, 4):
        for cvt in (None, 1):
            c = config.Cfg(copy.deepcopy(dict(cfg)))
            c.vehicle.threads, c.pipeline.cv_threads = vt, cvt
            try:
                r = Pipeline(c, source, frames=frames, warmup=warmup, output="none", quiet=True).run()
            except SystemExit as e:
                sys.exit(str(e))
            rows.append((r["fps"]["end_to_end_wall"], vt, cvt, r["stages_ms"]["total"]["p95"]))
            print("  vehicle threads %d, cv threads %-7s -> %.1f FPS (p95 frame %.1f ms)" % (vt, cvt if cvt else "default", rows[-1][0], rows[-1][3]), flush=True)
    best = max(rows)
    print("best measured: vehicle threads %d, cv threads %s  (%.1f FPS)" % (best[1], best[2] or "default", best[0]))
    json.dump({"vehicle": {"threads": best[1]}, "pipeline": {"cv_threads": best[2]}, "_measured_fps": best[0]}, open(config.ROOT / "config" / "tuned.json", "w"), indent=2)
    cfg.vehicle.threads, cfg.pipeline.cv_threads = best[1], best[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", "--source", dest="source", required=True, help="video file, or a camera index such as 0")
    ap.add_argument("--preset", default=None, help="config file in config/ (default: default.json)")
    ap.add_argument("--det-config", choices=list(DET), default=None)
    ap.add_argument("--output", choices=["show", "file", "none"], default=None)
    ap.add_argument("--out-file", default=None)
    ap.add_argument("--frames", type=int, default=None, help="stop after N frames")
    ap.add_argument("--start", type=float, default=None, help="start time in the video (s)")
    ap.add_argument("--autotune", action="store_true")
    ap.add_argument("--tuned", action="store_true", help="use config/tuned.json written by an earlier --autotune")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. vehicle.conf=0.3")
    a = ap.parse_args()
    cfg = config.load(a.preset, a.set)
    if a.det_config:
        apply_det(cfg, a.det_config)
    if a.start is not None:
        cfg.source.start_sec = a.start
    if a.tuned and (config.ROOT / "config" / "tuned.json").exists():
        t = json.load(open(config.ROOT / "config" / "tuned.json"))
        cfg.vehicle.threads, cfg.pipeline.cv_threads = t["vehicle"]["threads"], t["pipeline"]["cv_threads"]
    if a.autotune:
        print("auto-tuning thread settings (short real runs, no display) ...")
        autotune(cfg, a.source)
    out = a.output or cfg.pipeline.output
    res = Pipeline(cfg, a.source, frames=a.frames, warmup=0 if not a.frames else min(15, a.frames // 4), output=out, out_file=a.out_file).run()
    f = res["fps"]
    print("frames %d | end-to-end %.1f FPS | frame p95 %.1f ms | lane %.1f ms | detector %.1f ms | cpu %.2f cores | rss %.0f MB | lane/det calls %d/%d, dropped %d"
          % (res["frames_total"], f["end_to_end_wall"], res["stages_ms"]["total"]["p95"], res["stages_ms"]["lane"]["median"], res["detector_ms"]["median"],
             res["cpu"]["process_cores_used"], res["memory"]["rss_MB"], res["lane_calls"], res["det_calls"], res["dropped_frames"]))


if __name__ == "__main__":
    main()
