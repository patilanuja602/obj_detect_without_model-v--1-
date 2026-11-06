#!/usr/bin/env python3
"""
lane_vehicle_app.py -- integrated real-time lane + vehicle detection (CPU only).

    python3 lane_vehicle_app.py VIDEO.mp4 --show                 # live window (q to quit)
    python3 lane_vehicle_app.py VIDEO.mp4 --save out.mp4         # write an annotated video
    python3 lane_vehicle_app.py 0 --show                         # webcam index 0
    python3 lane_vehicle_app.py VIDEO.mp4 --mode lane --show     # lane only (baseline-equivalent loop)

Options:  --config my.json   --set detector.config=F   --set detector.threads=4   --mode par|seq|lane   --start S --duration D
          --cv-threads N     --det-threads N           --no-vehicles                --trace trace.json
The HUD's FPS is END-TO-END throughput of this loop (decode + lane + vehicles + drawing + output), averaged over the last 30 frames.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lvapp import config as cfgmod                       # noqa: E402
from lvapp.lane_v7 import LP                              # noqa: E402
from lvapp.pipeline import Engine                         # noqa: E402


def open_source(src, decode_threads):
    if decode_threads:
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;%d" % decode_threads)
    return cv2.VideoCapture(int(src) if str(src).isdigit() else src)


def main():
    ap = argparse.ArgumentParser(description="Integrated lane + vehicle detection (CPU)")
    ap.add_argument("src")
    ap.add_argument("--save", default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="a.b=value")
    ap.add_argument("--mode", choices=["async", "par", "seq", "lane"], default=None)
    ap.add_argument("--det-config", choices=["A", "B", "A2", "B2", "F", "F2"], default=None)
    ap.add_argument("--no-vehicles", action="store_true")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--cv-threads", type=int, default=None)
    ap.add_argument("--det-threads", type=int, default=None)
    ap.add_argument("--decode-threads", type=int, default=0, help="ffmpeg decoder threads (0 = ffmpeg default)")
    ap.add_argument("--trace", default=None, help="write per-frame lane/vehicle trace as JSON")
    a = ap.parse_args()

    cfg = cfgmod.load(a.config, a.set)
    if a.det_config is not None:
        cfg["detector"]["config"] = a.det_config
    if a.det_threads is not None:
        cfg["detector"]["threads"] = a.det_threads
    cvt = a.cv_threads if a.cv_threads is not None else cfg["pipeline"]["cv_threads"]
    if cvt:
        cv2.setNumThreads(int(cvt))
    cap = open_source(a.src, a.decode_threads)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    if a.start:
        cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
    ok, f = cap.read()
    if not ok:
        raise SystemExit("cannot read source: %s" % a.src)
    sh, sw = f.shape[:2]
    pp = cfg["pipeline"]
    ow = min(sw, pp["display_w"]) if pp.get("limit_display_upscale", True) else pp["display_w"]
    oh = int(round(sh * ow / sw))
    interp = cv2.INTER_AREA if pp["display_interp"] == "area" else cv2.INTER_LINEAR
    eng = Engine(cfg, sw, sh, mode=a.mode, with_vehicles=not a.no_vehicles)
    eng.warmup(f)
    print("source %dx%d  processing %dx%d  display %dx%d  mode=%s  detector=%s  ov-threads=%d  cv-threads=%d" % (
        sw, sh, eng.pw, eng.ph, ow, oh, eng.mode, cfg["detector"]["config"] if eng.vdet else "OFF", cfg["detector"]["threads"], cv2.getNumThreads()))
    tmp = (a.save + ".tmp.mp4") if a.save else None
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (ow, oh)) if tmp else None
    sink = LP._AsyncWriter(vw) if vw else None
    rd = LP._Prefetch(cap, depth=pp["reader_depth"])
    st = LP._Stage(rd.get, lambda fr: fr if (fr.shape[1] == ow and fr.shape[0] == oh) else cv2.resize(fr, (ow, oh), interpolation=interp))
    disp = f.copy() if (f.shape[1] == ow and f.shape[0] == oh) else cv2.resize(f, (ow, oh), interpolation=interp)
    n_max = a.max_frames or (int(a.duration * fps_in) if a.duration else 10 ** 9)
    n, trace, loop_t = 0, [], []
    t_loop = time.perf_counter()
    t_prev = t_loop
    fps_txt = ""
    fps_count = 0
    fps_start = t_loop
    current_fps = 0
    try:
        while f is not None and n < n_max:
            res = eng.process(f, n)
            out = eng.draw(disp, res, fps_txt)
            if sink:
                sink.write(out)
            if a.show:
                cv2.imshow("lane_vehicle_app (q to quit)", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if a.trace:
                trace.append(dict(f=res.fid, lane=res.lane["state"], n_veh=len(res.tracks),
                                  veh=[dict(id=t.id, cls=int(t.cls), score=round(t.score, 3), box=[round(float(v), 1) for v in t.box],
                                            gcp=[round(t.gcp[0], 1), round(t.gcp[1], 1)], zone=t.zone, coasted=bool(t.coasted)) for t in res.tracks]))
            n += 1
            now = time.perf_counter()
            loop_t.append(now - t_prev)
            t_prev = now

            fps_count += 1
            dt = now - fps_start
            if dt >= 1.0:
                current_fps = int(round(fps_count / dt))
                fps_count = 0
                fps_start = now
            elif current_fps == 0 and dt > 0.05:
                current_fps = int(round(fps_count / dt))
            fps_txt = "end-to-end %d FPS" % current_fps if current_fps > 0 else ""

            nxt = st.get()
            f, disp = nxt if nxt is not None else (None, None)
    finally:
        st.close()
        rd.close()
        if sink is not None:
            sink.close()
        eng.close()
    el = time.perf_counter() - t_loop
    if vw:
        vw.release()
    cap.release()
    if a.show:
        cv2.destroyAllWindows()
    if a.save and tmp:
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", a.save])
            os.remove(tmp)
        else:
            os.replace(tmp, a.save)
        print("saved", a.save)
    if a.trace:
        json.dump(trace, open(a.trace, "w"))
    lt = np.array(loop_t[5:] or loop_t)
    print("frames=%d  lane frames=%d  vehicle frames=%d  detector failures=%d" % (n, eng.n_lane, eng.n_det, eng.n_det_fail))
    print("END-TO-END: %d FPS over %d frames (%.2f s)   frame interval median %.1f ms  p95 %.1f ms" % (
        round(n / el), n, el, np.median(lt) * 1000, np.percentile(lt, 95) * 1000))


if __name__ == "__main__":
    main()
