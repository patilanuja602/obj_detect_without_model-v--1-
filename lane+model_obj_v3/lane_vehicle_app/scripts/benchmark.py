#!/usr/bin/env python3
"""
benchmark.py -- end-to-end benchmark of the integrated application (or of the baseline system) on a video.

  python3 scripts/benchmark.py --video V.mp4 --system v7 --mode par --frames 300
  python3 scripts/benchmark.py --video V.mp4 --system baseline --frames 300
  python3 scripts/benchmark.py --video V.mp4 --sweep            # (ov-threads, cv-threads, mode) sweep -> pick the best on YOUR machine

FPS = timed frames / wall-clock seconds of the whole loop (decode wait + lane + vehicles + drawing + display-size resize on its
stage thread + optional output).  Per-stage times are per frame (perf_counter).  Frame integrity (--audit, on by default):
every frame handed to the lane detector must carry the same fingerprint as the one handed to the vehicle detector, and after the
timed loop the video is decoded again sequentially and the fingerprints must match frame for frame (no drops, no reordering).
Output modes: none (default; everything is drawn to a display-size buffer, nothing shown), file (annotated video), show (window).
The JSON written to results/ records the machine (CPU model, governor, cores, versions) so sandbox and T480 numbers stay apart.
"""
import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lvapp import config as cfgmod                       # noqa: E402
from lvapp.baseline_system import BaselineEngine          # noqa: E402
from lvapp.lane_v7 import LP                              # noqa: E402
from lvapp.pipeline import Engine, fingerprint            # noqa: E402


def sysinfo():
    info = dict(python=platform.python_version(), platform=platform.platform(), cv2=cv2.__version__, numpy=np.__version__,
                logical_cpus=psutil.cpu_count(True), physical_cpus=psutil.cpu_count(False), ram_GB=round(psutil.virtual_memory().total / 2 ** 30, 1))
    try:
        import openvino as ov
        info["openvino"] = ov.__version__
    except Exception:
        pass
    try:
        for ln in open("/proc/cpuinfo"):
            if ln.startswith("model name"):
                info["cpu"] = ln.split(":", 1)[1].strip()
                break
        info["governor"] = open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip()
    except Exception:
        pass
    try:
        info["affinity_cpus"] = len(os.sched_getaffinity(0))
    except Exception:
        pass
    return info


def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def run(video, system, mode, frames, start, cfg, output, audit=True, cv_threads=None, quiet=False):
    if cv_threads is not None:
        cv2.setNumThreads(cv_threads)
    cap = cv2.VideoCapture(video)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, f = cap.read()
    if not ok:
        raise SystemExit("cannot read " + video)
    sh, sw = f.shape[:2]
    ow = cfg["pipeline"]["display_w"]
    oh = int(round(sh * ow / sw))
    interp = cv2.INTER_AREA if cfg["pipeline"]["display_interp"] == "area" else cv2.INTER_LINEAR
    eng = BaselineEngine(cfg, sw, sh) if system == "baseline" else Engine(cfg, sw, sh, mode=mode)
    if mode == "lane" and system != "baseline":
        pass
    if audit:
        eng.audit = {}
    eng.warmup(f)
    rd = LP._Prefetch(cap, depth=cfg["pipeline"]["reader_depth"])
    st = LP._Stage(rd.get, lambda fr: cv2.resize(fr, (ow, oh), interpolation=interp))
    disp = cv2.resize(f, (ow, oh), interpolation=interp)
    vw = None
    if output == "file":
        vw = cv2.VideoWriter("bench_out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (ow, oh))
    sink = LP._AsyncWriter(vw) if vw else None
    proc = psutil.Process()
    psutil.cpu_percent(None)
    c0, wall0 = proc.cpu_times(), time.perf_counter()
    rows, n, loop = [], 0, []
    t_prev = wall0
    try:
        while f is not None and n < frames:
            t0 = time.perf_counter()
            res = eng.process(f, n)
            t1 = time.perf_counter()
            out = eng.draw(disp, res, None)
            t2 = time.perf_counter()
            if sink:
                sink.write(out)
            if output == "show":
                cv2.imshow("bench", out)
                cv2.waitKey(1)
            t3 = time.perf_counter()
            n += 1
            tw = time.perf_counter()
            nxt = st.get()
            f, disp = nxt if nxt is not None else (None, None)
            now = time.perf_counter()
            r = dict(res.t)
            r.update(draw=t2 - t1, out=t3 - t2, wait_next=now - tw, process=t1 - t0, iteration=now - t_prev,
                     n_veh=len(res.tracks) if system != "baseline" else len(res.boxes))
            rows.append(r)
            t_prev = now
    finally:
        st.close()
        rd.close()
        if sink:
            sink.close()
        eng.close()
    wall = time.perf_counter() - wall0
    c1 = proc.cpu_times()
    cpu_cores = ((c1.user - c0.user) + (c1.system - c0.system)) / wall
    sys_cpu = psutil.cpu_percent(None)
    if vw:
        vw.release()
    cap.release()
    # ---- frame integrity: lane fingerprint == detector fingerprint, and == an independent sequential re-decode
    audit_ok, audit_msg = True, "audit off"
    if audit:
        a = eng.audit
        same = all(v.get("lane") == v.get("det") for v in a.values() if eng.vdet is not None)
        cap2 = cv2.VideoCapture(video)
        if start:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, start)
        redec = True
        for i in range(n):
            ok2, f2 = cap2.read()
            if not ok2 or fingerprint(f2) != a[i]["lane"]:
                redec = False
                break
        cap2.release()
        audit_ok = bool(same and redec and eng.n_lane == n and (eng.vdet is None or eng.n_det == n))
        audit_msg = "lane==det fingerprint: %s | matches independent re-decode: %s | lane calls %d, detector calls %d, frames %d" % (same, redec, eng.n_lane, eng.n_det, n)
    keys = ["rs", "lane", "det_pre", "det_inf", "det_post", "det_total", "det_wait", "track", "compute", "draw", "out", "wait_next", "process", "iteration"]
    M = {k: np.array([r.get(k, 0.0) for r in rows[3:]]) * 1000 for k in keys}
    out = dict(video=Path(video).name, system=system, mode=mode if system != "baseline" else "seq", frames=n, wall_s=wall, end_to_end_fps=n / wall,
               fps_steady=1000.0 / M["iteration"].mean() if len(M["iteration"]) else None,
               cpu_cores_used=cpu_cores, system_cpu_pct=sys_cpu, rss_MB=proc.memory_info().rss / 2 ** 20, peak_rss_MB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               lane_frames=eng.n_lane, vehicle_frames=eng.n_det, detector_failures=eng.n_det_fail, dropped_frames=max(0, n - eng.n_lane),
               audit_ok=audit_ok, audit=audit_msg, ms={k: dict(mean=float(v.mean()), median=float(np.median(v)), p95=pct(v, 95), max=float(v.max())) for k, v in M.items()},
               avg_vehicles=float(np.mean([r["n_veh"] for r in rows])), cfg=dict(detector=cfg["detector"], pipeline=cfg["pipeline"], cv_threads=cv2.getNumThreads()),
               machine=sysinfo(), is_t480=("8350U" in sysinfo().get("cpu", "")))
    if not quiet:
        m = out["ms"]
        print("%-9s %-5s %-14s frames=%d  END-TO-END %.1f FPS | lane %.1f ms (med %.1f) | det inf %.1f ms (pre %.1f post %.1f) | det wait %.1f | total/frame med %.1f p95 %.1f ms | CPU %.2f cores sys %.0f%% | RSS %.0f MB | lane=%d det=%d dropped=%d | audit %s%s" % (
            out["system"], out["mode"], out["video"], n, out["end_to_end_fps"], m["lane"]["mean"] + m["rs"]["mean"], m["lane"]["median"] + m["rs"]["median"],
            m["det_inf"]["mean"], m["det_pre"]["mean"], m["det_post"]["mean"], m["det_wait"]["mean"], m["iteration"]["median"], m["iteration"]["p95"],
            cpu_cores, sys_cpu, out["peak_rss_MB"], eng.n_lane, eng.n_det, out["dropped_frames"], "OK" if audit_ok else "FAILED", "" if audit_ok else "  " + audit_msg))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--system", choices=["baseline", "v7"], default="v7")
    ap.add_argument("--mode", choices=["par", "seq", "lane"], default=None)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--output", choices=["none", "file", "show"], default="none")
    ap.add_argument("--no-audit", action="store_true")
    ap.add_argument("--cv-threads", type=int, default=None)
    ap.add_argument("--sweep", action="store_true", help="try ov-threads x cv-threads x mode and print a ranking")
    ap.add_argument("--json", default=None, help="append the result as JSON to this file")
    a = ap.parse_args()
    cfg = cfgmod.load(a.config, a.set)
    mode = a.mode or cfg["pipeline"]["mode"]
    results = []
    if a.sweep:
        ncpu = psutil.cpu_count(False) or 2
        for mode_ in ("par", "seq"):
            for ovt in sorted({1, 2, max(1, ncpu - 1), ncpu}):
                for cvt in sorted({1, 2, ncpu}):
                    c = cfgmod.copy_cfg(cfg)
                    c["detector"]["threads"] = ovt
                    c["pipeline"]["cv_threads"] = cvt
                    r = run(a.video, "v7", mode_, a.frames, a.start, c, "none", not a.no_audit, cvt, quiet=True)
                    r["tag"] = "mode=%s ov=%d cv=%d" % (mode_, ovt, cvt)
                    results.append(r)
                    print("%-22s %.1f FPS  (cpu %.2f cores, audit %s)" % (r["tag"], r["end_to_end_fps"], r["cpu_cores_used"], r["audit_ok"]), flush=True)
        best = max(results, key=lambda r: r["end_to_end_fps"])
        print("BEST:", best["tag"], "%.1f FPS" % best["end_to_end_fps"], "-> use  --set detector.threads=%d --set pipeline.cv_threads=%d --set pipeline.mode=%s" % (
            best["cfg"]["detector"]["threads"], best["cfg"]["cv_threads"], best["mode"]))
    else:
        results.append(run(a.video, a.system, mode, a.frames, a.start, cfg, a.output, not a.no_audit, a.cv_threads if a.cv_threads is not None else cfg["pipeline"]["cv_threads"]))
    if a.json:
        with open(a.json, "a") as fh:
            for r in results:
                fh.write(json.dumps(r) + "\n")
    if any(not r["audit_ok"] for r in results):
        print("FRAME CHECK FAILED")
        sys.exit(2)


if __name__ == "__main__":
    main()
