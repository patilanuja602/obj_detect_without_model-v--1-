#!/usr/bin/env python3
"""
run_integrated.py -- System 2's lane detector (lane_path_v6.py, = lane_path_v6_p3, unmodified) + System 1's vehicle detector
(vehicle_detector.py, unmodified) + System 1's vehicle overlay (run_vehicle_pipeline.draw_vehicles, unmodified), in ONE pipeline.

Frame references: ONE processing-size frame `small` is made per source frame.  The lane detector and the vehicle detector both
read exactly that frame in the same loop iteration (no asynchronous detector, so no stale frame can ever be paired with newer
lane geometry).  All vehicle coordinates are processing-frame pixels and are scaled to the display frame only when drawing.

The single integration conflict that had to be solved (see README_INTEGRATION.md):
  System 2's _ProcResizer resizes only the lane ROI rows and leaves every other row ZERO.  The vehicle detector reads rows
  [int(band_top*h), h) over the full width, so with System 2's unmodified resizer it sees black rows.  Here the same ROI-only
  resize (still bit-identical, k x k box average) is widened to the union of the lane ROI and the vehicle band.

Everything else of System 2's pipeline is kept: background decoder thread, display-resize stage thread, async writer thread
(same frames, same order, nothing dropped), --display-interp, --cv-threads, --overlay-top-*.
Everything of System 1's runner is kept: --camera, --show, --debug, --debug-lane, --no-vehicles, --policy, --exclude, --cfg.

Examples
  python3 run_integrated.py video.mp4 out.mp4
  python3 run_integrated.py video.mp4 --show
  python3 run_integrated.py --camera 0 --show
  python3 run_integrated.py video.mp4 out.mp4 --start 20 --duration 10 --exclude 0.68 0.82 1 1 --debug
"""
import argparse
import os
import shutil
import subprocess
import time

import cv2
import numpy as np

import lane_path_v6 as V
from vehicle_detector import VehicleDetector, VCFG
from run_vehicle_pipeline import draw_vehicles          # System 1's overlay, unchanged


def make_resizer(det, vd, sw, sh, pw, ph, enabled):
    """System 2's ROI-only resizer, widened to the rows the vehicle detector reads (full width, rows yb..h)."""
    rs = V._ProcResizer(det, sw, sh, pw, ph, enabled=enabled)
    if rs.fast and vd is not None:
        y0, y1, x0, x1 = rs.roi
        rs.roi = (min(y0, vd.yb), ph, 0, pw)
    return rs


def stats_ms(a):
    a = np.asarray(a, np.float64) * 1000.0
    if len(a) == 0:
        return {}
    return dict(mean=float(a.mean()), median=float(np.median(a)), p95=float(np.percentile(a, 95)), max=float(a.max()))


def main():
    ap = argparse.ArgumentParser(description="Integrated lane (System 2) + vehicle (System 1) pipeline")
    ap.add_argument("src", nargs="?", help="input video")
    ap.add_argument("dst", nargs="?", help="output video (optional)")
    ap.add_argument("--camera", type=int, default=None, help="camera index instead of a video file")
    ap.add_argument("--show", action="store_true", help="live window (q quits)")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--out-w", type=int, default=1280)
    ap.add_argument("--legacy-vehicles", action="store_true", help="stage-1 vehicle detector only (640 px boxes, bbox-bottom contact), i.e. the pre-upgrade behaviour")
    ap.add_argument("--profile", choices=list(V.PROFILES), default="mountain")
    ap.add_argument("--debug", action="store_true", help="vehicle candidates, rejections, search region")
    ap.add_argument("--debug-lane", action="store_true", help="V6's own debug overlay")
    ap.add_argument("--no-vehicles", action="store_true", help="lane only (System 2 behaviour)")
    ap.add_argument("--policy", choices=["fallback", "suppress"], default=VCFG["unreliable_policy"])
    ap.add_argument("--exclude", type=float, nargs=4, action="append", default=[], metavar=("X0", "Y0", "X1", "Y1"),
                    help="fractions of the frame to blank for the VEHICLE detector (watermark/hood); repeatable")
    ap.add_argument("--cfg", nargs="*", default=[], metavar="KEY=VAL", help="override vehicle_detector.VCFG values")
    ap.add_argument("--top-y", type=float)
    ap.add_argument("--top-x", type=float, nargs=2)
    ap.add_argument("--overlay-top-y", type=float, default=None)
    ap.add_argument("--overlay-top-frac", type=float, default=None)
    ap.add_argument("--threads", choices=["auto", "on", "off"], default="auto",
                    help="decode / display-resize / write on background threads (same frames, same order). auto = on with >= 2 CPUs. "
                         "Ignored (off) for --camera so that no stale frames queue up.")
    ap.add_argument("--display-interp", choices=["area", "linear"], default="area")
    ap.add_argument("--no-roi-resize", action="store_true", help="always do the full-frame processing resize")
    ap.add_argument("--cv-threads", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=30, help="frames excluded from the printed statistics")
    ap.add_argument("--stats-json", default=None, help="write per-stage statistics to this file")
    ap.add_argument("--frame-times-csv", default=None, help="write per-frame stage times (ms) to this CSV")
    a = ap.parse_args()
    if a.cv_threads is not None:
        cv2.setNumThreads(a.cv_threads)
    if a.camera is None and not a.src:
        ap.error("give a video path or --camera N")
    dst = a.dst
    if not dst and not a.show:
        dst = "output_integrated.mp4"

    cap = cv2.VideoCapture(a.camera if a.camera is not None else a.src)
    live = a.camera is not None
    if live:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    elif a.start:
        cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    n_max = int(a.duration * fps_in) if a.duration else 10 ** 9
    ok, f = cap.read()
    if not ok:
        raise SystemExit("cannot read video/camera")
    sh, sw = f.shape[:2]

    cfg = dict(V.CFG)
    cfg.update(V.PROFILES[a.profile])
    if a.top_y is not None:
        cfg["top_y"] = a.top_y
    if a.top_x:
        cfg["top_x"] = tuple(a.top_x)
    pw = cfg["proc_w"]
    ph = int(round(sh * pw / sw))
    ow = a.out_w
    oh = int(round(sh * ow / sw))
    cfg6 = {}
    if a.overlay_top_y is not None:
        cfg6["overlay_top_y"] = a.overlay_top_y
    if a.overlay_top_frac is not None:
        cfg6["overlay_top_frac"] = a.overlay_top_frac
    det = V.PathDetectorV6(pw, ph, cfg, cfg6=cfg6)

    over = {}
    for kv in a.cfg:
        k, v = kv.split("=", 1)
        try:
            over[k] = float(v)
        except ValueError:
            over[k] = v
    over["exclude"] = tuple(tuple(e) for e in a.exclude)
    over["unreliable_policy"] = a.policy
    vd = None if a.no_vehicles else VehicleDetector(pw, ph, getattr(det, "poly", None), over)

    use_thr = (not live) and (a.threads == "on" or (a.threads == "auto" and (os.cpu_count() or 1) >= 2))
    rs = make_resizer(det, vd, sw, sh, pw, ph, enabled=not a.no_roi_resize)
    rs.self_check(f)                 # compares full vs ROI-only resize on the (widened) ROI; falls back to the full resize on mismatch
    interp = cv2.INTER_AREA if a.display_interp == "area" else cv2.INTER_LINEAR
    print("processing %dx%d  display %dx%d  roi-resize=%s threads=%s cv2-threads=%d vehicles=%s"
          % (pw, ph, ow, oh, "on(rows %d-%d)" % (rs.roi[0], rs.roi[1]) if rs.fast else "off", "on" if use_thr else "off",
             cv2.getNumThreads(), "on" if vd else "off"))

    tmp = (dst + ".tmp.mp4") if dst else None
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (ow, oh)) if tmp else None
    rd = V._Prefetch(cap) if use_thr else None
    st = V._Stage(rd.get, lambda fr: cv2.resize(fr, (ow, oh), interpolation=interp)) if use_thr else None
    disp = cv2.resize(f, (ow, oh), interpolation=interp)
    sink = V._AsyncWriter(vw) if (vw and use_thr) else vw

    T = {k: [] for k in ("resize_proc", "lane", "vehicles", "disp_resize", "draw_lane", "draw_veh", "sink", "wait_next", "frame")}
    n = 0
    t_loop0 = time.perf_counter()
    try:
        while f is not None and n < n_max:
            t_f = time.perf_counter()
            small = rs(f)                                       # ONE processing frame, shared by both detectors
            t_a = time.perf_counter()
            L, R = det.process(small)
            t_b = time.perf_counter()
            vf = vd.process(small, det.result, debug=a.debug, full=(None if a.legacy_vehicles else f)) if vd else None
            t_c = time.perf_counter()
            if st is None:
                disp = cv2.resize(f, (ow, oh), interpolation=interp)     # (with threads on this happens on the stage thread)
            t_r = time.perf_counter()
            disp = det.draw(disp, L, R, None, a.debug_lane)
            t_d = time.perf_counter()
            if vf is not None:
                lane_ms = (t_b - t_a) * 1000
                veh_ms = (t_c - t_b) * 1000
                tot = (T["lane"][-30:] + [t_b - t_a]) if T["lane"] else [t_b - t_a]
                tov = (T["vehicles"][-30:] + [t_c - t_b]) if T["vehicles"] else [t_c - t_b]
                fps_algo = 1.0 / max(np.mean(tot) + np.mean(tov), 1e-6)
                disp = draw_vehicles(disp, vf, pw, a.debug, fps_algo, (lane_ms, veh_ms))
            t_e = time.perf_counter()
            if sink:
                sink.write(disp)
            if a.show:
                cv2.imshow("lane + vehicles (q to quit)", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            t_g = time.perf_counter()
            n += 1
            if st is not None:
                nxt = st.get()
                f, disp = nxt if nxt is not None else (None, None)
            else:
                ok, f = cap.read()
                f = f if ok else None
            t_h = time.perf_counter()
            T["resize_proc"].append(t_a - t_f)
            T["lane"].append(t_b - t_a)
            T["vehicles"].append(t_c - t_b)
            T["disp_resize"].append(t_r - t_c)
            T["draw_lane"].append(t_d - t_r)
            T["draw_veh"].append(t_e - t_d)
            T["sink"].append(t_g - t_e)
            T["wait_next"].append(t_h - t_g)         # time blocked on the decoder (threads on) or decode itself (threads off)
            T["frame"].append(t_h - t_f)             # full iteration incl. waiting for the next decoded frame
    finally:
        if st is not None:
            st.close()
        if rd is not None:
            rd.close()
        if sink is not vw and sink is not None:
            sink.close()
    wall = time.perf_counter() - t_loop0
    if vw:
        vw.release()
    cap.release()
    if a.show:
        cv2.destroyAllWindows()
    if dst and tmp:
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", dst])
            os.remove(tmp)
        else:
            os.replace(tmp, dst)
        print("saved", dst)

    k = min(a.warmup, max(n // 4, 0))
    res = {"frames": n, "warmup_excluded": k, "wall_s": round(wall, 3),
           "e2e_fps_wall_incl_warmup": round(n / wall, 2) if wall > 0 else None}
    fr = np.asarray(T["frame"][k:], np.float64)
    if len(fr):
        res["output_fps"] = dict(mean=round(float(1 / fr.mean()), 1), median=round(float(1 / np.median(fr)), 1),
                                 min=round(float(1 / fr.max()), 1), p5=round(float(1 / np.percentile(fr, 95)), 1))
    res["stage_ms"] = {key: stats_ms(T[key][k:]) for key in T}
    print("frames=%d  proc %dx%d" % (n, pw, ph))
    if "output_fps" in res:
        o = res["output_fps"]
        print("output FPS (per-iteration, warm-up excluded): mean %.1f | median %.1f | min %.1f | 5th pct %.1f   [whole run incl. warm-up: %.1f]"
              % (o["mean"], o["median"], o["min"], o["p5"], res["e2e_fps_wall_incl_warmup"]))
    for key in ("resize_proc", "lane", "vehicles", "disp_resize", "draw_lane", "draw_veh", "sink", "wait_next", "frame"):
        s = res["stage_ms"][key]
        if s:
            print("  %-12s mean %6.2f  median %6.2f  p95 %6.2f  max %7.2f ms" % (key, s["mean"], s["median"], s["p95"], s["max"]))
    if a.stats_json:
        import json
        json.dump(res, open(a.stats_json, "w"), indent=1)
    if a.frame_times_csv:
        with open(a.frame_times_csv, "w") as fh:
            keys = list(T)
            fh.write("frame," + ",".join(keys) + "\n")
            for i in range(n):
                fh.write("%d," % i + ",".join("%.4f" % (T[kk][i] * 1000) for kk in keys) + "\n")


if __name__ == "__main__":
    main()
