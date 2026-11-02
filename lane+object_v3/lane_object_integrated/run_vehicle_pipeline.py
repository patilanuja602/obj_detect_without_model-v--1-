#!/usr/bin/env python3
"""
run_vehicle_pipeline.py -- lane_path_v6 (unmodified) + vehicle_detector on a video file or a camera.

V6 runs exactly ONCE per frame.  The same processing-resolution frame and V6's `det.result` are handed to the vehicle detector.
All vehicle coordinates are in V6 PROCESSING-frame pixels; they are scaled by (display_width / processing_width) only when drawing.

Examples
  python3 run_vehicle_pipeline.py video.mp4 out.mp4                    # write annotated video
  python3 run_vehicle_pipeline.py video.mp4 --show                     # live window (q quits)
  python3 run_vehicle_pipeline.py --camera 0 --show                    # webcam / car camera
  python3 run_vehicle_pipeline.py video.mp4 --benchmark 900            # algorithm-only timing, decode/display excluded
  python3 run_vehicle_pipeline.py video.mp4 --benchmark 900 --no-vehicles    # V6-only baseline, same frames
  python3 run_vehicle_pipeline.py video.mp4 out.mp4 --debug --exclude 0.68 0.82 1 1     # also show candidates/rejections; blank a watermark
"""
import argparse, json, os, shutil, subprocess, sys, time

import cv2
import numpy as np

import lane_path_v6 as V
from vehicle_detector import VehicleDetector, VCFG

STATE_COL = {"TRACKED": (60, 220, 60), "ACCEPTED": (0, 215, 255), "PREDICTED": (255, 200, 0), "CANDIDATE": (0, 140, 255)}


def _dashed_rect(img, p0, p1, col, step=5):
    x0, y0 = p0
    x1, y1 = p1
    for x in range(x0, x1, step * 2):
        cv2.line(img, (x, y0), (min(x + step, x1), y0), col, 1)
        cv2.line(img, (x, y1), (min(x + step, x1), y1), col, 1)
    for y in range(y0, y1, step * 2):
        cv2.line(img, (x0, y), (x0, min(y + step, y1)), col, 1)
        cv2.line(img, (x1, y), (x1, min(y + step, y1)), col, 1)


def draw_vehicles(disp, vf, proc_w, debug=False, fps=None, timing=None):
    """Draw on a display-resolution frame; coordinates are scaled from processing space here and ONLY here."""
    s = disp.shape[1] / float(proc_w)
    if debug:
        if vf.region is not None:
            cv2.polylines(disp, [(vf.region * s).astype(np.int32)], True, (255, 255, 0), 1)
        for d in vf.rejected:
            x0, y0, x1, y1 = (int(v * s) for v in d.box)
            cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 0, 160), 1)
            cv2.putText(disp, d.reject[:8], (x0, max(y0 - 2, 8)), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 220), 1)
        for d in vf.candidates:
            x0, y0, x1, y1 = (int(v * s) for v in d.box)
            cv2.rectangle(disp, (x0, y0), (x1, y1), STATE_COL["CANDIDATE"], 1)
    for d in vf.detections:
        x0, y0, x1, y1 = (int(round(v * s)) for v in d.box)
        col = STATE_COL[d.state]
        if d.state == "PREDICTED":
            _dashed_rect(disp, (x0, y0), (x1, y1), col)
        else:
            cv2.rectangle(disp, (x0, y0), (x1, y1), col, 2 if d.state == "TRACKED" else 1)
        cx, cy = int(round(d.contact[0] * s)), int(round(d.contact[1] * s))
        q = getattr(d, "contact_quality", "BBOX_BOTTOM")
        if q == "UNCERTAIN":                                  # no reliable tyre contact: hollow grey ring = coarse hint only
            cv2.circle(disp, (cx, cy), 6, (170, 170, 170), 1, cv2.LINE_AA)
        elif q == "ONE_TYRE":                                 # single visible contact: hollow orange ring (reduced confidence)
            cv2.circle(disp, (cx, cy), 4, (0, 165, 255), 2, cv2.LINE_AA)
        else:                                                 # axle ground point (two tyres) or legacy bbox bottom
            cv2.circle(disp, (cx, cy), 3, (0, 0, 255), -1, cv2.LINE_AA)
        if getattr(d, "src_scale", 0):                        # individual tyre contacts (source px -> display px)
            k = s / d.src_scale
            for (tx, ty) in (getattr(d, "tyres_src", None) or []):
                cv2.drawMarker(disp, (int(round(tx * k)), int(round(ty * k))), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 7, 1)
        lane = {"IN_CORRIDOR": "IN", "LEFT_OF_CORRIDOR": "L", "RIGHT_OF_CORRIDOR": "R", "AMBIGUOUS": "?", "UNKNOWN": "-"}[d.lane]
        txt = "#%d %s %.2f" % (d.track_id, lane, d.score)
        cv2.putText(disp, txt, (x0, max(y0 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    hud = ["V6 %s | road ref %s | scale %s" % (vf.lane_state, vf.road_ref, vf.scale_source or "-"),
           "vehicles: %d (%d observed)" % (len(vf.detections), sum(d.observed for d in vf.detections))]
    if timing:
        hud.append("algo %.1f ms (V6 %.1f + veh %.1f)  %s" % (timing[0] + timing[1], timing[0], timing[1],
                                                                ("%.0f FPS" % fps) if fps else ""))
    if vf.notes:
        hud.append(vf.notes[:70])
    for i, t in enumerate(hud):
        cv2.putText(disp, t, (10, 56 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(disp, t, (10, 56 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return disp


def stats(a):
    a = np.asarray(a, np.float64) * 1000.0
    return dict(median=float(np.median(a)), p95=float(np.percentile(a, 95)), mean=float(a.mean()), max=float(a.max()), n=int(len(a)))


def build(args, sw, sh):
    cfg6 = dict(V.CFG)
    cfg6.update(V.PROFILES[args.profile])
    pw = cfg6["proc_w"]
    ph = int(round(sh * pw / sw))
    det = V.PathDetectorV6(pw, ph, cfg6)
    over = {}
    for kv in args.cfg:
        k, v = kv.split("=", 1)
        try:
            over[k] = float(v)
        except ValueError:
            over[k] = v
    over["exclude"] = tuple(tuple(e) for e in args.exclude)
    over["unreliable_policy"] = args.policy
    vd = None if args.no_vehicles else VehicleDetector(pw, ph, getattr(det, "poly", None), over)
    return det, vd, pw, ph


def run_benchmark(args):
    cap = cv2.VideoCapture(args.src)
    if args.start:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    ok, f = cap.read()
    if not ok:
        raise SystemExit("cannot read %s" % args.src)
    sh, sw = f.shape[:2]
    det, vd, pw, ph = build(args, sw, sh)
    frames, t_dec = [], 0.0
    t0 = time.perf_counter()
    while ok and len(frames) < args.benchmark:                       # decode + resize BEFORE timing the algorithms
        frames.append(cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA))
        ok, f = cap.read()
    t_dec = time.perf_counter() - t0
    n = len(frames)
    warm = min(args.warmup, n // 4)
    t_v6, t_cand, t_score, t_track, t_veh, t_tot = [], [], [], [], [], []
    states = {}
    t_wall0 = time.perf_counter()
    for i, small in enumerate(frames):
        a = time.perf_counter()
        det.process(small)
        b = time.perf_counter()
        vf = vd.process(small, det.result) if vd else None
        c = time.perf_counter()
        st = det.result["lane"]["state"]
        if i >= warm:
            states[st] = states.get(st, 0) + 1
            t_v6.append(b - a)
            t_tot.append(c - a)
            if vf:
                t_cand.append(vf.timing["candidates"] / 1000)
                t_score.append(vf.timing["scoring"] / 1000)
                t_track.append(vf.timing["tracking"] / 1000)
                t_veh.append(c - b)
    wall = time.perf_counter() - t_wall0
    res = dict(video=os.path.basename(args.src), proc_size=[pw, ph], frames_decoded=n, warmup_frames_excluded=warm,
               frames_measured=n - warm, decode_resize_s=round(t_dec, 2), v6_state_counts=states,
               v6_ms=stats(t_v6), total_algorithm_ms=stats(t_tot),
               end_to_end_algorithm_fps_mean=round((n - warm) / sum(t_tot), 1),
               wall_loop_s_including_warmup=round(wall, 2), vehicles_enabled=bool(vd))
    if vd:
        res.update(vehicle_candidates_ms=stats(t_cand), vehicle_scoring_ms=stats(t_score), vehicle_tracking_ms=stats(t_track),
                   vehicle_total_ms=stats(t_veh))
    print(json.dumps(res, indent=1))
    if args.bench_json:
        json.dump(res, open(args.bench_json, "w"), indent=1)


def run_stream(args):
    cap = cv2.VideoCapture(args.camera if args.camera is not None else args.src)
    if args.camera is None and args.start:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    ok, f = cap.read()
    if not ok:
        raise SystemExit("cannot read video/camera")
    sh, sw = f.shape[:2]
    det, vd, pw, ph = build(args, sw, sh)
    ow = args.out_w
    oh = int(round(sh * ow / sw))
    dst = args.dst
    tmp = (dst + ".tmp.mp4") if dst else None
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (ow, oh)) if tmp else None
    n_max = int(args.duration * fps_in) if args.duration else 10 ** 9
    t6, tv, tall = [], [], []
    n = 0
    while ok and n < n_max:
        t_loop = time.perf_counter()
        small = cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA)
        a = time.perf_counter()
        L, R = det.process(small)
        b = time.perf_counter()
        vf = vd.process(small, det.result, debug=args.debug) if vd else None
        c = time.perf_counter()
        t6.append(b - a)
        tv.append(c - b)
        disp = cv2.resize(f, (ow, oh), interpolation=cv2.INTER_AREA)
        fps = 1.0 / max(np.mean(np.add(t6[-30:], tv[-30:])), 1e-6)
        disp = det.draw(disp, L, R, None, args.debug_lane)
        if vf:
            disp = draw_vehicles(disp, vf, pw, args.debug, fps, (t6[-1] * 1000, tv[-1] * 1000))
        if vw:
            vw.write(disp)
        if args.show:
            cv2.imshow("lane + vehicles (q to quit)", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        tall.append(time.perf_counter() - t_loop)
        n += 1
        ok, f = cap.read()
    if vw:
        vw.release()
    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    if dst:
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", dst])
            os.remove(tmp)
        else:
            os.replace(tmp, dst)
        print("saved", dst)
    k = min(5, max(len(t6) - 1, 0))
    s6, sv = stats(t6[k:]), stats(tv[k:]) if vd else None
    print("frames=%d  proc %dx%d" % (n, pw, ph))
    print("V6            median %.2f ms  p95 %.2f ms" % (s6["median"], s6["p95"]))
    if vd:
        print("vehicles      median %.2f ms  p95 %.2f ms" % (sv["median"], sv["p95"]))
    print("loop incl. decode/resize/draw/write/show: median %.2f ms -> %.1f FPS (NOT the algorithm-only figure)"
          % (np.median(tall[k:]) * 1000, 1.0 / np.mean(tall[k:])))


def main():
    ap = argparse.ArgumentParser(description="V6 lane detector + model-free vehicle detector")
    ap.add_argument("src", nargs="?", help="input video")
    ap.add_argument("dst", nargs="?", help="output video (optional)")
    ap.add_argument("--camera", type=int, default=None, help="camera index instead of a video file")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--out-w", type=int, default=1280)
    ap.add_argument("--profile", choices=list(V.PROFILES), default="mountain")
    ap.add_argument("--debug", action="store_true", help="show vehicle candidates, rejections, search region")
    ap.add_argument("--debug-lane", action="store_true", help="V6's own debug overlay")
    ap.add_argument("--no-vehicles", action="store_true", help="run V6 only (baseline)")
    ap.add_argument("--policy", choices=["fallback", "suppress"], default=VCFG["unreliable_policy"])
    ap.add_argument("--exclude", type=float, nargs=4, action="append", default=[], metavar=("X0", "Y0", "X1", "Y1"),
                    help="fractions of the frame to blank out (watermark/hood); repeatable")
    ap.add_argument("--cfg", nargs="*", default=[], metavar="KEY=VAL", help="override vehicle_detector.VCFG values")
    ap.add_argument("--benchmark", type=int, default=0, help="N frames: algorithm-only timing (decode excluded)")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--bench-json", default=None)
    a = ap.parse_args()
    if a.camera is None and not a.src:
        ap.error("give a video path or --camera N")
    if a.benchmark:
        if a.camera is not None:
            ap.error("--benchmark needs a video file")
        run_benchmark(a)
    else:
        if not a.dst and not a.show:
            a.dst = "output_vehicles.mp4"
        run_stream(a)


if __name__ == "__main__":
    main()
