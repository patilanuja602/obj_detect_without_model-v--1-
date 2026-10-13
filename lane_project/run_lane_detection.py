#!/usr/bin/env python3
"""Command-line runner: video file, webcam, or single image.

Examples
  python run_lane_detection.py --source road.mp4 --show
  python run_lane_detection.py --source road.mp4 --save out.mp4
  python run_lane_detection.py --source 0 --show            # webcam
  python run_lane_detection.py --source photo.jpg --save out.jpg
"""
import argparse
import csv
import os
import time

import cv2

from lane_detector import Config, LaneDetector
from lane_detector.overlay import draw


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="video path, image path, or camera index (0, 1, ...)")
    p.add_argument("--save", help="output video/image path")
    p.add_argument("--show", action="store_true", help="open a live window (press q to quit)")
    p.add_argument("--debug", action="store_true", help="draw horizon, raw segments and marking mask")
    p.add_argument("--proc-width", type=int, default=640, help="processing width (smaller = faster)")
    p.add_argument("--out-width", type=int, default=None, help="width of the rendered output")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--skip", type=int, default=0, help="skip this many frames at the start")
    p.add_argument("--car-x", type=float, default=0.5, help="car position across the image, 0..1 (0.5 = centred camera)")
    p.add_argument("--calibrate", action="store_true", help="click on the first frame where your car's centre is, then run")
    p.add_argument("--lane-width-m", type=float, default=3.7)
    p.add_argument("--log-csv", help="write per-frame offset / steering to this CSV")
    return p.parse_args()


def calibrate_car_x(source):
    """Show the first frame; the user clicks the car's centre-line at the bottom of the image."""
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Cannot read a frame for calibration")
    h, w = frame.shape[:2]
    scale = 960 / w
    view = cv2.resize(frame, None, fx=scale, fy=scale)
    picked = []

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            picked.append(x / view.shape[1])

    cv2.putText(view, "Click the CENTRE of your car (e.g. middle of the bonnet), then press any key",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.imshow("calibrate", view)
    cv2.setMouseCallback("calibrate", on_mouse)
    while not picked:
        if cv2.waitKey(30) & 0xFF == 27:
            break
    cv2.destroyWindow("calibrate")
    if not picked:
        return 0.5
    print(f"car-x = {picked[-1]:.3f}   (reuse with --car-x {picked[-1]:.3f})")
    return picked[-1]


def main():
    a = parse()
    if a.calibrate:
        a.car_x = calibrate_car_x(a.source)
    cfg = Config(proc_width=a.proc_width, car_x_frac=a.car_x, lane_width_m=a.lane_width_m)
    det = LaneDetector(cfg)

    is_cam = a.source.isdigit()
    is_img = (not is_cam) and a.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))

    if is_img:
        img = cv2.imread(a.source)
        res = det.process(img)
        out = draw(res, a.out_width, a.debug)
        if a.save:
            cv2.imwrite(a.save, out)
        if a.show:
            cv2.imshow("lanes", out)
            cv2.waitKey(0)
        return

    cap = cv2.VideoCapture(int(a.source) if is_cam else a.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {a.source}")
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    for _ in range(a.skip):
        cap.read()

    writer, log_f, log_w = None, None, None
    if a.log_csv:
        os.makedirs(os.path.dirname(os.path.abspath(a.log_csv)), exist_ok=True)
        log_f = open(a.log_csv, "w", newline="")
        log_w = csv.writer(log_f)
        log_w.writerow(["frame", "has_lane", "offset_m", "steer_deg", "horizon_y_frac", "curve", "confidence"])

    n, t_prev, fps_ema, t_start = 0, time.time(), None, time.time()
    while True:
        ok, frame = cap.read()
        if not ok or (a.max_frames and n >= a.max_frames):
            break
        res = det.process(frame)
        now = time.time()
        inst = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now
        fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst
        out = draw(res, a.out_width, a.debug, fps_ema)

        if a.save and writer is None:
            os.makedirs(os.path.dirname(os.path.abspath(a.save)), exist_ok=True)
            writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), in_fps,
                                     (out.shape[1], out.shape[0]))
        if writer is not None:
            writer.write(out)
        if log_w:
            hy = None if res.vp is None else res.vp[1] / res.frame.shape[0]
            log_w.writerow([n, int(res.has_lane), res.offset_m, res.steer_deg, hy, res.curve,
                            round(res.confidence, 3)])
        if a.show:
            cv2.imshow("lanes (q = quit)", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        n += 1

    cap.release()
    if writer:
        writer.release()
    if log_f:
        log_f.close()
    dt = time.time() - t_start
    print(f"{n} frames in {dt:.1f}s  ->  {n / max(dt, 1e-6):.1f} FPS average (incl. video decode + drawing)")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
