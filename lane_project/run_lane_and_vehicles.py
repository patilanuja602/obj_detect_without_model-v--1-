#!/usr/bin/env python3
"""Lane detection (unchanged) + vehicle detection (new, no model) on a video, camera or image.

  python run_lane_and_vehicles.py --source road.mp4 --show
  python run_lane_and_vehicles.py --source road.mp4 --save out.mp4
  python run_lane_and_vehicles.py --source 0 --show            # webcam
Same lane options as run_lane_detection.py, plus the vehicle options below.
"""
import argparse
import csv
import os
import time

import cv2

from lane_detector import Config, LaneDetector
from lane_detector.overlay import draw
from vehicle_detector import VConfig, VehicleDetector, draw_vehicles


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--save")
    p.add_argument("--show", action="store_true")
    p.add_argument("--debug", action="store_true", help="lane debug + vehicle search area and rejected blobs")
    p.add_argument("--proc-width", type=int, default=640)
    p.add_argument("--out-width", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--car-x", type=float, default=0.5)
    p.add_argument("--lane-width-m", type=float, default=3.7)
    # vehicle options
    p.add_argument("--max-distance", type=float, default=45.0, help="ignore vehicles farther than ~this many metres")
    p.add_argument("--side-lanes", type=int, default=3, help="lanes searched on each side of the ego lane")
    p.add_argument("--hood", type=float, default=0.93, help="rows below this height fraction are your own bonnet")
    p.add_argument("--log-csv")
    return p.parse_args()


def main():
    a = parse()
    cfg = Config(proc_width=a.proc_width, car_x_frac=a.car_x, lane_width_m=a.lane_width_m)
    lane = LaneDetector(cfg)                                   # lane code: used as-is
    veh = VehicleDetector(VConfig(max_distance_m=a.max_distance, side_lanes=a.side_lanes,
                                  hood_frac=a.hood, lane_width_m=a.lane_width_m))

    is_cam = a.source.isdigit()
    is_img = (not is_cam) and a.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))

    def render(res, vehicles, fps=None):
        out = draw(res, a.out_width, a.debug, fps)
        sc = 1.0 if not a.out_width else a.out_width / res.frame.shape[1]
        if a.debug and "roi" in veh.debug:
            cnt, _ = cv2.findContours(veh.debug["roi"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnt:
                cv2.polylines(out, [(c * sc).astype("int32")], True, (255, 255, 0), 1)
            for x1, y1, x2, y2, why in veh.debug["rejected"]:
                cv2.rectangle(out, (int(x1 * sc), int(y1 * sc)), (int(x2 * sc), int(y2 * sc)), (120, 120, 120), 1)
                cv2.putText(out, why, (int(x1 * sc), int(y2 * sc) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        draw_vehicles(out, vehicles, sc)
        return out

    if is_img:
        img = cv2.imread(a.source)
        res = lane.process(img)
        for _ in range(4):                                    # let the tiny tracker confirm on a still image
            vs = veh.process(res)
        out = render(res, vs)
        if a.save:
            cv2.imwrite(a.save, out)
        if a.show:
            cv2.imshow("lanes + vehicles", out)
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
        log_w.writerow(["frame", "id", "side", "dist_m", "x1", "y1", "x2", "y2"])

    n, t_prev, fps_ema, t_start = 0, time.time(), None, time.time()
    while True:
        ok, frame = cap.read()
        if not ok or (a.max_frames and n >= a.max_frames):
            break
        res = lane.process(frame)
        vs = veh.process(res)
        now = time.time()
        fps_ema = 1 / max(now - t_prev, 1e-6) if fps_ema is None else 0.9 * fps_ema + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        out = render(res, vs, fps_ema)
        if a.save and writer is None:
            os.makedirs(os.path.dirname(os.path.abspath(a.save)), exist_ok=True)
            writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), in_fps, (out.shape[1], out.shape[0]))
        if writer is not None:
            writer.write(out)
        if log_w:
            for v in vs:
                log_w.writerow([n, v.id, v.side, None if v.dist_m is None else round(v.dist_m, 1), *v.box])
        if a.show:
            cv2.imshow("lanes + vehicles (q = quit)", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        n += 1
    cap.release()
    if writer:
        writer.release()
    if log_f:
        log_f.close()
    dt = time.time() - t_start
    print(f"{n} frames in {dt:.1f}s -> {n / max(dt, 1e-6):.1f} FPS average")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
