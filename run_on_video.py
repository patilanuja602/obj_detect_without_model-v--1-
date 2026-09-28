import argparse
import csv
import queue
import sys
import threading
import time

import cv2

from detector import RoadObjectDetector, Params
from geometry import GroundGeometry

# color per source, for visual debugging: green=motion, cyan=static, yellow=both
SRC_COLOR = {"motion": (0, 255, 0), "static": (255, 255, 0), "motion+static": (0, 255, 255)}


def _reader(cap, q, stop):
    """Decode frames on a background thread so decoding overlaps detection."""
    try:
        while not stop.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            while not stop.is_set():
                try:
                    q.put(frame, timeout=0.1)
                    break
                except queue.Full:
                    continue
    finally:
        while True:                       # always deliver the end-of-stream marker
            try:
                q.put(None, timeout=0.1)
                return
            except queue.Full:
                if stop.is_set():
                    return


def _writer(writer, q):
    """Encode annotated frames on a background thread."""
    while True:
        vis = q.get()
        if vis is None:
            return
        writer.write(vis)


def main():
    ap = argparse.ArgumentParser(description="Run the road-object detector on a video.")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("csv")
    ap.add_argument("max_frames", nargs="?", type=int, default=None)
    ap.add_argument("--no-display", action="store_true",
                    help="don't open the live preview window (faster / headless)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("Could not open", args.input)
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps_in, (w, h))

    p = Params()
    det = RoadObjectDetector((h, w, 3), p)
    geo = GroundGeometry(w, h)
    last_print_str = ""

    csv_file = open(args.csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "contact_x", "contact_y", "source",
                         "lateral_m", "forward_m", "angle_deg"])

    # background decode / encode threads (cv2 releases the GIL in read/write)
    stop = threading.Event()
    in_q = queue.Queue(maxsize=16)
    out_q = queue.Queue(maxsize=16)
    t_in = threading.Thread(target=_reader, args=(cap, in_q, stop), daemon=True)
    t_out = threading.Thread(target=_writer, args=(writer, out_q), daemon=True)
    t_in.start()
    t_out.start()

    show = not args.no_display
    frame_idx = 0
    n_done = 0
    t_detect = 0.0
    t0 = time.time()
    while True:
        frame = in_q.get()
        if frame is None:
            break
        frame_idx += 1
        if args.max_frames and frame_idx > args.max_frames:
            break

        td = time.perf_counter()
        confirmed, fused, inter, flow_viz = det.process(frame)
        t_detect += time.perf_counter() - td
        n_done += 1

        vis = frame.copy()
        terminal_lines = []
        for t in confirmed:
            x1, y1, x2, y2 = [int(v) for v in t.box]
            cx, cy = [int(v) for v in t.contact]
            color = SRC_COLOR.get(t.sources, (0, 255, 0))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(vis, f"Obj{t.id}", (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            g = geo.locate(*t.contact)
            if g is not None:
                side_ab, side_cb, angle_deg = g
                terminal_lines.append(
                    f"Obj {t.id}: Lateral Distance (AB) = {side_ab:+.3f} m | "
                    f"Forward Distance (CB) = {side_cb:.3f} m | "
                    f"Horizontal Angle = {angle_deg:+.2f} deg")
                geo_cols = [f"{side_ab:.4f}", f"{side_cb:.4f}", f"{angle_deg:.3f}"]
            else:
                geo_cols = ["", "", ""]
            csv_writer.writerow([frame_idx, t.id, x1, y1, x2, y2, cx, cy, t.sources] + geo_cols)

        # print distances/angles in the terminal only when they change
        current_print_str = "\n".join(terminal_lines)
        if current_print_str and current_print_str != last_print_str:
            print("-" * 75)
            print(current_print_str)
            last_print_str = current_print_str

        cv2.putText(vis, f"frame {frame_idx}  objects={len(confirmed)}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, "green=motion  cyan=static  yellow=both", (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        out_q.put(vis)

        # Live display (main thread -- GUI calls must not run in a worker thread)
        if show:
            try:
                cv2.imshow("Live Detection - press q to quit", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error:
                print("No display available -- continuing without live preview "
                      "(use --no-display to silence this).")
                show = False

        if frame_idx % 50 == 0:
            elapsed = time.time() - t0
            print(f"frame {frame_idx}  ({frame_idx/elapsed:.1f} fps so far)")

    stop.set()
    out_q.put(None)
    t_out.join()
    t_in.join(timeout=2.0)
    cap.release()
    writer.release()
    csv_file.close()
    if show:
        cv2.destroyAllWindows()
    elapsed = time.time() - t0
    n = max(n_done, 1)
    print(f"Done. {n} frames in {elapsed:.1f}s -> {n/elapsed:.2f} FPS end-to-end "
          f"({n/max(t_detect, 1e-9):.2f} FPS detector-only, CPU, {w}x{h})")


if __name__ == "__main__":
    main()
