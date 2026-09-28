import cv2
import sys
import time
import csv
from detector import RoadObjectDetector, Params

# color per source, for visual debugging: green=motion, cyan=static, yellow=both
SRC_COLOR = {"motion": (0, 255, 0), "static": (255, 255, 0), "motion+static": (0, 255, 255)}


def main():
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    csv_path = sys.argv[3]
    max_frames = int(sys.argv[4]) if len(sys.argv) > 4 else None

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print("Could not open", in_path)
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps_in, (w, h))

    p = Params()
    det = RoadObjectDetector((h, w, 3), p)

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "contact_x", "contact_y", "source"])

    frame_idx = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        confirmed, fused, inter, flow_viz = det.process(frame)

        vis = frame.copy()
        for t in confirmed:
            x1, y1, x2, y2 = [int(v) for v in t.box]
            cx, cy = [int(v) for v in t.contact]
            color = SRC_COLOR.get(t.sources, (0, 255, 0))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(vis, f"Obj{t.id}", (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            csv_writer.writerow([frame_idx, t.id, x1, y1, x2, y2, cx, cy, t.sources])

        cv2.putText(vis, f"frame {frame_idx}  objects={len(confirmed)}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, "green=motion  cyan=static  yellow=both", (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(vis)

        # Live display
        cv2.imshow("Live Detection - press q to quit", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_idx % 50 == 0:
            elapsed = time.time() - t0
            print(f"frame {frame_idx}  ({frame_idx/elapsed:.1f} fps so far)")

    cap.release()
    writer.release()
    csv_file.close()
    cv2.destroyAllWindows()
    elapsed = time.time() - t0
    print(f"Done. {frame_idx} frames in {elapsed:.1f}s -> {frame_idx/elapsed:.2f} FPS (CPU, {w}x{h})")


if __name__ == "__main__":
    main()
