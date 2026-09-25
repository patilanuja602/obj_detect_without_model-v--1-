import cv2
import numpy as np
import glob, os, time, json
from detector import RoadObjectDetector, Params

SEQ_DIR = "/home/claude/work/seq"
OUT_DIR = "/home/claude/work/out"
os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(SEQ_DIR, "f_*.png")))
print("total frames:", len(files))

first = cv2.imread(files[0])
h, w = first.shape[:2]

p = Params()
det = RoadObjectDetector(first.shape, p)

# frame indices (1-based, matches filenames) we want full visualization for
viz_frames = set()
for a, b in [(1, 40), (440, 470), (500, 520), (840, 880), (1230, 1270), (1340, 1370)]:
    viz_frames.update(range(a, b + 1))

log = []
t0 = time.time()
n_processed = 0

for i, f in enumerate(files, start=1):
    frame = cv2.imread(f)
    confirmed, raw_dets, inter, flow_viz = det.process(frame)
    n_processed += 1

    log.append({"frame": i, "n_confirmed": len(confirmed), "n_raw": len(raw_dets)})

    if i in viz_frames:
        vis = frame.copy()
        for j, tr in enumerate(confirmed):
            x1, y1, x2, y2 = [int(v) for v in tr.box]
            cx, cy = [int(v) for v in tr.contact]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(vis, f"id{tr.id}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"frame {i}  objects={len(confirmed)}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT_DIR, f"frame_{i:05d}_final.png"), vis)

        fg = cv2.cvtColor(inter["fg_clean"], cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(OUT_DIR, f"frame_{i:05d}_mask.png"), fg)

        mog2raw = cv2.cvtColor(inter["mog2_raw"], cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(OUT_DIR, f"frame_{i:05d}_mog2raw.png"), mog2raw)

        if flow_viz is not None:
            pts, flow = flow_viz
            fvis = frame.copy()
            for (x, y), (dx, dy) in zip(pts, flow):
                x, y = int(x), int(y)
                cv2.arrowedLine(fvis, (x, y), (int(x + dx * 4), int(y + dy * 4)),
                                 (0, 200, 255), 1, tipLength=0.35)
            cv2.imwrite(os.path.join(OUT_DIR, f"frame_{i:05d}_flow.png"), fvis)

t1 = time.time()
elapsed = t1 - t0
fps = n_processed / elapsed
print(f"Processed {n_processed} frames in {elapsed:.1f}s -> {fps:.2f} FPS (CPU, {w}x{h})")

with open(os.path.join(OUT_DIR, "log.json"), "w") as fh:
    json.dump({"fps": fps, "elapsed_sec": elapsed, "n_frames": n_processed, "per_frame": log}, fh)
