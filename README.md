# High-Performance Classical Lane Detection — Best Lane (~29 FPS)

Single-file standalone classical lane detector (**no deep learning, no weights, pure OpenCV + NumPy**).

This version (`lane_path_v6_p4`) provides our **best lane detection performance**, reaching up to **~29 FPS** via:
- **Pipelined Multi-threading**: Background worker threads for decoding and display resizing running in parallel with the detection loop.
- **Fast Display Interpolation**: Optional `--display-interp linear` for ~5x faster background frame resizing without affecting detection accuracy.
- **ROI-Only Processing Resize**: Downscaling computed only for the required road region rather than the entire full-resolution frame.
- **Robust Classical Geometry**: Solid/dashed/dotted line styling, bounded corridor clipping at the first invalid row, and one-sided fallback.

---

## Quick Start (`lane_path_v6_p4`)

```bash
cd lane_path_v6_p4
source ../.venv/bin/activate
```

### 1. High-Performance Live Preview (~29 FPS)
```bash
python3 lane_path_v6_p3.py --source videos/1.mp4 --display-interp linear --show
```
*(Press `q` to quit)*

### 2. Standard Live Preview
```bash
python3 lane_path_v6_p3.py --source videos/1.mp4 --show
```

### 3. Debug View (Edges, Hough Lines, Corridor)
```bash
python3 lane_path_v6_p3.py --source videos/1.mp4 --show --debug
```

### 4. Save Video Output
```bash
python3 lane_path_v6_p3.py --source videos/1.mp4 --save output_v6_p4.mp4
```

---

## Tag History
- `lane_best_29fps` / `v10_lane_best_29fps`: High-performance pipelined lane detector (~29 FPS).
- `v9_final_best_lane` / `v6_final_best_lane`: Standalone classical lane path detector (lane_path_v6).
- `v8_lane`: Baseline single-file classical lane detector (`lane_path.py`).
- `v7_best`: Adaptive lane and vehicle detection (model-free).
