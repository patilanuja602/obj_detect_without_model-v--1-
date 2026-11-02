# Classical Lane & Vehicle Detection Pipeline (Pure OpenCV + NumPy)

Real-time classical computer vision pipeline (**no deep learning, no neural network weights, CPU only**).

---

## Latest Version: Lane + Vehicle Detection (~9 FPS) — `v11_lane_obj_9fps`

Located in [`lane+object_v3/`](lane+object_v3/):
- **Synchronous Dual Detection**: Pipelined V6 lane detector + model-free vehicle detector running in a unified loop (`run_integrated.py`).
- **Sub-Pixel Vehicle Refinement (`vehicle_refine.py`)**: Full-resolution under-body edge profiling, tyre ground-contact points (`TWO_TYRES` / `ONE_TYRE`), symmetry & side-edge filtering, and source-pixel Kalman tracking.
- **Visual Ground Contacts**: Green dot = two tyres detected, orange ring = one tyre, grey = uncertain contact hint.
- **Multi-threaded Architecture**: Background worker threads for decoding and display resizing.

### Usage (`lane+object_v3`):
```bash
cd "lane+object_v3"
source ../.venv/bin/activate

# 1. Live Preview (Lanes + Vehicles with tyre ground-contacts)
python3 run_integrated.py videos/1.mp4 --show

# 2. High-Performance Display
python3 run_integrated.py videos/1.mp4 --display-interp linear --show

# 3. Live Preview with Debug Overlays
python3 run_integrated.py videos/1.mp4 --show --debug

# 4. Save Video Output
python3 run_integrated.py videos/1.mp4 output_v11.mp4
```

---

## Lane-Only Detection: Best Lane (~29 FPS) — `v10_lane_best_29fps`

Located in [`lane_path_v6_p4/`](lane_path_v6_p4/):
- High-performance pipelined lane detector reaching up to **~29 FPS**.
- Multi-threaded display resize and background decode/write pipelining.
- ROI-only processing resize & bounding-box fill blending.

```bash
cd lane_path_v6_p4
source ../.venv/bin/activate

python3 lane_path_v6_p3.py --source videos/1.mp4 --display-interp linear --show
```

---

## Tag History
- `v11_lane_obj_9fps`: Best integrated lane + vehicle detector with tyre contacts (~9 FPS, `lane+object_v3`).
- `v10_lane_best_29fps`: High-performance pipelined lane detector (~29 FPS, `lane_path_v6_p4`).
- `v9_final_best_lane`: Standalone classical lane path detector (`lane_path_v6`).
- `v8_lane`: Baseline single-file classical lane detector (`lane_path.py`).
- `v7_best`: Adaptive lane and vehicle detection (model-free).
