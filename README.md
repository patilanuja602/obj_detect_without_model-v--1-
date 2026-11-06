# Classical Lane & Vehicle Detection Pipeline (Pure OpenCV + NumPy)

Real-time computer vision pipeline for lane detection and vehicle perception.

---

## Latest Version: Optimized Lane + Model Vehicle Detection — `v13_lane+model_obj_v3`

Located in [`lane+model_obj_v3/`](lane+model_obj_v3/):
- **Synchronous Dual Pipeline**: V6 lane detection (unmodified classical) combined with real-time YOLO26n vehicle detection via OpenVINO.
- **Hardware-Accelerated Preprocessing**: Native C++ OpenVINO `PrePostProcessor` integration (`ov_preprocess: true`) eliminating Python array copies.
- **Advanced Tracking & Deduplication**: Alpha-Beta velocity tracking (`vel_beta: 0.25`), IoU + center distance matching, class-switch hysteresis, and coasting with dashed boxes to prevent flickering.
- **Occlusion-Aware Display**: Restores vehicle bodies over lane overlays while preserving lane markings at the tyre ground-contact line.
- **Dual HUD Display**: Shows both lane algorithm rate (`25 FPS (algo)`) and genuine end-to-end throughput (`end-to-end 6 FPS`).

### Usage (`lane+model_obj_v3`):
```bash
cd "lane+model_obj_v3"
source ../.venv/bin/activate

# 1. Live Preview (Complete Video)
bash run.sh videos/1.mp4 --show

# 2. Faster Detector Mode (Higher FPS)
bash run.sh videos/1.mp4 --show --set detector.config=B2

# 3. Save Output Video to File
bash run.sh videos/1.mp4 --save results/output.mp4
```

---

## Previous Version: Lane + Model Vehicle Detection (~25 FPS) — `v12_lane+model_v2_25fps`

Located in [`lane+model_obj_v2/`](lane+model_obj_v2/):
- **Synchronous Dual Pipeline**: V6 lane detection (unmodified classical) combined with real-time YOLO26n vehicle detection via OpenVINO.
- **Enhanced Vehicle Detection**: Duplicate-free bounding boxes (class-agnostic NMS merge + class voting), vehicle ground-contact points, road zone association (`CUR`, `L`, `R`, `OUT`, `?`).
- **Vehicle-Aware Overlay**: Automatically cuts away lane overlay under vehicle bodies.
- **High Performance**: End-to-end processing at ~25 FPS with multithreaded OpenVINO inference.

### Usage (`lane+model_obj_v2`):
```bash
cd "lane+model_obj_v2"
source ../.venv/bin/activate

# 1. Live Preview (Complete Video)
bash run.sh --video videos/1.mp4

# 2. Faster Detector Mode (Higher FPS)
bash run.sh --video videos/1.mp4 --det-config B2

# 3. Save Output Video to File
bash run.sh --video videos/1.mp4 --output file --out-file results/output.mp4
```

---

## Previous Version: Lane + Vehicle Detection (~9 FPS) — `v11_lane_obj_9fps`

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
- `v13_lane+model_obj_v3`: Optimized lane + model vehicle detection pipeline (`lane+model_obj_v3`).
- `v12_lane+model_v2_25fps`: Best integrated lane + model vehicle detector (~25 FPS, `lane+model_obj_v2`).
- `v11_lane_obj_9fps`: Best integrated lane + vehicle detector with tyre contacts (~9 FPS, `lane+object_v3`).
- `v10_lane_best_29fps`: High-performance pipelined lane detector (~29 FPS, `lane_path_v6_p4`).
- `v9_final_best_lane`: Standalone classical lane path detector (`lane_path_v6`).
- `v8_lane`: Baseline single-file classical lane detector (`lane_path.py`).
- `v7_best`: Adaptive lane and vehicle detection (model-free).
