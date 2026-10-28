# Classical Lane Detection — Final Best Version (`lane_path_v6`)

Single-file standalone classical lane detector (**no deep learning, no weights, CPU only, OpenCV + NumPy**).

This version (`lane_path_v6`) represents the **final best lane detection** pipeline with:
- **Consistent overlay**: Smooth green lane corridor with line styling indicating status (solid = observed, dashed = tracked, dotted = inferred).
- **Graceful corridor clipping**: Corridor terminates at the first invalid row (crossing / diverging / flare) instead of dropping the entire lane.
- **One-sided / Inferred fallback**: Intelligently maintains guidance when one boundary is temporarily occluded or missing.
- **Outlier rejection & temporal smoothing**: Bounded tracking prevents sudden jumps, flickering, or saw-tooth top-edge artifacts.
- **100% Standalone**: Self-contained in [`lane_path_v6/lane_path_v6.py`](lane_path_v6/lane_path_v6.py) without dependencies on previous iterations.

---

## Usage (`lane_path_v6`)

```bash
cd lane_path_v6
source ../.venv/bin/activate
```

### 1. Live Preview Window (`--show`)
*(Press `q` anytime to quit)*
```bash
python3 lane_path_v6.py --source ../videos/1.mp4 --show
```

### 2. Live Debug View (`--debug`)
```bash
python3 lane_path_v6.py --source ../videos/1.mp4 --show --debug
```

### 3. Save Output Video
```bash
python3 lane_path_v6.py --source ../videos/1.mp4 --save output_v6.mp4
```

### 4. Road Profiles (Mountain vs Highway)
* **Mountain (default)**: `python3 lane_path_v6.py --source ../videos/1.mp4 --profile mountain --show`
* **Highway**: `python3 lane_path_v6.py --source ../videos/1.mp4 --profile highway --show`

---

## Previous Versions
- **`v8_lane`**: Baseline single-file classical lane detector ([`lane_path.py`](lane_path.py)).
- **`v7_best`**: Adaptive lane and vehicle detector (without ML model).
- **`v6`**: Adaptive classical lane detection with horizon tracking.
