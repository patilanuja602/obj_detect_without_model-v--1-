# Version 8: Classical Drivable-Path & Lane Detection (`v8_lane`)

Single-file standalone classical lane detector (**no deep learning, no weights, CPU only**).

This version contains **lane detection only** in a single self-contained script: [`lane_path.py`](lane_path.py).

---

## Pipeline
1. **Resolution downscaling** to 640px wide for real-time CPU performance (~25-35 FPS).
2. **Trapezoid ROI** bounding box crop.
3. **CLAHE** on LAB L-channel + Gaussian blur for illumination and contrast robustness.
4. **Canny Edge Detection** masked to the road trapezoid.
5. **HoughLinesP** + slope filtering to isolate left and right marking candidates while rejecting horizontal clutter.
6. **Dual fitting**:
   - **Method A (Straight)**: Length-weighted collinear segment averaging.
   - **Method B (Curved)**: Sliding window histogram + quadratic polynomial fit.
7. **Temporal smoothing & sanity checking** across frames.
8. **Drivable path corridor overlay**: Smooth `fillPoly` between the two detected boundaries.

---

## Usage

### 1. Live Preview Window (`--show`)
*(Press `q` anytime to quit)*
```bash
python3 lane_path.py --source videos/1.mp4 --show
```

### 2. Live Debug View (`--debug`)
Shows the trapezoid ROI, Canny edges, and Hough lines alongside the detected path:
```bash
python3 lane_path.py --source videos/1.mp4 --show --debug
```

### 3. Save Annotated Output Video
```bash
python3 lane_path.py --source videos/1.mp4 --save output_path.mp4 --show
```
*(Or positional syntax: `python3 lane_path.py videos/1.mp4 output_path.mp4 --show`)*

### 4. Road Profiles (Highway vs Mountain)
* **Mountain (default)**: `python3 lane_path.py --source videos/1.mp4 --profile mountain --show`
* **Highway**: `python3 lane_path.py --source videos/1.mp4 --profile highway --show`
