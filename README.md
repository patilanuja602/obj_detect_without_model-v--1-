# Adaptive Lane & Vehicle Detection (Classical CV — No Deep Learning)

> **Version 7 (Best Classical Version)**: This version combines **Adaptive Lane Detection** with **Road Vehicle Detection** in a unified, 100% classical computer vision pipeline (OpenCV + NumPy only). No deep learning, no neural network weights, no GPU required.

---

## Overview & Performance Note
* **Dual Detection Capability**: Unlike earlier versions that focused only on motion/optical flow or only on lane lines, this version detects **both ego lane boundaries and road vehicles simultaneously**.
* **Accuracy Notice**: Because this pipeline relies purely on classical geometric and photometric heuristics (Canny edges, Hough lines, RANSAC vanishing point estimation, Lab/HSV color contrast, asphalt models, and morphological filters), **its accuracy is lower than deep learning object detectors (e.g. YOLO)**. It is designed specifically for lightweight CPU environments where running neural networks is not possible.

---

## Key Features

1. **Adaptive Horizon & Vanishing Point Tracking (`vanishing.py`)**
   - Automatically estimates the road horizon on every frame using RANSAC line intersection.
   - Adapts to camera pitch, bumps, and slope without a hardcoded triangle ROI.

2. **Paint & Marking Detection (`markings.py`)**
   - Multi-scale top-hat contrast filter on LAB lightness for white solid/dashed lines.
   - HSV color filtering for yellow and orange lane markings.
   - Automatically adapts thresholds to frame brightness.

3. **Curved Lane Fitting & Tracking (`lanes.py`, `pipeline.py`)**
   - Quadratic polynomial fitting (`x(t) = vp_x + k·t + c·t²`).
   - Temporal smoothing, jump rejection, and a 12-frame memory bridge for missing dashes.
   - Computes lateral lane offset (meters) and approximate steering hint (degrees).

4. **Classical Vehicle Detection (`vehicle_detector/`)**
   - Uses ego-lane geometry to bound the vehicle search area (up to $N$ lane widths).
   - Asphalt background modeling to isolate candidate obstacle blobs.
   - Bounding box validation using ground shadow, aspect ratio, edge texture, and temporal tracking.
   - Labels targets as `FRONT` (ego lane), `LEFT`, or `RIGHT` with rough distance estimates.

---

## Installation

Ensure Python 3.10+ is available and install the lightweight requirements:
```bash
pip install -r lane_project/requirements.txt
```
*(Dependencies: `opencv-python>=4.8`, `numpy>=1.24`, `pytest>=7.0`)*

---

## How to Run

Navigate to `lane_project`:
```bash
cd lane_project
```

### 1. Run Lane + Vehicle Detection (Live Preview)
```bash
python3 run_lane_and_vehicles.py --source ../videos/1.mp4 --show
```
*(Press `q` inside the video window to quit at any time)*

### 2. Run with Debug Visualization
Displays the estimated horizon, raw Hough lines, lane marking mask, cyan vehicle search ROI, and rejected candidate boxes:
```bash
python3 run_lane_and_vehicles.py --source ../videos/1.mp4 --show --debug
```

### 3. Save Annotated Output Video & CSV Metrics
```bash
python3 run_lane_and_vehicles.py --source ../videos/1.mp4 --save output_vehicles.mp4 --log-csv vehicles_log.csv
```

### 4. Run Lane Detection Only
```bash
python3 run_lane_detection.py --source ../videos/1.mp4 --show
```

---

## Validation & Tests

Run the built-in synthetic test suites:
```bash
python3 lane_project/tests/test_synthetic.py
python3 lane_project/tests/test_vehicle_smoke.py
```
Both test suites verify horizon tracking under pitch changes and obstacle detection on synthetic road scenes.
