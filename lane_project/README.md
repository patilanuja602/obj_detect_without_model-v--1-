# Adaptive Lane Detection (classical CV, no model, no fixed ROI)

Real-time lane detection built only from OpenCV + NumPy.
It improves the Kaggle notebook / `Lane-Detection-master` approach (grayscale → blur → Canny → ROI → Hough)
by removing the **hard-coded triangle ROI** and adding **horizon tracking, white + yellow paint, lane-boundary
selection, temporal tracking and driving metrics**.

---

## 1. Setup (Ubuntu + VS Code, nothing assumed)

| Step | Command | What it does |
|---|---|---|
| 1 | `cd lane_project` | go into the project folder |
| 2 | `python3 -m venv .venv` | create an isolated Python environment |
| 3 | `source .venv/bin/activate` | switch it on (prompt shows `(.venv)`) |
| 4 | `pip install -r requirements.txt` | install OpenCV, NumPy, pytest |
| 5 | `python tests/test_synthetic.py` | self-check; prints `synthetic tests passed` |
| 6 | VS Code: *Ctrl+Shift+P → Python: Select Interpreter → .venv* | so VS Code uses it |

## 2. Run it

```bash
python run_lane_detection.py --source road.mp4 --show               # live window, q = quit
python run_lane_detection.py --source road.mp4 --save out.mp4       # write result video
python run_lane_detection.py --source 0 --show                      # webcam / camera 0
python run_lane_detection.py --source photo.jpg --save out.jpg      # single image
python run_lane_detection.py --source road.mp4 --show --debug       # horizon, raw segments, marking mask
python run_lane_detection.py --source road.mp4 --show --calibrate   # click where YOUR car's centre is
python run_lane_detection.py --source road.mp4 --save out.mp4 --log-csv log.csv
```

| Option | Meaning |
|---|---|
| `--car-x 0.8` | car centre-line across the image, 0 = left edge, 1 = right edge (default 0.5) |
| `--calibrate` | click the car's centre on the first frame instead of typing `--car-x` |
| `--proc-width 480/640/960` | processing size: smaller = faster (default 640) |
| `--out-width 960` | width of the rendered output |
| `--lane-width-m 3.7` | real lane width used to convert pixels to metres |

## 3. How it works

```
frame ─► 1 markings ─► 2 segments ─► 3 vanishing point ─► 4 lane boundaries ─► 5 tracking ─► 6 metrics
         white+yellow   Canny+Hough   RANSAC (= horizon)    pick left/right      smoothing,     offset,
         (local test)   (whole image) per frame             of the ego lane      short memory   steering hint
```

| Step | File | Idea |
|---|---|---|
| 1 | `markings.py` | White = thin structure brighter than its surroundings (multi-scale top-hat on LAB-L) and colourless. Yellow/orange = HSV hue test. Thresholds adapt to each frame. |
| 2 | `segments.py` | Canny + probabilistic Hough on the **whole image**, near-horizontal segments dropped. |
| 3 | `vanishing.py` | Lane lines, kerbs and guard-rails are parallel to the road, so they all point at one vanishing point. RANSAC finds it every frame. **Its y is the horizon**, so when the camera tilts up/down everything follows. |
| 4 | `lanes.py` | Keep segments pointing at the VP and below the horizon, group by where they hit the bottom row, fit each group as `x(t) = vp_x + k·t + c·t²` (lean + curvature). A `RoadModel` learns asphalt colour along the path from the VP to the car and rejects boundaries whose car-side is not road (guard-rails, sky, poles). Nearest valid boundary left and right of the car = the ego lane. |
| 5 | `lanes.py` | Smoothing, jump rejection (a new position must repeat 3 frames), 12-frame memory when a line briefly disappears. |
| 6 | `pipeline.py` | Offset from lane centre (m), steering hint (deg), curve direction. |

**Behaviour you asked for**
* Only **one** boundary found → only that line is drawn. **Two or more** found → the lane between the nearest left and right ones is filled green with a yellow centre path.
* Nothing found in a region → nothing is drawn there.
* Both white and yellow/orange paint, solid or dashed (shown in the HUD).

## 4. Things to know (honest limits)

| Topic | Detail |
|---|---|
| Car position | A camera not centred on the car sees the bonnet off to one side; no algorithm can know that without a hint. Use `--car-x` or `--calibrate` once per mounting. Default 0.5 assumes a centred camera. (I tested automatic bonnet detection; it was unreliable, so it is not included.) |
| Steering hint | Uncalibrated approximation (assumed 70° field of view). Good for display and trends, not for driving a real vehicle without calibration. "Straight ahead" is self-calibrated from the long-run average vanishing point. |
| Lane width | Offset in metres assumes `--lane-width-m` (3.7). |
| Hard cases | Night, heavy rain, no paint, sharp curves, lane changes (the old lane is held ≤ 12 frames) and strong glare can lose a line. This is why modern production systems use deep learning. |
| Speed | Detector alone ≈ 30 FPS at 640 px on 1080p input (CPU, this machine); 4K decoding and video writing dominate in the CLI. |

## 5. Tuning (`lane_detector/config.py`)

| Problem | Parameter |
|---|---|
| Faint paint missed | lower `min_contrast`, `contrast_sigma` |
| Orange / yellow line missed | lower `yellow_min_sat`, `yellow_min_val` |
| Too many false lines | raise `road_fraction_min`, `min_seg_len_frac` |
| Wobbly lines | lower `track_alpha` |
| Slow | lower `proc_width` |

## 6. Layout

```
lane_project/
├─ run_lane_detection.py      CLI (video / webcam / image, calibration, CSV log)
├─ lane_detector/
│  ├─ config.py   markings.py   segments.py   vanishing.py
│  ├─ lanes.py    pipeline.py   overlay.py
└─ tests/test_synthetic.py    horizon follows camera pitch; right lane chosen
```
