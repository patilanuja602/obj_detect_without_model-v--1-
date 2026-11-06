# lane_vehicle_app — integrated lane + vehicle detection (CPU only, ThinkPad T480 target)

Your verified V6 lane detector (`baseline/lane_path_v6_p3.py`, sha256 `e3c05550…`, **byte-identical, read-only, imported not copied**)
plus the OpenVINO YOLO26n vehicle detector from the T480 benchmark package, joined by a tracker, a vehicle-to-lane association and a
vehicle-aware edge mask.  Both detectors run on **every** frame (no skipping, no alternating, no stale detections).

## Install (Ubuntu 22.04)
```bash
bash install_ubuntu22.sh        # venv + pip + ffmpeg, then scripts/check_install.py must end with ALL GOOD
```
Manual: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python3 scripts/check_install.py`
(if `pip install openvino` fails on Python 3.10: `pip install "openvino==2024.6.0"`).  No internet is needed after installation; the ONNX models are in `models/`.

## Run
```bash
./run.sh VIDEO.mp4 --show                 # live window, q quits.  HUD FPS = end-to-end throughput of the whole loop
./run.sh VIDEO.mp4 --save out.mp4         # annotated video
./run.sh 0 --show                         # webcam 0
./run.sh VIDEO.mp4 --show --mode seq      # sequential instead of parallel
./run.sh VIDEO.mp4 --show --mode lane     # lane only (baseline-equivalent loop)
./run.sh VIDEO.mp4 --show --set detector.config=F      # full-frame detector (better far/edge coverage, slower)
./run.sh VIDEO.mp4 --trace trace.json --save out.mp4   # per-frame lane state + vehicles (ground-contact points, zones)
```
Settings live in `config/default.json` (every key is documented in TECHNICAL.md); override with `--config my.json` or `--set a.b=value`.

## Benchmark on the T480 (do this first — nothing here is a T480 measurement)
```bash
sudo cpupower frequency-set -g performance          # charger plugged in, other apps closed
python3 scripts/benchmark.py --video VIDEO.mp4 --system baseline --frames 300     # the original system (lane V6 + Phase-2.5 detector, sequential)
python3 scripts/benchmark.py --video VIDEO.mp4 --system v7 --mode par --frames 300
python3 scripts/benchmark.py --video VIDEO.mp4 --sweep --frames 200                # tries ov-threads x cv-threads x par/seq, prints the best setting
```
Each run prints end-to-end FPS, lane ms, detector pre/inference/post ms, wait for the detector, median and p95 frame time, CPU cores
used, RSS, frames processed by each detector, dropped frames, and a frame-integrity audit (lane and detector saw the same frame, and a
second sequential decode reproduces the same frames).  `FRAME CHECK FAILED` + exit code 2 if anything is wrong.  JSON goes to `--json FILE`.

Before/after quality comparison on identical frames: `bash scripts/run_comparisons.sh VIDEO_DIR results/runs` (see `results/REPORT.md`).
