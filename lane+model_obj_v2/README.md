# Lane + vehicle detection (CPU, OpenVINO) — improved application

Real-time lane detection (your V6 detector, **unmodified**) plus YOLO26n vehicle detection on every frame, with duplicate-free boxes, stable class labels,
a ground-contact point per vehicle, road association (current / left / right / outside / uncertain), and a frame-integrity benchmark.
Target machine: ThinkPad T480 (i5-8350U), Ubuntu 22.04, CPU only.  **No timing in this package was measured on that machine** (see TECHNICAL.md).

## Install (Ubuntu 22.04)
```bash
unzip lane_vehicle_app.zip && cd lane_vehicle_app
bash install_ubuntu22.sh          # creates .venv, installs requirements, runs the self-test (must end with ALL GOOD)
```
Manual equivalent: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python scripts/selftest.py`
(if the newest openvino does not install on Python 3.10: `pip install "openvino==2024.6.0"`).

## Run
```bash
bash run.sh --video /path/to/video.mp4                    # on-screen window, q to quit
bash run.sh --video /path/to/video.mp4 --output file      # writes results/output.mp4 (also works without a display)
bash run.sh --source 0                                    # camera 0
bash run.sh --video clip.mp4 --autotune --output none     # MEASURES OpenVINO/OpenCV thread settings on your machine, saves config/tuned.json
bash run.sh --video clip.mp4 --tuned                      # reuse the saved tuning
bash run.sh --video clip.mp4 --det-config B2              # smaller/faster detector input (A2 = default 960x352, B2 = 768x288, F = full frame 960x544)
bash run.sh --video clip.mp4 --preset baseline_reconstruction.json   # the old behaviour, for comparison
bash run.sh --video clip.mp4 --set vehicle.conf=0.25 --set association.outside_policy=hide
```
The window shows the lane overlay (V6, as before; the overlay is cut away under vehicle bodies), vehicle boxes coloured by zone with `CUR / L / R / OUT / ?` tags,
a dot at each vehicle's ground-contact point, and a second HUD line with the **end-to-end** FPS (the first line's "FPS (algo)" is the lane detector alone, as in the baseline).

## Benchmark (run this on the T480)
```bash
source .venv/bin/activate
python benchmark.py --video /path/to/video.mp4 --threads-list 1 2 3 4 --name t480
```
Runs the OLD behaviour (`baseline_reconstruction`) and the NEW default for every thread count, 300 frames each after a 30-frame warm-up, and prints FPS (timed frames / wall-clock
of the whole loop incl. decode wait, lane, detector, tracking, drawing), median / p95 frame time, lane and detector times, CPU cores used, RSS, lane/detector call counts,
dropped frames, and an **integrity** column: lane and detector must have received the same frame, and the processed frames must equal a plain sequential decode.
Results: `results/benchmark_t480.json`.  Add `--det-config B2` to compare the smaller detector input.

## Configuration
`config/default.json` (all paths relative).  Main switches:

| key | default | meaning |
|---|---|---|
| `vehicle.model_file`, `vehicle.band` | 960x352, rows 35–100 % | detector input; `--det-config` picks A2 / B2 / F |
| `vehicle.input_source` | `original` | detector reads the original frame (display-frame input lost 12 % of far detections) |
| `vehicle.nms_mode` | `agnostic_merge` | one NMS across classes + class vote (fixes car+truck duplicates); `class_aware` = old behaviour |
| `vehicle.threads` | 2 | OpenVINO inference threads (use `--autotune`) |
| `tracker.enter_score / confirm_score / confirm_hits / keep_score` | 0.25 / 0.45 / 2 / 0.20 | a detection below 0.45 must be seen twice in a row before it is shown; an existing track survives down to 0.20 |
| `association.outside_policy` | `dim` | `dim` / `hide` / `show` vehicles tagged outside the road |
| `lane.mask_mode` | **`off`** | `off` = lane output **bit-identical** to the V6 baseline.  `prev_tracks` / `same_frame` = EXPERIMENTAL vehicle-aware lane evidence (see TECHNICAL.md: consistent gains outside one unstable window, unresolved variability inside it) |
| `lane.overlay_cut_vehicles` | `true` | do not paint the lane overlay over vehicle bodies (display only) |
| `pipeline.parallel` | `true` | detector on a worker thread while the lane runs; joined on the same frame id |
| `pipeline.preprocess_in_stage` | `true` | detector preprocessing runs in the background stage thread (off the critical path) |

## Evaluation scripts (`eval/`)
`check_lane_identity.py` (lane bit-identical to baseline), `eval_vehicle_static.py` (Phase-2 reference frames), `agreement_display_vs_original.py`,
`run_sequence.py` + `run_sequences_all.sh` + `eval_sequence.py` (before/after lane and vehicle metrics), `ablate_lane.py`, `review_samples.py` (montages for manual checking).
Use `bash eval/run_sequences_all.sh UHD.mp4 HD.mp4 NMOUNTAIN.mp4`.

## What is verified, what is reconstructed
* Verified: lane file = your V6 (`lane_baseline/lane_path_v6_p3.py`, sha256 `e3c05550…`, never edited); lane output bit-identical with the mask off; every frame goes through both lane and detector;
  frame-integrity checks pass; the overlay of V6 matched your screen recording (median pixel overlap 0.94).
* Reconstructed: the "before" system (the source of your recorded app never arrived) = V6 + the Phase-2 package detector with class-aware NMS, conf 0.25, no tracker.
* Not tested: any run on the T480; Python 3.10; `--output show` (no display in the preparation sandbox).
