# lane_object_integrated -- System 2 lane detector + System 1 vehicle detector

Run (from this folder):
```
pip install -r requirements.txt
python3 run_integrated.py video.mp4 out.mp4 --start 5 --duration 10
python3 run_integrated.py video.mp4 --show            # q quits
python3 run_integrated.py --camera 0 --show
python3 run_integrated.py video.mp4 out.mp4 --display-interp linear      # opt-in faster display resize (background image only)
python3 run_integrated.py video.mp4 out.mp4 --exclude 0.68 0.82 1 1     # blank a watermark for the vehicle detector
python3 verify_integration.py video.mp4 --start 5 --frames 300           # bit-exact check vs System 1 behaviour
python3 bench_compare.py video.mp4 --start 5 --frames 330 --rounds 3     # S1 vs S2 vs integrated, same loop
python3 tests/test_v6.py ; python3 -m unittest -v tests.test_vehicle_detector
```

## What was found first (read this)
1. **The lane detection of the two systems is the same algorithm.** `lane_object_v1.zip` contains only the vehicle files; its lane file (`lane_path_v6.py`, "unmodified") was not in the zip.
   System 2's own reports state `process()` is unchanged code in every variant. I checked it: original V6 vs `lane_path_v6_p3.py` on 300 consecutive frames -> processing ROI, edge map, L channel and the complete
   `det.result` are identical (0 mismatches). Only `draw()` differs (display-only overlay clip). So "System 2 has better lane detection" cannot come from `process()`; if System 1's lane file
   in your folder differs from `baselines/lane_path_v6_original.py`, send it and I will diff it.
2. **`lane_path_v6_p4.zip` contains no p4 file.** Newest file is `lane_path_v6_p3.py` (the `lane_path_v6.py` beside it is older). The integrated project uses p3, byte-identical, as `lane_path_v6.py`.
3. **The one real conflict:** p3's ROI-only processing resize leaves all rows outside the lane ROI at zero. The vehicle detector reads rows `int(0.52*h)..h`. Measured: feeding it the ROI-only buffer changed its output on 295/300 frames.
   Fix: same ROI-only resize (still bit-identical k x k box average), widened to rows 187..360 (full width) = union of lane ROI and vehicle band. Not the full frame, so part of System 2's saving is kept.
   (Keeping rows >= 160 also matched on the test clip, but 187 is derived from `band_top`, not from one clip.)

## Architecture (smallest safe boundary)
camera/decoder (System 2 `_Prefetch`) -> ONE processing frame (`make_resizer`) -> `PathDetectorV6.process` (System 2) -> `VehicleDetector.process(small, det.result)` (System 1, unchanged)
-> display (System 2 `_Stage` resize, `det.draw`) -> System 1 `draw_vehicles` -> `_AsyncWriter` / window.
Both detectors read the same frame in the same iteration, synchronously: no stale-frame pairing. Vehicle boxes are in processing pixels and scaled only when drawn. Vehicles never write to lane state.
Files unchanged from the sources: `lane_path_v6.py` (= p3), `vehicle_detector.py`, `run_vehicle_pipeline.py` (System 1 runner, still works standalone and supplies `draw_vehicles`), `evaluate_vehicles.py`, both test files. New: `run_integrated.py`, `verify_integration.py`, `bench_compare.py`.

## Preservation checklist
Preserved from System 1: model-free vehicle detection, search region from the V6 corridor, road colour model, scoring, tracker (CANDIDATE/ACCEPTED/TRACKED/PREDICTED), contact point, IN/LEFT/RIGHT/AMBIGUOUS/UNKNOWN association, `--policy`, `--exclude`, `--cfg`, `--debug`, `--no-vehicles`, `--camera`, `--show`, HUD, benchmark-style timing (now per stage), 15 unit tests.
Adopted from System 2: the whole lane pipeline (ego lane corridor, one-sided/inferred/tracked boundaries, hysteresis, extent limiting, overlay clip), decoder/stage/writer threads, `--threads`, `--display-interp`, `--cv-threads`, `--overlay-top-*`, `--no-roi-resize`, 32 tests.
Modifications: widened resizer ROI; combined runner with per-stage timing; `--camera` forces threads off (a queue of 6 frames would add stale-frame latency).
System 1's own runner (`run_vehicle_pipeline.py`, incl. `--benchmark`) still works and now uses the p3 lane module. Not preserved / not possible: `eval/gt_A.json, gt_B.json` were not in the zip, so `evaluate_vehicles.py` cannot run. Adjacent lanes are still not detected independently (neither system does this).

## Verification (bit-exact, `verify_integration.py`)
| clip | frames | lane results mismatching | vehicle outputs mismatching | rows read by vehicle stage mismatching |
|---|---|---|---|---|
| 1080p, start 5 s | 300 | 0 | 0 (562 detections) | 0 |
| 1080p, start 40 s | 200 | 0 | 0 (441) | 0 |
| 2560x1440 (watermark excluded), start 20 s | 300 | 0 | 0 (498) | 0 |
So the integrated output equals System 1 / System 2 output; nothing was improved or degraded in detection quality, by construction. The tests (32 + 15) pass.

## Performance -- SINGLE-CORE SANDBOX, NOT the T480
1080p clip, frames 5 s..15 s, 300 measured frames x 3 interleaved rounds, same loop for all, with decode, resize, draw and mp4v write, no frame skipped.
| variant | mean FPS | median | min | 5th pct |
|---|---|---|---|---|
| System 1 (V6 + vehicles) | 20.0 | 21.6 | 11.6 | 14.8 |
| System 2 (lane only) | 27.9 | 30.3 | 14.0 | 20.0 |
| Integrated | 22.2 | 24.0 | 13.4 | 15.8 |
Stage medians, ms (integrated): decode 9.6 | processing resize 1.7 | lane 4.8 | vehicles 7.6 | display resize 9.6 | draw 1.5 | write 6.7.
Integrated CLI, 1 core, mean FPS: threads off 21.0 | + `--display-interp linear` 28.1 | threads on 22.1 | threads on + linear 27.1. Threads cannot add parallelism on one core.
**The 30-35 FPS target is NOT demonstrated.** Best measured: 28.1 mean (linear display resize, 1 core; min 17.8). Bottlenecks: display resize (9.6, removable with `linear`), decode 9.6, vehicles 7.6, write 6.7.
Calculation only (not measured): with decode, display resize and write on their own threads the main thread needs ~15.6 ms (resize+lane+vehicles+draw), i.e. a ceiling near 60 FPS on >= 3 free cores. Measure on the T480:
```
sudo cpupower frequency-set -g performance
python3 bench_compare.py $VIDEO --start 20 --frames 330 --rounds 5
for o in "--threads off" "--threads on" "--threads on --display-interp linear"; do python3 run_integrated.py $VIDEO /tmp/o.mp4 --start 20 --duration 10 $o --stats-json /tmp/s.json; done
```
Not done on purpose: frame skipping, lower detection frequency, lower resolution, running vehicles one frame behind lane (would pair stale boxes with newer corridor). A shared Lab conversion would save <= 0.5 ms and requires editing lane code: not worth it.

## Known limitations
- Vehicle detector accuracy is weak and was measured by System 1 only on 2 clips (P 0.50 / R 0.28 on video B; far vehicles < 25 px mostly missed; FP 0.4-0.6 per frame). Integration does not change this. In my overlay frames the detections are small, distant, and several are false boxes.
- Not tested: night, rain, glare, tunnels, dense curved traffic, true stop-and-go, live camera. Only two clips were available (no nD_20.mp4 / webcam recordings).
- ROI-only fast path needs an integer source/processing scale (1080p and 1440p at 640 px are); otherwise it falls back to the full resize automatically.
- All timings are one sandbox core with `cv2` using 1 thread.
