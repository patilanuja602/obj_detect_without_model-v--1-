# Technical notes

## Layout
`baseline/` untouched V6 lane file + Phase-2.5 detector (read-only, imported, never edited) · `lvapp/` new code · `config/default.json` · `scripts/` benchmark, comparison, install check · `models/` ONNX.

## Per-frame flow (frame id n)
1. Worker thread: vehicle detector on frame n (OpenVINO, band crop → letterbox → infer → class NMS → cross-class dedupe). Main thread, same time: lane V6 on frame n.
2. Join; the engine asserts both carry frame id n. Both detectors run on every frame (counters in the benchmark; audit compares frame fingerprints and an independent re-decode).
3. Tracker update with frame-n detections → association with frame-n lane geometry → drawing.
Modes: `par` (above), `seq`, `lane`. If the detector fails or cannot load, the lane detector continues unchanged (error shown in HUD).

## Components
- **Detector**: candidates at conf 0.20 (not a global threshold); preprocessing inside OpenVINO; cross-class NMS only at IoU>=0.65 so separate vehicles are not merged. Config A2 (960x352, band 35-100% of height) by default; `detector.config=F` = full frame.
- **Tracker**: one-to-one IoU+distance association; a vehicle is shown if score>=0.45, or if seen 3 times with mean score>=0.25 (keeps distant weak vehicles, hides one-frame ghosts); alpha-beta box smoothing; class voting with hysteresis; duplicate-track removal; aspect/size plausibility; stricter confirmation only for vehicles classified OUTSIDE. Missed boxes are drawn dashed for <=2 frames and flagged `coasted`. Ground-contact point = bottom centre of the box (`track.gcp`, also in `--trace`).
- **Association**: CURRENT / LEFT_ADJ / RIGHT_ADJ / OUTSIDE / UNCERTAIN from the ground point and the displayed corridor at that image row; UNCERTAIN when the lane state is not LANE, the row is beyond the corridor's valid extent, the corridor is flagged wide, the relevant boundary is only inferred, or the vehicle straddles a boundary.
- **Lane**: V6 unchanged. Display: vehicle bodies are restored over the lane overlay (display only; internal geometry untouched). Optional edge mask (`lane_vehicle.mask_enabled`, **off by default**, see below).

## Verified vs reconstructed
Verified: V6 lane file hash; ONNX models; lane output bit-identical to baseline with no vehicle input (450 frames, 3 videos). Reconstructed: the whole integration (no integrated source was supplied) and the baseline "before" system (V6 + Phase-2.5 detector, sequential).

## Why the lane edge mask is off
Boundaries touch vehicle bodies in most frames of the traffic video, so the mask can matter, but across 12 segments (1850 frames) it changed lane state inconsistently: overall LANE share 87% (baseline) vs 86%, worse on 99.mp4@2500, mixed on @1500, no gain elsewhere. No labels exist, so no claim of improvement is made. Enable with `--set lane_vehicle.mask_enabled=true` to experiment. The lane detector is sensitive to small edge perturbations.

## Not addressed / limitations
Shadows, road patches, corridor-width jumps and boundary selection are as in V6. Zone OUTSIDE is "more than 1.25 lane widths beyond a boundary", not a road-edge detector. All quality numbers are proxies (no ground truth). The mask prior uses tracks from frames <= n-1 (needed for parallelism).
