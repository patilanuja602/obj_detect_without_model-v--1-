# Model-free vehicle detector for the V6 lane pipeline

**Status: working prototype with weak, measured accuracy. Not reliable; do not use for safety decisions.**
`lane_path_v6.py` is **unmodified** and must sit in the same folder.

## Files (all new; nothing existing was changed)
| File | Purpose |
|---|---|
| `vehicle_detector.py` | the detector (OpenCV + NumPy only, no models/weights) |
| `run_vehicle_pipeline.py` | V6 + detector on a video/camera, overlay, benchmark mode |
| `evaluate_vehicles.py`, `eval/gt_A.json`, `eval/gt_B.json` | hand-annotated evaluation (16 + 8 frames) |
| `test_vehicle_detector.py` | 15 unit/smoke tests (synthetic data) |

Dependencies: `pip install opencv-python numpy` (ffmpeg optional, only to re-encode output).

## Run
```
python3 run_vehicle_pipeline.py in.mp4 out.mp4                 # annotated video
python3 run_vehicle_pipeline.py in.mp4 --show                  # live window, q quits
python3 run_vehicle_pipeline.py --camera 0 --show              # camera
python3 run_vehicle_pipeline.py in.mp4 out.mp4 --debug         # + candidates, rejection reasons, search region
python3 run_vehicle_pipeline.py in.mp4 --benchmark 900         # algorithm-only timing, decode excluded, 30 warm-up frames dropped
python3 run_vehicle_pipeline.py in.mp4 --benchmark 900 --no-vehicles   # V6-only baseline on the same frames
python3 -m unittest -v test_vehicle_detector
python3 evaluate_vehicles.py --video in.mp4 --gt eval/gt_B.json --exclude 0.68 0.82 1 1
```
`--exclude X0 Y0 X1 Y1` (fractions) blanks a watermark/hood. Video B has a "clideo.com" watermark, so all B runs used `0.68 0.82 1 1`.
`--policy suppress|fallback` chooses what happens when V6 is not in state LANE (default `fallback`: static ROI searched, boxes flagged `ROI_FALLBACK`, lane association `UNKNOWN`).
`--cfg key=value ...` overrides any `VCFG` value (every default is documented in `vehicle_detector.py`).

## How it works (short)
1. **Search region** from V6's corridor (`ys, xl, xr`, V6 processing pixels), widened sideways; a tight pass (corridor only) and a wide pass (adjacent lanes).
2. **Road colour model**: per-row median Lab colour sampled in the corridor core, update-gated so a vehicle in the lane is not learned as road.
3. **Candidates**: non-road pixels (colour/lightness deviation or dense edges) minus thin bright lane paint, run-width filtering per image row (kills paint/wall edges), dark underside seeds, column-profile splitting of merged neighbours, cut at the horizon.
4. **Scoring**: size vs. lane width at that row, aspect, fill, edge density, road below the box, lower-third horizontal edges. Heuristic 0..1, **not a probability**. Hard rejects are recorded with a reason (`--debug`).
5. **Tracker**: greedy IoU/distance, confirm after 3 hits in 5, coast max 6 frames. States: CANDIDATE / ACCEPTED / TRACKED / PREDICTED (`observed=False`).
6. **Contact point** (image space only, `bbox_bottom` or `lower_edge`) and **lane association** against the V6 corridor: IN / LEFT / RIGHT / AMBIGUOUS / UNKNOWN. Never extrapolated beyond V6's corridor; an `INFERRED` V6 side is not trusted. V6's corridor is the host path, not a map of all lanes: adjacent lanes are not independently detected.

## Measured results (sandbox: 1 Xeon core, NOT a ThinkPad T480)
Timing, video B, 870 measured frames after 30 warm-up, 640x360 processing:
| stage | median ms | p95 ms |
|---|---|---|
| V6 | 4.13 | 4.87 |
| vehicle candidates | 4.35 | 4.65 |
| vehicle scoring | 0.31 | 0.46 |
| vehicle tracking | 0.02 | 0.04 |
| vehicle total | 4.82 | 5.17 |
| **total algorithm** | **8.93** | **9.91** (about 112 FPS algorithm-only) |

Before one optimisation (cumsum/nonzero replaced in the run extraction, output bit-identical) the vehicle stage was 7.70 ms median and the total 11.88 ms. Video A: total median 8.55 ms.
Decode, resize, drawing, writing and display are excluded above; a full write-video loop ran at about 38 FPS here. **The 30 FPS target on the T480 is unverified**: run the `--benchmark` commands there.

Accuracy (IoU >= 0.30, observed ACCEPTED/TRACKED only, full sequential run):
| set | result |
|---|---|
| Video B, 36 annotated vehicles | precision 0.50, recall 0.28, 0.62 FP/frame |
| Video B, vehicles >= 25 px wide (15) | precision 0.46, recall 0.40, 0.44 FP/frame |
| Video A (no vehicle on host carriageway, 8 frames) | 5 FP, 0.62 FP/frame |

The annotation set is small, hand-made (+-2 px) and was **also used for tuning** (notably `edge_dens_min=0.50`), so these numbers are optimistic for unseen footage.

## Known limitations
- Low recall for vehicles narrower than about 25 px (640 px width); distant vehicles are mostly missed.
- Vehicles touching a wall/barrier or each other can merge or be lost; boxes are sometimes too wide.
- Far-field clutter (signs, guardrail ends, bridge parts) and wall bases still give false positives; the score saturates and discriminates poorly.
- Top edge of small vehicles is estimated from width (`parts["top_est"]=1`); trucks/buses are underestimated in height.
- Cold start: a vehicle filling the corridor core in the first frames is learned as road (the model adapts only slowly, gated).
- Dark vehicles on dark asphalt rely on edges and shadow and are less reliable.
- Not tested: night, rain, glare, tunnels, curved roads with dense traffic, true stopped/stationary queues, other cameras.
- Camera-motion compensation was **not implemented**: no measurement showed it was needed (detector is per-frame, appearance-based), and it was not benchmarked.
- No real-world coordinates or distance; contact points are image pixels only.

## Rollback
All files are additions. To remove: delete the five files above and `eval/`. `lane_path_v6.py` was never edited, so V6 behaves exactly as before.
