# Vehicle detector upgrade: tight boxes + tyre ground-contact (model-free)

STATUS: partial improvement, validated on a SMALL hand-annotated set. Read "Known limitations" before relying on it.

## What was wrong (measured)
1. Stage-1 road model is built from the corridor core, i.e. where vehicles are; at vehicle rows it was off by 30-60 grey levels.
2. Dark car on dark asphalt: only the dark underbody separates, so the candidate is a thin band and is rejected as "flat" (most misses).
3. Everything is at 640 px and scaled up (1 px = 4 source px at 2560); contact = bbox bottom-centre; EMA smoothing lags.

## What changed (lane_path_v6.py is byte-identical to the upload)
- `vehicle_refine.py` (new): full-resolution refinement per proposal: sub-pixel under-body edge profile, tyre contacts
  (TWO_TYRES / ONE_TYRE / UNCERTAIN; UNCERTAIN returns no precise point, only `contact_hint_src`), lateral edges from the dark
  mass, top edge searched around the typical rear-view height ratio, evidence gates (symmetry, side-edge strength).
- `vehicle_detector.py`: new fields (`box_src`, `contact_src`, `contact_quality`, `tyres_src`, ...), robust per-row road estimate,
  bottom-up under-body proposal scan, rescue of "flat/sparse/..." stage-1 rejections, source-pixel Kalman tracker.
  `process(..., full=frame)` enables it; without `full=` the old behaviour runs. `run_integrated.py --legacy-vehicles` = old path.
- `run_vehicle_pipeline.py`: overlay shows contact quality (filled = two tyres, orange ring = one tyre, grey ring = uncertain) and tyre marks.
- `evalkit/`: dump_detections.py, evaluate_gt.py, stability.py. `baselines/` = frozen original detector. `results/` = GT, dumps, summary.

## Measured results (video 9977360, 2560x1440, 6 frames / 11 annotated vehicles, annotation accuracy ~ +-3 px, source px)
| | original | new |
|---|---|---|
| vehicles matched | 5 / 11 | 9 / 11 |
| box IoU (mean, where top annotated) | 0.45 | 0.69 |
| left / right edge error, mean | 44 / 37 | 20.5 / 6.5 |
| bottom edge error, mean | 17.9 | 2.5 |
| top edge error, mean | 18.8 | 16.1 |
| contact-point error, mean / median | 30.7 / 25.6 | 10.1 / 7.8 |
| jitter RMS 2nd difference x0 / x1 / y1 / contact-x / contact-y | 17.3 / 14.3 / 6.3 / 10.8 / 12.3 | 6.9 / 6.4 / 3.8 / 6.7 / 3.4 |
| video B (near-empty road) observed detections / tracks >= 10 frames | 2655 / 54 | 1477 / 16 |
| vehicle stage ms (lane 5.2 ms unchanged) | 6.5-9.0 | 13.6-15.9 |
| end-to-end FPS, 1 core incl. decode+write, 1440p clip | 27.1 | 22.2 |

## Known limitations
- 35 FPS is NOT reached in this 1-core sandbox (22 FPS end-to-end; lane+vehicle = ~21 ms vs 14 ms before). Not measured on your CPU.
- GT is tiny (11 vehicles, 6 frames, one video) and annotated by eye; thresholds (symmetry >= 0.15, edge >= 1.5) were fitted on 8 of
  those vehicles plus video B, so there is no held-out set. The accuracy numbers are optimistic.
- Left/right edges still off by 20-50 px when a cast shadow or a visible side panel joins the dark mass (frames 869, 1049, 1858);
  white vans next to lane paint are still missed (2 of 11); top edge is only ~16 px accurate.
- Tyres are usually merged with the underbody shadow at this video quality: TWO_TYRES mostly means "both lower corners found",
  x then uses an assumed tyre width. The tyre-based midpoint (10.1 px) is only slightly better than the new box bottom-centre (11.6 px).
- Large oblique vehicles (tyres not visible) were excluded from GT and are not reliably handled.
- Video B still produces short false tracks (656 tracks); no precision/recall was measured on it, only counts.
- Adjacent lanes are inferred from the ego corridor geometry (lane detector not allowed to change), not detected independently.
