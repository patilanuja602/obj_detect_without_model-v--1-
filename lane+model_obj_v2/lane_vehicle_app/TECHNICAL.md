# Technical document — final implementation and measured results

All measurements below come from a **one-core Linux sandbox, not the T480**.  Quality metrics (what is detected / drawn) transfer; timings do not.

## 1. Architecture
```
decode thread -> display-resize + detector-preprocess (stage thread, frame N+1) -> main loop (frame N):
   detector worker (OpenVINO infer + decode)  ||  lane detector (V6, main thread)      [same frame id, joined before use]
   -> tracker -> road association -> drawing (lane overlay cut under vehicles) -> output (window / file)
```
* Lane: `lane_baseline/lane_path_v6_p3.py` is imported unchanged.  `lvapp/lane_engine.py` subclasses `PathDetectorV6` and overrides only `_edges()` (the single Canny edge
  map from which every Hough segment and candidate is derived) to optionally clear edge pixels on reliable vehicles.  With no mask set it is the baseline function
  (`eval/check_lane_identity.py`: 0 differing frames in 750 frames over three videos).
* Detector (`vehicle_detector.py`): YOLO26n ONNX via OpenVINO, band crop rows 35–100 %, letterbox; one class-agnostic NMS with a score-weighted class vote replaces per-class NMS
  (which allowed one vehicle to be boxed as car + truck + bus); size/aspect plausibility filter; ground-contact point = bottom-centre of the box.
* Tracker (`vehicle_tracker.py`): greedy IoU matching; entry 0.25, shown immediately at ≥ 0.45 else after 2 consecutive hits; kept down to 0.20; coasts 2 frames; class votes decay 0.85/frame and
  switch only when another class leads by 1.25×; light box smoothing (0.7 weight on the new detection).  The detector runs on every frame regardless.
* Association (`association.py`): ground point vs the lane detector's boundary curves of the same frame → current / left / right / outside / uncertain; "uncertain" whenever the lane
  state is not LANE or a boundary is unusable/contradicted/outlier/estimated, or the vehicle is far above the observed corridor.
* Synchronisation (`pipeline.py`): detector jobs carry the frame id and the loop asserts the returned id; fingerprints prove lane and detector saw the same frame and that the processed
  frames equal a sequential decode.  A detector exception yields no boxes for that frame and the lane continues (counted in `detector_failures`).

## 2. Measured results

### Vehicle detection (BEFORE = reconstructed old system, AFTER = default)
| UHD segment, 640 frames | BEFORE | AFTER |
|---|---|---|
| frames with duplicate boxes (IoU ≥ 0.6) | 27.7 % | 0 % |
| class flips per 100 linked boxes | 3.32 | 0.53 |
| box blinks (vanish 1–2 frames) per 100 boxes | 0.41 | 0.03 |
| boxes per frame | 4.62 | 4.50 |

HD segment (400 frames, almost empty road, small counts): duplicates 0.25 → 0 frames, class flips 8.3 → 3.9 per 100 links, boxes/frame 0.18 → 0.21.

Phase-2 reference frames (36 frames, 95 reference vehicles made by a larger model, reviewed by eye; recall ±5 points): recall 0.67 → 0.67, far recall 0.42 → 0.42, false positives per frame 0.14 → 0.11,
cross-class duplicate pairs 1 → 0.

**Trade-offs, reviewed by eye (UHD):** the tracker never showed 19 of 81 boxes the old system showed: about 6 were false positives (`bus`/`truck` on a building mural and signs), about 13 were real but marginal
(0.25–0.42) vehicles beyond the median barrier; none was on the driving road.  New low-confidence vehicles appear up to 5 frames later in 33 % of cases (mean −0.6 frames, i.e. often earlier).
HD: 8 of 30 not shown (not reviewed).  Two mural "bus" false positives (0.45–0.47) are still shown in a 24-track sample.  `tracker.confirm_hits=1` restores earlier/more marginal boxes.

**Detector input:** feeding the detector from the 1280-px display frame lost 12 % of far (7/60) and 3 % of mid-range detections versus the original frame, so the default keeps the original frame; the speed
benefit is obtained by running its preprocessing in the background stage thread instead (implemented, **not measured on the T480**).

### Road association (24 random UHD tracks, by eye)
No clearly wrong current/left/right tag; 7 of 24 were `uncertain` (correct while the lane is lost).  `outside` is rarely assigned (136 of 2,880 boxes): a car beyond the median barrier was tagged
`uncertain`, not `outside`, because the left boundary is usually a painted line, not a recognised road edge.

### Lane
Default = vehicle-aware lane evidence **off**: output bit-identical to the V6 baseline.  Improvements to the lane overlay are display-only (overlay not painted over vehicle bodies).
Experimental `lane.mask_mode=prev_tracks` (reliable near-field vehicles, height >= 15 px, clear edge pixels inside their boxes), UHD segment, 640 frames from video second 24
(stored run: `results/seq_mask_uhd.pkl`, metrics `results/lane_mask_vs_baseline_uhd.json`):

| metric | baseline lane | with mask |
|---|---|---|
| LANE state share | 90.6 % | 91.3 % |
| ONE_SIDED / NONE share | 6.7 % / 2.7 % | 7.7 % / 1.1 % |
| boundary jitter p95 at row 340 (L / R, processing px) | 2.90 / 1.94 | 2.50 / 1.85 |
| frames with a lane boundary through a reliable vehicle | 34 | 10 |
| longest non-LANE run (frames) | 33 | 47 |
| failure window (video frames 1093-1147): non-LANE / NO LANE frames | 48 / 13 | **40 / 6 in this run, 55 / 21 in four earlier runs of the same configuration** |

**Unresolved variability.** The mask runs are identical up to frame 375 and then diverge inside the failure window (a lane change by the ego vehicle while a car passes, where the V6 tracker is in an unstable state).
Four runs gave 85.4 % LANE over 480 frames and 55/21 in the window (`results/earlier_mask_run_states_480.pkl`), two runs gave 40/6.  I could not find the cause (the sandbox may have been moved to a different host between sessions, which
changes floating-point details); I therefore cannot claim the mask helps or hurts inside that window.  Outside it the effects were consistent: the early-segment regression (below) is gone and boundaries on vehicles fall from 34 to 10 frames.
Because lane output must stay as reliable as the baseline, the mask is **off by default**.

What was learned: (1) masking distant vehicles at the vanishing point deleted far-field evidence of both boundaries and lost the left boundary in frames where the baseline was correct, which is why only vehicles >= 15 px
high are used; (2) an occlusion-aware extension of the tracker's 6-frame hold (`lane.occlusion_hold`) never triggered in any test and is left at the baseline value; (3) the failure window is mainly a **lane change by the ego
vehicle** (the old right marking becomes the left boundary of the new lane); V6 does not handle that, before or after, and it is **not fixed**.

### Performance (sandbox, 1 core)
At equal thread counts BEFORE and AFTER ran at the same wall FPS (7.9 vs 7.9 and 9.3 vs 9.1 in two repeats): tracker, association and drawing add no measurable work.  The parallel worker and stage-thread
preprocessing cannot show a gain on one core.  Every run: lane calls = detector calls = frames, 0 dropped frames, integrity checks passed.  `--autotune` ran end to end (its result here is noise).
**No 35–40 FPS claim is made; run `benchmark.py` on the T480.**

## 3. Known limitations
Lane change handling (above); `outside` rarely assigned; residual mural false positives; hysteresis hides some marginal vehicles; lane state from the recording's app could not be matched exactly (source never provided);
no T480 measurements; Python 3.10 and `--output show` untested.
