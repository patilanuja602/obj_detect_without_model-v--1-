# Phase 3 — Road Object Detector (moving + stationary, no trained model)

## What changed from Phase 2
Phase 2 (`optical_beased__b__best_now_.zip`) only caught objects that move
relative to the road, because its whole pipeline was built on optical flow.
Phase 3 adds a **second, independent branch** that needs no motion at all:

- **Branch A — motion** (unchanged): Shi-Tomasi + Lucas-Kanade optical flow,
  MOG2 background subtraction, DBSCAN/KMeans clustering. Catches anything
  moving relative to the camera.
- **Branch B — static appearance** (new): every frame, samples a small
  "trusted road" patch near the bottom of the ROI (excluding any pixels the
  motion branch already flagged, so a vehicle right in front doesn't poison
  the model), builds a Lab-color statistical model of the road surface from
  it, then flags any region in the ROI whose color deviates from that model.
  A Sobel-edge cue is OR'd in to help solidify texture-distinct-but-similarly-
  colored objects. This has zero dependency on motion, so it fires on parked
  cars, trees, poles, curbs, dropped cargo — anything sitting still.

Both branches emit the same `(box, contact_point)` format and are merged
(IoU de-duplication) into **one shared tracker**, so a real object gets
exactly one stable track whether motion, appearance, or both branches saw it.
No trained weights, no pretrained model, no deep learning anywhere in the
pipeline — every threshold is either a fixed heuristic or a per-frame
statistic computed fresh.

## Files
- `detector.py` — the fused detector (`RoadObjectDetector`, `Params`)
- `run_on_video.py` — processes an .mp4, writes an annotated output video
  + a CSV log (frame, track id, box, ground-contact point, which branch(es)
  fired)
- `geometry.py` — ground-contact pixel → real-world forward/lateral
  distance + angle, using **115 cm** camera height as specified

## Running it
```
python3 run_on_video.py input.mp4 output.mp4 detections.csv [max_frames] [--no-display]
```
`--no-display` skips the live preview window (faster, and needed on headless
machines; without a display the script now falls back to this automatically).
For a live camera on the chip, swap `cv2.VideoCapture(in_path)` for
`cv2.VideoCapture(camera_index)` in `run_on_video.py` (or lift the frame
loop straight into your embedded main loop — `RoadObjectDetector.process()`
takes one BGR frame and returns confirmed tracks, nothing else changes).

## Validated against your footage
Ran on `1.mp4` (12s clip) — annotated result attached. Box colors:
`green` = motion branch, `cyan` = static branch only, `yellow` = both branches
agreed. On this clip it picked up a parked car and a stationary auto purely
from the static branch (no motion), alongside moving buses/bikes from the
motion branch, all through one tracker.

Measured **~13 FPS on this test machine's CPU at 640x480** with both branches
running in the original version (see *Speed (v2)* below for the optimised numbers). That's a starting point, not a final number for your chip — see
tuning notes below.

## Speed (v2) -- same output, ~2.9x faster detector
The detector was optimised **without changing any result**. On the 12 s test
clip (1410 frames, 640x480, single CPU core) the detections CSV is
byte-identical to the previous version and every annotated output frame is
pixel-identical, while:

| | before | after |
|---|---|---|
| detector only | 12.5 FPS | 35.8 FPS |
| end to end (decode + detect + draw + encode) | 13.4 FPS | 32.7 FPS |

What was changed (all exact -- no thresholds/params/algorithms touched):
- Pixel-wise stages (gamma, MOG2, Lab, shadow test, static branch) run only on
  the ROI rows instead of the full frame. Morphology runs on a zero-padded ROI
  window so borders behave exactly as on the full frame.
- Static branch: the Lab z-score uses three 256-entry lookup tables instead of
  float32 maths on every pixel (same float32 arithmetic, same result).
- Shadow suppression is evaluated only at foreground pixels (it could only ever
  clear pixels that were already foreground).
- Per-blob work uses the blob's bounding box instead of scanning the frame.
- The Sobel "edge cue" in the static branch was removed: it was combined as
  `dev | (edge & dev)`, which is always equal to `dev`, so it never influenced
  the output (it was pure wasted compute).
- Constants (gamma table, kernels, ROI area) are computed once, not per frame.
- `run_on_video.py` decodes and encodes on background threads so I/O overlaps
  detection; on multi-core machines `Params.PARALLEL_BRANCHES` also runs the
  static and motion branches concurrently (auto-disabled on 1 core; results are
  identical either way).

Deliberately left untouched: Shi-Tomasi corners, Lucas-Kanade flow, MOG2 and
DBSCAN/KMeans parameters and logic. What remains is mostly OpenCV's own MOG2,
corner detection and LK time. Further speed-ups (downscaling, skipping frames,
fewer corners) would change results, so they were not applied.

To re-verify equivalence after any future edit, use `verify_equivalence.py`
(runs the old and new detector side by side, frame by frame).

## Known limitations / next tuning steps
- Small, far-away objects (distant bikes near the horizon) are still missed
  by both branches — classical CV loses reliable signal at low pixel counts.
  `MIN_H_AT_HORIZON` in `Params` controls how aggressively far objects are
  filtered; lowering it catches more but raises false positives.
- The static branch can occasionally box part of a roadside hoarding/sign
  edge if it's within the ROI and has a sharp color/texture break — tighten
  `STATIC_Z_THRESH` (higher = stricter) or narrow `ROI_POLY_FRAC` to exclude
  more roadside clutter if this matters for your use case.
- `geometry.py`'s FX/FY/CX/CY intrinsics are carried over from the earlier
  calibration. If the chip's actual camera unit differs, redo a checkerboard
  calibration for it — the height (115 cm) is already set, but wrong focal
  length will silently give wrong distances even though detection is
  unaffected.
- For CPU budget on the actual embedded chip: the static branch is cheap
  (a few vectorized ops per frame); the motion branch (optical flow +
  clustering) is the heavier of the two. If the chip is tightly constrained,
  the static branch alone (skip `_motion_candidates`) is a viable
  moving+stationary-lite fallback since it re-evaluates every frame
  independently and does catch moving objects too, just with less refined
  boxes/tracking than the flow branch gives.
