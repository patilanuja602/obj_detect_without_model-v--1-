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
python3 run_on_video.py input.mp4 output.mp4 detections.csv [max_frames]
```
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
running. That's a starting point, not a final number for your chip — see
tuning notes below.

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
