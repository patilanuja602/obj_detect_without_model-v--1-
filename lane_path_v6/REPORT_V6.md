# lane_path_v6 — report (sandbox measurements only; NOT the ThinkPad T480; no ground truth => no accuracy claim)

## 1. Files received and what I could inspect
- `9977360-...mp4` (2560x1440, 249 s): identical (md5) to the earlier upload. Used from t=4.337 s (frame 130) for 93 s = 2787 frames.
- `Screencast_from_27-10-26_03_08_53_PM_IST.webm` (298 s, variable frame rate): fully decoded. It is **V5 in --debug mode** (ROI trapezoid + edge map visible, HUD "[to 87% of ROI]"). Only 2780 distinct frames (~9.3 displayed fps) = 93 s of source. I aligned it by image matching (screencast frame j = source frame 130+j, drift <= 6 frames); my V5 re-run is therefore a close reproduction, not pixel-exact (the screencast has the debug overlay).
- Older versions: V2, V3, V4, V5 sources from the earlier turns (still in my sandbox); you deleted yours. V6 needs none of them.
- I did not receive a "LinkedIn video" as such, only the files above.
- Viewed frame by frame (every consecutive frame): 219-239 (V3, V6), 2290-2310 (V5), 2412-2432 (V5, V6), 2322-2342 (V6), 2340-2360 (V6), 2540-2560 (V5), 2700-2720 (V5, V6); earlier turn: 140-160, 224-244, 560-580, 740-760, 1086-1106 (V4/V5). Everything else only through per-frame decision logs and statistics.

## 2. Which version is the baseline, and why
Measured on identical frames (V2 output == V3 output exactly):
- **V4 and V5 have bit-identical boundary geometry** (0 differences in 5574 boundary states); they differ only in what is displayed. So the "previous vs newer" difference you saw is the decision/drawing layer, not road-following.
- **V2/V3 are less stable at following the road**: 48 abrupt lane-edge jumps (>25 px) and 26 width jumps; e.g. frames 219-227 and 273-277: the left boundary hops 160-205 px between consecutive frames onto the dashed marking in the middle of the lane (viewed). V4/V5 have 0 / 5 jumps.
- V3 shows a lane in more frames (78.7%) than V4 (66.7%) but 1121 frames of crossed lines; V4's blanking comes from its decision layer (type "unsupported" / "contradicted" => UNCERTAIN).
=> Baseline = the V4/V5 measurement + tracking core (merged unchanged into V6; verified identical to V4 on 300 real frames before any V6 change). V5's decision layer is replaced. I cannot know which version you call "previous"; this is what the evidence supports.

## 3. Why the V5 overlay disappears (689 of 2787 frames = 24.7%)
boundary type "unsupported" 413 | "boundary unavailable" (second boundary missing) 216 | "contradicted" 57 | other 1. Plus 31 green<->orange switches and 331 amber frames. Crossing: V4 926 frames, V3 1121 frames of crossed/overlong lines; V5 0 (it clips).

## 4. Timestamped failures (V5 unless stated; source time = frame/29.97)
| time (frames) | observed | cause |
|---|---|---|
| 7.3-8.2 s (219-244) | V5 blank 236-244 (9 frames) while road unchanged; V3 left line hops to the centre dashes (219-220, 226-227) and curls (235-239) | left type "unsupported"/contradicted => UNCERTAIN (V5); false candidate (V3) |
| 77.7-77.9 s (2328-2333) | left boundary jumps onto the edge of the painted "ETC" symbol, lane halves for 6 frames (V4/V5 core) | road symbol accepted after 3 consistent frames. **V6: not displayed** (held at x=60, dashed, until the real boundary returns) |
| 78.2-78.8 s (2344-2359) | right boundary weak among orange chevrons; V5 blank for 16 frames (from its log; not viewed), V6 (viewed) keeps the lane <= 10 frames then one-sided | evidence "unsupported" |
| 80.0-80.7 s (2399-2424) | corridor ends flat at 40-50% of ROI (curved road, flare/cross in far field): correct clipping, V5 and V6 | valid part preserved |
| 80.9-90.5 s (2425-2712) | 288-frame blackout. 2425-2432: left boundary good, right lost, whole overlay removed. 2540-2560: toll-plaza apron, no markings at all (honest "no lane"). 2700-2712: only the right wall boundary exists | one boundary only => V5 removes everything |
Dashed centre marking inside the corridor (~7.3-8.2 s): whether the corridor is one lane or two is not decidable from the footage.

## 5. V6 changes (single file, `PathDetectorV6`, Part 5)
Always-green overlay (line style = source: solid observed, dashed tracked, dotted inferred) | corridor ends at the FIRST invalid row (converged/crossed/flare), found on a 128-row scan (~1 px), valid part kept, fill+lines+HUD share it, far end may rise <= 4 px/frame | one boundary: tracked position > ROI-edge (only if width plausible vs recent lane width AND the edge would not move the boundary >30 px from where it was last displayed) > thin one-sided line + note | outlier rejection (1-2 conflicting frames), genuine change after 3 consistent (6 if the corridor would become >1.5x wider or narrower) | new tracks need 2 consistent frames before display | type hysteresis 3 up / 5 down; an "unsupported" boundary stays displayable <= 10 frames | tracked boundaries <= max_hold2 = 6 frames, then removed. Classification (paint/edge) stays internal.

## 6. Tests (`python3 tests/test_v6.py`, run): 26/26 pass
Cases 1-18 of your list plus hysteresis, no-evidence, green-in-every-state, ROI-jump and narrowing-symbol cases; displayed pixels are checked, and every rendered frame is scanned for amber/blue/red. History: first run 21/24; failures = one real design gap (a new track after expiry was displayed at once => added 2-frame probation) and two test-design errors (no lane-width history; periodic jitter that repeated itself). Two tests (T19b, T21b) were added after real-footage problems. Vehicle/divider cases are geometric simulations, not real vehicle detection.

## 7. Same-frame comparison, 2787 frames (decision logs; proxies, not accuracy)
| | V3 | V4 | V5 | **V6** |
|---|---|---|---|---|
| lane displayed % | 78.7 | 66.7 | 75.3 | **80.4** |
| display on/off events | 158 | 321 | 73 | **32** |
| 1-frame gaps | 19 | 89 | 8 | **3** |
| gaps >= 10 frames / longest (frames) | 11 / 200 | 11 / 295 | 9 / 288 | 5 / 311 |
| lines crossed (w<0) frames | 1121 | 926 | 0 | **0** |
| lane edge jumps >25 px | 48 | 0 | 5 | 7 |
| width jumps >25% | 26 | 0 | 2 | **0** |
| orange frames / green<->orange switches | 353 / 267 | 310 / 161 | 331 / 31 | **0 / 0** |
| shown with tracked/inferred boundary | 240 | 26 | 199 | 284 |
| far-end change >8 px | 268 | 230 | 183 | **57** |
First 20 s (600 frames): displayed 94.0% (V5 84.0), on/off 13 (32), crossings 0. Remaining V6 gaps are mostly long and honest (toll apron). V6 shows a tracked/inferred boundary in 10.2% of frames (bounded). Edge jumps 7 > V4's 0: tracked boundaries re-acquired 25-45 px away (e.g. 638, 779, 888, 902, 2889) and one ROI-edge case (2741) - not fully solved.

## 8. ROI fallback: is it safe? Measured on frames with both boundaries (1958)
Median lane width 303 px (47% of frame). Hiding the right boundary: ROI right edge gives 1.77x the width, median error 233 px, passes the width gate 0.4%. Hiding the left: 1.35x, error 103 px, 33% pass the width gate but 0% within 30 px. => on this road the ROI edge is almost always an implausible stand-in (it includes the next lane). V6 implements it, gates it, and it was accepted in 1 frame; one-sided frames therefore show the observed boundary only (337 frames). Unit tests prove the mechanism on ROIs where it is plausible.

## 9. Performance (sandbox: 1 core x86_64, Python 3.12, OpenCV 4.13; 450 frames, interleaved)
process(): V6 6.23 ms (p95 8.78) | V5 6.07 (8.68) | V4 5.85 (8.25) => V6 ~ +6% vs V4. End-to-end (decode+resize+process+draw+write): V6 34.8 FPS (33.4-36.2), V5 36.8, V4 35.4; ~98% of one core in all. T480 not measured; run the benchmark there.

## 10. Remaining limitations
Boundary accuracy unmeasured (no ground truth) | false boundaries on road symbols/vehicles are only mitigated, not detected | toll apron/no-marking areas give no lane (by design) | 337 one-sided frames | ROI fallback rarely applicable here | 93 of 249 s of the source processed | `mountain` profile and start frame inferred | thresholds heuristic | tests simulate vehicles/dividers geometrically.

## 11. Commands
```
python3 lane_path_v6.py INPUT.mp4 OUTPUT.mp4 --start 4.337 --duration 93 [--debug] [--show]
python3 tests/test_v6.py
python3 code/harness_v6.py INPUT.mp4 out/v6 --start 4.337 --duration 93          # per-frame decision log + video
python3 code/bench_v6.py INPUT.mp4 --impl lane_path_v6.py:PathDetectorV6 --frames 450 --reps 5 --e2e-reps 3
```
