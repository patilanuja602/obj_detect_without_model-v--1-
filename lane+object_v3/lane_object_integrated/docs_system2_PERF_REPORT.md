# lane_path_v6 performance work

All numbers below were measured in a SINGLE-CORE sandbox (Xeon 2.8 GHz, OpenCV 4.13, 1 cv2 thread), NOT on the ThinkPad T480.
Video: 9977360-uhd-2560-1440-30fps, start 20 s, 300 measured frames after 30 warm-up, processing 640x360, display 1280x720.
No frame skipping, no resolution change, same frames for every variant. The 30-35 FPS target was NOT demonstrated; threaded
mode (the part expected to help on a multi-core T480) could not be speed-tested on one core.

## Files
- lane_path_v6.py            optimised (drop-in)        - lane_path_v6_pre_opt.py  rollback = your current working version (with clip)
- lane_path_v6_before.py     original V6 (benchmark A)   - perf_opt.patch           pre_opt -> optimised
- overlay_clip_original.patch  your original->clip patch
- bench_v6.py (stage profiler), bench_table.py (interleaved A/B/C), bench_cli.py (real main(), wall clock)
- verify_perf_equivalence.py, test_threads.py, tests/test_v6.py

## Original vs clipped source
Only draw(), the CLI options, CFG6 keys and a docstring changed (_clip_top is called only from draw()). Detection/tracking untouched.
Draw-only: original 3126 us vs clipped 3095 us per frame; internal results identical on 200 real frames (verify_overlay_clip.py).
Whole pipeline A vs B: 24.7 vs 24.6 FPS (no write), 20.0 vs 20.4 FPS (write). The overlay is not the cause.

## Profile (median ms/frame, sandbox)
decode 16 | write(mp4v) 7.5 | resize to proc 6.7 | process() 10 | resize to display 2.3 | draw 2.7
Inside process() (exclusive): HoughLinesP 3.2, _sliding2 1.5, _ridge_pairs 0.9, CLAHE 0.7, Canny 0.6, cvtColor 0.5, rest < 0.4 each.
Main bottleneck: frame I/O and resizing, not the lane algorithm. "FPS (algo)" on the HUD counts process() only.

## Changes
1. ROI-only processing resize (bit-identical for integer scale, self-checked, auto fall-back; --no-roi-resize disables). 6.7 -> 2.7 ms.
2. draw(): green fill blended only in the polygon bounding rectangle (_blend_fill). 2.7 -> 0.9 ms.
3. Background decode thread + background writer thread (order kept, nothing dropped, bounded queues). Auto-on with >= 2 CPUs; --threads on|off.
4. End-to-end FPS line printed; optional --cv-threads N.

## Results (median of 5 interleaved rounds)
| | A original | B current | C optimised (no threads) |
|---|---|---|---|
| pipeline w/o decode/write, ms | 21.25 | 21.78 | 16.63 |
| E2E FPS, no write | 24.7 | 24.6 | 28.1 |
| E2E FPS, with write | 20.0 | 20.4 | 22.3 |
Real CLI (10 s, write, 3 rounds): A 20.1/19.1/21.7, B 20.1/20.6/21.4, C threads off 23.3/24.2/25.3, C threads on (1 core) 21.4/24.8/23.7.
process() is unchanged code; 9.7 vs 10.4 ms between B and C is run-to-run noise.

## Verification
- verify_perf_equivalence.py: 600 consecutive frames, 4 segments (LANE/ONE_SIDED/NONE): processing input ROI, edges, L channel,
  full internal result (lane, corridor, partial, mode...) and drawn pixels (debug on/off) all bit-identical; 600 random-polygon blend tests identical.
- tests/test_v6.py 32/32 on both versions. test_threads.py: order/no-drop/byte-identical writer output.
- The internal corridor (result["lane"]["corridor"], pts_L/pts_R, extent_y) is unchanged and available.

## Caveats
- ROI-only resize leaves rows outside the ROI zero and reuses the buffer. A future vehicle detector needing the full processing frame
  must use _ProcResizer.full(frame) or --no-roi-resize.
- With threads on and a starved CPU the HUD "FPS (algo)" includes time taken by other threads.
- draw() blends into the array passed in (it already drew into it in other states).

## Commands (run from this folder, on the T480; set VIDEO)
VIDEO=/path/to/9977360-uhd-2560-1440-30fps-yn58shx6.mp4
# 0. best case: AC power, performance governor
sudo cpupower frequency-set -g performance
# 1. correctness
python3 tests/test_v6.py
python3 verify_perf_equivalence.py $VIDEO --before lane_path_v6_pre_opt.py --after lane_path_v6.py
python3 test_threads.py $VIDEO --module lane_path_v6.py
# 2. stage profile (use --write out.mp4 to include writing)
python3 bench_v6.py $VIDEO --variant lane_path_v6_pre_opt.py --mode fine --start 20 --frames 300 --warmup 30 --write /tmp/o.mp4
# 3. fair A/B/C table (sequential loop, same loop for all)
python3 bench_table.py $VIDEO --variants lane_path_v6_before.py lane_path_v6_pre_opt.py lane_path_v6.py --labels "A original" "B current" "C optimised" --rounds 5 --write
# 4. real CLI, wall clock, threading A/B (threads is where the T480 gain is expected)
for v in lane_path_v6_before.py lane_path_v6_pre_opt.py; do python3 bench_cli.py $v $VIDEO /tmp/o.mp4 --start 20 --duration 10; done
python3 bench_cli.py lane_path_v6.py $VIDEO /tmp/o.mp4 --start 20 --duration 10 --threads off
python3 bench_cli.py lane_path_v6.py $VIDEO /tmp/o.mp4 --start 20 --duration 10 --threads on
python3 bench_cli.py lane_path_v6.py $VIDEO /tmp/o.mp4 --start 20 --duration 10 --threads on --cv-threads 2   # also try 1, 4
# normal use
python3 lane_path_v6.py $VIDEO out.mp4 --start 20 --duration 10
# rollback
cp lane_path_v6_pre_opt.py lane_path_v6.py     (or: patch -R lane_path_v6.py < perf_opt.patch)
