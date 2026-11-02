# lane_path_v6_p3 -- phase 3 (display-resize pipeline stage)

All numbers: SINGLE-CORE sandbox, OpenCV 4.13, nD_20.mp4 (1920x1080), NOT the ThinkPad T480. No T480 number exists yet.

## What the profile of the previous version (lane_path_v6.py) showed (median ms/frame)
decode 12.8 | resize_proc 1.3 | process() 7.4 | resize to display 10.0 | draw 0.55   (HoughLinesP 2.5 is the biggest piece of process())
Pipeline without I/O = 19.5 ms (51 FPS); the largest single item left is the display resize (1080p->720p, INTER_AREA, non-integer 1.5x).
That resize only produces the background image of the output video; the detector never sees it.

## Changes (lane_path_v6_p3.py; v6 and v6_pre_opt untouched for rollback)
1. _Stage thread: display-size resize runs on its own thread between decoder and main loop (same frames, same order, nothing dropped,
   bounded queues). Output is bit-identical to v6 (150-frame check + full-stream count 450 = 450).
2. --display-interp linear (opt-in, default area): 2.3 ms instead of 11.0 ms; background-video pixels differ from INTER_AREA by
   mean 0.74/255 (max 16 at sharp edges). Detection is unaffected. Default stays bit-identical.
3. Tried and rejected: row-banded INTER_AREA (bit-identical, but 10.4 ms serial -> only useful with threads, which this sandbox cannot time).

## Measured here (1 core, 232 frames, with write, interleaved single runs -> indicative only)
v6 threads off 23.1 FPS | p3 threads off 23.2 | p3 threads on 24.2 | p3 threads on + linear 29.8
On one core threads cannot add parallelism, so the benefit of the stage thread is NOT demonstrated here.
Stage budget for a multi-core run: decode 12.8 | display resize 10 | main (resize_proc+process+draw) ~9.4 | write ~7.5, each on its own
thread -> the slowest stage (decode) would bound ~78 FPS in theory. That is a calculation, not a measurement.

## Verification
tests/test_v6.py 32/32 | verify_perf_equivalence.py (p3 vs pre_opt): 600 frames, all internal results and drawn pixels bit-identical |
stage order/frames/display equality over 150 frames. process() and draw() are unchanged code.

## Remaining bottlenecks
decode (12.8 ms, ffmpeg H.264, 1080p) | HoughLinesP 2.5 | video write (mp4v) 7.5 | resize_proc 1.3 | _sliding2 0.9

## Run on the T480
sudo cpupower frequency-set -g performance
python3 bench_cli.py lane_path_v6_pre_opt.py $VIDEO /tmp/o.mp4 --start 20 --duration 10
python3 bench_cli.py lane_path_v6_p3.py $VIDEO /tmp/o.mp4 --start 20 --duration 10 --threads on
python3 bench_cli.py lane_path_v6_p3.py $VIDEO /tmp/o.mp4 --start 20 --duration 10 --threads on --display-interp linear
# also try --cv-threads 1 / 2 / 4
