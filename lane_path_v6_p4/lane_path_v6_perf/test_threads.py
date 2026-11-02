#!/usr/bin/env python3
"""test_threads.py -- the background reader/writer must change NOTHING about which frames are processed/written or their order.
  1. _Prefetch yields exactly the frames cv2.VideoCapture.read() yields, in order, none dropped, then None; early close() does not hang.
  2. _AsyncWriter produces a byte-identical file to a direct cv2.VideoWriter fed the same frames.
  3. Errors in the writer thread surface to the caller.
Usage: python3 test_threads.py VIDEO [--module lane_path_v6.py]"""
import argparse, hashlib, importlib.util, os, sys, time
import cv2, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("--module", default="lane_path_v6.py"); a = ap.parse_args()
spec = importlib.util.spec_from_file_location("m", a.module); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
fails = 0
def check(c, msg):
    global fails
    print(("[PASS] " if c else "[FAIL] ") + msg); fails += 0 if c else 1
# 1 -------------------------------------------------------------------------------------------------------
N = 150
cap = cv2.VideoCapture(a.video); cap.set(cv2.CAP_PROP_POS_MSEC, 20000)
ref = []
for _ in range(N):
    ok, f = cap.read(); ref.append(f.copy())
cap.release()
cap = cv2.VideoCapture(a.video); cap.set(cv2.CAP_PROP_POS_MSEC, 20000)
pf = m._Prefetch(cap, depth=4); got = []
for _ in range(N):
    f = pf.get(); got.append(f)
    time.sleep(0.002)                     # slow consumer -> queue fills, reader must BLOCK, not drop
pf.close(); cap.release()
check(len(got) == N and all(g is not None and np.array_equal(g, r) for g, r in zip(got, ref)), "prefetch: %d frames, same content, same order (queue depth 4, slow consumer)" % N)
cap = cv2.VideoCapture(a.video); pf = m._Prefetch(cap, depth=2); pf.get(); t = time.time(); pf.close(); cap.release()
check(time.time() - t < 3, "prefetch: early close() returns promptly (%.2fs)" % (time.time() - t))
# end of stream -> None
short = "/tmp/_short.mp4"; vw = cv2.VideoWriter(short, cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
for i in range(7): vw.write(np.full((48, 64, 3), i * 30, np.uint8))
vw.release(); cap = cv2.VideoCapture(short); pf = m._Prefetch(cap); n = 0
while pf.get() is not None: n += 1
pf.close(); cap.release()
check(n == 7, "prefetch: end of stream gives None after exactly the %d frames in the file (got %d)" % (7, n))
# 2 -------------------------------------------------------------------------------------------------------
rng = np.random.default_rng(0); frames = [rng.integers(0, 256, (360, 640, 3), dtype=np.uint8) for _ in range(120)]
def write(path, async_):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360)); s = m._AsyncWriter(vw, depth=3) if async_ else vw
    for f in frames: s.write(f.copy())
    if async_: s.close()
    vw.release(); return hashlib.md5(open(path, "rb").read()).hexdigest()
h0, h1 = write("/tmp/_w0.mp4", False), write("/tmp/_w1.mp4", True)
check(h0 == h1, "async writer: output file byte-identical to the direct writer (120 frames, queue depth 3)  md5 %s" % h0[:10])
# 3 -------------------------------------------------------------------------------------------------------
class Bad:
    def write(self, f): raise RuntimeError("boom")
w = m._AsyncWriter(Bad()); w.write(frames[0])
try: w.close(); raised = False
except RuntimeError: raised = True
check(raised, "async writer: an exception in the writer thread is re-raised to the caller")
print("RESULT:", "PASS" if fails == 0 else "FAIL"); sys.exit(1 if fails else 0)
