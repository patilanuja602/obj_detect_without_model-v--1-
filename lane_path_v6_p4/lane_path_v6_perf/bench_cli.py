#!/usr/bin/env python3
"""bench_cli.py -- END-TO-END benchmark of a variant's real main() (decode -> resize -> detect -> draw -> write), wall-clock.

Runs the variant's own main() in this process with the same argv for every variant.  The final ffmpeg re-encode is disabled
(shutil.which patched) so that it is identical/absent for all variants; it is NOT part of the per-frame work.  Frames processed are
read from the variant's own 'frames=N' line.  The output file is hashed so identical output can be asserted.

  python3 bench_cli.py VARIANT.py VIDEO OUT.mp4 --start 20 --duration 10 [extra main() args, e.g. --threads on]
"""
import contextlib, hashlib, importlib.util, io, os, re, shutil, sys, time

import types
import cv2 as _cv2
variant, video, out = sys.argv[1:4]
extra = sys.argv[4:]
spec = importlib.util.spec_from_file_location("v6mod", variant)
mod = importlib.util.module_from_spec(spec); sys.modules["v6mod"] = mod; spec.loader.exec_module(mod)
# time the loop only: from the return of the FIRST read (after the seek to --start) to cap.release() (after the writer is flushed)
T = {}
class _Cap:
    def __init__(self, *a, **k): self._c = _cv2.VideoCapture(*a, **k)
    def read(self):
        r = self._c.read()
        T.setdefault("first", time.perf_counter())
        return r
    def release(self):
        T["end"] = time.perf_counter(); return self._c.release()
    def __getattr__(self, k): return getattr(self._c, k)
_ns = types.SimpleNamespace(**{k: getattr(_cv2, k) for k in dir(_cv2) if not k.startswith("__")})
_ns.VideoCapture = _Cap
mod.cv2 = _ns
shutil.which = lambda *a, **k: None                      # skip the libx264 re-encode for every variant
sys.argv = ["lane_path_v6.py", video, out] + extra
buf = io.StringIO()
t0 = time.perf_counter()
with contextlib.redirect_stdout(buf):
    mod.main()
wall = time.perf_counter() - t0
loop = T["end"] - T["first"]
txt = buf.getvalue()
m = re.search(r"frames=(\d+)", txt)
n = int(m.group(1)) if m else -1
algo = re.search(r"algo-only: mean (\d+) FPS", txt)
h = hashlib.md5(open(out, "rb").read()).hexdigest()[:12]
print("%-26s frames=%d loop=%.2fs (wall incl. seek/startup %.1fs)  END-TO-END %.2f FPS  (algo-only HUD %s FPS)  out-md5=%s  %s"
      % (os.path.basename(variant), n, loop, wall, n / loop, algo.group(1) if algo else "?", h,
         " ".join(re.findall(r"perf: .*", txt))))
