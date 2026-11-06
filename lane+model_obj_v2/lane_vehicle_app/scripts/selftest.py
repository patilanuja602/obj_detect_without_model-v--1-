#!/usr/bin/env python3
"""selftest.py -- run after installation.  Checks imports, file integrity, that every detector model loads and runs, and that the lane module loads.
Optionally:  python scripts/selftest.py --video clip.mp4   also verifies the lane output is bit-identical to the baseline when no vehicle mask is set."""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print("[%s] %s %s" % ("ok" if cond else "FAIL", name, extra))


try:
    import cv2, numpy, openvino, psutil
    check("imports", True, "cv2 %s, numpy %s, openvino %s" % (cv2.__version__, numpy.__version__, openvino.__version__))
except Exception as e:
    sys.exit("[FAIL] import error: %s (activate the virtual environment: source .venv/bin/activate)" % e)
for line in open(ROOT / "models" / "SHA256SUMS"):
    h, f = line.split()
    check("sha256 " + f, hashlib.sha256(open(ROOT / f, "rb").read()).hexdigest() == h)
from lvapp import config
from lvapp.lane_engine import load_lane_module
from lvapp.vehicle_detector import VehicleDetector
import numpy as np
for f, band in (("models/yolo26n_960x352.onnx", (0.35, 1.0)), ("models/yolo26n_768x288.onnx", (0.35, 1.0)), ("models/yolo26n_960x544.onnx", None)):
    d = VehicleDetector(ROOT / f, band=band, threads=2)
    d.detect(np.random.randint(0, 255, (1080, 1920, 3), np.uint8))
    check("detector " + f, True, "input %dx%d, compile %.1fs" % (d.w, d.h, d.compile_s))
lp = load_lane_module(config.load().lane.file)
check("lane module (V6) loads", hasattr(lp, "PathDetectorV6"), "sha256 %s..." % lp.__sha256__[:8])
ap = argparse.ArgumentParser()
ap.add_argument("--video", default=None)
a = ap.parse_args()
if a.video:
    r = subprocess.run([sys.executable, str(ROOT / "eval" / "check_lane_identity.py"), "--video", a.video, "--frames", "100", "--start", "5"], capture_output=True, text=True)
    print(r.stdout.strip())
    check("lane output identical to baseline when no vehicle mask is set", r.returncode == 0)
print("\nALL GOOD" if ok else "\nPROBLEMS FOUND")
sys.exit(0 if ok else 1)
