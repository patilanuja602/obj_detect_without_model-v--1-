#!/usr/bin/env python3
"""Checks the installation: imports, baseline file hash, models present and loadable, one end-to-end frame."""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ok = True


def check(name, cond, msg=""):
    global ok
    print("%-48s %s %s" % (name, "OK" if cond else "FAIL", msg))
    ok &= bool(cond)


try:
    import cv2, numpy, psutil, openvino
    check("imports (cv2, numpy, psutil, openvino)", True, "cv2 %s, openvino %s" % (cv2.__version__, openvino.__version__))
except Exception as e:
    check("imports", False, str(e))
    sys.exit(1)
h = hashlib.sha256((ROOT / "baseline" / "lane_path_v6_p3.py").read_bytes()).hexdigest()
check("baseline lane file unchanged", h.startswith("e3c05550"), h[:12])
for ln in (ROOT / "models" / "SHA256SUMS").read_text().splitlines():
    sha, name = ln.split()[:2]
    p = ROOT / "models" / Path(name).name
    check("model " + p.name, p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == sha)
import numpy as np
from lvapp import config
from lvapp.pipeline import Engine
cfg = config.load()
fr = np.random.randint(0, 255, (1080, 1920, 3), np.uint8)
try:
    e = Engine(cfg, 1920, 1080, mode="par")
    e.warmup(fr)
    r = e.process(fr, 0)
    e.close()
    check("one frame through the full pipeline", r.fid == 0 and e.vdet is not None)
except Exception as ex:
    check("pipeline", False, str(ex))
print("ALL GOOD" if ok else "PROBLEMS FOUND")
sys.exit(0 if ok else 1)
