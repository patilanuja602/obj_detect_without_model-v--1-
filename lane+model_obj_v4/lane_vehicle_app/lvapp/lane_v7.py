"""
lane_v7.py -- the verified V6 lane detector (baseline/lane_path_v6_p3.py, imported UNMODIFIED) plus a vehicle-aware edge mask.

PathDetectorV7 only overrides `_edges`: after the baseline computes its edge map, rectangles covering the BODIES of reliable vehicles
are cleared, so vehicle edges (bumpers, shadows under cars, roof lines, window frames, side-by-side car outlines) cannot become Hough
segments, seeds or boundary measurements.  Nothing else changes: with an empty mask the output is bit-identical to PathDetectorV6
(verified by scripts/compare_baseline.py --check-identical).

The mask is built from the tracker's PREDICTED boxes of reliable tracks (confirmed, not coasted, >= mask_min_hits hits,
score_ema >= mask_min_score), i.e. from frames <= t-1.  That is what lets the lane detector and the vehicle detector run in
parallel on the same frame t.  If the vehicle detector is missing / late / failing, the mask is empty and the lane detector behaves
exactly like V6.  Rectangles are shrunk horizontally (mask_inset_x) and at the bottom (mask_bottom_trim) so that lane paint right
beside a vehicle and at its ground-contact line stays visible to the detector.  The internal lane geometry is never altered by the
display clipping done in overlay.py.
"""
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASELINE_FILE = ROOT / "baseline" / "lane_path_v6_p3.py"


def load_baseline(path=BASELINE_FILE):
    """Import the baseline lane module from its file (never copied or edited)."""
    key = "lane_path_v6_p3_baseline"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[key] = m
    spec.loader.exec_module(m)
    return m


LP = load_baseline()


class PathDetectorV7(LP.PathDetectorV6):
    def __init__(self, w, h, cfg=None, cfg2=None, cfg3=None, cfg4=None, cfg6=None, mask_cfg=None):
        super().__init__(w, h, cfg, cfg2=cfg2, cfg3=cfg3, cfg4=cfg4, cfg6=cfg6)
        self.mc = dict(mask_enabled=True, mask_inset_x=0.06, mask_bottom_trim=0.04, mask_min_h_px=24)
        if mask_cfg:
            self.mc.update(mask_cfg)
        self._rects = []                                  # ROI-coordinate rectangles (x0, y0, x1, y1)
        self.mask_boxes_used = 0

    def set_vehicle_boxes(self, boxes_proc):
        """boxes_proc: iterable of (x1, y1, x2, y2) in PROCESSING-frame px.  Replaces the mask for the next process() call."""
        rects = []
        if self.mc["mask_enabled"]:
            for x1, y1, x2, y2 in boxes_proc:
                bh, bw = y2 - y1, x2 - x1
                if bh < self.mc["mask_min_h_px"]:
                    continue
                a = int(round(x1 + self.mc["mask_inset_x"] * bw)) - self.x0
                b = int(round(x2 - self.mc["mask_inset_x"] * bw)) - self.x0
                c = int(round(y1)) - self.y0
                d = int(round(y2 - self.mc["mask_bottom_trim"] * bh)) - self.y0
                a, c = max(a, 0), max(c, 0)
                b, d = min(b, self.x1 - self.x0), min(d, self.y1 - self.y0)
                if b > a and d > c:
                    rects.append((a, c, b, d))
        self._rects = rects
        self.mask_boxes_used = len(rects)

    def _edges(self, frame):
        e = super()._edges(frame)
        for a, c, b, d in self._rects:
            e[c:d, a:b] = 0
        return e
