"""Glue: runs steps 1-6 for one frame and returns a LaneResult."""
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .config import Config
from .lanes import BoundaryTrack, RoadModel, build_candidates, pick_side, t_of_y
from .markings import marking_masks
from .segments import extract_segments
from .vanishing import VanishingPointTracker


@dataclass
class LaneResult:
    frame: np.ndarray                      # processed-size BGR frame
    vp: Optional[np.ndarray] = None
    left: object = None                    # Boundary or None
    right: object = None
    y_top: float = 0.0
    y_bot: float = 0.0
    has_lane: bool = False                 # True only when BOTH boundaries are known
    car_x: float = 0.0                     # car position used (px, processed frame)
    offset_m: Optional[float] = None       # + = car right of lane centre
    lane_width_px: Optional[float] = None
    steer_deg: Optional[float] = None      # + = steer right (uncalibrated hint)
    curve: Optional[str] = None
    confidence: float = 0.0
    debug: dict = field(default_factory=dict)


class LaneDetector:
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        self._vpt = None
        self._left = self._right = None
        self._size = None
        self._ytop = None
        self._vpx_ref = None
        self._vpx_n = 0

    # ------------------------------------------------------------------
    def _init_size(self, w, h):
        self._size = (w, h)
        self._vpt = VanishingPointTracker(self.cfg, w, h)
        self._left, self._right = BoundaryTrack(self.cfg, w), BoundaryTrack(self.cfg, w)
        self._ytop = None

    def process(self, frame_bgr):
        cfg = self.cfg
        h0, w0 = frame_bgr.shape[:2]
        if w0 != cfg.proc_width:
            h = int(round(h0 * cfg.proc_width / w0))
            frame = cv2.resize(frame_bgr, (cfg.proc_width, h), interpolation=cv2.INTER_AREA)
        else:
            frame = frame_bgr
        h, w = frame.shape[:2]
        if self._size != (w, h):
            self._init_size(w, h)

        mask, yellow, lab = marking_masks(frame, cfg)
        segs = extract_segments(mask, cfg)
        vp = self._vpt.update(segs)

        res = LaneResult(frame=frame, vp=None if vp is None else vp.copy(),
                         confidence=self._vpt.confidence)
        res.car_x = cfg.car_x_frac * w
        res.debug.update(mask=mask, segs=segs)
        if vp is None:
            self._left.update(None)
            self._right.update(None)
            return res

        k3 = np.ones((3, 3), np.uint8)
        yellow_dil = cv2.dilate(yellow, k3)
        mark_dil = cv2.dilate(mask, k3)
        cands = build_candidates(segs, vp, yellow_dil, mark_dil, cfg, w, h)

        car_x = cfg.car_x_frac * w
        road = RoadModel(lab, vp, car_x, cfg, w, h)
        self._left.update(pick_side(cands, -1, car_x, vp, road, cfg, w, h))
        self._right.update(pick_side(cands, +1, car_x, vp, road, cfg, w, h))

        res.left, res.right = self._left.b, self._right.b
        res.debug["cands"] = cands
        self._finish(res, vp, car_x, w, h)
        return res

    # ------------------------------------------------------------------
    def _finish(self, res, vp, car_x, w, h):
        cfg = self.cfg
        L, R = res.left, res.right
        y_ref = h - 1.0
        res.y_bot = y_ref
        min_top = vp[1] + 0.10 * (y_ref - vp[1])
        # far end of what is drawn = what we actually saw (nearest of the two far ends)
        seen = [b.y_far for b in (L, R) if b is not None]
        if seen:
            raw = max(max(seen) - 0.03 * h, min_top)
            self._ytop = raw if self._ytop is None else 0.7 * self._ytop + 0.3 * raw
        res.y_top = float(self._ytop if self._ytop is not None else min_top)

        if L is None or R is None:
            return
        xl1, xr1 = L.x_at(1.0, vp[0]), R.x_at(1.0, vp[0])
        width = xr1 - xl1
        if not (cfg.min_lane_w_frac * w <= width <= cfg.max_lane_w_frac * w):
            return
        res.has_lane = True
        res.lane_width_px = float(width)
        centre = 0.5 * (xl1 + xr1)
        res.offset_m = float((car_x - centre) / width * cfg.lane_width_m)

        # steering hint: heading error (from VP) + lateral correction (from offset)
        f = (w / 2.0) / np.tan(np.radians(cfg.hfov_deg / 2.0))
        # 'straight ahead' = long-run average of the vanishing-point x (self-calibrating:
        # on average the car drives parallel to the road, so this absorbs camera yaw).
        self._vpx_n += 1
        a = max(0.01, 1.0 / self._vpx_n)
        self._vpx_ref = vp[0] if self._vpx_ref is None else (1 - a) * self._vpx_ref + a * vp[0]
        yaw = np.arctan((vp[0] - self._vpx_ref) / f)
        corr = np.arctan(-res.offset_m / cfg.lookahead_m)
        res.steer_deg = float(np.degrees(yaw + corr))

        # curvature of the lane centre-line (sign: + bends right)
        c_mid = 0.5 * (L.c + R.c)
        thr = 0.03 * w
        res.curve = "curve right" if c_mid > thr else "curve left" if c_mid < -thr else "straight"
        res.confidence = float(min(self._left.confidence, self._right.confidence))
