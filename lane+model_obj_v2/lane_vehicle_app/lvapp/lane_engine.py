"""
lane_engine.py -- wraps the UNMODIFIED V6 lane detector (lane_baseline/lane_path_v6_p3.py) and adds ONE optional input: a vehicle occlusion mask.

How the vehicle evidence is used (and how it is limited)
  * The V6 detector builds a single Canny edge map (`_edges`) and derives every Hough segment / candidate / fit from it.  The subclass below overrides
    only that method: after the normal edge map is computed, edge pixels inside (slightly dilated) boxes of RELIABLE vehicles are cleared.
    Result: vehicle sides, bumpers and shadows no longer create lane candidates, and lane markings that are hidden behind a vehicle simply provide no
    evidence (the existing tracker keeps the boundary alive from earlier frames, exactly as before).
  * Only reliable vehicles are used (the caller passes confirmed tracks with score >= mask_min_conf).  A mask that would cover more than
    mask_max_area_frac of the lane ROI is ignored entirely, so a wrong detection can never wipe out the lane evidence.
  * With no mask set (detector unavailable / late / failed / nothing reliable) `_edges` is the baseline function, so the output is bit-identical to
    the baseline (verified by eval/check_lane_identity.py).
  * Internal geometry is never altered; the display layer may hide the overlay under a vehicle, the lane model keeps its full corridor.
"""
import hashlib
import importlib.util

import cv2
import numpy as np

from . import config


def load_lane_module(path):
    p = config.resolve(path)
    spec = importlib.util.spec_from_file_location("lane_baseline_v6", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.__sha256__ = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return m


def make_detector_class(lp):
    class PathDetectorV6Veh(lp.PathDetectorV6):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._veh_mask = None
            self.veh_mask_frac = 0.0
            self._hold_base = self.c2["max_hold2"]          # the baseline's tracked-boundary hold limit (6 frames)
            self.occ_hold = self._hold_base                   # extended limit, used only while a reliable vehicle hides the boundary
            self.occ_min_frac = 0.25
            self.occ_hold_active = False
            self.protect_px = 0                               # >0: pixels within this distance of the TRACKED boundaries are never masked
            self._eff_mask = None

        def _build_effective_mask(self):
            """vehicle mask minus a protection band around the boundaries tracked in the previous frame: the mask may remove evidence that would create NEW
            lane candidates (vehicle edges, shadows) but never the evidence that supports an already established boundary (e.g. a road-edge barrier
            that a vehicle on the far side of it overlaps in the image)."""
            self._eff_mask = self._veh_mask
            if self._veh_mask is None or self.protect_px <= 0:
                return
            lane = self.result.get("lane") if isinstance(self.result, dict) else None
            if not lane:
                return
            prot = np.zeros_like(self._veh_mask)
            for s in ("L", "R"):
                pts = lane.get("pts_" + s)
                if lane.get(s) is not None and pts is not None and len(pts) > 1:
                    q = np.round(np.asarray(pts, float) - [self.x0, self.y0]).astype(np.int32)
                    cv2.polylines(prot, [q.reshape(-1, 1, 2)], False, 255, 2 * self.protect_px + 1)
            self._eff_mask = np.where(prot > 0, 0, self._veh_mask).astype(np.uint8)

        def _occluded_fraction(self, pts):
            """share of a boundary's points (processing px) that lie inside the vehicle mask."""
            if self._veh_mask is None or pts is None or len(pts) == 0:
                return 0.0
            h, w = self._veh_mask.shape
            xs = np.floor(np.asarray(pts)[:, 0] - self.x0).astype(int)
            ys = np.floor(np.asarray(pts)[:, 1] - self.y0).astype(int)
            ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            if not ok.any():
                return 0.0
            return float((self._veh_mask[ys[ok], xs[ok]] > 0).sum()) / len(pts)

        def process(self, frame):
            """Occlusion-aware hold: the baseline lets a boundary without measurements stay TRACKED for max_hold2 (6) frames.  If EVERY boundary that is
            currently being held lies mostly behind a reliable vehicle, the limit is raised to occ_hold for this frame; otherwise (nothing held, or a
            boundary vanished with no vehicle in front of it) the baseline limit applies unchanged."""
            hold = self._hold_base
            self.occ_hold_active = False
            if self._veh_mask is not None and self.occ_hold > self._hold_base:
                lane = self.result.get("lane") if isinstance(self.result, dict) else None
                if lane:
                    held = [(s, lane.get(s)) for s in ("L", "R") if lane.get(s) is not None and lane[s].get("age", 0) > 0]
                    if held and all(self._occluded_fraction(lane.get("pts_" + s)) >= self.occ_min_frac for s, _ in held):
                        hold = self.occ_hold
                        self.occ_hold_active = True
            self.c2["max_hold2"] = hold
            self._build_effective_mask()
            return super().process(frame)

        def set_vehicle_boxes(self, boxes, margin_x=0.08, margin_bottom=4, max_area_frac=0.35):
            """boxes: iterable of (x1, y1, x2, y2) in PROCESSING pixels.  Builds the ROI-shaped mask (or None)."""
            self._veh_mask, self.veh_mask_frac = None, 0.0
            if boxes is None or len(boxes) == 0:
                return
            h, w = self.y1 - self.y0, self.x1 - self.x0
            m = np.zeros((h, w), np.uint8)
            for b in boxes:
                bw = b[2] - b[0]
                xa = int(max(0, np.floor(b[0] - margin_x * bw - 2 - self.x0)))
                xb = int(min(w, np.ceil(b[2] + margin_x * bw + 2 - self.x0)))
                ya = int(max(0, np.floor(b[1] - 2 - self.y0)))
                yb = int(min(h, np.ceil(b[3] + margin_bottom - self.y0)))
                if xb > xa and yb > ya:
                    m[ya:yb, xa:xb] = 255
            frac = float((m > 0).mean())
            if 0 < frac <= max_area_frac:
                self._veh_mask, self.veh_mask_frac = m, frac

        def _edges(self, frame):
            e = super()._edges(frame)
            m = self._eff_mask if self._eff_mask is not None else self._veh_mask
            if m is not None:
                e[m > 0] = 0
            return e

    return PathDetectorV6Veh


class LaneEngine:
    def __init__(self, cfg, src_w, src_h):
        self.lp = load_lane_module(cfg.lane.file)
        lp = self.lp
        c = dict(lp.CFG)
        c.update(lp.PROFILES[cfg.lane.profile])
        self.pw = c["proc_w"]
        self.ph = int(round(src_h * self.pw / src_w))
        self.det = make_detector_class(lp)(self.pw, self.ph, c, cfg6={})
        self.rs = lp._ProcResizer(self.det, src_w, src_h, self.pw, self.ph, True)
        self.cfg = cfg.lane
        self.det.occ_hold = int(cfg.lane.get("occlusion_hold", self.det._hold_base))
        self.det.occ_min_frac = float(cfg.lane.get("occlusion_min_frac", 0.25))
        self.det.protect_px = int(cfg.lane.get("protect_tracked_px", 0))

    def self_check(self, frame):
        self.rs.self_check(frame)

    def set_mask(self, boxes_proc):
        self.det.set_vehicle_boxes(boxes_proc, self.cfg.mask_margin_x, self.cfg.mask_margin_bottom, self.cfg.mask_max_area_frac)

    def prepare(self, frame):
        """Processing-size frame for the lane detector (ROI-only fast resize of the baseline; rows outside the ROI are NOT valid)."""
        return self.rs(frame)

    def run(self, small):
        L, R = self.det.process(small)
        return L, R, self.det.result["lane"]

    def process(self, frame):
        return self.run(self.prepare(frame))

    def draw(self, disp, L, R, fps_algo):
        return self.det.draw(disp, L, R, fps_algo, False)
