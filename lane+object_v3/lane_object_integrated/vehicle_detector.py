#!/usr/bin/env python3
"""
vehicle_detector.py -- model-free (classical CV) vehicle candidate detector for the V6 lane pipeline.

OpenCV + NumPy only.  No pretrained weights, no learned classifier, no network access.

What it does, per frame (all coordinates are V6 PROCESSING-FRAME pixels, default 640 x 360):
  1. Builds a SEARCH REGION from V6's corridor (result["lane"]["corridor"]: ys, xl, xr) widened sideways so that
     neighbouring lanes are covered.  If V6 is not in state LANE, the policy `unreliable_policy` decides what happens
     ("suppress" = no detections, "fallback" = search the static V6 ROI trapezoid, detections flagged road_ref=ROI_FALLBACK).
  2. Estimates an adaptive ROAD APPEARANCE MODEL (median Lab colour per image row, sampled inside the corridor, EMA over time)
     so that anything that does NOT look like road surface is a candidate, whether it moves or not.
  3. Candidate mask = (deviation from road model) OR (dense edges), inside the search region -> closing -> connected components.
  4. Scores every candidate with explicit geometric / edge / road-contact rules (heuristic score, NOT a probability).
  5. Lightweight tracker (greedy IoU / centre-distance) -> track ids, confirmation, bounded coasting.
  6. Image-space ground-contact estimate and lane association against the V6 corridor (never extrapolated).

Detection states (see VehicleDetection.state):
  CANDIDATE   scored but below the accept threshold (only reported when debug=True)
  ACCEPTED    passed the heuristic accept threshold this frame, track not yet confirmed
  TRACKED     observed this frame AND its track is confirmed
  PREDICTED   NOT observed this frame; box is extrapolated from the track (coasting, bounded)

A box is a heuristic hypothesis, never proof of a vehicle.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ======================================================================================================
# Configuration (every value is documented; defaults were tuned on the two supplied videos only)
# ======================================================================================================
VCFG = dict(
    # ---- search band / region -------------------------------------------------------------------
    band_top=0.52,            # top row of the analysis band (fraction of frame height); vehicles' upper parts may lie above the horizon
    horizon_frac=0.60,        # horizon row used only when no width model is available (fraction of height)
    region_margin=0.7,        # wide pass: search region widened by this many LOCAL LANE WIDTHS on each side of the corridor
    tight_margin=0.10,        # tight pass: region = corridor widened by only this much (finds host-lane vehicles that touch a wall/barrier)        # search region is widened by this many LOCAL LANE WIDTHS on each side of the corridor
    unreliable_policy="fallback",   # V6 state != LANE: "suppress" (no detections) | "fallback" (search V6 ROI trapezoid, flagged)
    fallback_score_bonus=0.08,      # extra score required in fallback mode (less geometric evidence)
    exclude=(),               # tuple of (x0,y0,x1,y1) FRACTIONS to blank out (e.g. a watermark)
    # ---- road appearance model ------------------------------------------------------------------
    road_sample_cols=16,      # columns sampled per row inside the corridor core
    road_sample_step=6,       # row step of the sampled rows
    road_core=0.30,           # fraction of lane width kept on each side when sampling the road colour
    road_ema=0.15,            # temporal smoothing of the road model (slow: a vehicle must not become "road")
    road_gate_l=16, road_gate_c=10,   # a row's new road sample updates the model only if it is this close to the current model
                              # (rejects vehicles that cover the corridor core); ...
    road_force=60,            # ... unless it has been rejected this many consecutive frames (real lighting change: re-sync)
    # ---- candidate mask -------------------------------------------------------------------------
    dev_l=26,                 # |L - L_road| above this = not road-like (lightness)
    dev_c=16,                 # |a - a_road| + |b - b_road| above this = not road-like (colour)
    grad_t=60,                # |gx|+|gy| above this = strong edge pixel
    dens_k=9,                 # edge-density window (px)
    dens_t=70,                # edge density (0..255) above this joins the candidate mask
    close_w=5, close_h=5,     # morphological closing kernel (joins body/window/lights)
    paint_bright=38,          # L above (road L + paint_bright) AND near-neutral colour = paint candidate; only its THIN part is carved out
    paint_neutral=18,         # |a-128|+|b-128| below this = near-neutral (white paint); coloured/yellow paint is handled by width rules only
    run_min=0.22, run_max=1.20,   # a "bounded run" (non-road pixels in one image row, road-like on BOTH sides) must be between
                              # run_min and run_max * expected lane width at that row: kills lane paint (too thin), barrier walls and
                              # merged road-side clutter (touch non-road on one side / too wide)
        run_bands=8,              # row bands (each with its own width kernel) for the thin-structure filter
    flank=3,                  # px on each side of a run that must be road-like
    split_ratio=0.9,          # candidate boxes wider than split_ratio * lane width (at the row) are split at column-profile valleys
    split_valley=0.30,        # a column belongs to a part if its mask count >= split_valley * the box's max column count
    split_gap=3,              # valleys narrower than this (px) do not split
    dark_t=30,                # L below (road L - dark_t) = "dark seed" (underside shadow / tyres / dark body); needs only a NON-dark flank
    top_est_max_w=45,         # the top estimate is only allowed for boxes at most this wide (px): larger vehicles must show their body
    flat_min=0.45,            # larger boxes must be at least flat_min * width tall (measured), else rejected as "flat"
    top_est_penalty=0.85,     # score multiplier when the top was estimated (less evidence)
    height_ratio=0.80,        # when the measured box is lower than height_ratio*width (distant vehicle: body cannot be separated from the
                              # background), the top is ESTIMATED as bottom - height_ratio*width (flagged parts["top_est"]=1)
    above_horizon_px=22,      # search region may reach this many px above the horizon (upper part of distant vehicles)
    min_area=30,              # smallest connected component kept (px)
    # ---- scoring --------------------------------------------------------------------------------
    size_lo=0.18, size_ok_lo=0.35, size_ok_hi=1.05, size_hi=1.45,   # box width / expected lane width at the contact row
    aspect_lo=0.55, aspect_ok_lo=0.85, aspect_ok_hi=2.4, aspect_hi=3.6,   # box width / height
    fill_min=0.22,            # candidate mask fill ratio below which the shape is too sparse
    min_below_horizon=8,      # contact row must be at least this many px below the horizon (kills the vanishing-point clutter)
    min_box_w=12,             # narrower boxes are not reported (too little evidence at 640 px processing width)
    road_hard=0.25,           # hard reject if less than this fraction of the strip under the box looks like road
    edge_dens_min=0.50,       # minimum fraction of strong-edge pixels inside the box (hard reject below; also scales the edge sub-score).
                              # TUNED on the small annotated set (sweep 0.04..0.58): the strongest single separator found. Re-tune on new footage.
    road_below_min=0.35,      # fraction of the strip under the box that must look like road
    inside_min=0.50,          # fraction of the box's bottom span that must lie inside the search region
    thr_candidate=0.40,       # score to be reported as candidate (debug)
    thr_accept=0.55,          # score to be an accepted heuristic detection
    # ---- tracking -------------------------------------------------------------------------------
    track_iou=0.20, track_dist=0.9,   # association: IoU >= track_iou OR centre distance <= track_dist * max(box w)
    confirm_hits=3,           # observations needed (within confirm_window frames) to confirm a track
    confirm_window=5,
    max_coast=6,              # frames a CONFIRMED track may be predicted without observation
    max_coast_tentative=1,
    box_ema=0.6,              # weight of the new observation in the smoothed box
    nms_iou=0.45, nms_contain=0.8,    # duplicate suppression among candidates
    # ---- association ----------------------------------------------------------------------------
    assoc_tol=0.08,           # contact within this fraction of lane width of a boundary -> AMBIGUOUS
    width_hold=90,            # frames a previous width model may be reused when the corridor is lost
    # ---- accuracy upgrade (full-resolution refinement; active only when the source frame is passed as `full=`) --------------
    kf_q=0.012, kf_r_lat=2.0, kf_r_bot=1.5, kf_r_top=4.0,   # Kalman: accel. noise (x box width, px/frame^2), measurement sigmas (px)
    refine=True,              # run vehicle_refine.RearRefiner on every proposal (needs process(..., full=frame))
    ub_scan=True,             # extra proposals from the under-body edge scan (dark mass -> road transition)
    ub_min=0.18, ub_max=1.15, # under-body run width / expected lane width at that row
    ub_thick=3,               # dark rows required above an edge pixel (proc px)
    ub_road_dl=18,            # |L - L_road| below this = road-like (pixel under the dark mass)
    ub_dy=4,                  # max change of the under-body bottom row between neighbouring columns (proc px)
    ub_gap=3,                 # horizontal closing (proc px) joining the edge fragments of one under-body line
    rescue=("flat", "sparse", "no_edges", "size_small", "aspect"),   # stage-1 rejections that are re-examined by the refiner
    refine_max=8,             # at most this many proposals are refined per frame (best first)
    keep_unrefined=False,     # keep stage-1 accepted boxes that the refiner could not verify (flagged refined=False)
)


class _Kf1:
    """Scalar constant-velocity Kalman filter (position, velocity) used for every box edge / contact coordinate in SOURCE pixels.
    q = process noise (px^2 per frame^2 of acceleration), r = measurement variance (px^2)."""
    __slots__ = ("x", "v", "p00", "p01", "p11")

    def __init__(self, x, var=25.0):
        self.x, self.v, self.p00, self.p01, self.p11 = float(x), 0.0, var, 0.0, 25.0

    def predict(self, q):
        self.x += self.v
        p00 = self.p00 + 2 * self.p01 + self.p11 + 0.25 * q
        p01 = self.p01 + self.p11 + 0.5 * q
        self.p00, self.p01, self.p11 = p00, p01, self.p11 + q

    def update(self, z, r):
        sg = self.p00 + r
        k0, k1 = self.p00 / sg, self.p01 / sg
        y = z - self.x
        self.x += k0 * y
        self.v += k1 * y
        p00 = (1 - k0) * self.p00
        p01 = (1 - k0) * self.p01
        self.p11 -= k1 * self.p01
        self.p00, self.p01 = p00, p01


@dataclass
class VehicleDetection:
    """All coordinates: V6 processing-frame pixels (x right, y down).  `observed` tells observation vs. estimate."""
    track_id: int                       # -1 = untracked candidate
    state: str                          # CANDIDATE | ACCEPTED | TRACKED | PREDICTED
    box: Tuple[float, float, float, float]       # x0, y0, x1, y1 (smoothed for tracked detections)
    raw_box: Tuple[float, float, float, float]   # raw measurement of THIS frame (== box for PREDICTED it is the last one)
    center: Tuple[float, float]
    contact: Tuple[float, float]        # estimated ground-contact point, IMAGE-SPACE (not metric)
    contact_method: str                 # "bbox_bottom" | "lower_edge"
    score: float                        # heuristic 0..1 (NOT a probability)
    parts: dict                         # per-rule sub-scores, for debugging
    age: int = 0                        # frames since the track was created
    hits: int = 0                       # observations of the track
    miss: int = 0                       # consecutive frames without observation
    observed: bool = True               # False for PREDICTED
    lane: str = "UNKNOWN"               # IN_CORRIDOR | LEFT_OF_CORRIDOR | RIGHT_OF_CORRIDOR | AMBIGUOUS | UNKNOWN
    lane_reason: str = ""
    road_ref: str = "CORRIDOR"          # CORRIDOR | ROI_FALLBACK
    reject: str = ""                    # reason (only for rejected candidates)
    # ---- accuracy upgrade: SOURCE-pixel results of the full-resolution refiner (None / defaults when not refined) ----------
    refined: bool = False
    box_src: Optional[Tuple[float, float, float, float]] = None     # tight box, source px, sub-pixel
    top_estimated: bool = False
    contact_src: Optional[Tuple[float, float]] = None                # axle ground point, source px; None => UNCERTAIN
    contact_hint_src: Optional[Tuple[float, float]] = None           # coarse point (always set; NOT precise if UNCERTAIN)
    contact_quality: str = "BBOX_BOTTOM"                             # TWO_TYRES | ONE_TYRE | UNCERTAIN | BBOX_BOTTOM (legacy)
    contact_detail: str = ""
    contact_sigma: float = 0.0                                       # claimed 1-sigma pixel error of contact_src
    tyres_src: Optional[list] = None                                 # individual tyre contact points found (source px)
    evidence: float = 0.0
    src_scale: float = 0.0                                           # source px per processing px (for mapping *_src fields)


@dataclass
class VehicleFrame:
    detections: List[VehicleDetection] = field(default_factory=list)   # ACCEPTED / TRACKED / PREDICTED
    candidates: List[VehicleDetection] = field(default_factory=list)   # scored but not accepted (debug=True only)
    rejected: List[VehicleDetection] = field(default_factory=list)     # hard-rejected (debug=True only)
    lane_state: str = "NONE"
    road_ref: str = "NONE"              # CORRIDOR | ROI_FALLBACK | SUPPRESSED
    scale_source: str = ""              # corridor | held | prior
    notes: str = ""
    timing: dict = field(default_factory=dict)     # ms: candidates, scoring, tracking, total
    region: Optional[np.ndarray] = None            # search-region polygon (debug)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _ramp(x, lo, ok_lo, ok_hi, hi):
    """0 below lo, rising to 1 at ok_lo, 1 until ok_hi, falling to 0 at hi."""
    if x <= lo or x >= hi:
        return 0.0
    if x < ok_lo:
        return (x - lo) / max(ok_lo - lo, 1e-6)
    if x <= ok_hi:
        return 1.0
    return (hi - x) / max(hi - ok_hi, 1e-6)


class VehicleDetector:
    def __init__(self, w, h, roi_poly=None, cfg=None):
        """w,h: V6 processing size.  roi_poly: V6's `det.poly` (4x2, BL,TL,TR,BR) used only by the fallback policy."""
        self.w, self.h = int(w), int(h)
        self.c = dict(VCFG, **(cfg or {}))
        c = self.c
        self.yb = int(c["band_top"] * h)
        self.nb = self.h - self.yb
        self.roi_poly = (np.asarray(roi_poly, np.float32) if roi_poly is not None else
                         np.array([[0, 0.985 * h], [0.22 * w, 0.615 * h], [0.74 * w, 0.615 * h], [w - 1, 0.985 * h]], np.float32))
        self._rows_all = np.arange(self.yb, self.h, dtype=np.float32)
        self._road = None                                   # EMA road model, shape (nb, 3) float32
        self._wmodel = None                                 # (a, b): lane width ~= a*y + b
        self._wmodel_frame = -10 ** 9
        self._k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (c["close_w"], c["close_h"]))
        self._excl = np.ones((self.nb, self.w), np.uint8)
        for (x0, y0, x1, y1) in c["exclude"]:
            cv2.rectangle(self._excl, (int(x0 * w), int(y0 * h) - self.yb), (int(x1 * w), int(y1 * h) - self.yb), 0, -1)
        self._tracks: List[dict] = []
        self._rtracks: List[dict] = []
        self._next_id = 1
        self.fn = 0

    # ------------------------------------------------------------------------------------------------
    # geometry from V6
    # ------------------------------------------------------------------------------------------------
    def _geometry(self, lane):
        """Return dict with: mode (CORRIDOR|ROI_FALLBACK|SUPPRESSED), edge(y)->(xl,xr) for the SEARCH region (may extrapolate),
        valid rows [ext, yref] where association is allowed (no extrapolation), trusted sides, width model."""
        c, h, w = self.c, self.h, self.w
        g = dict(mode="SUPPRESSED", state=lane.get("state", "NONE"), corr=None, scale_source="", notes="")
        corr = lane.get("corridor") if lane else None
        ok = (lane is not None and lane.get("state") == "LANE" and corr is not None and corr.get("ys") is not None
              and len(corr["ys"]) >= 4)
        if ok:
            ys = np.asarray(corr["ys"], np.float64)
            xl = np.asarray(corr["xl"], np.float64)
            xr = np.asarray(corr["xr"], np.float64)
            fin = np.isfinite(ys) & np.isfinite(xl) & np.isfinite(xr)
            ys, xl, xr = ys[fin], xl[fin], xr[fin]
            if len(ys) >= 4 and np.all(xr - xl > 4):
                o = np.argsort(ys)                           # V6 stores ys bottom->top (descending); np.interp wants ascending
                ys, xl, xr = ys[o], xl[o], xr[o]
                g.update(mode="CORRIDOR", corr=(ys, xl, xr), ext=float(ys[0]), yref=float(ys[-1]))
                src = (lane.get("L") or {}).get("source"), (lane.get("R") or {}).get("source")
                g["src"] = src
                # width model w(y) = a*y + b  (perspective: width -> 0 at the horizon)
                a, b = np.polyfit(ys, xr - xl, 1)
                yh = -b / a if a > 1e-6 else None
                if a > 0.15 and yh is not None and 0.30 * h < yh < 0.80 * h:
                    self._wmodel, self._wmodel_frame = (float(a), float(b)), self.fn
                    g["scale_source"] = "corridor"
                # linear continuation of each boundary for the SEARCH region above the corridor (never used for association)
                g["lin_l"] = np.polyfit(ys, xl, 1)
                g["lin_r"] = np.polyfit(ys, xr, 1)
        if g["mode"] == "SUPPRESSED" and c["unreliable_policy"] == "fallback":
            g["mode"] = "ROI_FALLBACK"
        if g["mode"] == "SUPPRESSED":
            return g
        if not g["scale_source"]:
            if self._wmodel is not None and self.fn - self._wmodel_frame <= c["width_hold"]:
                g["scale_source"] = "held"
            else:
                bw = 0.55 * w
                yh = c["horizon_frac"] * h
                self._wmodel = (bw / (h - yh), -bw * yh / (h - yh))
                g["scale_source"] = "prior"
        a, b = self._wmodel
        g["wm"] = (a, b)
        g["yh"] = -b / a
        return g

    def _edges_at(self, g, y):
        """(xl, xr) of the SEARCH region's inner reference at rows y (np array).  Search only -- may extrapolate."""
        if g["mode"] == "CORRIDOR":
            ys, xl, xr = g["corr"]
            yc = np.clip(y, ys[0], ys[-1])
            L = np.interp(yc, ys, xl)
            R = np.interp(yc, ys, xr)
            above = y < ys[0]
            if above.any():
                L = np.where(above, np.polyval(g["lin_l"], y), L)
                R = np.where(above, np.polyval(g["lin_r"], y), R)
            return L, R
        p = self.roi_poly                                    # BL, TL, TR, BR
        yt, yb_ = p[1][1], p[0][1]
        yc = np.clip(y, yt, yb_)
        L = np.interp(yc, [yt, yb_], [p[1][0], p[0][0]])
        R = np.interp(yc, [yt, yb_], [p[2][0], p[3][0]])
        return L, R

    def _region(self, g, margin):
        """Search-region polygon (full-frame coordinates) and band mask."""
        c = self.c
        ys = np.linspace(max(g["yh"] - c["above_horizon_px"], self.yb), self.h - 1, 16).astype(np.float32)
        L, R = self._edges_at(g, ys)
        if g["mode"] == "CORRIDOR":
            wd = np.maximum(g["wm"][0] * ys + g["wm"][1], 8.0)           # modelled lane width (never crosses, unlike raw extrapolation)
            ctr = 0.5 * (L + R)
            half = (0.5 + margin) * wd
            L, R = ctr - half, ctr + half
        poly = np.concatenate([np.stack([L, ys], 1), np.stack([R, ys], 1)[::-1]]).astype(np.float32)
        poly[:, 0] = np.clip(poly[:, 0], 0, self.w - 1)
        m = np.zeros((self.nb, self.w), np.uint8)
        cv2.fillPoly(m, [np.round(poly - np.array([0, self.yb], np.float32)).astype(np.int32)], 1)
        return poly, m

    # ------------------------------------------------------------------------------------------------
    # road model + candidates
    # ------------------------------------------------------------------------------------------------
    def _road_model(self, lab, g):
        c = self.c
        yh = g["yh"]
        y0 = int(max(yh + 4, self.yb))
        rows = np.arange(y0, self.h, c["road_sample_step"], dtype=np.float32)
        if len(rows) < 3:
            return self._road
        L, R = self._edges_at(g, rows)
        wd = np.maximum(R - L, 6.0)
        if g["mode"] == "CORRIDOR":
            lo, hi = L + (0.5 - c["road_core"]) * wd, L + (0.5 + c["road_core"]) * wd
        else:                                              # fallback: central part of the ROI trapezoid
            lo, hi = L + 0.35 * wd, L + 0.65 * wd
        t = np.linspace(0, 1, c["road_sample_cols"], dtype=np.float32)
        xs = np.clip((lo[:, None] + (hi - lo)[:, None] * t[None, :]).astype(np.int32), 0, self.w - 1)
        yi = (rows.astype(np.int32) - self.yb)[:, None]
        samp = lab[yi, xs]                                  # (nrows, ncols, 3)
        med = np.median(samp, axis=1).astype(np.float32)    # (nrows, 3)
        full = np.empty((self.nb, 3), np.float32)
        ry = rows - self.yb
        for k in range(3):
            full[:, k] = np.interp(np.arange(self.nb), ry, med[:, k])
        if self._road is None:
            self._road, self._rej = full, np.zeros(self.nb, np.int32)
        else:
            diff = np.abs(full - self._road)
            ok = (diff[:, 0] < c["road_gate_l"]) & (diff[:, 1] + diff[:, 2] < c["road_gate_c"])
            self._rej = np.where(ok, 0, self._rej + 1)
            upd = ok | (self._rej > c["road_force"])
            self._rej[upd] = 0
            blend = c["road_ema"] * full + (1 - c["road_ema"]) * self._road
            self._road = np.where(upd[:, None], blend, self._road)
        return self._road

    def _thin(self, M, g):
        """Thin part of mask M: pixels removed by a scale-aware horizontal opening (kernel width = run_min * lane width at the row band)."""
        c = self.c
        m = M.astype(np.uint8)
        out = np.zeros_like(m)
        eb = np.linspace(0, self.nb, c["run_bands"] + 1).astype(int)
        for i in range(c["run_bands"]):
            a, b = eb[i], eb[i + 1]
            ymid = self.yb + 0.5 * (a + b)
            kw = int(max(3, round(c["run_min"] * max(g["wm"][0] * ymid + g["wm"][1], 0))))
            k = np.ones((1, kw), np.uint8)
            out[a:b] = m[a:b] - cv2.dilate(cv2.erode(m[a:b], k), k)
        return out

    def _bounded_runs(self, N, Nfull, g, flank=True):
        """Per image row: runs of non-road pixels whose width is plausible for that row (and, if `flank`, that have road-like pixels
        on both sides).  Fully vectorised.  Optimised after profiling: one flatnonzero instead of two nonzero calls, direct pixel
        look-ups for the flank test and index-based run filling instead of full-array cumsums (identical output, ~3x faster)."""
        c = self.c
        nb, w = N.shape
        out = np.zeros((nb, w), np.uint8)
        pad = np.zeros((nb, w + 2), np.int8)
        pad[:, 1:-1] = N
        d = np.diff(pad, axis=1)                                  # +1 at run start, -1 just after run end
        flat = np.flatnonzero(d)
        if flat.size == 0:
            return out
        val = d.ravel()[flat]
        r, col = np.divmod(flat, w + 1)
        st = val > 0
        rs, cs, ce = r[st], col[st], col[~st]                     # per row starts/ends alternate -> arrays align
        wexp = np.maximum(g["wm"][0] * (self.yb + np.arange(nb)) + g["wm"][1], 4.0)
        wd = ce - cs
        ok = (wd >= np.maximum(c["run_min"] * wexp[rs], 3)) & (wd <= c["run_max"] * wexp[rs])
        if flank:
            ok &= (cs > 0) & (ce < w)
        rs, cs, ce = rs[ok], cs[ok], ce[ok]
        if len(rs) == 0:
            return out
        if flank:
            left = np.zeros(len(rs), np.int16)
            right = np.zeros(len(rs), np.int16)
            for k in range(1, c["flank"] + 1):
                left += Nfull[rs, np.maximum(cs - k, 0)]
                right += Nfull[rs, np.minimum(ce + k - 1, w - 1)]
            ok = (left <= 1) & (right <= 1)
            rs, cs, ce = rs[ok], cs[ok], ce[ok]
            if len(rs) == 0:
                return out
        lens = ce - cs
        off = np.cumsum(lens) - lens
        rr = np.repeat(rs, lens)
        cc = np.arange(int(lens.sum())) - np.repeat(off, lens) + np.repeat(cs, lens)
        out[rr, cc] = 1
        return out

    def _candidates(self, bgr, g):
        """Returns (boxes (n,5) int x0,y0,x1,y1,area in FULL-frame coords, info dict with masks / integral images for scoring).
        Shared work (Lab, deviation, edges, paint carving) is done once; the run extraction + connected components are run for a
        TIGHT region (corridor only) and a WIDE region (adjacent lanes) and the boxes are merged by NMS later."""
        c = self.c
        band = bgr[self.yb:]
        lab = cv2.cvtColor(band, cv2.COLOR_BGR2LAB)
        road = self._road_model(lab, g)
        Lc = lab[:, :, 0].astype(np.int16)
        dl = np.abs(Lc - road[:, 0:1].astype(np.int16))
        dc = (np.abs(lab[:, :, 1].astype(np.int16) - road[:, 1:2].astype(np.int16))
              + np.abs(lab[:, :, 2].astype(np.int16) - road[:, 2:3].astype(np.int16)))
        dev = ((dl > c["dev_l"]) | (dc > c["dev_c"]))
        Lb = cv2.GaussianBlur(lab[:, :, 0], (3, 3), 0)
        gx = cv2.Sobel(Lb, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(Lb, cv2.CV_16S, 0, 1, ksize=3)
        E = ((np.abs(gx) + np.abs(gy)) > c["grad_t"]).astype(np.uint8)
        H = (np.abs(gy) > c["grad_t"]).astype(np.uint8)
        dens = cv2.blur(E * 255, (c["dens_k"], c["dens_k"]))
        Nfull = dev | (dens > c["dens_t"])                    # "not road-like" (colour/lightness deviation or dense edges)
        dark_full = (road[:, 0:1].astype(np.int16) - Lc) > c["dark_t"]
        excl = self._excl.astype(bool)
        # carve thin bright neutral structures (lane paint) out of the non-road mask BEFORE judging run widths
        neutral = (np.abs(lab[:, :, 1].astype(np.int16) - 128) + np.abs(lab[:, :, 2].astype(np.int16) - 128)) < c["paint_neutral"]
        bright = ((Lc - road[:, 0:1].astype(np.int16)) > c["paint_bright"]) & neutral & excl
        paint = self._thin(bright, g)
        if paint.any():
            Nfull = Nfull & ~cv2.dilate(paint, np.ones((3, 3), np.uint8)).astype(bool)
        cut = int(max(g["yh"], self.yb)) - self.yb          # nothing above the horizon can touch the road: cut there
        margins = [c["tight_margin"], c["region_margin"]] if g["mode"] == "CORRIDOR" else [0.0]
        all_boxes, m_all, poly_w, rmask_w = [], np.zeros((self.nb, self.w), np.uint8), None, None
        for margin in margins:
            poly, rmask = self._region(g, margin)
            allow = rmask.astype(bool) & excl
            allow[:cut] = False
            m = self._bounded_runs(Nfull & allow, Nfull, g, flank=False) | self._bounded_runs(dark_full & allow, dark_full, g)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._k_close)
            n, lbl, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            if n > 1:
                st = st[1:]
                st = st[st[:, cv2.CC_STAT_AREA] >= c["min_area"]]
                x0 = st[:, cv2.CC_STAT_LEFT]; y0 = st[:, cv2.CC_STAT_TOP]
                x1 = x0 + st[:, cv2.CC_STAT_WIDTH]; y1 = y0 + st[:, cv2.CC_STAT_HEIGHT]
                bx = np.stack([x0, y0 + self.yb, x1, y1 + self.yb, st[:, cv2.CC_STAT_AREA]], 1).astype(np.int32)
                all_boxes.append(self._split(bx, m, g))
            m_all |= m
            poly_w, rmask_w = poly, rmask                        # last pass = widest region (used for scoring / display)
        ub = np.zeros((0, 5), np.int32)
        if c["ub_scan"]:
            r2 = self._road_rows(Lc, allow)
            if r2 is not None:
                d2 = (r2[:, None] - Lc) > c["dark_t"]
                dl2 = np.abs(Lc - r2[:, None])
                ub = self._underbody(d2, dl2, g, allow, excl)
        info = dict(ub=ub, Lband=lab[:, :, 0], cut_row=cut + self.yb, dev=dev, E=E, H=H, m=m_all, poly=poly_w, rmask=rmask_w, dl=dl, road=road,
                    iiE=cv2.integral(E), iiH=cv2.integral(H), iiM=cv2.integral(m_all), iiD=cv2.integral(dev.astype(np.uint8)))
        boxes = np.concatenate(all_boxes) if all_boxes else np.zeros((0, 5), np.int32)
        return boxes, info

    def _road_rows(self, Lc, allow):
        """Robust per-row road lightness: median of L over the (wide) search region, sampled every 3rd row / column.  Unlike the
        corridor-core model it stays correct when vehicles cover the corridor core, because a vehicle covers well under half of the
        multi-lane search width.  Temporal EMA keeps it stable."""
        nb = Lc.shape[0]
        rows = np.arange(0, nb, 4)
        sub = Lc[rows][:, ::4].astype(np.float32)
        m = allow[rows][:, ::4]
        sub = np.where(m, sub, np.nan)
        cnt = m.sum(axis=1)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)           # rows with no allowed pixel are all-NaN by design
            med = np.nanmedian(sub, axis=1)
        ok = cnt >= 8
        if ok.sum() < 5:
            return None
        v = np.interp(rows, rows[ok], med[ok])
        v = cv2.blur(v.reshape(-1, 1), (1, 5)).ravel()
        full = np.interp(np.arange(nb), rows, v).astype(np.float32)
        r2 = getattr(self, "_road2", None)
        self._road2 = full if r2 is None else (0.35 * full + 0.65 * r2)
        return self._road2

    def _underbody(self, dark, dl, g, allow, excl):
        """Bottom-up proposals.  The lower end of a vehicle's dark under-body / tyre mass is a long, nearly horizontal dark->road
        transition.  Per image column: y_bot(x) = bottom of the lowest dark run (>= ub_thick dark rows) whose pixel 3 rows lower is
        road-like.  Neighbouring columns whose y_bot differs by <= ub_dy are grouped; a group whose width fits the lane width at
        its row becomes a proposal box (x0, y_bot - height_ratio*width, x1, y_bot).  Returns int array (n,5), proc coordinates."""
        c = self.c
        nb, w = dark.shape
        D = dark & allow & excl.astype(bool)
        k = c["ub_thick"]
        cnt = np.zeros(D.shape, np.uint8)
        for i in range(1, k + 1):                           # glints / reflections may interrupt the dark run: >= k-1 of the k rows above
            cnt[i:] += D[:-i]
        thick = D & (cnt >= k - 1)
        has = thick.any(axis=0)
        ybot = (nb - 1 - np.argmax(thick[::-1], axis=0)).astype(np.int32)
        yb3 = np.minimum(ybot + 3, nb - 1)
        valid = has & (dl[yb3, np.arange(w)] < c["ub_road_dl"])
        if valid.sum() < 6:
            return np.zeros((0, 5), np.int32)
        # group: contiguous valid columns whose bottom row changes by <= ub_dy (gaps up to ub_gap columns are bridged)
        idx = np.nonzero(valid)[0]
        out = []
        start = prev = idx[0]
        ys = [ybot[idx[0]]]
        def flush(x0, x1, ylist):
            wd = x1 - x0 + 1
            row = self.yb + int(np.median(ylist)) + 1
            wexp = max(g["wm"][0] * row + g["wm"][1], 4.0)
            if wd < c["ub_min"] * wexp or wd > c["ub_max"] * wexp or len(ylist) < 0.6 * wd:
                return
            out.append((x0, int(round(row - c["height_ratio"] * wd)), x1 + 1, row, len(ylist)))
        for x in idx[1:]:
            if x - prev <= c["ub_gap"] and abs(int(ybot[x]) - int(np.median(ys[-6:]))) <= c["ub_dy"]:
                ys.append(ybot[x]); prev = x
            else:
                flush(start, prev, ys); start = prev = x; ys = [ybot[x]]
        flush(start, prev, ys)
        return np.array(out, np.int32).reshape(-1, 5)

    def _split(self, boxes, m, g):
        """Split over-wide candidate boxes (e.g. two neighbouring vehicles) at valleys of the column projection of the mask."""
        c = self.c
        out = []
        for (x0, y0, x1, y1, area) in boxes:
            wexp = max(g["wm"][0] * y1 + g["wm"][1], 4.0)
            if (x1 - x0) <= c["split_ratio"] * wexp:
                out.append((x0, y0, x1, y1, area))
                continue
            sub = m[y0 - self.yb:y1 - self.yb, x0:x1]
            prof = sub.sum(axis=0).astype(np.int32)
            act = prof >= max(1.0, c["split_valley"] * prof.max())
            d = np.diff(np.concatenate([[0], act.astype(np.int8), [0]]))
            st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
            segs = []
            for a, b in zip(st, en):
                if segs and a - segs[-1][1] < c["split_gap"]:
                    segs[-1][1] = b
                else:
                    segs.append([a, b])
            for a, b in segs:
                part = sub[:, a:b]
                rows = np.nonzero(part.sum(axis=1) >= max(1, 0.25 * (b - a)))[0]
                if len(rows) == 0:
                    continue
                out.append((x0 + a, y0 + int(rows[0]), x0 + b, y0 + int(rows[-1]) + 1, int(part.sum())))
        return np.array(out, np.int32).reshape(-1, 5)

    # ------------------------------------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------------------------------------
    @staticmethod
    def _bsum(ii, x0, y0, x1, y1):
        return float(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])

    def _score(self, boxes, info, g):
        c = self.c
        out_ok, out_bad = [], []
        yb = self.yb
        for (x0, y0f, x1, y1f, area) in boxes:
            bw, bh = float(x1 - x0), float(y1f - y0f)
            top_est = 0.0
            truncated = y0f <= info["cut_row"] + 1               # touches the horizon cut: the real top is hidden
            if bh < c["height_ratio"] * bw and (bw <= c["top_est_max_w"] or truncated):
                y0f = max(int(round(y1f - c["height_ratio"] * bw)), yb)
                bh = float(y1f - y0f)
                top_est = 1.0
            y0, y1 = int(y0f - yb), int(y1f - yb)
            parts, reject = {}, ""
            wexp = max(g["wm"][0] * y1f + g["wm"][1], 4.0)        # expected lane width at the contact row
            if y1f <= g["yh"] + c["min_below_horizon"]:
                reject = "near_horizon"
            ratio = bw / wexp
            asp = bw / max(bh, 1.0)
            fill = area / max(bw * bh, 1.0) if not top_est else 1.0     # fill is only meaningful for a measured box
            area_box = bw * bh
            edens = self._bsum(info["iiE"], x0, y0, x1, y1) / max(area_box, 1.0)
            # strip under the box: does it look like road?
            sy0, sy1 = min(y1, self.nb - 1), min(y1 + 5, self.nb)
            if sy1 > sy0:
                road_below = 1.0 - self._bsum(info["iiD"], x0, sy0, x1, sy1) / max((x1 - x0) * (sy1 - sy0), 1)
            else:
                road_below = 1.0                                    # touches the bottom of the band: cannot be tested
            # horizontal-edge support in the lower third (underside / bumper)
            ly0 = y0 + int(0.6 * (y1 - y0))
            hlow = self._bsum(info["iiH"], x0, ly0, x1, y1) / max((x1 - x0) * max(y1 - ly0, 1), 1)
            # inside-region fraction of the bottom span
            span = info["rmask"][max(min(y1 - 1, self.nb - 1), 0), x0:x1]
            inside = float(span.mean()) if span.size else 0.0
            parts.update(size=_ramp(ratio, c["size_lo"], c["size_ok_lo"], c["size_ok_hi"], c["size_hi"]),
                         aspect=_ramp(asp, c["aspect_lo"], c["aspect_ok_lo"], c["aspect_ok_hi"], c["aspect_hi"]),
                         fill=min(1.0, fill / 0.6),
                         edge=min(1.0, edens / (2 * c["edge_dens_min"])),
                         road=min(1.0, road_below / c["road_below_min"]),
                         hedge=min(1.0, hlow / 0.10),
                         top_est=top_est, ratio=ratio, aspect_v=asp, fill_v=fill, edens=edens, road_below=road_below, inside=inside)
            if not reject:
                if bw < c["min_box_w"]:
                    reject = "tiny"
                elif road_below < c["road_hard"]:
                    reject = "not_on_road"
                elif bh < c["flat_min"] * bw:
                    reject = "flat"
                elif parts["size"] <= 0:
                    reject = "size_%s" % ("small" if ratio < c["size_ok_lo"] else "large")
                elif parts["aspect"] <= 0:
                    reject = "aspect"
                elif fill < c["fill_min"]:
                    reject = "sparse"
                elif edens < c["edge_dens_min"]:
                    reject = "no_edges"
                elif inside < c["inside_min"]:
                    reject = "outside_region"
            parts["sym"] = self._symmetry(info["Lband"], x0, y0, x1, y1)
            p = parts
            score = (p["size"] ** 0.30) * (p["aspect"] ** 0.20) * (p["fill"] ** 0.10) * (p["edge"] ** 0.10) \
                * (p["road"] ** 0.15) * ((0.5 + 0.5 * p["hedge"]) ** 0.15) * (c["top_est_penalty"] if top_est else 1.0)
            box = (float(x0), float(y0f), float(x1), float(y1f))
            d = VehicleDetection(track_id=-1, state="CANDIDATE", box=box, raw_box=box,
                                 center=((x0 + x1) / 2.0, (y0f + y1f) / 2.0),
                                 contact=((x0 + x1) / 2.0, float(y1f)), contact_method="bbox_bottom",
                                 score=float(score), parts=parts, road_ref=g["mode"], reject=reject)
            (out_bad if reject else out_ok).append(d)
        return out_ok, out_bad

    @staticmethod
    def _symmetry(L, x0, y0, x1, y1):
        """Normalised cross-correlation (-1..1) between the box's lightness patch and its left-right mirror.
        Rear views of vehicles are roughly mirror-symmetric; walls, patches and clutter usually are not."""
        y0, y1 = max(y0, 0), min(y1, L.shape[0])
        if y1 - y0 < 4 or x1 - x0 < 6:
            return 0.0
        p = L[y0:y1, x0:x1].astype(np.float32)
        p -= p.mean()
        f = p[:, ::-1]
        den = float(np.sqrt((p * p).sum() * (f * f).sum()))
        return float((p * f).sum() / den) if den > 1e-6 else 0.0

    # ------------------------------------------------------------------------------------------------
    # contact point + lane association
    # ------------------------------------------------------------------------------------------------
    def _refine_contact(self, d, info):
        """Lower edge: lowest row (within the lower 40% of the box) where >= 40% of the box width is non-road (deviation/edges)."""
        x0, y0, x1, y1 = (int(round(v)) for v in d.box)
        yb = self.yb
        ya = max(int(y0 + 0.6 * (y1 - y0)) - yb, 0)
        ye = min(y1 - yb, self.nb)
        if ye - ya < 3 or x1 - x0 < 4:
            return
        sub = info["m"][ya:ye, max(x0, 0):x1]
        rowfrac = sub.mean(axis=1)
        idx = np.nonzero(rowfrac >= 0.40)[0]
        if len(idx):
            d.contact = ((x0 + x1) / 2.0, float(ya + idx[-1] + yb + 1))
            d.contact_method = "lower_edge"

    def _associate(self, d, g):
        c = self.c
        if g["mode"] != "CORRIDOR":
            d.lane, d.lane_reason = "UNKNOWN", "NO_RELIABLE_CORRIDOR"
            return
        ys, xl, xr = g["corr"]
        cx, cy = d.contact
        if cy < ys[0] - 1.0:
            d.lane, d.lane_reason = "UNKNOWN", "BEYOND_CORRIDOR_EXTENT"
            return
        cy = min(cy, ys[-1])
        L, R = float(np.interp(cy, ys, xl)), float(np.interp(cy, ys, xr))
        tol = c["assoc_tol"] * (R - L)
        srcL, srcR = g.get("src", (None, None))
        trustL, trustR = srcL in ("OBSERVED", "TRACKED"), srcR in ("OBSERVED", "TRACKED")
        if cx < L - tol:
            d.lane, d.lane_reason = ("LEFT_OF_CORRIDOR", "") if trustL else ("UNKNOWN", "INFERRED_BOUNDARY")
        elif cx > R + tol:
            d.lane, d.lane_reason = ("RIGHT_OF_CORRIDOR", "") if trustR else ("UNKNOWN", "INFERRED_BOUNDARY")
        elif cx < L + tol or cx > R - tol:
            d.lane, d.lane_reason = "AMBIGUOUS", "NEAR_BOUNDARY"
        else:
            if trustL and trustR:
                d.lane, d.lane_reason = "IN_CORRIDOR", ""
            else:
                d.lane, d.lane_reason = "UNKNOWN", "INFERRED_BOUNDARY"
        if d.lane in ("IN_CORRIDOR", "LEFT_OF_CORRIDOR", "RIGHT_OF_CORRIDOR") and "TRACKED" in (srcL, srcR):
            d.lane_reason = "BOUNDARY_TRACKED"                # still reported, but one boundary is not freshly observed

    # ------------------------------------------------------------------------------------------------
    # full-resolution refinement (accuracy upgrade)
    # ------------------------------------------------------------------------------------------------
    def _refine_all(self, full, ok, bad, info, g, acc, weak):
        """Refine every proposal at full resolution; keep only those the refiner verifies.  Proposals = stage-1 accepted/weak boxes,
        stage-1 rejections whose reason is recoverable (e.g. 'flat': only the dark under-body was separated from a dark road) and
        the under-body scan.  Returns (accepted, weak) lists of VehicleDetection whose box/contact are the refined values (proc px)
        with the source-pixel truth in box_src / contact_src."""
        from vehicle_refine import RearRefiner
        if not hasattr(self, "_rr"):
            self._rr = RearRefiner(self.c.get("rcfg"))
        c = self.c
        s = full.shape[1] / float(self.w)
        props = []
        for d in acc + weak:
            props.append((d.box, d.score, "stage1"))
        for d in bad:
            if d.reject in c["rescue"] or d.reject.startswith("size_small"):
                props.append((d.box, 0.5, "rescued:" + d.reject))
        for (x0, y0, x1, y1, _a) in info["ub"]:
            props.append(((float(x0), float(y0), float(x1), float(y1)), 0.5, "underbody"))
        props.sort(key=lambda t: -t[1])
        out = []
        for (bx, sc, src) in props[:c["refine_max"]]:
            # proposals of the same vehicle are very similar: skip those already covered by an accepted refinement
            skip = False
            for o in out:
                a_, b_ = o.box, bx
                ix = max(0.0, min(a_[2], b_[2]) - max(a_[0], b_[0]))
                if ix / max(min(a_[2] - a_[0], b_[2] - b_[0]), 1.0) > 0.7 and abs(a_[3] - b_[3]) < 0.25 * (a_[2] - a_[0]):
                    skip = True; break
            if skip:
                continue
            wexp = max(g["wm"][0] * bx[3] + g["wm"][1], 4.0)
            rs_ = self._rr.refine(full, [v * s for v in bx], lane_w=wexp * s)
            for r in rs_:
              b = r.box
              ctr = r.contact if r.contact is not None else r.contact_hint
              box = tuple(v / s for v in b)
              d = VehicleDetection(track_id=-1, state="CANDIDATE", box=box, raw_box=box,
                                   center=((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0),
                                   contact=(ctr[0] / s, ctr[1] / s), contact_method="refined_" + r.quality.lower(),
                                   score=float(0.5 * r.evidence + 0.5 * min(1.0, sc + 0.3)),
                                   parts=dict(r.parts, src=src), road_ref=g["mode"])
              d.refined, d.box_src, d.top_estimated = True, tuple(b), r.top_estimated
              d.contact_src, d.contact_hint_src = r.contact, r.contact_hint
              d.contact_quality, d.contact_detail, d.contact_sigma = r.quality, r.detail, r.contact_sigma
              d.tyres_src, d.evidence = list(r.tyres), r.evidence
              out.append(d)
        out = self._nms(out)
        acc2 = [d for d in out if d.score >= c["thr_accept"]]
        weak2 = [d for d in out if d.score < c["thr_accept"]]
        return acc2, weak2

    # ------------------------------------------------------------------------------------------------
    # NMS + tracking
    # ------------------------------------------------------------------------------------------------
    def _nms(self, dets):
        c = self.c
        dets = sorted(dets, key=lambda d: -d.score)
        keep = []
        for d in dets:
            dup = False
            for k in keep:
                a, b = d.box, k.box
                ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                inter = ix * iy
                sm = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
                if _iou(a, b) > c["nms_iou"] or (sm > 0 and inter / sm > c["nms_contain"]):
                    dup = True
                    break
            if not dup:
                keep.append(d)
        return keep

    def _track(self, accepted, weak):
        """Greedy association.  New tracks only from `accepted`; existing CONFIRMED tracks may also be sustained by `weak` candidates."""
        c = self.c
        tr = self._tracks
        for t in tr:
            t["obs"] = None
            px0, py0, px1, py1 = t["box"]
            vx, vy = t["vel"]
            t["pred"] = (px0 + vx, py0 + vy, px1 + vx, py1 + vy)
        pool = [(d, True) for d in accepted] + [(d, False) for d in weak]
        pairs = []
        for ti, t in enumerate(tr):
            pc = ((t["pred"][0] + t["pred"][2]) / 2, (t["pred"][1] + t["pred"][3]) / 2)
            tw = t["pred"][2] - t["pred"][0]
            for di, (d, acc) in enumerate(pool):
                if not acc and not t["confirmed"]:
                    continue
                iou = _iou(t["pred"], d.box)
                dist = np.hypot(d.center[0] - pc[0], d.center[1] - pc[1]) / max(tw, d.box[2] - d.box[0], 1.0)
                if iou >= c["track_iou"] or dist <= c["track_dist"] * 0.5:
                    pairs.append((-(iou + (0.0 if acc else -0.1)), ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            tr[ti]["obs"] = pool[di][0]
        # update tracks
        for t in tr:
            t["age"] += 1
            d = t["obs"]
            if d is not None:
                nb = d.box
                ob = t["box"]
                e = c["box_ema"]
                sm = tuple(e * nb[i] + (1 - e) * ob[i] for i in range(4))
                t["vel"] = (0.5 * t["vel"][0] + 0.5 * ((sm[0] + sm[2]) - (ob[0] + ob[2])) / 2,
                            0.5 * t["vel"][1] + 0.5 * ((sm[1] + sm[3]) - (ob[1] + ob[3])) / 2)
                t["box"] = sm
                t["hits"] += 1
                t["miss"] = 0
                t["recent"].append(1)
                t["last"] = d
            else:
                t["miss"] += 1
                t["recent"].append(0)
                t["box"] = t["pred"]
            t["recent"] = t["recent"][-c["confirm_window"]:]
            if not t["confirmed"] and sum(t["recent"]) >= c["confirm_hits"]:
                t["confirmed"] = True
        # new tracks from unmatched ACCEPTED detections
        for di, (d, acc) in enumerate(pool):
            if di in used_d or not acc:
                continue
            tr.append(dict(id=self._next_id, box=d.box, vel=(0.0, 0.0), pred=d.box, obs=d, last=d, age=1, hits=1, miss=0,
                           recent=[1], confirmed=False))
            self._next_id += 1
        # expire + de-duplicate
        alive = []
        for t in tr:
            lim = c["max_coast"] if t["confirmed"] else c["max_coast_tentative"]
            if t["miss"] > lim:
                continue
            alive.append(t)
        alive.sort(key=lambda t: -t["hits"])
        out = []
        for t in alive:
            if any(_iou(t["box"], o["box"]) > 0.5 for o in out):
                continue
            out.append(t)
        self._tracks = out

    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # refinement-aware tracking: Kalman filters in SOURCE pixels (no stale stage-1 tracks, no EMA lag)
    # ------------------------------------------------------------------------------------------------
    _KEYS = ("x0", "y0", "x1", "y1", "cx", "cy")

    def _track_refined(self, acc, weak, s):
        c = self.c
        tr = self._rtracks
        qf = c["kf_q"]
        for t in tr:
            t["obs"] = None
            w_ = max(t["kf"]["x1"].x - t["kf"]["x0"].x, 8.0)
            for k, f in t["kf"].items():
                f.predict((qf * w_) ** 2)
            t["pred"] = tuple(t["kf"][k].x / s for k in ("x0", "y0", "x1", "y1"))
        pool = [(d, True) for d in acc] + [(d, False) for d in weak]
        pairs = []
        for ti, t in enumerate(tr):
            pc = ((t["pred"][0] + t["pred"][2]) / 2, (t["pred"][1] + t["pred"][3]) / 2)
            tw = t["pred"][2] - t["pred"][0]
            for di, (d, a_) in enumerate(pool):
                if not a_ and not t["confirmed"]:
                    continue
                iou = _iou(t["pred"], d.box)
                dist = np.hypot(d.center[0] - pc[0], d.center[1] - pc[1]) / max(tw, d.box[2] - d.box[0], 1.0)
                if iou >= c["track_iou"] or dist <= c["track_dist"] * 0.5:
                    pairs.append((-(iou + (0.0 if a_ else -0.1)), ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti); used_d.add(di)
            tr[ti]["obs"] = pool[di][0]
        for t in tr:
            t["age"] += 1
            d = t["obs"]
            if d is not None:
                bx = d.box_src
                w_ = max(bx[2] - bx[0], 8.0)
                kf = t["kf"]
                rl = (c["kf_r_lat"] + 0.02 * w_) ** 2
                rb = (c["kf_r_bot"] + (0.0 if d.contact_quality == "TWO_TYRES" else 1.5)) ** 2
                rt = (0.12 * w_) ** 2 if d.top_estimated else (c["kf_r_top"] + 0.03 * w_) ** 2
                kf["x0"].update(bx[0], rl); kf["x1"].update(bx[2], rl); kf["y1"].update(bx[3], rb); kf["y0"].update(bx[1], rt)
                if d.contact_src is not None and d.contact_quality == "TWO_TYRES":
                    rc = max(d.contact_sigma, 1.0) ** 2
                    kf["cx"].update(d.contact_src[0], rc); kf["cy"].update(d.contact_src[1], rc)
                t["miss"] = 0; t["hits"] += 1; t["recent"].append(1); t["last"] = d
            else:
                t["miss"] += 1; t["recent"].append(0)
            t["recent"] = t["recent"][-c["confirm_window"]:]
            if not t["confirmed"] and sum(t["recent"]) >= c["confirm_hits"]:
                t["confirmed"] = True
        for di, (d, a_) in enumerate(pool):
            if di in used_d or not a_:
                continue
            bx, ct = d.box_src, (d.contact_src or d.contact_hint_src)
            kf = {"x0": _Kf1(bx[0]), "y0": _Kf1(bx[1]), "x1": _Kf1(bx[2]), "y1": _Kf1(bx[3]), "cx": _Kf1(ct[0]), "cy": _Kf1(ct[1])}
            tr.append(dict(id=self._next_id, kf=kf, pred=d.box, obs=d, last=d, age=1, hits=1, miss=0, recent=[1], confirmed=False))
            self._next_id += 1
        alive = []
        for t in tr:
            lim = c["max_coast"] if t["confirmed"] else c["max_coast_tentative"]
            if t["miss"] <= lim:
                alive.append(t)
        alive.sort(key=lambda t: -t["hits"])
        out = []
        for t in alive:
            cur = tuple(t["kf"][k].x / s for k in ("x0", "y0", "x1", "y1"))
            if any(_iou(cur, tuple(o["kf"][k].x / s for k in ("x0", "y0", "x1", "y1"))) > 0.5 for o in out):
                continue
            out.append(t)
        self._rtracks = out

    def _emit_refined(self, out, g, s):
        c = self.c
        for t in self._rtracks:
            d = t["obs"]
            kf = t["kf"]
            bs = tuple(kf[k].x for k in ("x0", "y0", "x1", "y1"))
            if d is not None:
                if not t["confirmed"] and d.score < c["thr_accept"]:
                    continue
                det = d
                det.state = "TRACKED" if t["confirmed"] else "ACCEPTED"
                det.observed = True
                det.raw_box = det.box
                if det.contact_quality == "TWO_TYRES" and det.contact_src is not None:
                    det.contact_src = (kf["cx"].x, kf["cy"].x)           # filtered axle ground point
            elif t["confirmed"]:
                last = t["last"]
                det = VehicleDetection(track_id=t["id"], state="PREDICTED", box=tuple(v / s for v in bs), raw_box=last.raw_box,
                                       center=(0, 0), contact=(0, 0), contact_method="predicted", score=last.score, parts=last.parts,
                                       miss=t["miss"], observed=False, lane=last.lane, lane_reason="STALE", road_ref=last.road_ref)
                det.refined = True
                det.contact_quality = "PREDICTED"; det.contact_sigma = 0.0
                det.contact_src = (kf["cx"].x, kf["cy"].x) if last.contact_quality == "TWO_TYRES" else None
                det.contact_hint_src = (0.5 * (bs[0] + bs[2]), bs[3])
                det.tyres_src = []
            else:
                continue
            det.track_id, det.age, det.hits, det.miss = t["id"], t["age"], t["hits"], t["miss"] if d is None else 0
            det.box_src = bs
            det.src_scale = s
            det.box = tuple(v / s for v in bs)
            det.center = ((det.box[0] + det.box[2]) / 2, (det.box[1] + det.box[3]) / 2)
            ctr = det.contact_src if det.contact_src is not None else det.contact_hint_src
            det.contact = (ctr[0] / s, ctr[1] / s)
            if g["mode"] == "CORRIDOR" and d is None:
                self._associate(det, g)
            out.detections.append(det)
        return out

    def process(self, bgr, lane_result, debug=False, full=None):
        """bgr: the SAME processing-resolution frame given to V6.  lane_result: det.result (dict with "lane").
        full: optional ORIGINAL-resolution frame.  When given (and cfg refine=True) every proposal is refined at full resolution
        (tight box + tyre ground contacts in SOURCE pixels, see vehicle_refine.py); otherwise the legacy stage-1 behaviour runs."""
        c = self.c
        t_start = time.perf_counter()
        self.fn += 1
        out = VehicleFrame()
        lane = (lane_result or {}).get("lane") if lane_result else None
        g = self._geometry(lane or {})
        out.lane_state = g["state"]
        out.road_ref = g["mode"]
        out.scale_source = g.get("scale_source", "")
        if g["mode"] == "SUPPRESSED":
            out.notes = "V6 state %s: vehicle detection suppressed by policy" % g["state"]
            # tracks are not fed: they age out
            self._track([], [])
            out.timing = dict(candidates=0.0, scoring=0.0, tracking=0.0, total=(time.perf_counter() - t_start) * 1000)
            return self._emit(out, g, None, debug)
        t0 = time.perf_counter()
        boxes, info = self._candidates(bgr, g)
        t1 = time.perf_counter()
        ok, bad = self._score(boxes, info, g)
        bonus = c["fallback_score_bonus"] if g["mode"] == "ROI_FALLBACK" else 0.0
        ok = self._nms(ok)
        acc = [d for d in ok if d.score >= c["thr_accept"] + bonus]
        weak = [d for d in ok if c["thr_candidate"] <= d.score < c["thr_accept"] + bonus]
        if full is not None and c["refine"]:
            acc, weak = self._refine_all(full, ok, bad, info, g, acc, weak)
        else:
            for d in acc + weak:
                self._refine_contact(d, info)
        for d in acc + weak:
            self._associate(d, g)
        self._last = dict(info=info, g=g)               # kept for debugging / visualisation only
        t2 = time.perf_counter()
        use_ref = full is not None and c["refine"]
        if use_ref:
            self._track_refined(acc, weak, full.shape[1] / float(self.w))
        else:
            self._track(acc, weak)
        t3 = time.perf_counter()
        out.region = info["poly"]
        out.timing = dict(candidates=(t1 - t0) * 1000, scoring=(t2 - t1) * 1000, tracking=(t3 - t2) * 1000,
                          total=(t3 - t_start) * 1000)
        if g["mode"] == "ROI_FALLBACK":
            out.notes = "V6 state %s: searching static ROI (policy=fallback); lane association UNKNOWN" % g["state"]
        if debug:
            out.candidates = [d for d in weak if all(d is not t["obs"] for t in (self._rtracks if use_ref else self._tracks))]
            out.rejected = bad
        if use_ref:
            return self._emit_refined(out, g, full.shape[1] / float(self.w))
        return self._emit(out, g, info, debug)

    def _emit(self, out, g, info, debug):
        c = self.c
        for t in self._tracks:
            d = t["obs"]
            if d is not None:
                if not t["confirmed"] and d.score < c["thr_accept"]:
                    continue                                         # weak observation of an unconfirmed track: not shown
                det = d
                det.track_id, det.age, det.hits, det.miss = t["id"], t["age"], t["hits"], 0
                det.state = "TRACKED" if t["confirmed"] else "ACCEPTED"
                det.raw_box = d.box
                det.box = tuple(t["box"])
                det.center = ((det.box[0] + det.box[2]) / 2, (det.box[1] + det.box[3]) / 2)
                dx = det.center[0] - (d.raw_box[0] + d.raw_box[2]) / 2
                det.contact = (det.contact[0] + dx, det.contact[1])
                det.observed = True
                if det.refined:                                     # source-pixel results follow the SMOOTHED proc box
                    sc_ = det.box_src[2] / max(det.raw_box[2], 1e-6) if False else None
            elif t["confirmed"]:
                last = t["last"]
                b = t["box"]
                det = VehicleDetection(track_id=t["id"], state="PREDICTED", box=tuple(b), raw_box=last.raw_box,
                                       center=((b[0] + b[2]) / 2, (b[1] + b[3]) / 2),
                                       contact=((b[0] + b[2]) / 2, last.contact[1] + (b[3] - last.box[3])),
                                       contact_method=last.contact_method, score=last.score, parts=last.parts,
                                       age=t["age"], hits=t["hits"], miss=t["miss"], observed=False,
                                       lane=last.lane, lane_reason="STALE_" + last.lane_reason if last.lane_reason else "STALE",
                                       road_ref=last.road_ref)
                if g["mode"] == "CORRIDOR":
                    self._associate(det, g)                          # re-associate the predicted contact with the CURRENT corridor
            else:
                continue
            out.detections.append(det)
        return out

    def reset(self):
        self._tracks, self._rtracks, self._next_id, self._road = [], [], 1, None
        self._rej = None
