#!/usr/bin/env python3
"""
lane_path_v6.py -- STANDALONE lane-path detector (OpenCV + NumPy only; no other project file is needed).

Pipeline (classical CV, no learned weights):  CLAHE/LAB -> Canny -> Hough segments -> seed clustering -> ridge scoring -> quadratic
boundary fits -> per-boundary tracking -> corridor geometry -> overlay.  Parts 1-4 below are the measurement/tracking code of the
earlier lane_path / v2 / v3 / v4 files merged into ONE file without behavioural change (verified bit-identical on 300 real frames);
Part 5 is new in V6.

V6 changes (all in PART 5, `PathDetectorV6`):
  1. The overlay is ALWAYS green (fill and boundary lines).  Boundary source is shown by line style only: solid = observed,
     dashed = tracked from earlier frames, dotted = inferred (ROI edge).  Paint/edge classification never changes a colour.
  2. A lane is never removed because one part is invalid.  The corridor ends at the FIRST row where the geometry becomes invalid
     (crossed / converged / diverging-flare); everything before it is kept.  Fill, both lines and HUD use that one extent.
  3. One boundary only: the missing side is (a) its tracked position if the tracker still holds it, else (b) a provisional ROI edge,
     accepted only if the resulting width is plausible (vs. recent lane width, or vs. frame width when there is no history), else
     (c) only the observed boundary is drawn as a thin line (state ONE_SIDED).  Provisional boundaries are marked INFERRED.
  4. Temporary misses: tracked boundaries are kept for max_hold2 frames (dashed); a single inconsistent measurement is a rejected
     outlier (not blended, lane kept); 3 consistent conflicting measurements replace the boundary (genuine change); a jump to a
     much wider corridor needs 2x as long.  A boundary whose evidence turns 'unsupported' stays displayable for unsup_grace frames
     only.  Boundary-type hysteresis: up 3 / down 5 frames.  Everything is bounded: nothing is drawn forever.
  5. The far end of the displayed corridor may move up the image at most ext_grow px/frame (shrinking is immediate), which removes
     the saw-tooth in the fill's top edge without ever drawing beyond the valid extent.

  6. Display-only overlay clip (overlay_top_y / overlay_top_frac in CFG6): the green fill and boundary lines are not drawn above that row.
     Detection, tracking, result["lane"], pts_L/pts_R, corridor and extent_y are untouched (only draw() changed).

Run:   python3 lane_path_v6.py INPUT.mp4 OUTPUT.mp4 [--start S] [--duration D] [--profile mountain] [--debug] [--show]
"""
import argparse
import os
import shutil
import subprocess
import time

import cv2
import numpy as np

# ======================================================================================================
# PART 1 - measurement front end (edges, Hough segments, seed clustering, curve fit)  [ex lane_path.py]
# ======================================================================================================
CFG = dict(
    proc_w=640,
    # trapezoid as fractions of the processed frame
    top_y=0.615, bot_y=0.985,
    top_x=(0.22, 0.74), bot_x=(0.0, 1.0),
    clahe_clip=2.5, clahe_grid=(8, 8),
    blur_k=5,
    hough_thresh=18, hough_min_len=20, hough_max_gap=12,
    min_abs_slope=0.20,          # |dy/dx| below this = horizontal clutter
    min_abs_m=0.08,              # |dx/dy| below this = vertical clutter (car edges)
    exclude=None,                # optional (x0,y0,x1,y1) fractions to blank out
    near_frac=0.45,              # "near field" = bottom 45 % of the ROI
    seed_min_len=50,             # min total segment length to accept a seed
    seed_gap=11,                 # px gap (at ref row) that separates clusters
    straight_cover=0.50,         # inliers must span >=60 % of ROI height
    straight_rms=6.0,            # ... with RMS residual below this (px)
    n_windows=14, win_margin=14, win_minpix=6,
    ema_new=0.45, ema_new_jump=0.12, max_hold=12,
)

PROFILES = {
    # elevated highway, vanishing region left of centre, watermark bottom-right
    "highway": dict(top_y=0.615, top_x=(0.28, 0.64),
                    exclude=(0.60, 0.86, 1.0, 1.0)),
    # winding mountain road, orange centre line, wet asphalt
    "mountain": dict(top_y=0.615, top_x=(0.22, 0.74)),
}

GREEN = (80, 220, 80)
LEFT_C = (255, 140, 30)
RIGHT_C = (40, 60, 255)

PAINTED, ROAD_EDGE, EST, UNKNOWN = "PAINTED_LANE_MARKING", "ROAD_EDGE", "ESTIMATED_LANE_BOUNDARY", "UNKNOWN"
OBSERVED, TRACKED, INFERRED = "OBSERVED", "TRACKED", "INFERRED"
CURRENT_LANE, ROAD_CORRIDOR, UNCERTAIN = "CURRENT_LANE", "ROAD_CORRIDOR", "UNCERTAIN"
GREEN, AMBER = (80, 220, 80), (40, 180, 255)

class _BaseDetector:
    def __init__(self, w, h, cfg=CFG):
        self.c = cfg
        self.w, self.h = w, h
        c = cfg
        ty, by = int(c["top_y"] * h), int(c["bot_y"] * h)
        poly = np.array([
            [c["bot_x"][0] * w, by], [c["top_x"][0] * w, ty],
            [c["top_x"][1] * w, ty], [c["bot_x"][1] * w - 1, by]], np.int32)
        self.poly = poly
        self.x0 = int(max(poly[:, 0].min(), 0))
        self.x1 = int(min(poly[:, 0].max() + 1, w))
        self.y0, self.y1 = ty, by
        self.H = self.y1 - self.y0
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [poly], 255)
        # erode so the artificial ROI border does not create edges
        if c.get("exclude"):
            ex = c["exclude"]
            cv2.rectangle(m, (int(ex[0] * w), int(ex[1] * h)),
                          (int(ex[2] * w), int(ex[3] * h)), 0, -1)
        m = cv2.erode(m, np.ones((5, 5), np.uint8))
        self.mask = m[self.y0:self.y1, self.x0:self.x1]
        self.clahe = cv2.createCLAHE(c["clahe_clip"], c["clahe_grid"])
        self.prev = {"L": None, "R": None}   # (a,b,c) in normalised y
        self.hold = {"L": 0, "R": 0}
        self.mode = {"L": "-", "R": "-"}
        self.edges = None

    # ------------------------------------------------------- preprocessing
    def _edges(self, frame):
        c = self.c
        crop = frame[self.y0:self.y1, self.x0:self.x1]
        L = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[:, :, 0]
        L = self.clahe.apply(L)
        L = cv2.GaussianBlur(L, (c["blur_k"], c["blur_k"]), 0)
        med = float(np.median(L[::4, ::4]))
        lo = int(np.clip(0.66 * med, 30, 90))
        e = cv2.Canny(L, lo, int(lo * 2.6))
        return cv2.bitwise_and(e, self.mask)

    # --------------------------------------------------- hough + filtering
    def _segments(self, edges):
        c = self.c
        ls = cv2.HoughLinesP(edges, 1, np.pi / 180, c["hough_thresh"],
                             minLineLength=c["hough_min_len"],
                             maxLineGap=c["hough_max_gap"])
        segs = []
        if ls is None:
            return segs
        if ls.ndim == 3:
            ls = ls[:, 0]
        # Hough works in crop coordinates -> shift back to full-frame coords
        ls = ls.astype(np.float32) + np.array(
            [self.x0, self.y0, self.x0, self.y0], np.float32)
        for x1, y1, x2, y2 in ls:
            dx, dy = x2 - x1, y2 - y1
            if dy == 0:
                continue
            if abs(dy) / (abs(dx) + 1e-6) < c["min_abs_slope"]:
                continue                      # crosswalk / shadow / horizon
            if y1 > y2:                       # make (x1,y1) the upper point
                x1, y1, x2, y2 = x2, y2, x1, y1
                dx, dy = -dx, -dy
            m = dx / dy                       # dx per dy  (x = m*y + b form)
            length = float(np.hypot(dx, dy))
            if abs(m) < c["min_abs_m"]:
                continue                      # vertical clutter (car edges, poles)
            segs.append((x1, y1, x2, y2, m, length))
        # NOTE: left/right is decided by position relative to the camera
        # centre (see _seed), not by slope sign: when the vanishing point is
        # left of the centre line, that line leans like a "right" line.
        return segs

    @staticmethod
    def _xat(seg, y):
        x1, y1, x2, y2, m, _ = seg
        return x1 + m * (y - y1)

    # ---------------------------------------------------------------- seed
    def _seed(self, segs, side, cx, yref):
        """Nearest-to-centre lane boundary in the near field."""
        c = self.c
        near_y = yref - c["near_frac"] * self.H
        cand = []
        for s in segs:
            if s[3] < near_y:                 # lower endpoint not in near field
                continue
            xr = self._xat(s, yref)
            if side == "L" and xr < cx or side == "R" and xr > cx:
                cand.append((xr, s))
        if not cand:
            return None
        cand.sort(key=lambda t: t[0])
        # 1-D clustering on x at the reference row
        clusters, cur = [], [cand[0]]
        for item in cand[1:]:
            if item[0] - cur[-1][0] > self.c["seed_gap"]:
                clusters.append(cur)
                cur = [item]
            else:
                cur.append(item)
        clusters.append(cur)
        clusters = [k for k in clusters
                    if sum(s[5] for _, s in k) >= c["seed_min_len"]]
        if not clusters:
            return None
        k = clusters[-1] if side == "L" else clusters[0]    # closest to centre
        wsum = sum(s[5] for _, s in k)
        xr = sum(x * s[5] for x, s in k) / wsum
        m = sum(s[4] * s[5] for _, s in k) / wsum
        return xr, m

    # ------------------------------------------------- Method A (straight)
    def _try_straight(self, segs, xr, m, yref):
        """Length-weighted average of every segment collinear with the seed."""
        tol = 10 + 0.04 * self.H
        inl = []
        for s in segs:
            ym = 0.5 * (s[1] + s[3])
            xm = 0.5 * (s[0] + s[2])
            xp = xr + m * (ym - yref)
            if abs(xm - xp) < tol and abs(s[4] - m) < 0.35 + 0.25 * abs(m):
                inl.append(s)
        if len(inl) < 2:
            return None
        ys = np.array([[s[1], s[3]] for s in inl])
        cover = (ys.max() - ys.min()) / self.H
        if cover < self.c["straight_cover"]:
            return None
        w = np.array([s[5] for s in inl])
        m2 = float(np.sum([s[4] * s[5] for s in inl]) / w.sum())
        xr2 = float(np.sum([self._xat(s, yref) * s[5] for s in inl]) / w.sum())
        res = []
        for s in inl:
            for (x, y) in ((s[0], s[1]), (s[2], s[3])):
                res.append(x - (xr2 + m2 * (y - yref)))
        if np.sqrt(np.mean(np.square(res))) > self.c["straight_rms"]:
            return None
        return (0.0, m2 * self.H, xr2)        # normalised-y coefficients

    # --------------------------------------- Method B (sliding window+poly)
    def _sliding(self, edges, xr, m, yref, side):
        c = self.c
        nw = c["n_windows"]
        wh = self.H / nw
        xs_all, ys_all = [], []
        x = xr
        dx_step = -m * wh                     # predicted shift per window going up
        hits = 0
        for i in range(nw):
            yb = yref - i * wh
            yt = yb - wh
            xp = x + dx_step
            marg = c["win_margin"] + 0.30 * abs(dx_step)
            xa = int(max(xp - marg - self.x0, 0))
            xb = int(min(xp + marg - self.x0, edges.shape[1]))
            ya = int(max(yt - self.y0, 0))
            yb_i = int(min(yb - self.y0, edges.shape[0]))
            if xb - xa < 4 or yb_i - ya < 2:
                x = xp
                continue
            ny, nx = np.nonzero(edges[ya:yb_i, xa:xb])
            if len(nx) >= c["win_minpix"]:
                nxf = nx + xa + self.x0
                nyf = ny + ya + self.y0
                xs_all.append(nxf)
                ys_all.append(nyf)
                # Gaussian weight around the prediction: the paint line wins
                # over denser clutter (grass, wall edges) at the window border
                wgt = np.exp(-((nxf - xp) / (0.55 * marg)) ** 2)
                new_x = float(np.sum(nxf * wgt) / np.sum(wgt))
                dx_step = 0.6 * dx_step + 0.4 * (new_x - x)
                x = new_x
                hits += 1
            else:
                x = xp
        if hits < 4:
            return None
        xs = np.concatenate(xs_all).astype(np.float32)
        ys = np.concatenate(ys_all).astype(np.float32)
        t = (ys - yref) / self.H
        coef = np.polyfit(t, xs, 2)
        r = xs - np.polyval(coef, t)
        keep = np.abs(r) < max(2.5 * r.std(), 4)
        if keep.sum() > 20:
            coef = np.polyfit(t[keep], xs[keep], 2)
        if abs(coef[0]) > 0.45 * self.w:      # implausible curvature
            coef = np.concatenate([[0.0], np.polyfit(t, xs, 1)])
        return tuple(float(v) for v in coef)

    # ------------------------------------------------------------- smoothing
    def _smooth(self, side, new):
        prev = self.prev[side]
        if new is None:
            self.hold[side] += 1
            if self.hold[side] > self.c["max_hold"]:
                self.prev[side] = None
            return self.prev[side]
        if prev is None:
            self.prev[side] = new
        else:
            jump = abs(new[2] - prev[2]) > 70
            a = self.c["ema_new_jump"] if (jump and self.hold[side] < 8) \
                else self.c["ema_new"]
            self.prev[side] = tuple(a * n + (1 - a) * p
                                    for n, p in zip(new, prev))
        self.hold[side] = 0 if not (new is None) else self.hold[side]
        return self.prev[side]

    # ------------------------------------------------------------------ main
    def process(self, frame):
        edges = self._edges(frame)
        self.edges = edges
        segs = self._segments(edges)
        yref = self.y1 - 1
        cx = self.w * 0.5
        out = {}
        for side in ("L", "R"):
            new = None
            seed = self._seed(segs, side, cx, yref)
            if seed is not None:
                xr, m = seed
                new = self._try_straight(segs, xr, m, yref)
                if new is not None:
                    self.mode[side] = "A"
                else:
                    new = self._sliding(edges, xr, m, yref, side)
                    self.mode[side] = "B" if new is not None else "-"
            else:
                self.mode[side] = "-"
            out[side] = self._smooth(side, new)

        # sanity (at the bottom row only): sensible lane width and order.
        # Near the vanishing point the boundaries converge, so the drawn path
        # is truncated there instead of rejecting the whole detection.
        L, R = out["L"], out["R"]
        if L is not None and R is not None:
            wd0 = R[2] - L[2]
            if wd0 < 0.10 * self.w or wd0 > 1.6 * self.w:
                bad = "L" if self.hold["L"] > self.hold["R"] else "R"
                self.prev[bad] = None
                out[bad] = None
        return out["L"], out["R"]

    # ------------------------------------------------------------- drawing
    def curve_pts(self, coef, t):
        ys = self.y1 - 1 + t * self.H
        xs = np.polyval(coef, t)
        return np.stack([xs, ys], 1)

    def draw(self, disp, L, R, fps=None, debug=False):
        s = disp.shape[1] / self.w
        ov = disp.copy()
        tt = np.linspace(-1, 0, 40)
        keep = None
        if L is not None and R is not None:
            wd = np.polyval(R, tt) - np.polyval(L, tt)
            bad = np.nonzero(wd < 8)[0]
            i0 = bad.max() + 1 if len(bad) else 0
            keep = tt[i0:] if len(tt) - i0 >= 5 else None
        if keep is not None:
            pl, pr = self.curve_pts(L, keep) * s, self.curve_pts(R, keep) * s
            poly = np.concatenate([pl, pr[::-1]]).astype(np.int32)
            cv2.fillPoly(ov, [poly], GREEN)
            disp = cv2.addWeighted(ov, 0.38, disp, 0.62, 0)
            cv2.polylines(disp, [pl.astype(np.int32)], False, LEFT_C, 3, cv2.LINE_AA)
            cv2.polylines(disp, [pr.astype(np.int32)], False, RIGHT_C, 3, cv2.LINE_AA)
            # centre line + lateral offset at the bottom of the ROI
            pc = ((pl + pr) / 2).astype(np.int32)
            cv2.polylines(disp, [pc], False, (255, 255, 255), 2, cv2.LINE_AA)
            wd = R[2] - L[2]
            off = (self.w / 2 - (L[2] + R[2]) / 2) / max(wd, 1)
            txt = f"offset {off:+.2f} lane (+ = right of centre)"
        else:
            cv2.polylines(disp, [(self.poly * s).astype(np.int32)], True,
                          (0, 0, 255), 1)
            txt = "no lane"
        hud = f"L:{self.mode['L']} R:{self.mode['R']}  {txt}"
        if fps:
            hud = f"{fps:5.0f} FPS (algo)  " + hud
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(disp, hud, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if debug and self.edges is not None:
            e = cv2.cvtColor(self.edges, cv2.COLOR_GRAY2BGR)
            e = cv2.resize(e, None, fx=s * 0.6, fy=s * 0.6)
            h, w = e.shape[:2]
            disp[34:34 + h, -w - 4:-4] = e
        return disp


# ======================================================================================================
# PART 2 - boundary tracking / evidence / classification  [ex lane_path_v2.py]
# ======================================================================================================
CFG2 = dict(
    ridge_min=8.0,          # cross-section ridge score for "thin bright stripe"
    edgeB_frac=2.0,         # sliding-window (Method B) edge accepted as ROAD_EDGE only above this edge_frac (2.0 = disabled)
    edgeB_cover=0.7,
    edge_min_len=250.0,     # a NON-painted (step-edge) seed cluster needs this much total segment length (~1.9x ROI height)
    ridge_frac_min=0.5,     # fraction of samples along a curve that must be ridge-like
    edge_frac_min=0.6,      # ROAD_EDGE: fraction of samples lying on a Canny edge
    edge_cover_min=0.5,     # ROAD_EDGE: supported span / ROI height
    max_hold2=6,            # frames a boundary may be TRACKED (original: 12)
    gate_px=18.0,           # shape gate: max |dx| over rows before measurements conflict
    reacquire_n=3,          # consistent conflicting measurements needed to replace the state
    wide_ratio=1.5,         # bottom width / recent CURRENT_LANE median width -> ROAD_CORRIDOR
    ema_new2=0.45,
)

class _DetectorV2(_BaseDetector):
    def __init__(self, w, h, cfg=None, cfg2=None):
        super().__init__(w, h, cfg or CFG)
        self.c2 = dict(CFG2, **(cfg2 or {}))
        self.state = {"L": None, "R": None}      # tracked boundary states
        self.conf = {"L": None, "R": None}       # last conflicting measurement
        self.conf_n = {"L": 0, "R": 0}
        self.width_hist = []
        self.result = None
        self.Lraw = None
        from collections import Counter
        self.stats = Counter()                    # diagnostic counters (why measurements were lost)

    # ------------------------------------------------------------ edges
    def _edges(self, frame):                      # identical to original + keeps raw L
        c = self.c
        crop = frame[self.y0:self.y1, self.x0:self.x1]
        L = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[:, :, 0]
        self.Lraw = L
        L = self.clahe.apply(L)
        L = cv2.GaussianBlur(L, (c["blur_k"], c["blur_k"]), 0)
        med = float(np.median(L[::4, ::4]))
        lo = int(np.clip(0.66 * med, 30, 90))
        e = cv2.Canny(L, lo, int(lo * 2.6))
        return cv2.bitwise_and(e, self.mask)

    # ------------------------------------------------- ridge (F1, F5)
    def _ridge_pairs(self, P0, P1, n=8):
        """cross-section ridge score for many short segments at once.
        P0,P1: (N,2) full-frame points. score = peak(centre) - max(flank), >0 thin bright stripe,
        ~0 or <0 step edge / uniform."""
        N = len(P0)
        if N == 0:
            return np.zeros(0)
        d = P1 - P0
        ln = np.hypot(d[:, 0], d[:, 1]) + 1e-6
        nx, ny = -d[:, 1] / ln, d[:, 0] / ln
        ts = np.linspace(0.1, 0.9, n)[None, :]
        px = P0[:, 0:1] + ts * d[:, 0:1] - self.x0
        py = P0[:, 1:2] + ts * d[:, 1:2] - self.y0
        offs = np.arange(-14, 15, 1.0)
        X = px[:, :, None] + offs[None, None, :] * nx[:, None, None]
        Y = py[:, :, None] + offs[None, None, :] * ny[:, None, None]
        H, W = self.Lraw.shape
        X = np.clip(X, 0, W - 1.001).astype(np.float32).reshape(N * n, 29)
        Y = np.clip(Y, 0, H - 1.001).astype(np.float32).reshape(N * n, 29)
        prof = cv2.remap(self.Lraw.astype(np.float32), X, Y, cv2.INTER_LINEAR).reshape(N, n, 29).mean(1)
        ymid = 0.5 * (P0[:, 1] + P1[:, 1])
        k = np.clip(0.35 + 0.035 * (ymid - self.y0), 0.35, 1.0)
        w = np.maximum(2, np.floor(3.0 * k + 1)).astype(int)
        cs = np.concatenate([np.zeros((N, 1), np.float32), np.cumsum(prof, 1)], 1)   # prefix sums -> O(1) window means
        def wmean(a, b):
            return (np.take_along_axis(cs, b[:, None], 1) - np.take_along_axis(cs, a[:, None], 1))[:, 0] / (b - a)
        best = np.full(N, -1e9)
        for ci in range(-5, 6):
            c = ci + 14
            peak = wmean(c - w // 2, c + w // 2 + 1)
            l1 = wmean(np.maximum(c - w - 6, 0), np.maximum(c - w - 1, 1))
            l2 = wmean(np.minimum(c + w + 1, 28), np.minimum(c + w + 6, 29))
            best = np.maximum(best, peak - np.maximum(l1, l2))
        return best

    # ------------------------------------------------ seeds (F1)
    def _seed_candidates(self, segs, side, cx, yref):
        """Original clustering rule, but returns ALL clusters nearest-to-centre first, after removing short non-painted
        (step-edge) clusters."""
        c = self.c
        near_y = yref - c["near_frac"] * self.H
        cand = []
        for s in segs:
            if s[3] < near_y:
                continue
            xr = self._xat(s, yref)
            if (side == "L" and xr < cx) or (side == "R" and xr > cx):
                cand.append((xr, s))
        if not cand:
            return []
        if self.near_ridge is None:
            ns = [s for s in segs if s[3] >= near_y]
            r = self._ridge_pairs(np.array([[s[0], s[1]] for s in ns], np.float32),
                                  np.array([[s[2], s[3]] for s in ns], np.float32)) if ns else []
            self.near_ridge = {id(s): v for s, v in zip(ns, r)}
        cand.sort(key=lambda t: t[0])
        clusters, cur = [], [cand[0]]
        for item in cand[1:]:
            if item[0] - cur[-1][0] > c["seed_gap"]:
                clusters.append(cur)
                cur = [item]
            else:
                cur.append(item)
        clusters.append(cur)
        out = []
        for k in clusters:
            tot = sum(s[5] for _, s in k)
            if tot < c["seed_min_len"]:
                continue
            xr = sum(x * s[5] for x, s in k) / tot
            m = sum(s[4] * s[5] for _, s in k) / tot
            ridge = sum(self.near_ridge[id(s)] * s[5] for _, s in k) / tot
            out.append(dict(xr=xr, m=m, length=tot, ridge=ridge, painted=ridge >= self.c2["ridge_min"]))
        order = out[::-1] if side == "L" else out            # nearest to centre first (original preference kept)
        # F1: a short step edge (car side, gore chevron, post) is not credible evidence of a boundary
        return [k for k in order if k["painted"] or k["length"] >= self.c2["edge_min_len"]]

    # ----------------------------------------- sliding + support (F4)
    def _sliding2(self, edges, xr, m, yref):
        """Original _sliding() logic, additionally returning the supported y-range."""
        c = self.c
        nw = c["n_windows"]
        wh = self.H / nw
        xs_all, ys_all = [], []
        x = xr
        dx_step = -m * wh
        hits = 0
        for i in range(nw):
            yb = yref - i * wh
            yt = yb - wh
            xp = x + dx_step
            marg = c["win_margin"] + 0.30 * abs(dx_step)
            xa = int(max(xp - marg - self.x0, 0))
            xb = int(min(xp + marg - self.x0, edges.shape[1]))
            ya = int(max(yt - self.y0, 0))
            yb_i = int(min(yb - self.y0, edges.shape[0]))
            if xb - xa < 4 or yb_i - ya < 2:
                x = xp
                continue
            ny, nx = np.nonzero(edges[ya:yb_i, xa:xb])
            if len(nx) >= c["win_minpix"]:
                nxf = nx + xa + self.x0
                nyf = ny + ya + self.y0
                xs_all.append(nxf)
                ys_all.append(nyf)
                wgt = np.exp(-((nxf - xp) / (0.55 * marg)) ** 2)
                new_x = float(np.sum(nxf * wgt) / np.sum(wgt))
                dx_step = 0.6 * dx_step + 0.4 * (new_x - x)
                x = new_x
                hits += 1
            else:
                x = xp
        if hits < 4:
            return None
        xs = np.concatenate(xs_all).astype(np.float32)
        ys = np.concatenate(ys_all).astype(np.float32)
        t = (ys - yref) / self.H
        coef = np.polyfit(t, xs, 2)
        r = xs - np.polyval(coef, t)
        keep = np.abs(r) < max(2.5 * r.std(), 4)
        if keep.sum() > 20:
            coef = np.polyfit(t[keep], xs[keep], 2)
        if abs(coef[0]) > 0.45 * self.w:
            coef = np.concatenate([[0.0], np.polyfit(t, xs, 1)])
        y_top = float(np.percentile(ys[keep] if keep.sum() > 20 else ys, 2))
        return tuple(float(v) for v in coef), y_top, hits

    def _fit_side(self, segs, edges, cand, yref):
        """straight (Method A) else sliding (Method B); returns measurement dict or None."""
        xr, m = cand["xr"], cand["m"]
        st = self._try_straight(segs, xr, m, yref)
        if st is not None:
            tol = 10 + 0.04 * self.H
            inl = [s for s in segs if abs(0.5 * (s[0] + s[2]) - (xr + m * (0.5 * (s[1] + s[3]) - yref))) < tol
                   and abs(s[4] - m) < 0.35 + 0.25 * abs(m)]
            y_top = float(min(min(s[1], s[3]) for s in inl)) if inl else yref - 0.5 * self.H
            return dict(coef=st, y_top=y_top, mode="A", windows=None)
        sl = self._sliding2(edges, xr, m, yref)
        if sl is None:
            return None
        return dict(coef=sl[0], y_top=sl[1], mode="B", windows=sl[2])

    # ---------------------------------- curve evidence typing (F5)
    def _curve_evidence(self, meas, edges_dil, yref):
        ys = np.arange(yref, meas["y_top"] - 1, -9.0)
        if len(ys) < 3:
            return 0.0, 0.0
        t = (ys - yref) / self.H
        xs = np.polyval(meas["coef"], t)
        P = np.stack([xs, ys], 1).astype(np.float32)
        r = self._ridge_pairs(P[:-1], P[1:], n=6)
        ridge_frac = float(np.mean(r >= self.c2["ridge_min"]))
        xi = np.clip(np.round(xs - self.x0).astype(int), 0, edges_dil.shape[1] - 1)
        yi = np.clip(np.round(ys - self.y0).astype(int), 0, edges_dil.shape[0] - 1)
        edge_frac = float(np.mean(edges_dil[yi, xi] > 0))
        return ridge_frac, edge_frac

    # ------------------------------------------------- tracking (F2, F3)
    def _shape_dist(self, a, b, yref):
        top = max(min(a["y_top"], b["y_top"]), self.y0)       # compare where both are supported
        ts = np.linspace((top - yref) / self.H, 0, 5)
        return float(np.max(np.abs(np.polyval(a["coef"], ts) - np.polyval(b["coef"], ts))))

    def _track(self, side, meas, yref):
        st = self.state[side]
        c2 = self.c2
        if meas is None:
            self.stats[side + ":no_measurement"] += 1
            if st is None:
                return None
            st["age"] += 1
            if st["age"] > c2["max_hold2"]:
                self.state[side] = None
                self.conf_n[side] = 0
                return None
            st["source"] = TRACKED
            return st
        meas["age"] = 0
        meas["source"] = OBSERVED
        if st is None:
            self.state[side] = meas
            self.conf_n[side] = 0
            return meas
        d = self._shape_dist(st, meas, yref)
        if d <= c2["gate_px"]:                                  # consistent -> blend like the original EMA
            a = c2["ema_new2"]
            st["coef"] = tuple(a * n + (1 - a) * p for n, p in zip(meas["coef"], st["coef"]))
            st["y_top"] = a * meas["y_top"] + (1 - a) * st["y_top"] if st["age"] == 0 else meas["y_top"]
            for k in ("mode", "windows", "ridge_frac", "edge_frac"):
                st[k] = meas.get(k)
            st["age"], st["source"] = 0, OBSERVED
            self.conf_n[side] = 0
            self.stats[side + ":blend"] += 1
            return st
        # inconsistent: do not blend. Re-acquire only when the same new curve persists.
        if self.conf[side] is not None and self._shape_dist(self.conf[side], meas, yref) <= c2["gate_px"]:
            self.conf_n[side] += 1
        else:
            self.conf_n[side] = 1
        self.conf[side] = meas
        if self.conf_n[side] >= c2["reacquire_n"] or st["age"] > 0:
            self.stats[side + ":replace"] += 1
            self.state[side] = meas
            self.conf_n[side] = 0
            return meas
        self.stats[side + ":conflict_hold"] += 1
        st["age"] += 1                                          # keep old curve, but say so
        st["source"] = TRACKED
        return st

    # ------------------------------------------------------------ main
    def process(self, frame):
        edges = self._edges(frame)
        self.edges = edges
        self.near_ridge = None
        segs = self._segments(edges)
        yref = self.y1 - 1
        cx = self.w * 0.5
        edges_dil = cv2.dilate(edges, np.ones((5, 5), np.uint8))
        meas = {"L": None, "R": None}
        for side in ("L", "R"):
            cands = self._seed_candidates(segs, side, cx, yref)[:2]
            if not cands:
                self.stats[side + ":no_seed_candidate"] += 1
            for cand in cands:
                mm = self._fit_side(segs, edges, cand, yref)
                if mm is not None:
                    mm["ridge_frac"], mm["edge_frac"] = self._curve_evidence(mm, edges_dil, yref)
                    mm["seed_ridge"], mm["seed_len"] = cand["ridge"], cand["length"]
                    meas[side] = mm
                    break
        # pair validation on raw measurements (no fixed width / parallelism): ordering + no divergence upward
        L, R = meas["L"], meas["R"]
        if L is not None and R is not None:
            top = max(L["y_top"], R["y_top"])
            ts = np.linspace(0, (top - yref) / self.H, 12)                  # bottom -> top of common support
            wd = np.polyval(R["coef"], ts) - np.polyval(L["coef"], ts)
            conv = np.nonzero(wd < 8)[0]                                    # convergence at the vanishing region is expected
            wd_v = wd[:conv[0]] if len(conv) else wd                        # only judge rows before convergence
            bad = (wd[0] < 0.10 * self.w or wd[0] > 1.6 * self.w           # bottom ordering / extreme width (original rule)
                   or (len(wd_v) > 1 and wd_v.max() > 1.25 * wd[0] + 10))   # corridor widening upward = diverging pair
            if bad:
                weaker = "L" if (L["seed_len"] * (1 + L["ridge_frac"])) < (R["seed_len"] * (1 + R["ridge_frac"])) else "R"
                meas[weaker] = None
                self.stats[weaker + ":pair_drop"] += 1
        out = {}
        for side in ("L", "R"):
            out[side] = self._track(side, meas[side], yref)
            self.mode[side] = "-" if out[side] is None else ("A" if out[side].get("mode") == "A" else "B")
        self.result = self._classify(out, yref)
        L, R = out["L"], out["R"]
        return (None if L is None else L["coef"]), (None if R is None else R["coef"])

    # ----------------------------------------- classification (F5)
    def _btype(self, b):
        c2 = self.c2
        if b is None:
            return UNKNOWN
        if b["ridge_frac"] >= c2["ridge_frac_min"]:
            return PAINTED
        cover = (self.y1 - 1 - b["y_top"]) / self.H
        # ROAD_EDGE additionally needs straight-line support (Method A: collinear, low RMS) - barrier bases/curbs
        # are straight; bollard rows, queued vehicles and booth edges fitted by sliding windows are not credible edges
        if b["edge_frac"] >= c2["edge_frac_min"] and cover >= c2["edge_cover_min"] and (b.get("mode") == "A" or (b["edge_frac"] >= c2["edgeB_frac"] and cover >= c2["edgeB_cover"])):
            return ROAD_EDGE
        return UNKNOWN

    def _classify(self, st, yref):
        res = dict(mode=UNCERTAIN, boundaries={}, offset=None, reason="")
        for side in ("L", "R"):
            b = st[side]
            if b is None:
                res["boundaries"][side] = None
                continue
            ys = np.linspace(yref, b["y_top"], 24)
            pts = np.stack([np.polyval(b["coef"], (ys - yref) / self.H), ys], 1)
            res["boundaries"][side] = dict(id=side, type=self._btype(b), source=b["source"], points=pts,
                                           support=dict(span_frac=(yref - b["y_top"]) / self.H, windows=b.get("windows"),
                                                        method=b.get("mode"), age=b["age"]),
                                           quality=dict(ridge_frac=b["ridge_frac"], edge_frac=b["edge_frac"],
                                                        note="heuristic score, not a probability"))
        bl, br = res["boundaries"]["L"], res["boundaries"]["R"]
        if bl is None or br is None:
            res["reason"] = "boundary unavailable"
            return res
        tl, tr = bl["type"], br["type"]
        if UNKNOWN in (tl, tr):
            res["reason"] = "boundary type unsupported"
            return res
        wd0 = st["R"]["coef"][2] - st["L"]["coef"][2]
        both_obs = bl["source"] == OBSERVED and br["source"] == OBSERVED
        h = self.width_hist
        ref = float(np.median(h[:-30])) if len(h) >= 45 else None      # widths older than ~1 s
        wide = ref is not None and wd0 > self.c2["wide_ratio"] * ref
        if tl == PAINTED and tr == PAINTED and both_obs:
            self.width_hist = (h + [wd0])[-120:]
        if tl == PAINTED and tr == PAINTED and not wide:
            res["mode"] = CURRENT_LANE
            res["offset"] = (self.w / 2 - (st["L"]["coef"][2] + st["R"]["coef"][2]) / 2) / max(wd0, 1)
        else:
            res["mode"] = ROAD_CORRIDOR
            res["reason"] = "wide (%.2fx lane width ~1-3 s ago)" % (wd0 / ref) if wide else "includes road-edge boundary"
        return res

    # ------------------------------------------------------------- drawing
    def draw(self, disp, L=None, R=None, fps=None, debug=False):
        s = disp.shape[1] / self.w
        res = self.result
        ov = disp.copy()
        bl, br = res["boundaries"]["L"], res["boundaries"]["R"]
        if res["mode"] != UNCERTAIN:
            top = max(self.state["L"]["y_top"], self.state["R"]["y_top"])      # corridor only where BOTH are supported
            ys = np.linspace(self.y1 - 1, top, 30)
            tt = (ys - (self.y1 - 1)) / self.H
            xl, xr = np.polyval(self.state["L"]["coef"], tt), np.polyval(self.state["R"]["coef"], tt)
            ok = (xr - xl) >= 8
            if ok.sum() >= 5:
                i = np.nonzero(ok)[0].max() + 1
                pl = np.stack([xl[:i], ys[:i]], 1) * s
                pr = np.stack([xr[:i], ys[:i]], 1) * s
                col = GREEN if res["mode"] == CURRENT_LANE else AMBER
                cv2.fillPoly(ov, [np.concatenate([pl, pr[::-1]]).astype(np.int32)], col)
                disp = cv2.addWeighted(ov, 0.30, disp, 0.70, 0)
        for b, colr in ((bl, LEFT_C), (br, RIGHT_C)):
            if b is None or (b["type"] == UNKNOWN and not debug) or (res["mode"] == UNCERTAIN and not debug):
                continue          # unsupported boundaries / UNCERTAIN frames are not drawn (HUD still reports them)
            p = (b["points"] * s).astype(np.int32)
            th = 3 if b["type"] == PAINTED else 2
            if b["source"] == OBSERVED:
                cv2.polylines(disp, [p], False, colr, th, cv2.LINE_AA)
            else:                                               # TRACKED: dashed
                for i in range(0, len(p) - 1, 2):
                    cv2.line(disp, tuple(p[i]), tuple(p[i + 1]), colr, th, cv2.LINE_AA)
        def lab(b):
            return "-" if b is None else {PAINTED: "paint", ROAD_EDGE: "edge", UNKNOWN: "unsupported"}[b["type"]] + ("" if b["source"] == OBSERVED else "/trk")
        txt = res["mode"] + ("" if res["offset"] is None else f" off {res['offset']:+.2f}")
        hud = f"L:{lab(bl)} R:{lab(br)}  {txt}"
        if res["reason"] and res["mode"] != CURRENT_LANE:
            hud += f"  ({res['reason']})"
        if fps:
            hud = f"{fps:5.0f} FPS (algo)  " + hud
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(disp, hud, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if debug:
            cv2.polylines(disp, [(self.poly * s).astype(np.int32)], True, (0, 0, 255), 1)   # ROI only in debug
            if self.edges is not None:
                e = cv2.cvtColor(self.edges, cv2.COLOR_GRAY2BGR)
                e = cv2.resize(e, None, fx=s * 0.6, fy=s * 0.6)
                h, w = e.shape[:2]
                disp[34:34 + h, -w - 4:-4] = e
        return disp


# ======================================================================================================
# PART 3 - speed-optimised ridge scoring / seed search  [ex lane_path_v3.py]
# ======================================================================================================
CFG3 = dict(
    ridge_n_seed=8,        # longitudinal samples per segment for seed scoring   (V2: 8)
    ridge_n_curve=6,       # longitudinal samples per curve piece                (V2: 6)
    ridge_half=14,         # cross-section half width in px (profile = 2*half+1) (V2: 14)
    ridge_centres=5,       # +/- centre search range                             (V2: 5)
    ridge_flank=6,         # flank window length                                 (V2: 6)
    max_cands=2,           # candidates tried per side                           (V2: 2)
)

class _DetectorV3(_DetectorV2):
    def __init__(self, w, h, cfg=None, cfg2=None, cfg3=None):
        super().__init__(w, h, cfg, cfg2)
        self.c3 = dict(CFG3, **(cfg3 or {}))
        self.Lf = None
        self._k5 = np.ones((5, 5), np.uint8)
        self._ts = {}
        self._build_ridge_tables()

    # ----------------------------------------------------------- tables
    def _build_ridge_tables(self):
        """Index tables for the O(1) prefix-sum windows of _ridge_pairs, one row per window width w (2..4)."""
        c3 = self.c3
        half, nc, fl = c3["ridge_half"], c3["ridge_centres"], c3["ridge_flank"]
        self._L = 2 * half + 1
        cen = np.arange(-nc, nc + 1) + half                       # candidate centres (11 in V2)
        K = len(cen)
        A = np.zeros((5, 6 * K), np.int64)                        # [a_peak, a_l1, a_l2 | b_peak, b_l1, b_l2] per w
        D = np.ones((5, 3 * K), np.int64)
        for w in range(2, 5):
            a0, b0 = cen - w // 2, cen + w // 2 + 1
            a1, b1 = np.maximum(cen - w - fl, 0), np.maximum(cen - w - 1, 1)
            a2, b2 = np.minimum(cen + w + 1, self._L - 1), np.minimum(cen + w + fl, self._L)
            A[w] = np.concatenate([a0, a1, a2, b0, b1, b2])
            D[w] = np.concatenate([b0 - a0, b1 - a1, b2 - a2])
        self._A, self._D, self._K = A, D, K
        self._offs = np.arange(-half, half + 1, 1.0)

    # --------------------------------------------------------- edges (O2, O6)
    def _edges(self, frame):
        c = self.c
        crop = frame[self.y0:self.y1, self.x0:self.x1]
        L = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[:, :, 0]
        self.Lraw = L
        self.Lf = None                                         # float32 copy made lazily, once per frame
        L = self.clahe.apply(L)
        L = cv2.GaussianBlur(L, (c["blur_k"], c["blur_k"]), 0)
        sub = L[::4, ::4]
        h = np.bincount(sub.reshape(-1), minlength=256)         # exact median of a uint8 sample (== np.median)
        n = sub.size
        cs = np.cumsum(h)
        a = int(np.searchsorted(cs, (n - 1) // 2, "right"))
        b = int(np.searchsorted(cs, n // 2, "right"))
        med = (a + b) / 2.0
        lo = int(np.clip(0.66 * med, 30, 90))
        e = cv2.Canny(L, lo, int(lo * 2.6))
        return cv2.bitwise_and(e, self.mask)

    # ---------------------------------------------------------- segments (O5)
    def _segments(self, edges):
        c = self.c
        ls = cv2.HoughLinesP(edges, 1, np.pi / 180, c["hough_thresh"], minLineLength=c["hough_min_len"],
                             maxLineGap=c["hough_max_gap"])
        if ls is None:
            return []
        if ls.ndim == 3:
            ls = ls[:, 0]
        ls = ls.astype(np.float32) + np.array([self.x0, self.y0, self.x0, self.y0], np.float32)
        x1, y1, x2, y2 = ls[:, 0], ls[:, 1], ls[:, 2], ls[:, 3]
        dx, dy = x2 - x1, y2 - y1
        keep = (dy != 0) & (np.abs(dy) / (np.abs(dx) + 1e-6) >= c["min_abs_slope"])
        sw = y1 > y2                                           # make (x1,y1) the upper point
        X1, Y1 = np.where(sw, x2, x1), np.where(sw, y2, y1)
        X2, Y2 = np.where(sw, x1, x2), np.where(sw, y1, y2)
        DX, DY = np.where(sw, -dx, dx), np.where(sw, -dy, dy)
        with np.errstate(divide="ignore", invalid="ignore"):
            M = DX / DY
        keep &= np.abs(M) >= c["min_abs_m"]
        if not keep.any():
            return []
        LEN = np.hypot(DX[keep], DY[keep]).tolist()
        return list(zip(X1[keep], Y1[keep], X2[keep], Y2[keep], M[keep], LEN))

    # ----------------------------------------------------- ridge (O1, O2)
    def _ridge_pairs(self, P0, P1, n=8):
        """Same quantity as V2._ridge_pairs (peak(centre) - max(flank) of the mean cross-section, best over centres)."""
        N = len(P0)
        if N == 0:
            return np.zeros(0)
        d = P1 - P0
        ln = np.hypot(d[:, 0], d[:, 1]) + 1e-6
        nx, ny = -d[:, 1] / ln, d[:, 0] / ln
        ts = self._ts.get(n)
        if ts is None:
            ts = self._ts[n] = np.linspace(0.1, 0.9, n)[None, :]
        px = P0[:, 0:1] + ts * d[:, 0:1] - self.x0
        py = P0[:, 1:2] + ts * d[:, 1:2] - self.y0
        offs = self._offs
        Lw = self._L
        X = px[:, :, None] + offs[None, None, :] * nx[:, None, None]
        Y = py[:, :, None] + offs[None, None, :] * ny[:, None, None]
        if self.Lf is None:
            self.Lf = self.Lraw.astype(np.float32)             # once per frame
        H, W = self.Lf.shape
        X = np.clip(X, 0, W - 1.001).astype(np.float32).reshape(N * n, Lw)
        Y = np.clip(Y, 0, H - 1.001).astype(np.float32).reshape(N * n, Lw)
        prof = cv2.remap(self.Lf, X, Y, cv2.INTER_LINEAR).reshape(N, n, Lw).mean(1)
        ymid = 0.5 * (P0[:, 1] + P1[:, 1])
        k = np.clip(0.35 + 0.035 * (ymid - self.y0), 0.35, 1.0)
        w = np.maximum(2, np.floor(3.0 * k + 1)).astype(int)
        cs = np.empty((N, Lw + 1), np.float32)
        cs[:, 0] = 0
        np.cumsum(prof, 1, out=cs[:, 1:])
        K = self._K
        g = np.take_along_axis(cs, self._A[w], 1)               # all window ends for all centres in one gather
        den = self._D[w]
        peak = (g[:, 3 * K:4 * K] - g[:, 0:K]) / den[:, 0:K]
        l1 = (g[:, 4 * K:5 * K] - g[:, K:2 * K]) / den[:, K:2 * K]
        l2 = (g[:, 5 * K:6 * K] - g[:, 2 * K:3 * K]) / den[:, 2 * K:3 * K]
        return np.maximum(-1e9, (peak - np.maximum(l1, l2)).max(1))

    # ------------------------------------------------- seeds (O3)
    def _clusters(self, segs, side, cx, yref):
        """Original clustering rule; geometry only, no appearance work.  Nearest-to-centre first."""
        c = self.c
        near_y = yref - c["near_frac"] * self.H
        cand = []
        for i, s in enumerate(segs):
            if s[3] < near_y:
                continue
            xr = self._xat(s, yref)
            if (side == "L" and xr < cx) or (side == "R" and xr > cx):
                cand.append((xr, s, i))
        if not cand:
            return []
        cand.sort(key=lambda t: t[0])
        clusters, cur = [], [cand[0]]
        for item in cand[1:]:
            if item[0] - cur[-1][0] > c["seed_gap"]:
                clusters.append(cur)
                cur = [item]
            else:
                cur.append(item)
        clusters.append(cur)
        out = []
        for k in clusters:
            tot = sum(s[5] for _, s, _ in k)
            if tot < c["seed_min_len"]:
                continue
            xr = sum(x * s[5] for x, s, _ in k) / tot
            m = sum(s[4] * s[5] for _, s, _ in k) / tot
            out.append(dict(xr=xr, m=m, length=tot, members=k))
        return out[::-1] if side == "L" else out

    def _seed_candidates_both(self, segs, cx, yref):
        c2 = self.c2
        order = {sd: self._clusters(segs, sd, cx, yref) for sd in ("L", "R")}
        # clusters whose appearance can still change which (<= max_cands) candidates are returned
        need = {}
        for sd, cl in order.items():
            sure, upto = 0, len(cl)
            for j, k in enumerate(cl):
                if k["length"] >= c2["edge_min_len"]:
                    sure += 1
                    if sure >= self.c3["max_cands"]:
                        upto = j + 1
                        break
            need[sd] = cl[:upto]
        segsel = [(sd, k, mem) for sd in ("L", "R") for k in need[sd] for mem in k["members"]]
        if segsel:
            P0 = np.array([[m[1][0], m[1][1]] for _, _, m in segsel], np.float32)
            P1 = np.array([[m[1][2], m[1][3]] for _, _, m in segsel], np.float32)
            r = self._ridge_pairs(P0, P1, n=self.c3["ridge_n_seed"])
            acc = {}
            for (sd, k, (_, s, _)), v in zip(segsel, r):
                acc.setdefault(id(k), []).append(v * s[5])
            for sd in need:
                for k in need[sd]:
                    k["ridge"] = sum(acc[id(k)]) / k["length"]
                    k["painted"] = k["ridge"] >= c2["ridge_min"]
        out = {}
        for sd in ("L", "R"):
            ok = [k for k in need[sd] if k["painted"] or k["length"] >= c2["edge_min_len"]]
            out[sd] = ok[:self.c3["max_cands"]]
        return out

    # ------------------------------------------------ evidence (O4)
    def _evidence_batch(self, meas, edges, yref):
        sides = [s for s in ("L", "R") if meas[s] is not None]
        if not sides:
            return
        edges_dil = cv2.dilate(edges, self._k5)
        p0, p1, spans = [], [], []
        for sd in sides:
            mm = meas[sd]
            ys = np.arange(yref, mm["y_top"] - 1, -9.0)
            if len(ys) < 3:
                mm["ridge_frac"], mm["edge_frac"] = 0.0, 0.0
                continue
            t = (ys - yref) / self.H
            xs = np.polyval(mm["coef"], t)
            P = np.stack([xs, ys], 1).astype(np.float32)
            xi = np.clip(np.round(xs - self.x0).astype(int), 0, edges_dil.shape[1] - 1)
            yi = np.clip(np.round(ys - self.y0).astype(int), 0, edges_dil.shape[0] - 1)
            mm["edge_frac"] = float(np.mean(edges_dil[yi, xi] > 0))
            p0.append(P[:-1])
            p1.append(P[1:])
            spans.append((sd, len(P) - 1))
        if not p0:
            return
        r = self._ridge_pairs(np.concatenate(p0), np.concatenate(p1), n=self.c3["ridge_n_curve"])
        i = 0
        for sd, m in spans:
            meas[sd]["ridge_frac"] = float(np.mean(r[i:i + m] >= self.c2["ridge_min"]))
            i += m

    # ------------------------------------------------------------ main
    def process(self, frame):
        edges = self._edges(frame)
        self.edges = edges
        segs = self._segments(edges)
        yref = self.y1 - 1
        cx = self.w * 0.5
        cands = self._seed_candidates_both(segs, cx, yref)
        meas = {"L": None, "R": None}
        for side in ("L", "R"):
            if not cands[side]:
                self.stats[side + ":no_seed_candidate"] += 1
            for cand in cands[side]:
                mm = self._fit_side(segs, edges, cand, yref)
                if mm is not None:
                    mm["seed_ridge"], mm["seed_len"] = cand["ridge"], cand["length"]
                    meas[side] = mm
                    break
        self._evidence_batch(meas, edges, yref)
        # ---- pair validation: verbatim from V2 -------------------------------------------------------------
        L, R = meas["L"], meas["R"]
        if L is not None and R is not None:
            top = max(L["y_top"], R["y_top"])
            ts = np.linspace(0, (top - yref) / self.H, 12)
            wd = np.polyval(R["coef"], ts) - np.polyval(L["coef"], ts)
            conv = np.nonzero(wd < 8)[0]
            wd_v = wd[:conv[0]] if len(conv) else wd
            bad = (wd[0] < 0.10 * self.w or wd[0] > 1.6 * self.w
                   or (len(wd_v) > 1 and wd_v.max() > 1.25 * wd[0] + 10))
            if bad:
                weaker = "L" if (L["seed_len"] * (1 + L["ridge_frac"])) < (R["seed_len"] * (1 + R["ridge_frac"])) else "R"
                meas[weaker] = None
                self.stats[weaker + ":pair_drop"] += 1
        out = {}
        for side in ("L", "R"):
            out[side] = self._track(side, meas[side], yref)
            self.mode[side] = "-" if out[side] is None else ("A" if out[side].get("mode") == "A" else "B")
        self.result = self._classify(out, yref)
        L, R = out["L"], out["R"]
        return (None if L is None else L["coef"]), (None if R is None else R["coef"])


# ======================================================================================================
# PART 4 - common-support shape distance, reacquisition, paint evidence, corridor  [ex lane_path_v4.py]
# ======================================================================================================
CFG4 = dict(
    paint_edge_min=0.40,    # PAINTED needs >= this fraction of curve samples on a Canny edge
    paint_span_min=0.35,    # ... and a supported span of at least this fraction of the ROI height
    paint_slope_min=0.25,   # ... and |dx/dy| at the bottom >= this (near-vertical boundaries: poles, vehicle sides)
    paint_persist=2,        # ... and the evidence must have accumulated: paint_n >= this (+1 per passing observed frame, -1 per failing one)
    paint_cap=4,            # paint_n saturates here, so one bad frame cannot erase an established boundary's history
    edge_slope_min=0.25,    # ROAD_EDGE also may not be near-vertical
    contradicted_blocks=True,  # a boundary held against contradicting measurements cannot support a lane claim
    corr_n=30,              # rows sampled along the corridor
    corr_min_w=8.0,         # px: corridor narrower than this = converged (end of useful corridor)
    corr_growth=1.25,       # width may not exceed growth*w0 + corr_grow_add going up the image (diverging = invalid)
    corr_grow_add=10.0,
    corr_min_pts=5,         # fewer valid rows than this -> no corridor -> UNCERTAIN
)

class _DetectorV4(_DetectorV3):
    def __init__(self, w, h, cfg=None, cfg2=None, cfg3=None, cfg4=None):
        super().__init__(w, h, cfg, cfg2, cfg3)
        self.c4 = dict(CFG4, **(cfg4 or {}))
        self.conf_paint = {"L": 0, "R": 0}


    # ------------------------------------------------------------ C1
    def _shape_dist(self, a, b, yref):
        top = max(a["y_top"], b["y_top"], self.y0)             # common support ends at the NEARER of the two tops
        if top >= yref - 1.0:
            return float("inf")                                  # no common supported region
        ts = np.linspace((top - yref) / self.H, 0, 5)
        return float(np.max(np.abs(np.polyval(a["coef"], ts) - np.polyval(b["coef"], ts))))

    # ------------------------------------------------------------ C3 helpers
    def _paint_static(self, m):
        """Evidence beyond the ridge score that a boundary can be a painted marking."""
        c2, c4 = self.c2, self.c4
        cover = (self.y1 - 1 - m["y_top"]) / self.H
        slope = abs(m["coef"][1]) / self.H
        return bool(m["ridge_frac"] >= c2["ridge_frac_min"] and m["edge_frac"] >= c4["paint_edge_min"]
                    and cover >= c4["paint_span_min"] and slope >= c4["paint_slope_min"])

    # ------------------------------------------------------------ C2
    def _track(self, side, meas, yref):
        st = self.state[side]
        c2 = self.c2
        if meas is None:
            self.stats[side + ":no_measurement"] += 1
            self.conf[side], self.conf_n[side], self.conf_paint[side] = None, 0, 0     # a gap breaks a conflict streak
            if st is None:
                return None
            st["age"] += 1
            if st["age"] > c2["max_hold2"]:
                self.state[side] = None
                return None
            st["source"] = TRACKED
            return st
        meas["age"] = 0
        meas["source"] = OBSERVED
        ok = self._paint_static(meas)
        if st is None:
            meas["paint_n"] = 1 if ok else 0
            self.state[side] = meas
            self.conf[side], self.conf_n[side], self.conf_paint[side] = None, 0, 0
            return meas
        d = self._shape_dist(st, meas, yref)
        if d <= c2["gate_px"]:                                   # consistent -> blend like V2
            a = c2["ema_new2"]
            st["coef"] = tuple(a * n + (1 - a) * p for n, p in zip(meas["coef"], st["coef"]))
            st["y_top"] = a * meas["y_top"] + (1 - a) * st["y_top"] if st["age"] == 0 else meas["y_top"]
            for k in ("mode", "windows", "ridge_frac", "edge_frac"):
                st[k] = meas.get(k)
            pn = st.get("paint_n", 0)
            st["paint_n"] = min(self.c4["paint_cap"], pn + 1) if ok else max(0, pn - 1)
            st["age"], st["source"], st["contradicted"] = 0, OBSERVED, False
            self.conf[side], self.conf_n[side], self.conf_paint[side] = None, 0, 0
            self.stats[side + ":blend"] += 1
            return st
        # inconsistent: never blend.  Replace ONLY after reacquire_n consecutive measurements that agree with each other.
        prev = self.conf[side]
        if prev is not None and self._shape_dist(prev, meas, yref) <= c2["gate_px"]:
            self.conf_n[side] += 1
            self.conf_paint[side] = self.conf_paint[side] + 1 if ok else 0
        else:
            self.conf_n[side] = 1
            self.conf_paint[side] = 1 if ok else 0
        self.conf[side] = meas
        if self.conf_n[side] >= c2["reacquire_n"]:
            self.stats[side + ":replace"] += 1
            meas["paint_n"] = self.conf_paint[side]
            self.state[side] = meas
            self.conf[side], self.conf_n[side], self.conf_paint[side] = None, 0, 0
            return meas
        self.stats[side + ":conflict_hold"] += 1
        st["contradicted"] = True                                # held AGAINST current measurements (not merely unmeasured)
        st["age"] += 1                                           # keep the old curve, but say so ...
        if st["age"] > c2["max_hold2"]:                          # ... and do not hold it forever
            self.stats[side + ":conflict_expire"] += 1
            self.state[side] = None
            return None
        st["source"] = TRACKED
        return st

    # ------------------------------------------------------------ C3
    def _btype(self, b):
        c2, c4 = self.c2, self.c4
        if b is None:
            return UNKNOWN
        if self._paint_static(b) and b.get("paint_n", 0) >= c4["paint_persist"]:
            return PAINTED
        cover = (self.y1 - 1 - b["y_top"]) / self.H
        slope = abs(b["coef"][1]) / self.H
        # ROAD_EDGE: straight-line (Method A) support, long, edge-supported, not near-vertical
        if (b["edge_frac"] >= c2["edge_frac_min"] and cover >= c2["edge_cover_min"] and slope >= c4["edge_slope_min"]
                and (b.get("mode") == "A" or (b["edge_frac"] >= c2["edgeB_frac"] and cover >= c2["edgeB_cover"]))):
            return ROAD_EDGE
        return UNKNOWN

    # ------------------------------------------------------------ C4
    def _corridor(self, L, R, yref):
        """Extent of the region between two boundaries that is supported by BOTH and geometrically valid.
        Stops at the FIRST invalid row (never at a later valid one)."""
        c4 = self.c4
        top = max(L["y_top"], R["y_top"], self.y0)
        out = dict(valid=False, reason="", kind="", stop=0, ys=None, xl=None, xr=None)
        if top >= yref - 2:
            out["reason"] = "no common support"
            return out
        n = c4["corr_n"]
        ys = np.linspace(yref, top, n)
        tt = (ys - yref) / self.H
        xl, xr = np.polyval(L["coef"], tt), np.polyval(R["coef"], tt)
        wd = xr - xl
        w0 = wd[0]
        if w0 < 0.10 * self.w or w0 > 1.6 * self.w:
            out["reason"] = "bottom width %.0f px implausible" % w0
            return out
        small = wd < c4["corr_min_w"]
        big = wd > c4["corr_growth"] * w0 + c4["corr_grow_add"]
        bad = small | big
        stop = int(np.argmax(bad)) if bad.any() else n
        if stop < n:
            # natural convergence never recovers; an invalid section followed by valid rows again is a GAP (non-monotonic)
            out["kind"] = "diverged" if big[stop] else ("gap" if (~bad[stop:]).any() else "converged")
        out.update(stop=stop, ys=ys[:stop], xl=xl[:stop], xr=xr[:stop])
        if stop < c4["corr_min_pts"]:
            out["reason"] = "corridor %s after %d rows" % (out["kind"] or "short", stop)
            return out
        out["valid"] = True
        return out

    def _classify(self, st, yref):
        res = dict(mode=UNCERTAIN, boundaries={}, offset=None, reason="", corridor=None)
        for side in ("L", "R"):
            b = st[side]
            if b is None:
                res["boundaries"][side] = None
                continue
            ys = np.linspace(yref, b["y_top"], 24)
            pts = np.stack([np.polyval(b["coef"], (ys - yref) / self.H), ys], 1)
            res["boundaries"][side] = dict(id=side, type=self._btype(b), source=b["source"], points=pts,
                                           support=dict(span_frac=(yref - b["y_top"]) / self.H, windows=b.get("windows"), contradicted=bool(b.get("contradicted")),
                                                        method=b.get("mode"), age=b["age"], paint_n=b.get("paint_n", 0)),
                                           quality=dict(ridge_frac=b["ridge_frac"], edge_frac=b["edge_frac"],
                                                        note="heuristic score, not a probability"))
        bl, br = res["boundaries"]["L"], res["boundaries"]["R"]
        if bl is None or br is None:
            res["reason"] = "boundary unavailable"
            return res
        tl, tr = bl["type"], br["type"]
        if UNKNOWN in (tl, tr):
            res["reason"] = "boundary type unsupported"
            return res
        if self.c4["contradicted_blocks"] and (st["L"].get("contradicted") or st["R"].get("contradicted")):
            res["reason"] = "boundary contradicted by current measurements (held, not confirmed)"
            return res
        corr = self._corridor(st["L"], st["R"], yref)
        res["corridor"] = corr
        if not corr["valid"]:
            res["reason"] = "corridor invalid: " + corr["reason"]            # HUD and rendering now agree: nothing is filled
            return res
        wd0 = st["R"]["coef"][2] - st["L"]["coef"][2]
        both_obs = bl["source"] == OBSERVED and br["source"] == OBSERVED
        h = self.width_hist
        ref = float(np.median(h[:-30])) if len(h) >= 45 else None
        wide = ref is not None and wd0 > self.c2["wide_ratio"] * ref
        if tl == PAINTED and tr == PAINTED and both_obs:
            self.width_hist = (h + [wd0])[-120:]
        if tl == PAINTED and tr == PAINTED and not wide:
            res["mode"] = CURRENT_LANE
            res["offset"] = (self.w / 2 - (st["L"]["coef"][2] + st["R"]["coef"][2]) / 2) / max(wd0, 1)
        else:
            res["mode"] = ROAD_CORRIDOR
            res["reason"] = "wide (%.2fx lane width ~1-3 s ago)" % (wd0 / ref) if wide else "includes road-edge boundary"
        return res

    # ------------------------------------------------------------ drawing (C4)
    def draw(self, disp, L=None, R=None, fps=None, debug=False):
        s = disp.shape[1] / self.w
        res = self.result
        ov = disp.copy()
        bl, br = res["boundaries"]["L"], res["boundaries"]["R"]
        corr = res.get("corridor")
        cut_y = None
        if res["mode"] != UNCERTAIN and corr is not None and corr["valid"]:
            pl = np.stack([corr["xl"], corr["ys"]], 1) * s
            pr = np.stack([corr["xr"], corr["ys"]], 1) * s
            col = GREEN if res["mode"] == CURRENT_LANE else AMBER
            cv2.fillPoly(ov, [np.concatenate([pl, pr[::-1]]).astype(np.int32)], col)
            disp = cv2.addWeighted(ov, 0.30, disp, 0.70, 0)
            if corr["kind"] in ("diverged", "gap"):
                cut_y = corr["ys"][-1]                           # lines also stop where the pair becomes invalid
                                                                 # (natural convergence at the far end keeps full-length lines)
        for b, colr in ((bl, LEFT_C), (br, RIGHT_C)):
            if b is None or (b["type"] == UNKNOWN and not debug) or (res["mode"] == UNCERTAIN and not debug):
                continue
            pts = b["points"] if cut_y is None else b["points"][b["points"][:, 1] >= cut_y]
            if len(pts) < 2:
                continue
            p = (pts * s).astype(np.int32)
            th = 3 if b["type"] == PAINTED else 2
            if b["source"] == OBSERVED:
                cv2.polylines(disp, [p], False, colr, th, cv2.LINE_AA)
            else:
                for i in range(0, len(p) - 1, 2):
                    cv2.line(disp, tuple(p[i]), tuple(p[i + 1]), colr, th, cv2.LINE_AA)

        def lab(b):
            return "-" if b is None else {PAINTED: "paint", ROAD_EDGE: "edge", UNKNOWN: "unsupported"}[b["type"]] + ("" if b["source"] == OBSERVED else "/trk")
        txt = res["mode"] + ("" if res["offset"] is None else f" off {res['offset']:+.2f}")
        hud = f"L:{lab(bl)} R:{lab(br)}  {txt}"
        if res["reason"] and res["mode"] != CURRENT_LANE:
            hud += f"  ({res['reason']})"
        if fps:
            hud = f"{fps:5.0f} FPS (algo)  " + hud
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(disp, hud, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if debug:
            cv2.polylines(disp, [(self.poly * s).astype(np.int32)], True, (0, 0, 255), 1)
            if self.edges is not None:
                e = cv2.cvtColor(self.edges, cv2.COLOR_GRAY2BGR)
                e = cv2.resize(e, None, fx=s * 0.6, fy=s * 0.6)
                h, w = e.shape[:2]
                disp[34:34 + h, -w - 4:-4] = e
        return disp


# ======================================================================================================
# PART 5 - V6: display policy, partial-lane preservation, one-sided fallback, bounded tracking, green overlay
# ======================================================================================================
CFG6 = dict(
    block_n=3,           # consecutive conflicting measurements before a held boundary is flagged 'contradicted' (1..n-1 = rejected outlier)
    type_up_n=3,         # hysteresis: consecutive frames of better raw evidence before the reported boundary type improves
    type_dn_n=5,         # hysteresis: consecutive frames of worse raw evidence before it degrades
    acq_n=2,             # a NEW track must be observed this many consecutive consistent frames before it is displayed (probation)
    unsup_grace=10,      # frames a boundary that WAS supported (paint/edge) stays displayable while its evidence is 'unsupported'
    flare_abs=6.0,       # corridor 'flare' rule: width may not grow going up by more than flare_abs + flare_rel*(smallest width so far)
    flare_rel=0.12,
    min_extent=0.10,     # the valid corridor must span at least this fraction of the ROI height to be displayed
    ext_grow=4.0,        # px/frame: the displayed far end may move UP the image at most this fast (shrinking is immediate)
    wide_reacq_mult=2,   # a replacement that would make the corridor much wider or narrower (>1.5x) needs this many times more consistent frames
    roi_fallback=True,   # one boundary only: try the ROI edge as a provisional boundary (only if the resulting width is plausible)
    fb_ratio=(0.75, 1.30),  # plausible corridor width relative to the recent lane width (median of up to 120 supported frames)
    fb_abs=(0.20, 0.70),    # ... or, with < fb_min_hist samples, relative to the frame width
    fb_min_hist=45,
    fb_recent=45,        # ROI fallback: a boundary displayed within this many frames must not jump more than fb_jump_px when replaced by the ROI edge
    fb_jump_px=30.0,
    scan_n=128,          # rows used to find the first invalid row of the corridor
    line_bgr=(40, 215, 40),  # displayed boundary colour; the fill uses GREEN.  NEVER changes with state.
    # ---- DISPLAY-ONLY overlay clip (does not touch detection, tracking, result["lane"], pts_L/pts_R or corridor) ----
    # overlay_top_y: image ROW, in PROCESSING-FRAME pixels (the frame passed to process(); row 0 = top, y grows downward).
    #   The green fill and the boundary lines are NOT drawn above this row.  E.g. processing height 360 and overlay_top_y=180
    #   -> nothing green above row 180.  draw() converts it to the display frame with the same scale it uses for the corridor.
    #   0 (or any value <= the corridor's far end) = no clipping = the original V6 overlay.
    #   None = use overlay_top_frac * processing height (resolution independent default).
    overlay_top_y=None,
    overlay_top_frac=0.72,   # default clip row as a fraction of the processing height (ROI spans 0.615 .. 0.985 by default)
)
_RANK = {UNKNOWN: 0, ROAD_EDGE: 1, PAINTED: 2}


def _clip_top(pts, y_clip):
    """DISPLAY-ONLY clip of an upward-running polyline (rows DEcrease along the array, as pts_L / pts_R / partial points do).
    pts: (N,2) float [x, y] in processing px.  Keeps the part at or below row y_clip (y >= y_clip) and replaces the first point
    above it by the linearly interpolated point ON the row y_clip.  Returns a NEW array (input never modified), the SAME object if
    nothing lies above y_clip, or None if nothing is left.  Used for the fill's two sides with the same ys, so both end on one row."""
    y = pts[:, 1]
    if y_clip is None or y.min() >= y_clip:
        return pts
    if y[0] <= y_clip:
        return None
    i = int(np.argmax(y < y_clip))                      # first point strictly above the clip row (i >= 1 here)
    a, b = pts[i - 1], pts[i]
    t = (a[1] - y_clip) / (a[1] - b[1])
    return np.vstack([pts[:i], [a[0] + t * (b[0] - a[0]), y_clip]])


class PathDetectorV6(_DetectorV4):
    """Same measurement pipeline as the old versions (bit-identical edges/segments/fits/tracking), different DECISION layer.

    result["lane"] is the single source of truth for the overlay, the HUD and any downstream consumer:
      state     LANE | ONE_SIDED | NONE
      extent_y  row (processing px) where the displayed corridor ends; fill, both lines and HUD all use it
      L, R      per boundary: source (OBSERVED / TRACKED / INFERRED), type, usable, conf, age, ...
      pts_L/pts_R/corridor   the exact polylines that are drawn
    """

    def __init__(self, w, h, cfg=None, cfg2=None, cfg3=None, cfg4=None, cfg6=None):
        super().__init__(w, h, cfg, cfg2, cfg3, cfg4)
        self.c6 = dict(CFG6, **(cfg6 or {}))
        self.fn = 0                 # frame counter
        self._ext_prev = None       # displayed far-end row of the previous frame (rate limiting)
        self._last_disp = {}        # side -> (x at ROI bottom, frame) of the last displayed observed/tracked boundary

    # ------------------------------------------------------------------ tracking: outlier vs genuine change
    def _widens(self, side, meas, yref):
        other = self.state["R" if side == "L" else "L"]
        h = self.width_hist
        if other is None or len(h) < self.c6["fb_min_hist"]:
            return False
        ref = float(np.median(h))
        w = abs(meas["coef"][2] - other["coef"][2])
        return w > self.c2["wide_ratio"] * ref or w < ref / self.c2["wide_ratio"]     # much wider (neighbour lane / divider) OR much narrower (road symbol / vehicle edge)

    def _track(self, side, meas, yref):
        n0 = self.c2["reacquire_n"]
        if meas is not None and self._widens(side, meas, yref):      # a sudden jump to a much wider / narrower corridor (neighbour lane, divider, road symbol?)
            self.c2["reacquire_n"] = n0 * self.c6["wide_reacq_mult"]  # needs longer agreement before it replaces the track
        prev = self.state[side]
        hold0, blend0, rep0 = self.stats[side + ":conflict_hold"], self.stats[side + ":blend"], self.stats[side + ":replace"]
        try:
            st = super()._track(side, meas, yref)
        finally:
            self.c2["reacquire_n"] = n0
        if st is None:
            return None
        if st is not prev:                                            # new track (after expiry / first sight) or replacement
            st["obs_n"] = n0 if self.stats[side + ":replace"] > rep0 else 1   # a replacement was already confirmed by reacquire_n frames
        elif self.stats[side + ":blend"] > blend0:
            st["obs_n"] = min(st.get("obs_n", 1) + 1, 9)
        if self.stats[side + ":conflict_hold"] > hold0:               # measurement rejected, old curve kept
            st["conf_run"] = st.get("conf_run", 0) + 1
            st["contradicted"] = st["conf_run"] >= self.c6["block_n"]
            st["outlier"] = not st["contradicted"]
        elif self.stats[side + ":blend"] > blend0:
            st["conf_run"], st["outlier"] = 0, False
        return st

    # ------------------------------------------------------------------ boundary type with hysteresis
    def _btype(self, b):
        raw = super()._btype(b)
        if b is None:
            return UNKNOWN
        c6 = self.c6
        cur = b.get("btype")
        if cur is None:
            b["btype"], b["bt_up"], b["bt_dn"] = raw, 0, 0
        elif _RANK[raw] > _RANK[cur]:
            b["bt_up"], b["bt_dn"] = b.get("bt_up", 0) + 1, 0
            if b["bt_up"] >= c6["type_up_n"]:
                b["btype"], b["bt_up"] = raw, 0
        elif _RANK[raw] < _RANK[cur]:
            b["bt_dn"], b["bt_up"] = b.get("bt_dn", 0) + 1, 0
            if b["bt_dn"] >= c6["type_dn_n"]:
                b["btype"], b["bt_dn"] = raw, 0
        else:
            b["bt_up"] = b["bt_dn"] = 0
        b["bt_held"] = b["btype"] != raw
        if b["btype"] != UNKNOWN:
            b["last_sup"] = self.fn
        return b["btype"]

    # ------------------------------------------------------------------ geometry
    def _corridor(self, L, R, yref):
        """First invalid row ends the corridor; everything before it is kept.  kind: converged | crossed | diverged | ''."""
        c4, c6 = self.c4, self.c6
        top = max(L["y_top"], R["y_top"], self.y0)
        out = dict(valid=False, reason="", kind="", stop=0, ys=None, xl=None, xr=None, extent=0.0)
        if top >= yref - 2:
            out["reason"] = "no common support"
            return out
        n = max(c4["corr_n"], c6["scan_n"])                 # fine scan: the clip row is found to ~1 px, not to the 4-5 px of a 30-row grid
        ys = np.linspace(yref, top, n)
        tt = (ys - yref) / self.H
        xl, xr = np.polyval(L["coef"], tt), np.polyval(R["coef"], tt)
        wd = xr - xl
        w0 = wd[0]
        if w0 < 0.10 * self.w or w0 > 1.6 * self.w:
            out["reason"] = "bottom width %.0f px implausible" % w0
            return out
        small = wd < c4["corr_min_w"]
        big = wd > c4["corr_growth"] * w0 + c4["corr_grow_add"]
        rm = np.minimum.accumulate(wd)
        flare = wd > rm + c6["flare_abs"] + c6["flare_rel"] * np.maximum(rm, 0)
        bad = small | big | flare
        stop = int(np.argmax(bad)) if bad.any() else n
        if stop < n:
            out["kind"] = "diverged" if (big[stop] or flare[stop]) else ("crossed" if wd[stop] < 0 else "converged")
        ext = float(ys[stop - 1]) if stop > 0 else float(yref)
        out.update(stop=stop, ys=ys[:stop], xl=xl[:stop], xr=xr[:stop], extent=(yref - ext) / self.H, ext_y=ext)
        if out["extent"] < c6["min_extent"]:
            out["reason"] = "valid part too short (%s after %.0f%% of ROI)" % (out["kind"] or "short", 100 * out["extent"])
            return out
        out["valid"] = True
        return out

    def _roi_edge(self, side, yref, y_top):
        p = self.poly                                   # bottom-left, top-left, top-right, bottom-right
        (xb, yb), (xt, yt) = (p[0], p[1]) if side == "L" else (p[3], p[2])
        slope = float(xt - xb) / float(yt - yb)         # dx/dy
        return dict(coef=(0.0, slope * self.H, float(xb) + slope * (yref - yb)), y_top=float(y_top), source=INFERRED)

    def _fallback(self, have, yref):
        """Provisional opposite boundary from the ROI edge.  Returns (virtual boundary | None, info)."""
        c6 = self.c6
        miss = "R" if have == "L" else "L"
        info = dict(tried=True, side=miss, accepted=False, why="")
        D = self.state[have]
        V = self._roi_edge(miss, yref, D["y_top"])
        w0 = abs(V["coef"][2] - D["coef"][2])
        h = self.width_hist
        if len(h) >= c6["fb_min_hist"]:
            ref = float(np.median(h)); ratio = w0 / ref; lo, hi = c6["fb_ratio"]
            info["ratio"] = ratio
            if not lo <= ratio <= hi:
                info["why"] = "ROI edge would give %.2fx the recent lane width" % ratio
                return None, info
        else:
            lo, hi = c6["fb_abs"]; fr = w0 / self.w; info["ratio"] = fr
            if not lo <= fr <= hi:
                info["why"] = "ROI edge gives %.0f%% of frame width (no lane-width history)" % (100 * fr)
                return None, info
        L, R = (D, V) if have == "L" else (V, D)
        if not self._corridor(L, R, yref)["valid"]:
            info["why"] = "ROI-edge corridor geometry invalid"
            return None, info
        ld = self._last_disp.get(miss)
        if ld is not None and self.fn - ld[1] <= c6["fb_recent"] and abs(V["coef"][2] - ld[0]) > c6["fb_jump_px"]:
            info["why"] = "ROI edge would move the %s boundary %.0f px from where it was last displayed" % (miss, abs(V["coef"][2] - ld[0]))
            return None, info
        info["accepted"] = True
        return V, info

    # ------------------------------------------------------------------ decision layer
    def _usable(self, b, typ):
        if b is None or b.get("obs_n", 1) < self.c6["acq_n"]:         # unproven new track: not displayed yet
            return False
        if typ != UNKNOWN:
            return True
        ls = b.get("last_sup")
        return ls is not None and self.fn - ls <= self.c6["unsup_grace"]       # bounded: never forever

    def _describe(self, b, typ, usable):
        mx = self.c2["max_hold2"] + 1
        return dict(source=b["source"], type=typ, usable=bool(usable), age=int(b["age"]), outlier=bool(b.get("outlier", False)),
                    contradicted=bool(b.get("contradicted", False)), type_held=bool(b.get("bt_held", False)),
                    conf=round(max(0.0, 1.0 - b["age"] / mx) * (1.0 if typ != UNKNOWN else 0.5), 3), coef=tuple(b["coef"]), y_top=float(b["y_top"]))

    def _classify(self, st, yref):
        self.fn += 1
        res = dict(mode=UNCERTAIN, boundaries={}, offset=None, reason="", corridor=None, extent_y=None, partial={})
        typ, use, B = {}, {}, {}
        for side in ("L", "R"):
            b = st[side]
            typ[side] = self._btype(b) if b is not None else UNKNOWN
            use[side] = self._usable(b, typ[side])
            if b is not None:
                ys = np.linspace(yref, b["y_top"], 24)
                B[side] = dict(id=side, type=typ[side], source=b["source"], points=np.stack([np.polyval(b["coef"], (ys - yref) / self.H), ys], 1),
                               support=dict(span_frac=(yref - b["y_top"]) / self.H, contradicted=bool(b.get("contradicted")), age=int(b["age"]),
                                            paint_n=b.get("paint_n", 0)))
            else:
                B[side] = None
        res["boundaries"] = B
        lane = dict(state="NONE", extent_y=None, ext_valid=None, L=None, R=None, pts_L=None, pts_R=None, corridor=None, note="", fb=dict(tried=False))
        for side in ("L", "R"):
            if st[side] is not None:
                lane[side] = self._describe(st[side], typ[side], use[side])
        Lb, Rb = st["L"], st["R"]
        corr, fbinfo, virt_side = None, None, None
        if use["L"] and use["R"]:
            c = self._corridor(Lb, Rb, yref)
            if c["valid"]:
                corr = c
            else:
                lane["note"] = "pair rejected: " + c["reason"]
        elif use["L"] or use["R"]:
            have = "L" if use["L"] else "R"
            miss = "R" if have == "L" else "L"
            lane["note"] = "%s boundary unavailable" % miss
            if self.c6["roi_fallback"]:
                V, fbinfo = self._fallback(have, yref)
                lane["fb"] = fbinfo
                if V is not None:
                    virt_side = miss
                    corr = self._corridor(*((Lb, V) if have == "L" else (V, Rb)), yref)
                    lane["note"] = "%s inferred from ROI edge" % miss
                    lane[miss] = dict(source=INFERRED, type=EST, usable=True, age=0, outlier=False, contradicted=False, type_held=False,
                                      conf=0.4, coef=tuple(V["coef"]), y_top=float(V["y_top"]))
                else:
                    lane["note"] += "; ROI fallback rejected (%s)" % fbinfo["why"]
            else:
                lane["note"] += "; ROI fallback disabled"
        if corr is not None:
            ext_v = float(corr["ext_y"])
            ext = ext_v if self._ext_prev is None else max(ext_v, self._ext_prev - self.c6["ext_grow"])   # never above the valid extent
            self._ext_prev = ext
            ys = np.linspace(yref, ext, self.c4["corr_n"])
            tt = (ys - yref) / self.H
            Lc = Lb if virt_side != "L" else V
            Rc = Rb if virt_side != "R" else V
            xl, xr = np.polyval(Lc["coef"], tt), np.polyval(Rc["coef"], tt)
            lane.update(state="LANE", extent_y=ext, ext_valid=ext_v, corridor=dict(ys=ys, xl=xl, xr=xr),
                        pts_L=np.stack([xl, ys], 1), pts_R=np.stack([xr, ys], 1), kind=corr["kind"])
            for sd, xx in ("L", xl[0]), ("R", xr[0]):
                if sd != virt_side:
                    self._last_disp[sd] = (float(xx), self.fn)
            res["extent_y"] = ext
            res["corridor"] = dict(valid=True, kind=corr["kind"], stop=corr["stop"], ys=ys, xl=xl, xr=xr, reason="")
            # --- classification kept for diagnostics only: it NEVER changes the displayed colour
            wd0 = float(xr[0] - xl[0])
            h = self.width_hist
            ref = float(np.median(h)) if len(h) >= self.c6["fb_min_hist"] else None
            both_obs = virt_side is None and Lb["source"] == OBSERVED and Rb["source"] == OBSERVED and typ["L"] != UNKNOWN and typ["R"] != UNKNOWN
            if both_obs:
                self.width_hist = (h + [wd0])[-120:]
            wide = ref is not None and wd0 > self.c2["wide_ratio"] * ref
            if typ["L"] == PAINTED and typ["R"] == PAINTED and virt_side is None and not wide:
                res["mode"] = CURRENT_LANE
                res["offset"] = (self.w / 2 - (xl[0] + xr[0]) / 2) / max(wd0, 1)
            else:
                res["mode"] = ROAD_CORRIDOR
            lane["wide"] = bool(wide)
        else:
            self._ext_prev = None
            side = "L" if use["L"] else ("R" if use["R"] else None)
            if side is not None and not (use["L"] and use["R"]):
                lane["state"] = "ONE_SIDED"
                res["partial"] = {side: B[side]["points"]}
            elif use["L"] and use["R"]:
                lane["state"] = "NONE"
            res["reason"] = lane["note"]
        res["lane"] = lane
        return res

    # ------------------------------------------------------------------ drawing: always green
    def draw(self, disp, L=None, R=None, fps=None, debug=False):
        s = disp.shape[1] / self.w                      # processing -> display scale (same one used for every overlay coordinate)
        lane = self.result["lane"]
        c6 = self.c6
        lc = c6["line_bgr"]
        yc = c6["overlay_top_y"]                        # display-only clip row, PROCESSING-frame px (clipped BEFORE scaling by s)
        yc = c6["overlay_top_frac"] * self.h if yc is None else float(yc)
        if lane["state"] == "LANE":
            cr = lane["corridor"]
            pl, pr = _clip_top(np.stack([cr["xl"], cr["ys"]], 1), yc), _clip_top(np.stack([cr["xr"], cr["ys"]], 1), yc)
            if pl is not None and pr is not None:       # both sides share the same ys -> identical cut row, flat top edge
                ov = disp.copy()
                cv2.fillPoly(ov, [np.concatenate([pl * s, (pr * s)[::-1]]).astype(np.int32)], GREEN)
                disp = cv2.addWeighted(ov, 0.30, disp, 0.70, 0)

        def line(pts, source, th):
            pts = _clip_top(pts, yc)                    # same visible extent as the fill; lane["pts_*"] itself is not modified
            if pts is None:
                return
            p = (pts * s).astype(np.int32)
            if source == OBSERVED:
                cv2.polylines(disp, [p], False, lc, th, cv2.LINE_AA)
            elif source == TRACKED:                                   # retained from earlier frames: dashed
                for i in range(0, len(p) - 1, 2):
                    cv2.line(disp, tuple(p[i]), tuple(p[i + 1]), lc, th, cv2.LINE_AA)
            else:                                                     # inferred (ROI edge): dotted
                for i in range(0, len(p) - 1, 3):
                    cv2.line(disp, tuple(p[i]), tuple(p[min(i + 1, len(p) - 1)]), lc, th, cv2.LINE_AA)

        if lane["state"] == "LANE":
            line(lane["pts_L"], lane["L"]["source"], 3)
            line(lane["pts_R"], lane["R"]["source"], 3)
        elif lane["state"] == "ONE_SIDED":
            for side, pts in self.result["partial"].items():
                line(pts, lane[side]["source"], 2)

        def lab(side):
            d = lane[side]
            if d is None:
                return "-"
            t = {PAINTED: "paint", ROAD_EDGE: "edge", UNKNOWN: "weak", EST: "roi"}[d["type"]]
            return t + {OBSERVED: "", TRACKED: "/trk", INFERRED: "/inf"}[d["source"]] + ("~" if d["outlier"] else "")
        if lane["state"] == "LANE":
            hud = "LANE [to %.0f%% of ROI] L:%s R:%s %s" % (100 * (self.y1 - 1 - lane["extent_y"]) / self.H, lab("L"), lab("R"), self.result["mode"])
            if lane.get("note"):
                hud += "  (" + lane["note"] + ")"
        elif lane["state"] == "ONE_SIDED":
            hud = "ONE BOUNDARY ONLY L:%s R:%s  (%s)" % (lab("L"), lab("R"), lane["note"])
        else:
            hud = "NO LANE  L:%s R:%s  (%s)" % (lab("L"), lab("R"), lane["note"] or "no boundary evidence")
        if fps:
            hud = "%5.0f FPS (algo)  " % fps + hud
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(disp, hud, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if debug:
            cv2.polylines(disp, [(self.poly * s).astype(np.int32)], True, (0, 0, 255), 1)
            if self.edges is not None:
                e = cv2.cvtColor(self.edges, cv2.COLOR_GRAY2BGR)
                e = cv2.resize(e, None, fx=s * 0.6, fy=s * 0.6)
                hh, ww = e.shape[:2]
                disp[34:34 + hh, -ww - 4:-4] = e
        return disp


# ======================================================================================================
# command line
# ======================================================================================================
def main():
    ap = argparse.ArgumentParser(description="Lane Path Detector v6 (standalone)")
    ap.add_argument("src", nargs="?", default=None, help="input video path")
    ap.add_argument("dst", nargs="?", default=None, help="output video path")
    ap.add_argument("--source", dest="source_opt", default=None)
    ap.add_argument("--save", dest="save_opt", default=None)
    ap.add_argument("--show", action="store_true", help="open a live window (press q to quit)")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--out-w", type=int, default=1280)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--profile", choices=list(PROFILES), default="mountain")
    ap.add_argument("--top-y", type=float)
    ap.add_argument("--top-x", type=float, nargs=2)
    ap.add_argument("--overlay-top-y", type=float, default=None,
                    help="display-only: do not draw the green overlay above this row (PROCESSING-frame px, row 0 = top; 0 = no clipping)")
    ap.add_argument("--overlay-top-frac", type=float, default=None,
                    help="same, as a fraction of the processing height (used when --overlay-top-y is not given; default %.2f)" % CFG6["overlay_top_frac"])
    a = ap.parse_args()
    src = a.src or a.source_opt
    if not src:
        ap.error("an input video is required (e.g. lane_path_v6.py video.mp4 output.mp4)")
    dst = a.dst or a.save_opt
    if not dst and not a.show:
        dst = "output_v6.mp4"
    cfg = dict(CFG)
    cfg.update(PROFILES[a.profile])
    if a.top_y is not None:
        cfg["top_y"] = a.top_y
    if a.top_x:
        cfg["top_x"] = tuple(a.top_x)
    cap = cv2.VideoCapture(src)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    if a.start:
        cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
    n_max = int(a.duration * fps_in) if a.duration else 10 ** 9
    ok, f = cap.read()
    if not ok:
        raise SystemExit("cannot read video: %s" % src)
    sh, sw = f.shape[:2]
    pw = cfg["proc_w"]
    ph = int(round(sh * pw / sw))
    ow = a.out_w
    oh = int(round(sh * ow / sw))
    cfg6 = {}
    if a.overlay_top_y is not None:
        cfg6["overlay_top_y"] = a.overlay_top_y
    if a.overlay_top_frac is not None:
        cfg6["overlay_top_frac"] = a.overlay_top_frac
    det = PathDetectorV6(pw, ph, cfg, cfg6=cfg6)
    print("processing %dx%d  display %dx%d  overlay clip row = %.1f (processing px) -> %.1f (display px)" % (
        pw, ph, ow, oh, det.c6["overlay_top_y"] if det.c6["overlay_top_y"] is not None else det.c6["overlay_top_frac"] * ph,
        (det.c6["overlay_top_y"] if det.c6["overlay_top_y"] is not None else det.c6["overlay_top_frac"] * ph) * ow / pw))
    tmp = (dst + ".tmp.mp4") if dst else None
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (ow, oh)) if tmp else None
    times, n = [], 0
    while ok and n < n_max:
        small = cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        L, R = det.process(small)
        times.append(time.perf_counter() - t0)
        disp = cv2.resize(f, (ow, oh), interpolation=cv2.INTER_AREA)
        fps = 1.0 / np.mean(times[-30:])
        out_frame = det.draw(disp, L, R, fps, a.debug)
        if vw:
            vw.write(out_frame)
        if a.show:
            cv2.imshow("lane_path_v6 (q to quit)", out_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        n += 1
        ok, f = cap.read()
    if vw:
        vw.release()
    cap.release()
    if a.show:
        cv2.destroyAllWindows()
    if dst and tmp:
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", dst])
            os.remove(tmp)
        else:
            os.replace(tmp, dst)
        print("saved output to %s" % dst)
    t = np.array(times[5:] or times)
    if len(t) > 0:
        print("frames=%d  algo-only: mean %.0f FPS, p95 latency %.1f ms, worst %.1f ms  (processing size %dx%d)"
              % (n, 1 / t.mean(), np.percentile(t, 95) * 1000, t.max() * 1000, pw, ph))


if __name__ == "__main__":
    main()
