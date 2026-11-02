#!/usr/bin/env python3
"""
vehicle_refine.py -- full-resolution, model-free refinement + verification of rear-view vehicle proposals.

Input : the ORIGINAL camera frame (BGR, source pixels) and a coarse proposal box (also source pixels) from the stage-1 detector,
        plus the expected lane width (source px) at the proposal's row.
Output: a RefinedVehicle (all coordinates SOURCE pixels, sub-pixel floats) or None if the evidence for a vehicle is too weak.

What it measures (no trained model; only gradients, thresholds relative to the local road, and geometry):
  1. Local road luminance from the road strip below the proposal (median of the lowest rows of the crop).
  2. UNDER-BODY EDGE PROFILE: per image column, the sub-pixel row where dark (shadow/tyre/bumper underside) turns into road.
     The tyre-road contact is the lower end of the dark mass, so tyre contacts are read from this profile.
  3. TYRE CONTACTS: left/right lower corners of the dark mass; each tyre's x is the centre of the dark run just above the contact row
     (resolved tyre) or the outer corner + half an assumed tyre width (tyre merged with the shadow -> flagged).
     Both -> axle ground midpoint; one -> that point, reduced confidence; none reliable -> UNCERTAIN (no precise point is invented).
  4. LATERAL EDGES from non-road columns over the lower body (run-length filtered so lane paint / neighbours do not extend the box).
  5. TOP EDGE from the strongest horizontal edge above the body inside plausible height/width bounds, else estimated (flagged).
  6. EVIDENCE: dark-bottom coverage + sharpness, left-right symmetry, side-edge strength, plausible width -> accept/reject.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import cv2
import numpy as np

RCFG = dict(
    work_w=200,            # crop is processed at <= this vehicle width (px); wider vehicles are down-scaled for body analysis
    margin_x=0.30, margin_top=0.40, margin_bot=0.45,       # crop margins (fractions of proposal width)
    road_rows=0.14,        # lowest fraction of the crop used for the road reference
    dark_abs=14.0, dark_rel=0.28,   # dark = G < Lroad - max(dark_abs, dark_rel*Lroad)
    edge_win_up=0.30, edge_win_dn=0.30,   # search window for the under-body edge around the proposal bottom (x proposal width)
    edge_min_contrast=10.0, # min dark->road step (grey levels) for a valid edge column
    passes=1, use_thick=False, grow_frac=0.08, thick_frac=0.6,       # a column belongs to the dark mass when its dark thickness >= this x the 90th-percentile thickness
    tyre_zone=0.30,        # tyre search zone at each end of the dark mass (fraction of mass width)
    tyre_w=0.13,           # assumed tyre width / mass width when the tyre is merged with the shadow (only for the MERGED case)
    body_rows=(0.08, 0.55),# rows (fractions of width above contact) used for lateral extents
    run_min=0.25,          # a lateral run must be at least this fraction of width
    sym_min=0.15, edge2_min=1.5, evid_min=0.45,   # hard gates, set just under the 10th percentile of the annotated vehicles (n=8)
    top_est_ratio=0.84, top_win=0.15, top_min_grad=14.0,   # rear-view height/width ~0.84 (annotated: 0.76-0.96); edge searched +-0.15*W around it
)

@dataclass
class RefinedVehicle:
    box: Tuple[float, float, float, float]           # x0,y0,x1,y1 source px (sub-pixel)
    top_estimated: bool
    contact: Optional[Tuple[float, float]]           # axle ground point; None when UNCERTAIN
    contact_hint: Tuple[float, float]                # coarse point (always set; NOT precise when quality==UNCERTAIN)
    quality: str                                     # TWO_TYRES | ONE_TYRE | UNCERTAIN
    detail: str                                      # RESOLVED | MERGED | SINGLE_SIDE | NO_EDGE ...
    tyres: List[Tuple[float, float]]                 # individual tyre/ corner contact points found (0..2)
    contact_sigma: float                             # rough 1-sigma pixel uncertainty claimed for `contact`
    evidence: float                                  # 0..1 (heuristic, not a probability)
    parts: dict = field(default_factory=dict)


def _subpix(y, v):
    """parabolic refinement of a peak at integer index y of the 1-D array v"""
    if y <= 0 or y >= len(v) - 1: return float(y)
    a, b, c = v[y - 1], v[y], v[y + 1]
    d = a - 2 * b + c
    return float(y) if abs(d) < 1e-6 else float(y + 0.5 * (a - c) / d)


class RearRefiner:
    def __init__(self, cfg=None):
        self.c = dict(RCFG, **(cfg or {}))

    # ------------------------------------------------------------------
    def refine(self, F, prior, lane_w=None, debug=None):
        """Returns a LIST of RefinedVehicle (0, 1 or several: a proposal that spans two separate dark under-body masses is split)."""
        self.dbg = {}
        r = self._core(F, prior, lane_w, 0)
        if r is None: return []
        r = r if isinstance(r, list) else [r]
        if self.c["passes"] >= 2:                         # second pass: re-run on the tight box so windows/zones are anchored to the vehicle
            out = []
            for v in r:
                b = v.box
                w_ = b[2] - b[0]
                r2 = self._core(F, (b[0], b[1], b[2], b[3]), lane_w, 1)
                out.append(r2 if (r2 is not None and not isinstance(r2, list)) else v)
            r = out
        return r

    def _core(self, F, prior, lane_w, depth):
        c = self.c
        H, Wf = F.shape[:2]
        X0, Y0, X1, Y1 = [float(v) for v in prior]
        W0 = max(X1 - X0, 8.0)
        cx0 = int(max(X0 - c["margin_x"] * W0, 0)); cx1 = int(min(X1 + c["margin_x"] * W0, Wf))
        cy0 = int(max(Y0 - c["margin_top"] * W0, 0)); cy1 = int(min(Y1 + c["margin_bot"] * W0, H))
        if cx1 - cx0 < 16 or cy1 - cy0 < 16: return None
        f = min(1.0, c["work_w"] / W0)
        patch = cv2.cvtColor(F[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2GRAY)        # gray first: the resize below is then 1 channel
        if f < 1.0:
            patch = cv2.resize(patch, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        G = cv2.GaussianBlur(patch.astype(np.float32), (0, 0), 0.8)
        h, w = G.shape
        # prior in patch coords (working scale)
        px0, px1 = (X0 - cx0) * f, (X1 - cx0) * f
        py0, py1 = (Y0 - cy0) * f, (Y1 - cy0) * f
        Wp = px1 - px0
        # ---- 1. road reference: lowest rows of the crop, columns outside the central proposal span + centre (median is robust)
        r0 = int(h * (1 - c["road_rows"]))
        Lr = float(np.median(G[r0:, :]))
        thr_d = max(c["dark_abs"], c["dark_rel"] * Lr)
        # ---- 2. under-body edge profile in a window around the proposal bottom
        ya = int(max(py1 - c["edge_win_up"] * Wp, 4)); yb = int(min(py1 + c["edge_win_dn"] * Wp, h - 4))
        if yb - ya < 8: return None
        xa = int(max(px0 - 0.10 * Wp, 0)); xb = int(min(px1 + 0.10 * Wp, w))
        ys = np.arange(ya, yb)
        up = G[ys - 3, xa:xb]; dn = G[ys + 3, xa:xb]           # intensity 3 rows above / below
        step = dn - up                                           # >0 when going dark -> bright (downwards)
        dark_above = G[ys - 3, xa:xb] < (Lr - thr_d)
        road_below = np.abs(G[ys + 3, xa:xb] - Lr) < 0.6 * thr_d + 6
        S = np.where(dark_above & road_below, step, 0.0)        # bottom-edge response
        arg = S.argmax(axis=0)
        val = S[arg, np.arange(S.shape[1])]
        valid = val >= c["edge_min_contrast"]
        if valid.sum() < 0.25 * Wp: return None
        # vectorised parabolic sub-pixel refinement of the edge row of every column
        nC = S.shape[1]; cols = np.arange(nC)
        am1 = np.clip(arg - 1, 0, S.shape[0] - 1); ap1 = np.clip(arg + 1, 0, S.shape[0] - 1)
        a_, b_, c_ = S[am1, cols], S[arg, cols], S[ap1, cols]
        dd = a_ - 2 * b_ + c_
        off = np.where(np.abs(dd) < 1e-6, 0.0, 0.5 * (a_ - c_) / np.where(np.abs(dd) < 1e-6, 1.0, dd))
        off = np.clip(off, -1.0, 1.0) * ((arg > 0) & (arg < S.shape[0] - 1))
        yedge = np.where(valid, ya + arg + off, np.nan).astype(np.float32)
        # keep the largest contiguous valid segment that overlaps the proposal centre
        vm = valid.astype(np.int8); d = np.diff(np.concatenate([[0], vm, [0]]))
        st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        best, bl = None, 0
        for a_, b_ in zip(st, en):
            gap = 0
            if b_ - a_ > bl: best, bl = (a_, b_), b_ - a_
        # merge small gaps (<= 3 columns)
        segs = [list(p) for p in zip(st, en)]
        merged = []
        for sgm in segs:
            if merged and sgm[0] - merged[-1][1] <= max(3, int(0.03 * Wp)): merged[-1][1] = sgm[1]
            else: merged.append(sgm)
        min_w = max(0.15 * Wp, (0.18 * lane_w * f) if lane_w else 0.0, 6.0)
        sig = [m for m in merged if (m[1] - m[0]) >= min_w]
        if depth == 0 and len(sig) >= 2:
            outs = []
            for m in sig:
                sub = (cx0 + (xa + m[0]) / f - 0.03 * Wp / f, Y0, cx0 + (xa + m[1]) / f + 0.03 * Wp / f, Y1)
                r = self._core(F, sub, lane_w, 1)
                if r is not None: outs.append(r)
            return outs if outs else None
        cen = 0.5 * (px0 + px1) - xa
        cand = [m for m in sig if m[0] - 0.1 * Wp <= cen <= m[1] + 0.1 * Wp] or sig
        if not cand: return None
        seg = max(cand, key=lambda m: m[1] - m[0])
        sl, sr = seg
        if sr - sl < 0.35 * Wp: return None
        ye = yedge[sl:sr].copy()
        nn = np.isfinite(ye)
        if nn.sum() < 4: return None
        ye = np.interp(np.arange(len(ye)), np.nonzero(nn)[0], ye[nn]).astype(np.float32)   # bridge the small gaps
        # median-filter the profile along x
        yf = cv2.medianBlur(ye.reshape(1, -1), 5).ravel().astype(np.float32)
        gx = np.arange(sl, sr) + xa
        mass_l, mass_r = float(gx[0]), float(gx[-1] + 1)
        Wm = mass_r - mass_l
        # ---- 2b. refine the mass by DARK THICKNESS above the contact row: under-body + tyres are thick, a cast shadow beside the car
        #          is thin (trimmed); lit tyre corners that failed the edge test are re-attached (grown).
        yc_m = int(round(float(np.median(yf))))
        top_r = max(yc_m - max(4, int(0.14 * Wm)), 0)
        Dk = (G[top_r:min(yc_m + 3, h), :] < (Lr - thr_d))
        thick = Dk.sum(axis=0).astype(np.float32)
        lo_x, hi_x = int(max(mass_l - c["grow_frac"] * Wm, 0)), int(min(mass_r + c["grow_frac"] * Wm, w))
        inside = thick[int(mass_l):int(mass_r)]
        tmax = float(np.percentile(inside, 90)) if inside.size else 0.0
        if c["use_thick"] and tmax >= 3:
            good = thick >= max(2.5, c["thick_frac"] * tmax)
            good[:lo_x] = False; good[hi_x:] = False
            cx_ = int(0.5 * (mass_l + mass_r))
            if good[cx_] or good[max(cx_ - 3, 0):cx_ + 4].any():
                l_ = cx_; r_ = cx_
                gap = max(2, int(0.05 * Wm))
                miss = 0
                while l_ - 1 >= lo_x and miss <= gap:
                    l_ -= 1; miss = 0 if good[l_] else miss + 1
                while r_ + 1 < hi_x and miss <= gap:
                    r_ += 1; miss = 0 if good[r_] else miss + 1
                # (the loops above reuse `miss`; recompute both ends cleanly)
                l_ = cx_; miss = 0; best_l = cx_
                while l_ - 1 >= lo_x and miss <= gap:
                    l_ -= 1
                    if good[l_]: best_l = l_; miss = 0
                    else: miss += 1
                r_ = cx_; miss = 0; best_r = cx_
                while r_ + 1 < hi_x and miss <= gap:
                    r_ += 1
                    if good[r_]: best_r = r_; miss = 0
                    else: miss += 1
                if best_r - best_l >= 0.5 * Wm:
                    nl, nr = float(best_l), float(best_r + 1)
                    # extend the edge profile by replicating its end values over the grown columns
                    pad_l = max(0, int(round(mass_l - nl))); pad_r = max(0, int(round(nr - mass_r)))
                    if pad_l or pad_r:
                        yf = np.concatenate([np.full(pad_l, yf[0], np.float32), yf, np.full(pad_r, yf[-1], np.float32)])
                    cut_l = max(0, int(round(nl - mass_l))); cut_r = max(0, int(round(mass_r - nr)))
                    yf = yf[cut_l:len(yf) - cut_r] if (cut_l or cut_r) else yf
                    mass_l, mass_r = nl, nr
                    Wm = mass_r - mass_l
                    gx = (np.arange(len(yf)) + int(mass_l)).astype(np.int64)
        # ---- 3. tyre contacts at the two ends of the dark mass
        tz = max(3, int(c["tyre_zone"] * Wm))
        def side(sl_slice, outer_is_left):
            yy = yf[sl_slice]; xx = gx[sl_slice].astype(np.float32)
            if len(yy) < 3: return None
            # contact row: lowest (largest y) robust value => 85th percentile of the profile in the tyre zone
            yc = float(np.percentile(yy, 85))
            # x: centre of the dark run in the band just above the contact row
            band0, band1 = int(max(yc - 0.06 * Wm, 1)), int(max(yc - 0.015 * Wm, 2))
            seg_dark = (G[band0:band1 + 1, int(xx[0]):int(xx[-1]) + 1].mean(axis=0) < (Lr - thr_d))
            return yc, seg_dark, xx
        L = side(slice(0, tz), True); R = side(slice(len(yf) - tz, len(yf)), False)
        tyres = []; det = []
        for nm, s_ in (("L", L), ("R", R)):
            if s_ is None: continue
            yc, dk, xx = s_
            if dk.sum() < 2: continue
            idx = np.nonzero(dk)[0]
            if nm == "L":
                # first contiguous dark run starting at the outer end
                run_end = idx[0]
                while run_end + 1 < len(dk) and dk[run_end + 1]: run_end += 1
                a_, b_ = idx[0], run_end
            else:
                run_start = idx[-1]
                while run_start - 1 >= 0 and dk[run_start - 1]: run_start -= 1
                a_, b_ = run_start, idx[-1]
            rw = (b_ - a_ + 1)
            resolved = (rw <= 0.20 * Wm) and (rw < len(dk) - 1)   # run ends inside the zone: a lighter gap follows -> separate tyre
            if resolved:
                x = float(xx[0] + 0.5 * (a_ + b_) + 0.5)
            else:
                half = 0.5 * c["tyre_w"] * Wm
                x = (mass_l + half) if nm == "L" else (mass_r - half)
            tyres.append((nm, x, yc, resolved)); det.append("R" if resolved else "M")
        # ---- 4. lateral extents of the body
        yc_all = max([t[2] for t in tyres], default=float(np.median(yf)))
        rb0 = int(max(yc_all - c["body_rows"][1] * Wm, 0)); rb1 = int(max(yc_all - c["body_rows"][0] * Wm, rb0 + 2))
        band = G[rb0:rb1, :]
        nonroad = (np.abs(band - Lr) > max(18.0, 0.22 * Lr)).astype(np.float32)
        colfrac = nonroad.mean(axis=0)
        q = colfrac >= 0.35
        qd = np.diff(np.concatenate([[0], q.astype(np.int8), [0]])); s2, e2 = np.nonzero(qd == 1)[0], np.nonzero(qd == -1)[0]
        runs = [[a_, b_] for a_, b_ in zip(s2, e2)]
        mrg = []
        for r_ in runs:
            if mrg and r_[0] - mrg[-1][1] <= max(2, int(0.04 * Wm)): mrg[-1][1] = r_[1]
            else: mrg.append(r_)
        mrg = [r_ for r_ in mrg if (r_[1] - r_[0]) >= c["run_min"] * Wm and r_[0] - 0.25 * Wm <= mass_r - cx0 * 0 and r_[1] + 0.25 * Wm >= mass_l]
        if mrg:
            ov = [r_ for r_ in mrg if r_[1] > mass_l + 0.3 * Wm and r_[0] < mass_r - 0.3 * Wm]
            rr = max(ov, key=lambda r_: min(r_[1], mass_r) - max(r_[0], mass_l)) if ov else None
        else:
            rr = None
        # lateral edges = outer ends of the dark under-body/tyre mass.  (The non-road body run was tried and rejected: lane paint,
        # shadows and neighbouring vehicles extend it by up to 40 px; it is kept only in the debug output.)
        bx0, bx1 = mass_l, mass_r
        Wb = bx1 - bx0
        self.dbg.update(mass=(cx0 + mass_l / f, cx0 + mass_r / f), body_run=None if rr is None else (cx0 + rr[0] / f, cx0 + rr[1] / f), tyres=[(t[0], cx0 + t[1] / f, cy0 + t[2] / f, t[3]) for t in tyres], Lr=Lr, thr=thr_d)
        yb_box = max([t[2] for t in tyres], default=float(np.percentile(yf, 85)))
        # ---- 5. top edge
        tw0, tw1 = int(bx0 + 0.2 * Wb), int(bx1 - 0.2 * Wb)
        top_est = True; by0 = yb_box - c["top_est_ratio"] * Wb
        ty_lo, ty_hi = int(max(yb_box - (c["top_est_ratio"] + c["top_win"]) * Wb, 2)), int(yb_box - (c["top_est_ratio"] - c["top_win"]) * Wb)
        if ty_hi - ty_lo > 6 and tw1 - tw0 > 6 and ty_hi < h - 3:
            gy = np.abs(cv2.Sobel(G[ty_lo - 2:ty_hi + 3, tw0:tw1], cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1)[2:-2]
            if gy.max() > c["top_min_grad"]:
                by0 = float(ty_lo + _subpix(int(gy.argmax()), gy)); top_est = False
        # ---- 6. evidence
        # dark-bottom coverage and sharpness
        cover = min(1.0, (sr - sl) / max(Wb, 1.0))
        sharp = min(1.0, float(np.median(val[sl:sr])) / 40.0)
        # symmetry of the lower body
        sy0, sy1 = int(max(by0, 0)), int(min(yb_box, h))
        sym = 0.0
        if sy1 - sy0 > 4 and Wb > 8:
            p = G[sy0:sy1, int(bx0):int(bx1)]
            if p.size:
                p = p - p.mean(); fl = p[:, ::-1]
                dd = float(np.sqrt((p * p).sum() * (fl * fl).sum()))
                sym = float((p * fl).sum() / dd) if dd > 1e-6 else 0.0
        ar = (yb_box - by0) / max(Wb, 1.0)
        # body texture and side-edge strength (walls / shadows / road patches are smooth, vehicles are structured)
        tex = edge2 = 0.0
        by_a, by_b, bx_a, bx_b = int(max(by0, 0)), int(min(yb_box - 0.08 * Wb, h - 1)), int(max(bx0, 0)), int(min(bx1, w))
        if by_b - by_a > 4 and bx_b - bx_a > 8:
            pb = G[by_a:by_b, bx_a:bx_b]
            gxm = np.abs(cv2.Sobel(pb, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)
            gym = np.abs(cv2.Sobel(pb, cv2.CV_32F, 0, 1, ksize=3)).mean()
            tex = float(0.5 * (gxm.mean() + gym))
            k_ = max(2, int(0.04 * len(gxm)))
            inner = float(np.median(gxm[len(gxm) // 5: -len(gxm) // 5])) + 1.0
            edge2 = float(0.5 * (gxm[:k_ + 2].max() + gxm[-(k_ + 2):].max()) / inner)
        if lane_w:
            wr = (Wb / f) / lane_w
            geom = 1.0 if 0.30 <= wr <= 1.15 else 0.0
        else:
            geom = 1.0
        evid = float(geom * (0.40 * cover + 0.30 * sharp + 0.30 * max(0.0, min(1.0, (sym + 0.1) / 0.5))))
        parts = dict(cover=cover, sharp=sharp, sym=sym, geom=geom, Lr=Lr, thr_d=thr_d, ar=float(ar), Wb=float(Wb / f), tex=tex, edge2=edge2)
        if evid < c["evid_min"] or geom == 0.0 or sym < c["sym_min"] or edge2 < c["edge2_min"]:
            return None
        # ---- 7. contact point assembly
        def to_src(x, y): return (cx0 + x / f, cy0 + y / f)
        pts = [to_src(x, y) for (_, x, y, _) in tyres]
        merged_ = any(not t[3] for t in tyres)
        if len(tyres) == 2:
            q_, det_ = "TWO_TYRES", ("RESOLVED" if not merged_ else "MERGED")
            cxy = (0.5 * (pts[0][0] + pts[1][0]), 0.5 * (pts[0][1] + pts[1][1])); sig = 2.5 if not merged_ else 4.0
        elif len(tyres) == 1:
            q_, det_ = "ONE_TYRE", "SINGLE_SIDE"; cxy = pts[0]; sig = 7.0
        else:
            q_, det_ = "UNCERTAIN", "NO_TYRE"; cxy = None; sig = 15.0
        hint = to_src(0.5 * (bx0 + bx1), yb_box)
        box = (cx0 + bx0 / f, cy0 + by0 / f, cx0 + bx1 / f, cy0 + yb_box / f)
        return RefinedVehicle(box=box, top_estimated=top_est, contact=cxy, contact_hint=hint, quality=q_, detail=det_, tyres=pts,
                              contact_sigma=sig, evidence=evid, parts=parts)
