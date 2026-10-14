"""Vehicle detection without any trained model.

Idea
----
1. The lane detector (untouched) gives the horizon (vanishing point) and the lane lines.
   From them we build a road-only search area: the ego lane plus up to N lane-widths of
   *asphalt* on each side, from the bonnet up to a maximum distance.  Sky, buildings,
   trees and everything beyond the road edge are never looked at.
2. Inside that area we learn what asphalt looks like *row by row* (robust median in LAB),
   so near/far brightness changes and a car covering part of a row do not matter.
   Pixels that differ from the asphalt (and are not lane paint) are candidate vehicle pixels.
3. Candidate blobs are accepted only if they look like a vehicle:
   size and shape consistent with the local lane width (the lane is our ruler),
   a dark contact/shadow band on the ground (or a dark body), enough texture.
   Long thin things (walls, dividers, guard-rails) fail the shape test.
4. A small tracker confirms boxes over a few frames and keeps them steady.
"""
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from lane_detector.lanes import t_of_y

from .config import VConfig
from .tracker import Track, Tracker


@dataclass
class Vehicle:
    id: int
    box: tuple            # x1, y1, x2, y2 in the processed frame
    side: str             # FRONT / LEFT / RIGHT
    dist_m: Optional[float]
    hits: int = 0


class VehicleDetector:
    def __init__(self, cfg: VConfig = None):
        self.cfg = cfg or VConfig()
        self.tracker = Tracker(self.cfg)
        self._Wd = None                       # lane width at the bottom row (px), remembered
        self._coefs = {}
        self._offs = {}
        self._hist = []                       # recent horizon rows, for a spike-proof horizon
        self._held = 0
        self._last_g = None
        self._slot_ok = {}                    # slot index -> smoothed "looks like road"
        self.debug = {}

    # ------------------------------------------------------------------ geometry
    def _lines(self, res, h, w):
        """Return (vp, xs_fn) where xs_fn(j, t) = x of lane line j (0 = left ego line, 1 = right ego line,
        -1 = one lane further left, 2 = one lane further right ...)."""
        vp, L, R = res.vp, res.left, res.right
        if vp is None or (L is None and R is None):
            return None
        Wd = None
        if L is not None and R is not None:
            wd = R.x_at(1.0, vp[0]) - L.x_at(1.0, vp[0])
            if 0.12 * w <= wd <= 1.7 * w:
                Wd = wd
                self._Wd = wd if self._Wd is None else 0.9 * self._Wd + 0.1 * wd
        if Wd is None:
            Wd = self._Wd
        if Wd is None:
            return None
        base_k, base_c = (L.k, L.c) if L is not None else (R.k - Wd, R.c)

        def lane_px(t):
            if L is not None and R is not None:
                return np.maximum(R.x_at(t, vp[0]) - L.x_at(t, vp[0]), 0.05 * Wd * t)
            return Wd * t

        def x(j, t):
            return vp[0] + base_k * t + base_c * t * t + j * Wd * t
        x.lane_px = lane_px
        x.img_w = w
        return vp, x, Wd

    # ------------------------------------------------------------------ main
    def process(self, res) -> List[Vehicle]:
        cfg = self.cfg
        frame = res.frame
        h, w = frame.shape[:2]
        g = self._lines(res, h, w)
        if g is None:
            return self._hold(None, h)
        vp, X, Wd = g
        y_ref = h - 1.0
        self._hist = (self._hist + [float(vp[1])])[-31:]
        hor = float(np.median(self._hist))
        if len(self._hist) >= 8 and abs(vp[1] - hor) > cfg.horizon_glitch * h:
            return self._hold(g, h)               # lane code's horizon spiked this frame: don't trust the geometry
        self._held = 0
        self._last_g = g
        f = (w / 2.0) / np.tan(np.radians(cfg.hfov_deg / 2.0))

        # ---- search rows -------------------------------------------------
        y_bot = int(cfg.hood_frac * h)
        t_far = max(0.05, cfg.lane_width_m * f / (Wd * cfg.max_distance_m))
        y_top = int(max(hor + t_far * (y_ref - hor), hor + 3))
        if y_top >= y_bot - 8:
            return self._hold(g, h)
        y_top_c = y_top
        ys = np.arange(y_top_c, y_bot + 1)
        ts = np.maximum(t_of_y(ys.astype(np.float32), vp[1], y_ref), 0.03)
        g_top = y_top - y_top_c                           # index in ys where the ground area starts

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        Lc = cv2.GaussianBlur(lab[:, :, 0], (0, 0), 1.0).astype(np.float32)
        a = lab[:, :, 1].astype(np.float32) - 128.0
        b = lab[:, :, 2].astype(np.float32) - 128.0
        paint = self._thin_paint(res.debug.get("mask"), vp, Wd, y_ref, h, w)

        S = cfg.side_lanes
        slots = list(range(-S, S + 1))        # slot s lies between lane lines s and s+1; s=0 is the ego lane

        # a vehicle reaches above its contact row, so the lateral limits of row y come from a row a
        # vehicle-height lower (the cone of the road widens downwards)
        ts_lat = ts * (1.0 + cfg.veh_h_ratio * Wd / max(y_ref - vp[1], 1.0))

        def poly(j0, j1, lateral=False):
            tt = ts_lat if lateral else ts
            xl, xr = X(j0, tt), X(j1, tt)
            pts = np.concatenate([np.stack([xl, ys], 1), np.stack([xr, ys], 1)[::-1]])
            return pts.astype(np.int32)

        # ---- one asphalt model per lane slot --------------------------------------------------
        # (a smooth robust curve over distance, fitted to that slot's own rows, so cars covering
        #  part of a lane are outliers; per-slot because e.g. a concrete patch can lie in only one lane)
        step = 3
        rows = np.arange(0, len(ys), step)
        rows = rows[ts[rows] > 0.04]
        nR = len(ys)

        def row_medians(s):
            xa = np.clip(X(s + 0.2, ts[rows]), 0, w - 1).astype(int)
            xb = np.clip(X(s + 0.8, ts[rows]), 0, w - 1).astype(int)
            meds = np.full((len(rows), 3), np.nan, np.float32)
            for k, r in enumerate(rows):
                if xb[k] - xa[k] < 5:
                    continue
                y = ys[r]
                ok = paint[y, xa[k]:xb[k]] == 0
                if ok.sum() < 5:
                    continue
                sl = slice(xa[k], xb[k])
                meds[k] = (np.median(Lc[y, sl][ok]), np.median(a[y, sl][ok]), np.median(b[y, sl][ok]))
            return meds

        # ego lane: smooth robust curve over distance (rows with a car in them are outliers)
        meds0 = row_medians(0)
        good0 = ~np.isnan(meds0[:, 0])
        if good0.sum() < 10:
            return self._hold(g, h)
        tt0 = ts[rows][good0]
        ego = np.zeros((nR, 3), np.float32)
        mad_ego = 1.0
        for c in range(3):
            vv = meds0[good0, c]
            keep = np.ones(len(tt0), bool)
            for _ in range(4):
                coef = np.polyfit(tt0[keep], vv[keep], 2)
                r_ = vv - np.polyval(coef, tt0)
                mad = max(np.median(np.abs(r_[keep])) * 1.4826, 1.0)
                keep = np.abs(r_) < 2.5 * mad
                if keep.sum() < 8:
                    break
            old = self._coefs.get(c)
            if old is not None:
                coef = 0.6 * old + 0.4 * coef
            self._coefs[c] = coef
            ego[:, c] = np.polyval(coef, ts)
            if c == 0:
                mad_ego = mad
        ego_rows = ego[rows]

        # every other lane: the ego curve plus ONE constant offset (median over its rows, so a car
        # covering part of the lane cannot drag it) -- this handles e.g. a concrete patch in one lane only
        REF = np.zeros((len(slots), nR, 3), np.float32)
        MAD = np.full(len(slots), mad_ego, np.float32)
        slot_chroma = {}
        for si, s in enumerate(slots):
            off = np.zeros(3, np.float32)
            if s == 0:
                REF[si] = ego
                slot_chroma[s] = float(np.median(np.hypot(meds0[good0, 1], meds0[good0, 2])))
                continue
            m = row_medians(s)
            gd = ~np.isnan(m[:, 0])
            if gd.sum() >= 8:
                d = m[gd] - ego_rows[gd]
                off = np.median(d, axis=0)
                key = (s,)
                if key in self._offs:
                    off = 0.5 * self._offs[key] + 0.5 * off
                self._offs[key] = off
                slot_chroma[s] = float(np.median(np.hypot(m[gd, 1], m[gd, 2])))
            REF[si] = ego + off

        slotmap = np.zeros((h, w), np.uint8)
        for si, s in enumerate(slots):
            cv2.fillPoly(slotmap, [poly(s, s + 1, lateral=True)], si + 1)
        sm = slotmap[ys]                                          # (nR, w)
        idx = np.maximum(sm.astype(np.int32) - 1, 0)
        rr = np.arange(nR)[:, None]
        refL_img = np.zeros((h, w), np.float32); refa_img = refL_img.copy(); refb_img = refL_img.copy()
        thr_img = refL_img.copy()
        refL_img[ys] = REF[idx, rr, 0]
        refa_img[ys] = REF[idx, rr, 1]
        refb_img[ys] = REF[idx, rr, 2]
        thr_img[ys] = np.maximum(cfg.min_dL, cfg.sigma_dL * MAD)[idx]
        dL = Lc - refL_img
        dC = np.hypot(a - refa_img, b - refb_img)
        deviates = ((np.abs(dL) > thr_img) | (dC > cfg.min_dC))

        # ---- which side lanes are really road?  (neutral grey surface, contiguous with the ego lane)
        for s in slots:
            if s == 0:
                self._slot_ok[s] = 1.0
                continue
            ok_now = 1.0 if (s in slot_chroma and slot_chroma[s] < cfg.road_max_chroma) else 0.0
            old = self._slot_ok.get(s, ok_now)
            self._slot_ok[s] = (1 - cfg.slot_alpha) * old + cfg.slot_alpha * ok_now
        active = {0}
        for s in range(1, S + 1):
            if self._slot_ok.get(s, 0) >= 0.5 and (s - 1) in active:
                active.add(s)
        for s in range(-1, -S - 1, -1):
            if self._slot_ok.get(s, 0) >= 0.5 and (s + 1) in active:
                active.add(s)
        lo, hi = min(active), max(active)
        roi = np.zeros((h, w), np.uint8)
        cv2.fillPoly(roi, [poly(lo if lo < 0 else 0, hi + 1 if hi > 0 else 1, lateral=True)], 255)
        roi[:y_top_c] = 0
        roi[y_bot + 1:] = 0
        self.debug_chroma = slot_chroma

        # ---- candidate pixels -------------------------------------------------------
        cand = (deviates & (paint == 0) & (roi > 0)).astype(np.uint8) * 255
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((cfg.strip_open, 3), np.uint8))   # drop thin horizontal strips
        cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7)))
        gray = lab[:, :, 0]
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 140)
        gy = np.abs(cv2.Sobel(cv2.GaussianBlur(Lc, (0, 0), 1.0), cv2.CV_32F, 0, 1, ksize=3)) / 8.0

        n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        dets, rejected = [], []
        for i in range(1, n):
            bx, by, bw, bh, area = stats[i]
            by0, bh0 = by, bh                         # what the mask really covers
            if by <= y_top + 1:                       # blob touches the far limit: the roof is above the rows we search
                ny = self._extend_top(bx, by, bw, by + bh, vp, X, y_ref, gy)
                bh, by = by + bh - ny, ny
            sub = labels[by0:by0 + bh0, bx:bx + bw] == i
            reason = self._check(bx, by, bw, bh, bh0, area, vp, X, y_ref, y_top, refL_img, Lc, edges, sub, h)
            if reason is None:
                box = self._refine(bx, by, bw, bh, by0 - by, sub)
                dets.extend(self._split(box, X, y_ref, vp, Lc, sub, bx, by0))
            else:
                rejected.append((bx, by, bx + bw, by + bh, reason))
        dets = self._nms(dets)
        self.debug = dict(roi=roi, cand=cand, rejected=rejected, active=sorted(active),
                          y_top=y_top, y_bot=y_bot, chroma=slot_chroma)
        self.tracker.update(dets)
        return self._finish(g, h)

    def _hold(self, g, h):
        """No trustworthy lane geometry this frame: keep the current boxes for a short while."""
        self._held += 1
        if self._held > self.cfg.hold_frames:
            self.tracker.tracks = []
        return self._finish(self._last_g if self._last_g is not None else g, h)

    def _finish(self, g, h):
        """Confirmed tracks -> Vehicle list, with duplicates (a box mostly inside a bigger one) removed."""
        vs = self.tracker.confirmed(self._locate_fn(g, h))
        vs.sort(key=lambda v: -(v.box[2] - v.box[0]) * (v.box[3] - v.box[1]))
        keep = []
        for v in vs:
            a = (v.box[2] - v.box[0]) * (v.box[3] - v.box[1])
            dup = False
            for k in keep:
                ix = max(0, min(v.box[2], k.box[2]) - max(v.box[0], k.box[0]))
                iy = max(0, min(v.box[3], k.box[3]) - max(v.box[1], k.box[1]))
                if ix * iy > 0.7 * a:
                    dup = True
                    break
            if not dup:
                keep.append(v)
        return keep

    # ------------------------------------------------------------------ refine
    def _refine(self, bx, by, bw, bh, pad_top, sub):
        """Trim thin slivers of road attached to the blob (keep the dense core).  `pad_top` rows above
        the measured mask were added by the roof search and are left alone."""
        m = sub.astype(np.float32)
        col = cv2.blur(m.mean(0).reshape(1, -1), (5, 1)).ravel()
        row = cv2.blur(m.mean(1).reshape(-1, 1), (1, 5)).ravel()
        cx = np.flatnonzero(col >= 0.35 * col.max())
        ry = np.flatnonzero(row >= 0.25 * row.max())
        if len(cx) < 4 or len(ry) < 4:
            return (bx, by, bx + bw, by + bh)
        top = by if pad_top > 0 else by + int(ry[0])
        return (bx + int(cx[0]), top, bx + int(cx[-1]) + 1, by + pad_top + int(ry[-1]) + 1)

    # ------------------------------------------------------------------ split
    def _split(self, box, X, y_ref, vp, Lc, sub, bx0, by0):
        """Two vehicles side by side can form one blob.  If the blob is wider than one vehicle and the
        body brightness steps cleanly (e.g. black car next to a white one), cut it there."""
        cfg = self.cfg
        x1, y1, x2, y2 = box
        t2 = max(float(t_of_y(float(y2), vp[1], y_ref)), 0.03)
        lane = float(X.lane_px(t2))
        bw = x2 - x1
        if bw < cfg.split_min_ratio * lane:
            return [box]
        ya, yb = max(y1, by0), max(max(y1, by0) + 3, int(y2 - 0.2 * (y2 - y1)))
        sig = Lc[ya:yb, x1:x2].mean(0)
        minw = max(6, int(cfg.split_part_ratio * lane))
        if bw < 2 * minw + 2:
            return [box]
        best, bk = 0.0, None
        for k in range(minw, bw - minw):
            l, r = sig[:k], sig[k:]
            sc = abs(l.mean() - r.mean())
            if sc > best and l.std() < 0.5 * sc and r.std() < 0.5 * sc:
                best, bk = sc, k
        if bk is None or best < cfg.split_min_step:
            return [box]
        out = []
        for a_, b_ in ((x1, x1 + bk), (x1 + bk, x2)):
            seg = sub[:, max(a_ - bx0, 0):max(b_ - bx0, 1)].astype(np.float32)
            if seg.size:
                col = cv2.blur(seg.mean(0).reshape(1, -1), (5, 1)).ravel()
                cx = np.flatnonzero(col >= 0.35 * col.max())
                if len(cx) >= 4:
                    a_, b_ = a_ + int(cx[0]), a_ + int(cx[-1]) + 1
            out.append((a_, y1, b_, y2))
        return out

    # ------------------------------------------------------------------ roof
    def _extend_top(self, bx, by, bw, y2, vp, X, y_ref, gy):
        """The search rows stop at the distance limit, but a vehicle sticks up above its contact row by
        about 0.4 lane-widths.  Look for the roof edge (a horizontal edge across the box) between 0.25 and
        0.75 lane-widths above the contact line; if none is clear, use the typical car height."""
        cfg = self.cfg
        t2 = max(float(t_of_y(float(y2), vp[1], y_ref)), 0.03)
        lane = float(X.lane_px(t2))
        ya, yb = max(0, int(y2 - cfg.roof_max_ratio * lane)), int(y2 - cfg.roof_min_ratio * lane)
        if by <= ya:
            return by
        top = int(y2 - cfg.roof_prior_ratio * lane)
        xa, xb = bx + int(0.15 * bw), bx + max(int(0.85 * bw), int(0.15 * bw) + 1)
        yend = min(yb, by)
        if yend - ya >= 3:
            prof = gy[ya:yend, xa:xb].mean(1)
            prof = np.convolve(prof, np.ones(3) / 3, mode="same")
            idx = np.flatnonzero(prof > cfg.roof_edge_min)
            if len(idx):
                top = ya + int(idx[np.argmin(np.abs(ya + idx - top))])      # edge nearest to the typical height
        return int(min(by, max(top, ya)))

    # ------------------------------------------------------------------ paint
    def _thin_paint(self, mask, vp, Wd, y_ref, h, w):
        """Lane paint = thin bright/yellow components.  The lane code's mask also flags white cars
        (a bright blob is 'thin' for its structuring element), so keep only components whose
        thickness is small compared with the lane width at that row."""
        out = np.zeros((h, w), np.uint8)
        if mask is None:
            return out
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        for i in range(1, n):
            x, y, bw, bh, area = st[i]
            sl = (slice(y, y + bh), slice(x, x + bw))
            thick = 2.0 * float(dt[sl][lab[sl] == i].max())
            t = max(float(t_of_y(float(y + bh), vp[1], y_ref)), 0.03)
            if thick <= self.cfg.paint_thick_ratio * Wd * t + 2.0:
                out[sl][lab[sl] == i] = 255
        return cv2.dilate(out, np.ones((5, 5), np.uint8))

    # ------------------------------------------------------------------ checks
    def _check(self, bx, by, bw, bh, bh_mask, area, vp, X, y_ref, y_top, refL, Lc, edges, sub, h_img):
        cfg = self.cfg
        if bw < cfg.min_px_w or bh < cfg.min_px_h:
            return "tiny"
        y2 = by + bh
        if y2 < y_top:
            return "above_ground_area"
        t2 = float(t_of_y(float(y2), vp[1], y_ref))
        lane_px = max(float(X.lane_px(t2)), 1.0)
        wr, hr = bw / lane_px, bh / lane_px
        if not (cfg.min_w_ratio <= wr <= cfg.max_w_ratio):
            return f"w_ratio {wr:.2f}"
        if not (cfg.min_h_ratio <= hr <= cfg.max_h_ratio):
            return f"h_ratio {hr:.2f}"
        f_px = 0.5 * X.img_w / np.tan(np.radians(cfg.hfov_deg / 2.0))
        if cfg.lane_width_m * f_px / lane_px > 1.25 * cfg.max_distance_m:
            return "too_far"
        if bw > cfg.max_width_frac * X.img_w:
            return "too_wide"
        asp = bw / bh
        if not (cfg.min_aspect <= asp <= cfg.max_aspect):
            return f"aspect {asp:.2f}"
        fill = area / float(bw * bh_mask)
        if fill < cfg.min_fill:
            return f"fill {fill:.2f}"
        # walls, dividers and guard-rails run ALONG the road: their base keeps one lateral position
        # (a fixed number of lane-widths from the lane lines) while rows change.  A vehicle's
        # ground-contact line spans several lateral positions at (almost) one distance.
        cols = np.flatnonzero(sub.any(0))
        if len(cols) > 12:
            xs = cols[int(0.08 * len(cols)):max(int(0.92 * len(cols)), int(0.08 * len(cols)) + 2)]
            bot = bh_mask - 1 - np.argmax(sub[::-1, :][:, xs], axis=0)
            yy = (bot + by + (bh - bh_mask)).astype(np.float32)
            tt = np.maximum(t_of_y(yy, vp[1], y_ref), 0.03)
            u = ((xs + bx) - X(0, tt)) / np.maximum(X.lane_px(tt), 1.0)
            if np.ptp(yy) >= cfg.wall_min_rows and np.ptp(u) < cfg.wall_u_range:
                return f"wall_edge u_range={np.ptp(u):.2f}"
        # ground-contact shadow / dark body
        yb0 = max(by, y2 - max(3, int(0.18 * bh)))
        band = Lc[yb0:y2 + 1, bx:bx + bw].mean(1)
        drop = float((refL[yb0:y2 + 1, bx:bx + bw].mean(1) - band).max())
        body = float(Lc[by:y2, bx:bx + bw].mean())
        body_ref = float(refL[by:y2, bx:bx + bw].mean())
        if drop < cfg.shadow_drop and body > body_ref - cfg.shadow_drop:
            return f"no_shadow {drop:.0f}"
        ed = float((edges[by:y2, bx:bx + bw] > 0).mean())
        if ed < cfg.min_edge_density:
            return f"flat {ed:.3f}"
        # smooth surfaces (concrete wall, road-surface seams) have almost no horizontal intensity change;
        # a vehicle has lights, plates, windows, wheel arches
        reg = Lc[by:y2, bx:bx + bw]
        gx = float(np.abs(cv2.Sobel(reg, cv2.CV_32F, 1, 0, ksize=3)).mean()) / 8.0
        if gx < cfg.min_grad_x:
            return f"smooth {gx:.1f}"
        return None

    @staticmethod
    def _nms(boxes):
        boxes = sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
        keep = []
        for b in boxes:
            dup = False
            for k in keep:
                ix = max(0, min(b[2], k[2]) - max(b[0], k[0]))
                iy = max(0, min(b[3], k[3]) - max(b[1], k[1]))
                inter = ix * iy
                a = (b[2] - b[0]) * (b[3] - b[1])
                if inter > 0.6 * a:
                    dup = True
                    break
            if not dup:
                keep.append(b)
        return keep

    # ------------------------------------------------------------------ labels
    def _locate_fn(self, g, h):
        cfg = self.cfg
        def loc(box):
            if g is None:
                return "FRONT", None
            vp, X, Wd = g
            y_ref = h - 1.0
            t = float(t_of_y(float(box[3]), vp[1], y_ref))
            cx = 0.5 * (box[0] + box[2])
            xl, xr = float(X(0, t)), float(X(1, t))
            side = "LEFT" if cx < xl else "RIGHT" if cx > xr else "FRONT"
            f = 0.5 * X.img_w / np.tan(np.radians(cfg.hfov_deg / 2.0))
            dist = cfg.lane_width_m * f / max(float(X.lane_px(t)), 1.0)
            return side, float(dist)
        return loc
