#!/usr/bin/env python3
"""
Classical-CV drivable-path detector (no deep learning, CPU only).

Pipeline (as specified):
  1. Downscale to 640 px wide                      (speed: resolution)
  2. Trapezoid ROI, crop to its bounding box       (speed: ROI)
  3. CLAHE on the L channel -> Gaussian blur       (illumination robustness)
  4. Canny edges, masked to the trapezoid          (gradient extraction)
  5. HoughLinesP + slope filtering (left/right, reject near-horizontal)
  6. Per side: straight?  -> Method A: length-weighted average line
               curved?    -> Method B: sliding windows + quadratic fit
  7. Temporal smoothing + sanity checks
  8. cv2.fillPoly between the two fitted lanes = drivable path

Usage:
  python lane_path.py input.mp4 output.mp4 [--start 36 --duration 20] [--debug]
"""
import argparse
import os
import subprocess
import shutil
import time

import cv2
import numpy as np

# ---------------------------------------------------------------- config ----
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


class PathDetector:
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


# ------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", default=None, help="input video path")
    ap.add_argument("dst", nargs="?", default=None, help="output video path")
    ap.add_argument("--source", dest="source_opt", default=None, help="input video path (alternative to positional src)")
    ap.add_argument("--save", dest="save_opt", default=None, help="output video path (alternative to positional dst)")
    ap.add_argument("--show", action="store_true", help="open a live window (press q to quit)")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--out-w", type=int, default=1280)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--profile", choices=list(PROFILES), default="mountain")
    ap.add_argument("--top-y", type=float)
    ap.add_argument("--top-x", type=float, nargs=2)
    a = ap.parse_args()

    src = a.src or a.source_opt
    if not src:
        ap.error("an input video is required (e.g. lane_path.py video.mp4 output.mp4 or --source video.mp4)")

    dst = a.dst or a.save_opt
    if not dst and not a.show:
        dst = "output.mp4"

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
        raise SystemExit(f"cannot read video: {src}")
    sh, sw = f.shape[:2]
    pw = cfg["proc_w"]
    ph = int(round(sh * pw / sw))
    ow = a.out_w
    oh = int(round(sh * ow / sw))
    det = PathDetector(pw, ph, cfg)

    tmp = (dst + ".tmp.mp4") if dst else None
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (ow, oh)) if tmp else None
    times, n = [], 0
    while ok and n < n_max:
        small = cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        L, R = det.process(small)
        dt = time.perf_counter() - t0
        times.append(dt)
        disp = cv2.resize(f, (ow, oh), interpolation=cv2.INTER_AREA)
        fps = 1.0 / np.mean(times[-30:])
        out_frame = det.draw(disp, L, R, fps, a.debug)
        if vw:
            vw.write(out_frame)
        if a.show:
            cv2.imshow("lane_path (q to quit)", out_frame)
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
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c:v",
                            "libx264", "-pix_fmt", "yuv420p", "-crf", "23", dst])
            os.remove(tmp)
        else:
            os.replace(tmp, dst)
        print(f"saved output to {dst}")

    t = np.array(times[5:] or times)
    if len(t) > 0:
        print(f"frames={n}  algo-only: mean {1/t.mean():.0f} FPS, "
              f"p95 latency {np.percentile(t, 95)*1000:.1f} ms, "
              f"worst {t.max()*1000:.1f} ms  (processing size {pw}x{ph})")


if __name__ == "__main__":
    main()
