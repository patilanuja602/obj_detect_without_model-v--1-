"""Step 4 + 5 - from segments to the two boundaries of the lane the car is in.

Every boundary is modelled *relative to the vanishing point*:

    x(t) = vp_x + k * t + c * t^2 ,   t = (y - vp_y) / (y_ref - vp_y)   (t=0 horizon, t=1 bottom)

k = lean of the line, c = curvature.  Because the model hangs off the VP, it follows the
horizon automatically when the camera pitches up/down.
"""
from dataclasses import dataclass, replace

import cv2
import numpy as np

from .segments import angle_error_to_point, seg_geometry


# ----------------------------------------------------------------------------
@dataclass
class Boundary:
    k: float
    c: float
    support: float            # summed segment length (px)
    y_far: float              # farthest (smallest y) supporting pixel
    y_near: float
    colour: str = "white"     # 'white' | 'yellow'
    solid: float = 0.0        # 0..1 paint coverage along the line (>0.75 = solid)

    @property
    def d1(self):             # x offset from VP at the bottom row - the stable quantity
        return self.k + self.c

    def x_at(self, t, vp_x):
        return vp_x + self.k * t + self.c * t * t


def t_of_y(y, vp_y, y_ref):
    return (y - vp_y) / max(1.0, (y_ref - vp_y))


# ----------------------------------------------------------------------------
def build_candidates(segs, vp, yellow_dil, mark_dil, cfg, w, h):
    """Group VP-consistent segments into candidate boundaries."""
    if len(segs) == 0 or vp is None:
        return []
    y_ref = h - 1.0
    mid, d, length = seg_geometry(segs)
    err, dist = angle_error_to_point(mid, d, vp)
    ok = (
        (err < np.radians(cfg.seg_vp_angle_tol_deg))
        & (mid[:, 1] > vp[1] + cfg.near_vp_frac * (y_ref - vp[1]))
    )
    if not ok.any():
        return []
    segs, mid, length = segs[ok], mid[ok], length[ok]

    # intercept of the VP->midpoint ray with the bottom row
    t_mid = t_of_y(mid[:, 1], vp[1], y_ref)
    xr = vp[0] + (mid[:, 0] - vp[0]) / np.maximum(t_mid, 1e-3)
    weight = length * np.maximum(t_mid, 0.05)

    order = np.argsort(xr)
    tol = cfg.cluster_tol_frac * w
    groups, cur = [], [order[0]]
    for idx in order[1:]:
        gx = np.average(xr[cur], weights=weight[cur])
        if xr[idx] - gx < tol:
            cur.append(idx)
        else:
            groups.append(cur)
            cur = [idx]
    groups.append(cur)

    out = []
    for g in groups:
        b = _fit_group(segs[g], vp, yellow_dil, mark_dil, cfg, w, h)
        if b is not None:
            out.append(b)
    return out


def _fit_group(segs, vp, yellow_dil, mark_dil, cfg, w, h):
    y_ref = h - 1.0
    pts = []
    for x1, y1, x2, y2 in segs:
        n = max(3, int(np.hypot(x2 - x1, y2 - y1) / 4))
        pts.append(np.stack([np.linspace(x1, x2, n), np.linspace(y1, y2, n)], axis=1))
    pts = np.concatenate(pts)
    t = t_of_y(pts[:, 1], vp[1], y_ref)
    xrel = pts[:, 0] - vp[0]

    A = np.stack([t, t * t], axis=1)
    reg = np.diag([0.0, cfg.curv_ridge * len(t)])
    try:
        k, c = np.linalg.solve(A.T @ A + reg + np.eye(2) * 1e-6, A.T @ xrel)
    except np.linalg.LinAlgError:
        return None
    lim = cfg.max_curv_frac * w
    c = float(np.clip(c, -lim, lim))
    k = float(np.sum(t * (xrel - c * t * t)) / max(np.sum(t * t), 1e-6)) if abs(c) == lim else float(k)

    xs = np.clip(pts[:, 0].astype(int), 0, w - 1)
    ys = np.clip(pts[:, 1].astype(int), 0, h - 1)
    yellow_frac = float((yellow_dil[ys, xs] > 0).mean())
    colour = "yellow" if yellow_frac > 0.35 else "white"

    b = Boundary(k=k, c=c,
                 support=float(sum(np.hypot(s[2] - s[0], s[3] - s[1]) for s in segs)),
                 y_far=float(pts[:, 1].min()), y_near=float(pts[:, 1].max()),
                 colour=colour)
    b.solid = _paint_coverage(b, vp, mark_dil, w, h)
    return b


def _paint_coverage(b, vp, mark_dil, w, h):
    y_ref = h - 1.0
    ys = np.linspace(b.y_far, b.y_near, 40)
    t = t_of_y(ys, vp[1], y_ref)
    xs = b.x_at(t, vp[0])
    xi = np.clip(xs.astype(int), 0, w - 1)
    yi = np.clip(ys.astype(int), 0, h - 1)
    return float((mark_dil[yi, xi] > 0).mean())


# ----------------------------------------------------------------------------
class RoadModel:
    """Learns 'what asphalt looks like right now' from a thin strip along the ego path.

    The strip is *derived from the vanishing point*, not hard-coded: it is the ray from the
    horizon down to the car's position at the bottom of the image.  Later we only accept a
    boundary if the pixels on the car's side of it look like that asphalt.  This is what
    rejects guard-rails, walls, sky edges and sign poles that also converge to the VP.
    """

    def __init__(self, lab, vp, car_x, cfg, w, h):
        self.lab, self.w, self.h = lab, w, h
        y_ref = h - 1.0
        ts = np.linspace(0.2, 0.85, 36)
        offs = np.linspace(-1, 1, 5)
        ys = vp[1] + ts * (y_ref - vp[1])
        cx = vp[0] + (car_x - vp[0]) * ts
        X = (cx[:, None] + offs[None, :] * 0.02 * w * ts[:, None]).ravel()
        Y = np.repeat(ys, len(offs))
        px = self._take(X, Y)
        self.med = np.median(px, axis=0)
        mad = np.median(np.abs(px - self.med), axis=0) * 1.4826
        self.sL = max(20.0, 3.5 * mad[0])
        self.sC = max(12.0, 3.5 * float(np.hypot(mad[1], mad[2])))

    def _take(self, X, Y):
        xi = np.clip(X.astype(int), 0, self.w - 1)
        yi = np.clip(Y.astype(int), 0, self.h - 1)
        return self.lab[yi, xi].astype(np.float32)

    def fraction(self, X, Y):
        px = self._take(X, Y)
        dL = np.abs(px[:, 0] - self.med[0])
        dC = np.hypot(px[:, 1] - self.med[1], px[:, 2] - self.med[2])
        return float(((dL < self.sL) & (dC < self.sC)).mean())

    def inner_fraction(self, b, side, vp, cfg):
        """Road-likeness on the car's side of boundary `b` (side=+1: interior is to its right)."""
        w, h = self.w, self.h
        y_ref = h - 1.0
        y0 = max(b.y_far, vp[1] + 0.12 * (y_ref - vp[1]))
        y1 = min(b.y_near, 0.92 * h)
        if y1 - y0 < 6:
            y0, y1 = y1 - 6, y1
        ys = np.linspace(y0, y1, 26)
        t = t_of_y(ys, vp[1], y_ref)
        xs = b.x_at(t, vp[0])
        X, Y = [], []
        for off in (0.025, 0.04, 0.06):
            X.append(xs + side * off * w * t)
            Y.append(ys)
        return self.fraction(np.concatenate(X), np.concatenate(Y))


# ----------------------------------------------------------------------------
class BoundaryTrack:
    """Temporal smoothing + short memory for one side (left or right)."""

    def __init__(self, cfg, w):
        self.cfg, self.w = cfg, w
        self.b = None
        self.missed = 0
        self._pend = None
        self._pend_n = 0

    @property
    def alive(self):
        return self.b is not None

    def update(self, meas):
        cfg = self.cfg
        if meas is None:
            self.missed += 1
            if self.missed > cfg.track_max_hold:
                self.b = None
            return
        if self.b is None:
            self.b, self.missed = replace(meas), 0
            return
        if abs(meas.d1 - self.b.d1) > cfg.track_jump_frac * self.w:
            if self._pend is not None and abs(meas.d1 - self._pend.d1) < 0.04 * self.w:
                self._pend_n += 1
            else:
                self._pend, self._pend_n = meas, 1
            if self._pend_n >= cfg.track_jump_confirm:
                self.b, self.missed, self._pend = replace(meas), 0, None
            else:
                self.missed += 1
                if self.missed > cfg.track_max_hold:
                    self.b = None
            return
        self._pend = None
        a = cfg.track_alpha
        b = self.b
        b.k = (1 - a) * b.k + a * meas.k
        b.c = (1 - a) * b.c + a * meas.c
        b.y_far = (1 - a) * b.y_far + a * meas.y_far
        b.y_near = (1 - a) * b.y_near + a * meas.y_near
        b.support, b.colour, b.solid = meas.support, meas.colour, meas.solid
        self.missed = 0

    @property
    def confidence(self):
        if self.b is None:
            return 0.0
        return max(0.0, 1.0 - self.missed / (self.cfg.track_max_hold + 1))


# ----------------------------------------------------------------------------
def pick_side(cands, side, car_x, vp, road, cfg, w, h):
    """Nearest valid boundary on one side of the car. side=-1 -> left, +1 -> right."""
    y_ref = h - 1.0
    margin = cfg.car_margin_frac * w
    pool = []
    for b in cands:
        x1 = b.x_at(1.0, vp[0])
        if side < 0 and x1 < car_x - margin:
            pool.append((car_x - x1, b))
        elif side > 0 and x1 > car_x + margin:
            pool.append((x1 - car_x, b))
    pool.sort(key=lambda p: p[0])
    for _, b in pool[: cfg.max_candidates_per_side]:
        # interior of a LEFT boundary lies to its right (+1) and vice-versa
        if road.inner_fraction(b, -side, vp, cfg) >= cfg.road_fraction_min:
            return b
    return None
