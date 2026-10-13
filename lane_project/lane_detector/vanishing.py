"""Step 3 - adaptive horizon.

Every lane marking, kerb and guard-rail is parallel to the road, so in the image they
all point at one *vanishing point*.  We find it per frame with RANSAC over the segments.
Its y-coordinate is the horizon: when the camera pitches up/down the horizon moves, and
everything that depends on it (which segments are 'road', where the lane ends) follows.
This replaces the hard-coded triangle of the original notebook.
"""
import numpy as np

from .segments import angle_error_to_point, seg_geometry


class VanishingPointTracker:
    def __init__(self, cfg, w, h):
        self.cfg, self.w, self.h = cfg, w, h
        self.vp = None            # np.array([x, y])
        self.missed = 0
        self._pending = None
        self._pending_n = 0
        self.rng = np.random.default_rng(0)
        self.confidence = 0.0

    # ------------------------------------------------------------------
    def _plausible(self, p):
        lo, hi = self.cfg.vp_y_range
        return (lo * self.h <= p[1] <= hi * self.h) and (-self.w <= p[0] <= 2 * self.w)

    @staticmethod
    def _intersect(m1, d1, m2, d2):
        A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
        if abs(np.linalg.det(A)) < 1e-6:
            return None
        t = np.linalg.solve(A, m2 - m1)
        return m1 + t[0] * d1

    def _inliers(self, p, mid, d, length):
        err, dist = angle_error_to_point(mid, d, p)
        tol = np.radians(self.cfg.vp_angle_tol_deg)
        return (err < tol) & (dist > 0.05 * self.h), err

    def _score(self, p, mid, d, length):
        inl, _ = self._inliers(p, mid, d, length)
        return float(length[inl].sum()), inl

    def _refine(self, p, mid, d, length):
        for _ in range(2):
            inl, _ = self._inliers(p, mid, d, length)
            if inl.sum() < 2:
                break
            n = np.stack([-d[inl, 1], d[inl, 0]], axis=1)       # normals
            wgt = length[inl]
            A = (n[:, :, None] * n[:, None, :] * wgt[:, None, None]).sum(0)
            rhs = (n * ((n * mid[inl]).sum(1) * wgt)[:, None]).sum(0)
            A += np.eye(2) * 1e-3
            try:
                p = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                break
        return p

    def _has_both_sides(self, inl, mid, d):
        """Need lines leaning left AND right (a real converging pair)."""
        slope_sign = np.sign(d[inl, 0] * d[inl, 1])
        return (slope_sign > 0).any() and (slope_sign < 0).any()

    # ------------------------------------------------------------------
    def _ransac(self, segs):
        """Vectorised RANSAC: all hypotheses are scored against all segments at once."""
        cfg = self.cfg
        n = len(segs)
        if n < 2:
            return None, 0.0
        mid, d, length = seg_geometry(segs)
        need = cfg.vp_min_support_frac * self.h

        prob = length / length.sum()
        i = self.rng.choice(n, cfg.vp_iters, p=prob)
        j = self.rng.choice(n, cfg.vp_iters, p=prob)
        cross = d[i, 0] * d[j, 1] - d[i, 1] * d[j, 0]
        good = (i != j) & (np.abs(cross) > np.sin(np.radians(cfg.vp_min_pair_angle_deg)))
        i, j, cross = i[good], j[good], cross[good]
        if len(i):
            dm = mid[j] - mid[i]
            sc = (dm[:, 0] * d[j, 1] - dm[:, 1] * d[j, 0]) / cross
            P = mid[i] + sc[:, None] * d[i]
            lo, hi = cfg.vp_y_range
            P = P[(P[:, 1] >= lo * self.h) & (P[:, 1] <= hi * self.h)
                  & (P[:, 0] >= -self.w) & (P[:, 0] <= 2 * self.w)]
        else:
            P = np.zeros((0, 2))
        if self.vp is not None:
            P = np.vstack([self.vp[None, :], P])
        if len(P) == 0:
            return None, 0.0

        V = P[:, None, :] - mid[None, :, :]                       # (M, N, 2)
        dist = np.hypot(V[..., 0], V[..., 1]) + 1e-6
        sin = np.abs(d[None, :, 0] * V[..., 1] - d[None, :, 1] * V[..., 0]) / dist
        inl = (sin < np.sin(np.radians(cfg.vp_angle_tol_deg))) & (dist > 0.05 * self.h)
        score = inl @ length
        pos = (np.sign(d[:, 0] * d[:, 1]) > 0)
        neg = ~pos
        valid = (inl @ pos.astype(np.float32) > 0) & (inl @ neg.astype(np.float32) > 0) & (score >= need)
        if not valid.any():
            return None, float(score.max())
        score = np.where(valid, score, -1)
        k = int(np.argmax(score))
        best = self._refine(P[k].copy(), mid, d, length)
        if not self._plausible(best):
            return None, float(score[k])
        return best, float(score[k])

    # ------------------------------------------------------------------
    def update(self, segs):
        """Feed this frame's segments; returns the (smoothed) vanishing point or None."""
        cand, support = self._ransac(segs)
        cfg = self.cfg
        if cand is None:
            self.missed += 1
            self.confidence *= 0.9
            if self.missed > cfg.vp_max_hold:
                self.vp = None
            return self.vp

        self.missed = 0
        self.confidence = min(1.0, support / (1.5 * self.h))
        if self.vp is None:
            self.vp = cand
            return self.vp

        jump = np.hypot(*(cand - self.vp))
        if jump > cfg.vp_jump_frac * self.h:
            # large jump: accept only if seen repeatedly (rejects one-frame clutter)
            if self._pending is not None and np.hypot(*(cand - self._pending)) < 0.05 * self.h:
                self._pending_n += 1
            else:
                self._pending, self._pending_n = cand, 1
            if self._pending_n >= cfg.vp_jump_confirm:
                self.vp = 0.4 * self.vp + 0.6 * cand
                self._pending = None
            return self.vp

        self._pending = None
        a = cfg.vp_alpha
        self.vp = (1 - a) * self.vp + a * cand
        return self.vp
