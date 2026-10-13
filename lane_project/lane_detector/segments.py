"""Step 2 - turn the marking mask into straight line segments (Canny + probabilistic Hough)."""
import cv2
import numpy as np


def extract_segments(mask, cfg):
    """Return an (N, 4) float array of [x1, y1, x2, y2]; N may be 0."""
    h, w = mask.shape[:2]
    edges = cv2.Canny(mask, cfg.canny_low, cfg.canny_high)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, cfg.hough_threshold,
        minLineLength=max(8, int(cfg.min_seg_len_frac * h)),
        maxLineGap=max(4, int(cfg.max_seg_gap_frac * h)),
    )
    if lines is None:
        return np.zeros((0, 4), np.float32)
    segs = lines.reshape(-1, 4).astype(np.float32)

    dx = segs[:, 2] - segs[:, 0]
    dy = segs[:, 3] - segs[:, 1]
    ang = np.degrees(np.arctan2(np.abs(dy), np.abs(dx)))     # 0 = horizontal, 90 = vertical
    keep = (ang >= cfg.min_seg_angle_deg) & (ang <= cfg.max_seg_angle_deg)
    return segs[keep]


def seg_geometry(segs):
    """Midpoints, unit directions and lengths for an (N, 4) segment array."""
    p1, p2 = segs[:, :2], segs[:, 2:]
    d = p2 - p1
    length = np.hypot(d[:, 0], d[:, 1]) + 1e-6
    d = d / length[:, None]
    mid = (p1 + p2) / 2.0
    return mid, d, length


def angle_error_to_point(mid, d, pt):
    """Angle (radians) between each segment direction and the ray from its midpoint to `pt`."""
    v = pt[None, :] - mid
    dist = np.hypot(v[:, 0], v[:, 1]) + 1e-6
    sin = np.abs(d[:, 0] * v[:, 1] - d[:, 1] * v[:, 0]) / dist
    return np.arcsin(np.clip(sin, 0, 1)), dist
