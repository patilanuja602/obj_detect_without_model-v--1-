"""Synthetic check of the 'camera moves up and down' requirement.

We draw a flat road with painted lines whose vanishing point sits at a chosen horizon row,
then verify the detector finds that horizon (within 5 % of image height) and locks the lane.
Run:  python -m pytest -q      (or)   python tests/test_synthetic.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lane_detector import Config, LaneDetector  # noqa: E402

W, H = 640, 360


def make_frame(horizon_y, vp_x=320, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), (200, 170, 120), np.uint8)               # sky
    img[int(horizon_y):] = (95, 95, 95)                                # asphalt
    img = cv2.add(img, rng.integers(0, 12, img.shape, dtype=np.uint8))  # texture noise
    vp = np.array([vp_x, horizon_y], np.float32)

    def ray(x_bottom, colour, thick, dashed=False):
        bottom = np.array([x_bottom, H - 1], np.float32)
        n = 14 if dashed else 1
        for i in range(n):
            a, b = (i / n, (i + 0.5) / n) if dashed else (0.0, 1.0)
            p1 = bottom + (vp - bottom) * (a * 0.9)
            p2 = bottom + (vp - bottom) * (b * 0.9)
            th = max(1, int(thick * (1 - a * 0.9)))
            cv2.line(img, tuple(map(int, p1)), tuple(map(int, p2)), colour, th, cv2.LINE_AA)

    ray(60, (235, 235, 235), 8)                  # left edge: white solid
    ray(600, (60, 200, 235), 8)                  # right edge: yellow solid
    ray(330, (235, 235, 235), 7, dashed=True)    # an adjacent dashed line (must NOT be chosen)
    return img


def run(horizon_y):
    det = LaneDetector(Config(proc_width=W))
    res = None
    for k in range(25):                           # let temporal filters settle
        res = det.process(make_frame(horizon_y, seed=k))
    return res


def test_horizon_follows_camera_pitch():
    for hy in (0.30 * H, 0.45 * H, 0.60 * H):
        res = run(hy)
        assert res.vp is not None, f"no vanishing point for horizon {hy}"
        assert abs(res.vp[1] - hy) < 0.05 * H, (hy, res.vp)
        assert res.has_lane, f"lane not locked at horizon {hy}"


def test_lane_is_between_outer_lines():
    res = run(0.45 * H)
    vp_x = res.vp[0]
    xl, xr = res.left.x_at(1.0, vp_x), res.right.x_at(1.0, vp_x)
    assert xl < 320 < xr
    assert abs(xl - 60) < 0.06 * W and abs(xr - 600) < 0.06 * W


if __name__ == "__main__":
    test_horizon_follows_camera_pitch()
    test_lane_is_between_outer_lines()
    print("synthetic tests passed")
