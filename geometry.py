"""
Ground-contact pixel -> real-world distance + angle.

Camera calibration and geometry taken from main.py (the live-camera system):
  - Forward distance  (side CB) = (fy * H) / (v - horizon_y)
  - Horizontal angle  (theta)   = atan((u - cx) / fx)
  - Lateral distance  (side AB) = CB * tan(theta)

Detection is NOT touched. The detector runs on the raw frame exactly as
before; only the single ground-contact point of each confirmed track is
lens-undistorted here (cv2.undistortPoints) before the maths is applied.
"""

import math

import cv2
import numpy as np

# ------------------------------------------------------------------
# Calibration (640x480 capture) -- values copied from main.py / detector 1.py
# ------------------------------------------------------------------
CALIB_WIDTH = 640
CALIB_HEIGHT = 480

FX = 462.5529
FY = 462.24689384
CX = 329.3637
CY = 252.93461492

# Camera height above the ground/table plane, in metres (from main.py)
CAMERA_HEIGHT_M = 0.30

# OpenCV order: k1, k2, p1, p2, k3
DIST_COEFFS = np.array(
    [-0.04617136, -0.00448782, 0.01069043, 0.00344335, 0.01316230],
    dtype=np.float64,
)


class GroundGeometry:
    """Converts a ground-contact pixel (u, v) into (lateral_m, forward_m, angle_deg)."""

    def __init__(self, frame_w=CALIB_WIDTH, frame_h=CALIB_HEIGHT,
                 camera_height_m=CAMERA_HEIGHT_M):
        # If the video is not 640x480, scale the intrinsics to match.
        sx = frame_w / float(CALIB_WIDTH)
        sy = frame_h / float(CALIB_HEIGHT)
        self.fx, self.cx = FX * sx, CX * sx
        self.fy, self.cy = FY * sy, CY * sy
        self.frame_h = frame_h
        self.camera_height_m = camera_height_m
        self.K = np.array([[self.fx, 0.0, self.cx],
                           [0.0, self.fy, self.cy],
                           [0.0, 0.0, 1.0]], dtype=np.float64)

    def _undistort_point(self, u, v):
        pt = np.array([[[u, v]]], dtype=np.float64)
        out = cv2.undistortPoints(pt, self.K, DIST_COEFFS, P=self.K)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def locate(self, contact_x, contact_y):
        """Returns (lateral_m, forward_m, angle_deg), or None if the point is
        at/above the horizon (no ground intersection)."""
        u, v = self._undistort_point(contact_x, contact_y)

        horizon_y = self.frame_h / 2.0            # same as main.py
        vertical_pixel_distance = v - horizon_y
        if vertical_pixel_distance <= 0:
            return None

        forward_m = (self.fy * self.camera_height_m) / vertical_pixel_distance
        angle_rad = math.atan2(u - self.cx, self.fx)
        lateral_m = forward_m * math.tan(angle_rad)
        return lateral_m, forward_m, math.degrees(angle_rad)
