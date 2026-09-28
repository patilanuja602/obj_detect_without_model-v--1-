"""
Ground-contact-point -> real-world distance/angle, carried over from the
earlier main.py, with the camera height updated to 115 cm as specified.

IMPORTANT: FX, FY, CX, CY, DIST_COEFFS below are the intrinsics that were
calibrated for the ORIGINAL camera/mount (see detector_1.py / main.py).
If the camera that actually ships in the chip is a different unit, or the
same unit but reflashed/refocused, these must be recalibrated (checkerboard
calibration) -- the FOCAL LENGTH values are what turn a pixel offset into a
real-world distance, so wrong intrinsics silently produce wrong distances
even though detection itself still works fine.
"""

import math
import numpy as np

# --- camera height above the road surface ---
CAMERA_HEIGHT_M = 1.15   # 115 cm, as specified

# --- intrinsics (carried over from detector_1.py / main.py -- RECALIBRATE
#     if the deployed chip camera differs from the one these came from) ---
FX = 462.5529
FY = 462.24689384
CX = 329.3637
CY = 252.93461492

K = np.array([
    [FX, 0.0, CX],
    [0.0, FY, CY],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

DIST_COEFFS = np.array([
    -0.04617136, -0.00448782, 0.01069043, 0.00344335, 0.01316230
], dtype=np.float64)


def calculate_object_geometry(contact_x, contact_y, frame_height):
    """Forward distance (CB), lateral distance (AB), and horizontal angle
    (theta) from a ground-contact pixel, assuming a flat road and the
    camera height above."""
    horizon_y = frame_height / 2.0
    vertical_pixel_distance = contact_y - horizon_y
    if vertical_pixel_distance <= 0:
        return None, None, None

    side_cb_m = (FY * CAMERA_HEIGHT_M) / vertical_pixel_distance
    horizontal_pixel_offset = contact_x - CX
    angle_rad = math.atan2(horizontal_pixel_offset, FX)
    angle_deg = math.degrees(angle_rad)
    side_ab_m = side_cb_m * math.tan(angle_rad)

    return side_ab_m, side_cb_m, angle_deg
