"""Step 1 - find painted road markings (white AND yellow/orange) without any ROI.

White paint:  a thin structure that is locally brighter than its surroundings
              (morphological top-hat on the L channel) and nearly colourless.
Yellow paint: hue/saturation test in HSV.
Both tests are *local / relative*, so they adapt to shadows, exposure and camera tilt.
"""
import cv2
import numpy as np


def _odd(n):
    n = int(n)
    return max(3, n if n % 2 else n + 1)


def marking_masks(bgr, cfg):
    """Return (marking_mask, yellow_mask, lab_image), all at the input resolution."""
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = cv2.GaussianBlur(lab[:, :, 0], (0, 0), 1.2)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    chroma = np.hypot(a, b)

    # multi-scale top-hat: thin far-away lines AND wide near-camera lines
    tophat = np.zeros_like(L)
    for s in cfg.tophat_scales:
        k = _odd(s * w)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        tophat = np.maximum(tophat, cv2.morphologyEx(L, cv2.MORPH_TOPHAT, se))

    thr = max(cfg.min_contrast, float(tophat.mean() + cfg.contrast_sigma * tophat.std()))
    bright_enough = L > cfg.white_min_L          # local contrast is already enforced by the top-hat
    white = (tophat > thr) & bright_enough & (chroma < cfg.max_white_chroma)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo, hi = cfg.yellow_hue
    yellow = (
        (hsv[:, :, 0] >= lo) & (hsv[:, :, 0] <= hi)
        & (hsv[:, :, 1] >= cfg.yellow_min_sat) & (hsv[:, :, 2] >= cfg.yellow_min_val)
    )

    mask = (white | yellow).astype(np.uint8) * 255
    yellow = yellow.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mask, yellow, lab
