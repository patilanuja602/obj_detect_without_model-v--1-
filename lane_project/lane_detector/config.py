"""All tunable numbers live here so the other modules contain no magic values."""
from dataclasses import dataclass


@dataclass
class Config:
    # ---- input ---------------------------------------------------------
    proc_width: int = 640            # frames are resized to this width for processing

    # ---- Step 1: painted-marking mask (markings.py) --------------------
    tophat_scales: tuple = (0.02, 0.07)   # structuring-element size as fraction of width
    contrast_sigma: float = 1.0           # adaptive threshold = mean + sigma * std of top-hat
    min_contrast: float = 20.0            # ...but never lower than this (L units)
    white_min_L: int = 95                 # paint is at least this bright in absolute terms
    max_white_chroma: float = 28.0        # white paint is nearly colourless (LAB chroma)
    yellow_hue: tuple = (5, 35)           # OpenCV hue range for yellow / orange paint
    yellow_min_sat: int = 60
    yellow_min_val: int = 95

    # ---- Step 2: line segments (segments.py) ---------------------------
    canny_low: int = 50
    canny_high: int = 150
    hough_threshold: int = 22
    min_seg_len_frac: float = 0.035       # fraction of frame height
    max_seg_gap_frac: float = 0.025
    min_seg_angle_deg: float = 18.0       # reject near-horizontal segments
    max_seg_angle_deg: float = 89.0

    # ---- Step 3: vanishing point / horizon (vanishing.py) --------------
    vp_iters: int = 250
    vp_angle_tol_deg: float = 2.0
    vp_min_pair_angle_deg: float = 10.0
    vp_min_support_frac: float = 0.25     # of frame height, summed segment length
    vp_alpha: float = 0.35                # temporal smoothing
    vp_jump_frac: float = 0.10            # a jump bigger than this (of h) needs confirmation
    vp_jump_confirm: int = 2
    vp_max_hold: int = 20                 # frames to keep last VP when none found
    vp_y_range: tuple = (-0.3, 0.85)      # allowed horizon position (fraction of h)

    # ---- Step 4: lane boundaries (lanes.py) ----------------------------
    seg_vp_angle_tol_deg: float = 3.0
    near_vp_frac: float = 0.07            # ignore segments this close to horizon
    cluster_tol_frac: float = 0.04        # merge intercepts closer than this (of w)
    curv_ridge: float = 0.15
    max_curv_frac: float = 0.20
    car_x_frac: float = 0.5               # car centre-line across the image (0..1); see --calibrate
    car_margin_frac: float = 0.03
    min_lane_w_frac: float = 0.12
    max_lane_w_frac: float = 1.7
    road_fraction_min: float = 0.50
    max_candidates_per_side: int = 4

    # ---- Step 5: tracking ----------------------------------------------
    track_alpha: float = 0.40
    track_max_hold: int = 12
    track_jump_frac: float = 0.08
    track_jump_confirm: int = 3

    # ---- Step 6: vehicle metrics ---------------------------------------
    lane_width_m: float = 3.7
    hfov_deg: float = 70.0
    lookahead_m: float = 12.0
