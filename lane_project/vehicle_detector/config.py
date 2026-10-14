"""All tunable numbers for the vehicle detector."""
from dataclasses import dataclass


@dataclass
class VConfig:
    # ---- where to look (derived from the lane lines) ---------------------
    side_lanes: int = 3              # how many lane-widths of road to search on each side of the ego lane
    max_distance_m: float = 45.0     # ignore anything farther than this (approx, from lane width)
    hood_frac: float = 0.93          # rows below this fraction of the height are the car's own bonnet
    hfov_deg: float = 70.0           # only used to turn lane width into a rough distance
    lane_width_m: float = 3.7
    road_max_chroma: float = 18.0    # a side lane is 'road' if its surface is this neutral grey (LAB a,b distance)
    slot_alpha: float = 0.15         # smoothing of that decision over time

    # ---- "not asphalt" test ------------------------------------------------
    min_dL: float = 26.0             # lightness difference to the asphalt of the same row (LAB-L units, 0..255)
    sigma_dL: float = 4.0            # ...or this many robust std-devs, whichever is larger
    min_dC: float = 22.0             # colour difference (a,b plane)
    road_blur: int = 3
    paint_thick_ratio: float = 0.10  # lane paint is thinner than this fraction of the lane width

    # ---- blob -> vehicle checks (all relative to the local lane width) -----
    veh_h_ratio: float = 0.5         # vehicle height as a fraction of the lane width (cars ~0.4, trucks more)
    min_w_ratio: float = 0.12        # box width / lane width at that row
    max_w_ratio: float = 1.9
    min_h_ratio: float = 0.09
    max_h_ratio: float = 1.30
    min_aspect: float = 0.45         # width / height
    max_aspect: float = 3.8
    min_fill: float = 0.38           # blob area / box area
    min_px_w: int = 12
    min_px_h: int = 9
    shadow_drop: float = 12.0        # contact band must be this much darker than asphalt (L), or the body is dark
    min_edge_density: float = 0.065   # vehicles are textured (edges, windows, lights)

    # ---- tracking ----------------------------------------------------------
    confirm_hits: int = 3
    max_missed: int = 6
    iou_match: float = 0.15
    smooth: float = 0.5
    roof_min_ratio: float = 0.25     # roof search window above the contact line, in lane widths
    roof_max_ratio: float = 0.75
    roof_prior_ratio: float = 0.42   # typical car height / lane width, used when no roof edge is clear
    roof_edge_min: float = 9.0       # horizontal-edge strength (mean |dL/dy|) that counts as a roof line
    wall_u_range: float = 0.22       # base spans less than this many lane-widths sideways while changing rows = wall/divider
    wall_min_rows: int = 8
    strip_open: int = 9              # horizontal strips thinner than this many rows are not vehicles
    horizon_glitch: float = 0.06     # horizon this far (x height) from its recent median = lane-code glitch
    hold_frames: int = 20            # keep boxes this long when the lane geometry is unusable
    split_min_ratio: float = 0.95    # try to split blobs wider than this many lane-widths
    split_part_ratio: float = 0.28   # each part at least this wide
    split_min_step: float = 60.0     # brightness step (L units) required to split
    min_grad_x: float = 2.2          # mean |dL/dx| inside the box; walls and road seams are smoother than this
    max_width_frac: float = 0.6      # no vehicle box is wider than this fraction of the image
