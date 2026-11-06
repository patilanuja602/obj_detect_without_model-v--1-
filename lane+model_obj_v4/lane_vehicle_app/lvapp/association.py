"""
association.py -- vehicle-to-road association from the V6 lane geometry.

Zones: CURRENT, LEFT_ADJ, RIGHT_ADJ, OUTSIDE, UNCERTAIN.

The ground-contact point (bottom centre of the vehicle box, original-frame px) is converted to the lane detector's processing
frame and compared with the displayed corridor (lane["corridor"]: ys descending from the ROI bottom to the far end, xl / xr per row)
at the vehicle's own image row.  A classification is only made when the geometry supports it:

  * lane state must be LANE (a two-boundary corridor exists);
  * the vehicle's ground row must lie inside the corridor's valid extent (never extrapolated beyond the far end; rows below the ROI
    bottom use the bottom row's x values);
  * the corridor must not be flagged `wide` (it probably spans several lanes -> "current lane" is meaningless);
  * ADJACENT-lane claims need the boundary on that side to be OBSERVED or TRACKED (an INFERRED boundary is only an ROI-edge guess);
  * a vehicle straddling a boundary (less than overlap_current of its bottom edge inside the corridor, but not mostly in the
    neighbour either) stays UNCERTAIN -- this is also what a lane change looks like.
OUTSIDE means "more than adjacent_width_mult x lane width beyond a boundary": the road beyond one neighbouring lane, a barrier-side
structure, or a parallel road.  Only the tracker's confirmation rule uses it (stricter score/hits); nothing is deleted because of it.
"""
import numpy as np

CURRENT, LEFT_ADJ, RIGHT_ADJ, OUTSIDE, UNCERTAIN = "CURRENT", "LEFT_ADJ", "RIGHT_ADJ", "OUTSIDE", "UNCERTAIN"
SHORT = {CURRENT: "CUR", LEFT_ADJ: "L-ADJ", RIGHT_ADJ: "R-ADJ", OUTSIDE: "OUT", UNCERTAIN: "?"}


def associate(tracks, lane, scale, cfg):
    """tracks: list[Track] (original-frame px).  lane: PathDetector result["lane"].  scale: processing px / original px.
    Sets track.zone and track.zone_info in place."""
    if not cfg["enabled"] or not tracks:
        for t in tracks:
            t.zone, t.zone_info = UNCERTAIN, "association off"
        return
    cr = lane.get("corridor") if lane else None
    if not lane or lane.get("state") != "LANE" or cr is None:
        for t in tracks:
            t.zone, t.zone_info = UNCERTAIN, "no lane corridor"
        return
    ys, xl, xr = np.asarray(cr["ys"], float), np.asarray(cr["xl"], float), np.asarray(cr["xr"], float)
    yb, yfar = ys[0], ys[-1]                                   # ys runs from the ROI bottom (large y) up to the far end (small y)
    wide = bool(lane.get("wide"))
    src_l = lane["L"]["source"] if lane.get("L") else None
    src_r = lane["R"]["source"] if lane.get("R") else None
    ok_l, ok_r = src_l in ("OBSERVED", "TRACKED"), src_r in ("OBSERVED", "TRACKED")
    ysr, xlr, xrr = ys[::-1], xl[::-1], xr[::-1]               # np.interp needs ascending x
    for t in tracks:
        x1, y1, x2, y2 = (t.box * scale).tolist()              # processing-frame px
        gx, gy = (x1 + x2) * 0.5, y2
        if gy < yfar - cfg["far_margin_px"]:
            t.zone, t.zone_info = UNCERTAIN, "beyond reliable lane extent"
            continue
        if wide:
            t.zone, t.zone_info = UNCERTAIN, "wide corridor"
            continue
        gyc = min(max(gy, ysr[0]), ysr[-1])                    # below ROI bottom -> bottom row values; above far end within margin -> far row
        L, R = float(np.interp(gyc, ysr, xlr)), float(np.interp(gyc, ysr, xrr))
        wd = R - L
        if wd < 8:
            t.zone, t.zone_info = UNCERTAIN, "degenerate width"
            continue
        bw = max(x2 - x1, 1.0)
        inside = max(0.0, min(x2, R) - max(x1, L)) / bw
        left_part = max(0.0, min(x2, L) - x1) / bw
        right_part = max(0.0, x2 - max(x1, R)) / bw
        if inside >= cfg["overlap_current"] and L <= gx <= R:
            t.zone, t.zone_info = CURRENT, "in %.0f%%" % (100 * inside)
        elif gx < L:
            if gx >= L - cfg["adjacent_width_mult"] * wd:
                if left_part >= cfg["overlap_adjacent"] and ok_l:
                    t.zone, t.zone_info = LEFT_ADJ, "left %.0f%%" % (100 * left_part)
                elif not ok_l:
                    t.zone, t.zone_info = UNCERTAIN, "left boundary inferred"
                else:
                    t.zone, t.zone_info = UNCERTAIN, "straddles left boundary"
            else:
                t.zone, t.zone_info = OUTSIDE, "%.1f lane widths left" % ((L - gx) / wd)
        elif gx > R:
            if gx <= R + cfg["adjacent_width_mult"] * wd:
                if right_part >= cfg["overlap_adjacent"] and ok_r:
                    t.zone, t.zone_info = RIGHT_ADJ, "right %.0f%%" % (100 * right_part)
                elif not ok_r:
                    t.zone, t.zone_info = UNCERTAIN, "right boundary inferred"
                else:
                    t.zone, t.zone_info = UNCERTAIN, "straddles right boundary"
            else:
                t.zone, t.zone_info = OUTSIDE, "%.1f lane widths right" % ((gx - R) / wd)
        else:
            t.zone, t.zone_info = UNCERTAIN, "straddles boundary"
