"""
association.py -- decide where each tracked vehicle is relative to the detected road:
    current | left | right | outside | uncertain

Uses the lane detector's geometry of the SAME frame (boundary point lists in processing pixels, boundary types and reliability flags) and the
vehicle's ground-contact point (bottom-centre of its box).  Nothing is forced: when the lane geometry is not reliable the answer is "uncertain".

  r = (x_ground - x_left(y)) / (x_right(y) - x_left(y))      evaluated at the vehicle's ground row; 0..1 = inside the ego corridor.
  current   : -0.05 <= r <= 1.05
  left      : left boundary is a painted marking and  -adjacent_range <= r < -0.05      (a lane marking implies a lane beyond it)
  right     : right boundary is a painted marking and  1.05 < r <= 1+adjacent_range
  outside   : beyond a ROAD_EDGE boundary (barrier / kerb) on that side, or farther than outside_range lane widths from the corridor
  uncertain : lane state not LANE, a boundary unusable / contradicted / an outlier / estimated, the lane is flagged `wide`, the vehicle is
              farther above the observed corridor than max_extrap_px (extrapolation not trusted), or the position is in a grey zone.
Boundaries are extrapolated above their observed extent along their local tangent (lines converge towards the vanishing point).
A zone only changes after `zone_persist` consecutive agreeing frames, so labels do not flicker.
"""
import numpy as np


def _curve(pts):
    """x(y) callable from boundary points (x, y) in processing px; quadratic fit, tangent extrapolation above the observed top."""
    pts = np.asarray(pts, float)
    if pts is None or len(pts) < 4:
        return None
    ys, xs = pts[:, 1], pts[:, 0]
    deg = 2 if len(pts) >= 8 else 1
    c = np.polyfit(ys, xs, deg)
    ytop = float(ys.min())
    slope = float(np.polyval(np.polyder(c), ytop))
    slope = max(-3.0, min(3.0, slope))
    xtop = float(np.polyval(c, ytop))

    def f(y):
        return float(np.polyval(c, y)) if y >= ytop else xtop + slope * (y - ytop)
    f.ytop = ytop
    return f


def snapshot(lane):
    """Copy of what association needs (the lane detector's result dict is reused across frames)."""
    def side(b):
        return None if b is None else dict(type=b.get("type"), usable=bool(b.get("usable")), outlier=bool(b.get("outlier")), contradicted=bool(b.get("contradicted")), source=b.get("source"))
    return dict(state=lane["state"], wide=bool(lane.get("wide")), L=side(lane.get("L")), R=side(lane.get("R")),
                pts_L=None if lane.get("pts_L") is None else np.array(lane["pts_L"]), pts_R=None if lane.get("pts_R") is None else np.array(lane["pts_R"]))


class RoadAssociator:
    def __init__(self, zone_persist=2, max_extrap_px=80, adjacent_range=1.15, outside_range=2.3, **_):
        self.persist, self.max_extrap, self.adj, self.out = zone_persist, max_extrap_px, adjacent_range, outside_range

    def classify_point(self, snap, gx_proc, gy_proc):
        """-> (zone, r)"""
        if snap is None or snap["state"] != "LANE" or snap["wide"] or snap["L"] is None or snap["R"] is None:
            return "uncertain", None
        for b in (snap["L"], snap["R"]):
            if not b["usable"] or b["contradicted"] or b["outlier"] or b["type"] in ("ESTIMATED_LANE_BOUNDARY", "UNKNOWN"):
                return "uncertain", None
        fl, fr = _curve(snap["pts_L"]), _curve(snap["pts_R"])
        if fl is None or fr is None:
            return "uncertain", None
        xl, xr = fl(gy_proc), fr(gy_proc)
        w = xr - xl
        if w < 8:
            return "uncertain", None
        r = (gx_proc - xl) / w
        extrap = max(0.0, max(fl.ytop, fr.ytop) - gy_proc)
        l_edge, r_edge = snap["L"]["type"] == "ROAD_EDGE", snap["R"]["type"] == "ROAD_EDGE"
        if extrap > self.max_extrap and -0.5 < r < 1.5:
            return "uncertain", r
        if -0.05 <= r <= 1.05:
            return "current", r
        if r < -0.05:
            if l_edge:
                return "outside", r
            if r >= -self.adj:
                return "left", r
            return ("outside" if r < -self.out else "uncertain"), r
        if r_edge:
            return "outside", r
        if r <= 1 + self.adj:
            return "right", r
        return ("outside" if r > 1 + self.out else "uncertain"), r

    def update(self, tracks, snap, k):
        """k = display_px / processing_px.  Sets track.zone / track.zone_r with persistence."""
        for t in tracks:
            gx, gy = t.ground
            zone, r = self.classify_point(snap, gx / k, gy / k)
            t.zone_r = r
            h = t.zone_hist
            h.append(zone)
            del h[:-self.persist]
            if t.zone == "uncertain" and t.first_fid == t.last_fid and len(h) == 1:
                t.zone = zone                                        # first sighting: take it
            elif len(h) >= self.persist and all(z == h[-1] for z in h):
                t.zone = h[-1]
        return tracks
