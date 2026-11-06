"""
overlay.py -- drawing.  Lane overlay = the baseline's draw() (unchanged), followed by

  * occlusion restore: the pixels of every visible vehicle's BODY are put back from the un-annotated display frame, so the green
    fill / boundary lines are never painted over a vehicle.  Only the DISPLAY is affected; lane["pts_*"], corridor and extent_y keep
    the full geometry (result["lane"] is untouched);
  * vehicle boxes, coloured by lane zone, with class, confidence, track id and zone, plus the ground-contact dot.
"""
import cv2
import numpy as np

from .association import SHORT
from .vehicle_detector import NAMES

ZONE_COLOR = {"CURRENT": (0, 0, 255), "LEFT_ADJ": (255, 170, 0), "RIGHT_ADJ": (0, 200, 255), "OUTSIDE": (150, 150, 150), "UNCERTAIN": (255, 0, 200)}


def snapshot_bodies(disp, tracks, scale, inset_x, bottom_trim):
    """Copy the body pixels of the visible tracks BEFORE the lane overlay is drawn.  scale = display px / original px."""
    H, W = disp.shape[:2]
    out = []
    for t in tracks:
        x1, y1, x2, y2 = t.box * scale
        bw, bh = x2 - x1, y2 - y1
        a, b = int(max(x1 + inset_x * bw, 0)), int(min(x2 - inset_x * bw, W))
        c, d = int(max(y1, 54)), int(min(y2 - bottom_trim * bh, H))       # rows 0..53 belong to the HUD
        if b > a and d > c:
            out.append((a, c, b, d, disp[c:d, a:b].copy()))
    return out


def restore_bodies(disp, patches):
    for a, c, b, d, p in patches:
        disp[c:d, a:b] = p


def draw_vehicles(disp, tracks, scale, show_gcp=True):
    for t in tracks:
        x1, y1, x2, y2 = (t.box * scale).tolist()
        col = ZONE_COLOR.get(t.zone, (255, 255, 255))
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        if t.coasted:                                       # not detected in this frame: dashed
            for x in range(p1[0], p2[0], 10):
                cv2.line(disp, (x, p1[1]), (min(x + 5, p2[0]), p1[1]), col, 1)
                cv2.line(disp, (x, p2[1]), (min(x + 5, p2[0]), p2[1]), col, 1)
            for y in range(p1[1], p2[1], 10):
                cv2.line(disp, (p1[0], y), (p1[0], min(y + 5, p2[1])), col, 1)
                cv2.line(disp, (p2[0], y), (p2[0], min(y + 5, p2[1])), col, 1)
        else:
            cv2.rectangle(disp, p1, p2, col, 2)
        lab = "%s %.2f #%d %s" % (NAMES.get(t.cls, "?"), t.score, t.id, SHORT.get(t.zone, "?"))
        (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y = max(p1[1], th + 58)
        cv2.rectangle(disp, (p1[0], y - th - 4), (p1[0] + tw + 2, y), col, -1)
        cv2.putText(disp, lab, (p1[0] + 1, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if show_gcp:
            cv2.circle(disp, (int((x1 + x2) / 2), int(y2)), 3, (255, 255, 255), -1)
            cv2.circle(disp, (int((x1 + x2) / 2), int(y2)), 4, (0, 0, 0), 1)
    return disp


def draw_hud(disp, text):
    """Second HUD line (the baseline draws its own lane line in rows 0-30)."""
    cv2.rectangle(disp, (0, 30), (disp.shape[1], 52), (0, 0, 0), -1)
    cv2.putText(disp, text, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return disp
