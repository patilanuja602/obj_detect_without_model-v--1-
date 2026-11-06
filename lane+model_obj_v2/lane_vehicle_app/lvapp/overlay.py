"""
overlay.py -- drawing helpers (display only; nothing here changes lane geometry or detection results).

Vehicle colours (BGR):  current lane = amber, thick | left/right adjacent = blue-orange | outside road = grey, thin (policy dim/hide/show)
                        uncertain = amber, thin    | coasting (carried through a missed frame) = lower contrast.
Plain style (association disabled) reproduces the old look: amber 2-px box and a "car 0.78" label.
"""
import cv2
import numpy as np

AMBER = (0, 200, 255)
ADJ = (255, 170, 40)
GREY = (150, 150, 150)
ZONE_TAG = {"current": "CUR", "left": "L", "right": "R", "outside": "OUT", "uncertain": "?"}


def cut_regions(disp, tracks, min_score=0.45):
    """Save the pixels under reliable vehicles BEFORE the lane overlay is drawn, to restore them afterwards (no lane paint on vehicle bodies)."""
    H, W = disp.shape[:2]
    saved = []
    for t in tracks:
        if t.score < min_score:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in t.box]
        x1, x2, y1, y2 = max(x1, 0), min(x2, W), max(y1, 31), min(y2, H)           # keep the HUD bar intact
        if x2 > x1 and y2 > y1:
            saved.append((x1, y1, x2, y2, disp[y1:y2, x1:x2].copy()))
    return saved


def restore_regions(disp, saved):
    for x1, y1, x2, y2, px in saved:
        disp[y1:y2, x1:x2] = px


def draw_vehicles(img, tracks, style="zones", outside_policy="dim"):
    H, W = img.shape[:2]
    for t in tracks:
        x1, y1, x2, y2 = [int(round(v)) for v in t.box]
        zone = t.zone if style == "zones" else "uncertain"
        if style == "zones" and zone == "outside" and outside_policy == "hide":
            continue
        if style == "plain":
            col, th = AMBER, 2
        elif zone == "current":
            col, th = AMBER, 2
        elif zone in ("left", "right"):
            col, th = ADJ, 2
        elif zone == "outside":
            col, th = (GREY, 1) if outside_policy == "dim" else (AMBER, 2)
        else:
            col, th = AMBER, 1
        if style == "zones" and t.coasting:
            col = tuple(int(c * 0.6) for c in col)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, th)
        lab = "%s %.2f" % (t.name, t.score)
        if style == "zones":
            lab += " " + ZONE_TAG[zone]
            gx, gy = t.ground
            cv2.circle(img, (int(gx), int(gy)), 3, col, -1)                          # ground-contact point
        (tw, thh), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(y1, thh + 34)
        cv2.rectangle(img, (x1, y - thh - 4), (min(x1 + tw + 2, W - 1), y), col, -1)
        cv2.putText(img, lab, (x1 + 1, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def hud_line(img, text):
    """Second HUD line below the lane detector's own HUD (rows 31-52)."""
    img[31:52, :] = (img[31:52, :] * 0.25).astype(np.uint8)
    cv2.putText(img, text, (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img
