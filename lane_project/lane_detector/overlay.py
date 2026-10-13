"""Drawing helpers (kept separate so the detector never depends on the GUI)."""
import cv2
import numpy as np

from .lanes import t_of_y

LEFT_COL = (255, 120, 0)       # BGR  blue-ish
RIGHT_COL = (60, 60, 255)      # red-ish
FILL_COL = (0, 200, 0)
PATH_COL = (0, 255, 255)


def _polyline(b, vp, y_top, y_bot, h, scale):
    ys = np.linspace(y_top, y_bot, 28)
    t = t_of_y(ys, vp[1], h - 1.0)
    xs = b.x_at(t, vp[0])
    return np.stack([xs, ys], 1) * scale


def draw(res, out_width=None, debug=False, fps=None):
    frame = res.frame
    h, w = frame.shape[:2]
    scale = 1.0 if not out_width else out_width / w
    img = frame if scale == 1.0 else cv2.resize(frame, None, fx=scale, fy=scale)
    img = img.copy()
    ov = img.copy()
    t = max(2, int(3 * scale))

    if res.vp is not None:
        L, R, vp = res.left, res.right, res.vp
        if res.has_lane:
            pl = _polyline(L, vp, res.y_top, res.y_bot, h, scale)
            pr = _polyline(R, vp, res.y_top, res.y_bot, h, scale)
            poly = np.concatenate([pl, pr[::-1]]).astype(np.int32)
            cv2.fillPoly(ov, [poly], FILL_COL)
            # centre path (where the car should drive)
            pc = (pl + pr) / 2
            cv2.polylines(ov, [pc.astype(np.int32)], False, PATH_COL, t, cv2.LINE_AA)
        img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
        for b, col in ((L, LEFT_COL), (R, RIGHT_COL)):
            if b is not None:
                p = _polyline(b, vp, res.y_top, res.y_bot, h, scale)
                cv2.polylines(img, [p.astype(np.int32)], False, col, t + 2, cv2.LINE_AA)
        if debug:
            hy = int(vp[1] * scale)
            cv2.line(img, (0, hy), (img.shape[1], hy), (255, 0, 255), 1)
            cv2.circle(img, (int(vp[0] * scale), hy), 6, (255, 0, 255), -1)
            for x1, y1, x2, y2 in res.debug.get("segs", []):
                cv2.line(img, (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale)), (0, 165, 255), 1)

    _hud(img, res, fps, scale)
    if debug and "mask" in res.debug:
        m = cv2.resize(res.debug["mask"], (img.shape[1] // 4, img.shape[0] // 4))
        img[-m.shape[0]:, -m.shape[1]:] = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    return img


def _hud(img, res, fps, scale):
    lines = []
    if res.has_lane:
        lines.append(f"Lane: LOCKED ({res.curve})")
        lines.append(f"Offset: {res.offset_m:+.2f} m   Steer hint: {res.steer_deg:+.1f} deg")
    elif res.left is not None or res.right is not None:
        lines.append("Lane: one boundary only")
    else:
        lines.append("Lane: none detected")
    for tag, b in (("L", res.left), ("R", res.right)):
        if b is not None:
            kind = "solid" if b.solid > 0.75 else "dashed"
            lines.append(f"{tag}: {b.colour} {kind}")
    if res.vp is not None:
        lines.append(f"Horizon y: {res.vp[1] / res.frame.shape[0] * 100:.0f}% of height")
    if fps is not None:
        lines.append(f"{fps:.0f} FPS")
    fs = 0.55 * max(scale, 0.8) * (img.shape[1] / 960) ** 0.3
    y = int(24 * fs * 1.6)
    box_w = int(330 * fs * 1.2)
    cv2.rectangle(img, (6, 6), (6 + box_w, 12 + int(len(lines) * 24 * fs * 1.5)), (0, 0, 0), -1)
    for s in lines:
        cv2.putText(img, s, (14, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
        y += int(24 * fs * 1.5)
