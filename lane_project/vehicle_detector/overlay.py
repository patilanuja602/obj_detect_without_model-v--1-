import cv2

COL = {"FRONT": (0, 0, 255), "LEFT": (255, 160, 0), "RIGHT": (0, 200, 255)}


def draw_vehicles(img, vehicles, scale=1.0):
    for v in vehicles:
        x1, y1, x2, y2 = [int(c * scale) for c in v.box]
        col = COL.get(v.side, (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, max(2, int(2 * scale)))
        label = v.side + (f" ~{v.dist_m:.0f}m" if v.dist_m else "")
        fs = 0.5 * max(scale, 0.8)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), col, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)
    return img
