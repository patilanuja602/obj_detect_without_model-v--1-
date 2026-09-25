import cv2
import numpy as np
import sys
from detector import RoadObjectDetector, Params

# ---------------------------------------------------------------
# STEP 1: Let the user draw a Region of Interest polygon on the
#         first frame. Click points to form the polygon, press
#         ENTER to confirm, or 'r' to reset and redraw.
# ---------------------------------------------------------------
roi_points = []
drawing_done = False


def mouse_callback(event, x, y, flags, param):
    global roi_points
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))


def select_roi(frame):
    global roi_points, drawing_done
    roi_points = []
    clone = frame.copy()
    cv2.namedWindow("Draw ROI - click points, ENTER to confirm, R to reset")
    cv2.setMouseCallback("Draw ROI - click points, ENTER to confirm, R to reset", mouse_callback)

    print("\n=== ROI SELECTION ===")
    print("  Click on the video to place polygon points.")
    print("  Press ENTER or SPACE to confirm the ROI.")
    print("  Press 'r' to reset and start over.")
    print("  Press 'q' to quit.\n")

    while True:
        display = clone.copy()

        # Draw the points and lines
        if len(roi_points) > 0:
            for i, pt in enumerate(roi_points):
                cv2.circle(display, pt, 5, (0, 255, 0), -1)
                cv2.putText(display, str(i + 1), (pt[0] + 8, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            for i in range(1, len(roi_points)):
                cv2.line(display, roi_points[i - 1], roi_points[i], (0, 255, 0), 2)
            # Close the polygon visually
            if len(roi_points) > 2:
                cv2.line(display, roi_points[-1], roi_points[0], (0, 255, 0), 2)

        cv2.putText(display, "Click to add points | ENTER=confirm | R=reset | Q=quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imshow("Draw ROI - click points, ENTER to confirm, R to reset", display)

        key = cv2.waitKey(30) & 0xFF
        if key == 13 or key == 32:  # ENTER or SPACE
            if len(roi_points) >= 3:
                break
            else:
                print("  Need at least 3 points to form a polygon!")
        elif key == ord('r'):
            roi_points = []
            print("  ROI reset. Click new points.")
        elif key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyWindow("Draw ROI - click points, ENTER to confirm, R to reset")
    return np.array(roi_points, dtype=np.int32)


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
if len(sys.argv) < 2:
    print("Usage: python3 play_video.py <video_file>")
    sys.exit(1)

video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    sys.exit(1)

ret, frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    sys.exit(1)

# Let user draw ROI on the first frame
roi_poly = select_roi(frame)
print(f"ROI confirmed with {len(roi_poly)} points: {roi_poly.tolist()}")

# Build a binary mask from the ROI polygon
h, w = frame.shape[:2]
roi_mask = np.zeros((h, w), dtype=np.uint8)
cv2.fillPoly(roi_mask, [roi_poly], 255)

# Initialize the detector
p = Params()
detector = RoadObjectDetector(frame.shape, p)

print(f"Starting live detection for {video_path}... Press 'q' to quit.")

while True:
    confirmed, raw_dets, inter, flow_viz = detector.process(frame)

    vis = frame.copy()

    # Draw the ROI polygon outline
    cv2.polylines(vis, [roi_poly], isClosed=True, color=(255, 255, 0), thickness=2)

    for tr in confirmed:
        x1, y1, x2, y2 = [int(v) for v in tr.box]
        cx, cy = [int(v) for v in tr.contact]

        # Only show objects whose ground-contact point is INSIDE the ROI
        if cv2.pointPolygonTest(roi_poly.astype(np.float32), (float(cx), float(cy)), False) >= 0:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(vis, f"id{tr.id}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Live Detection", vis)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()
    if not ret:
        break

cap.release()
cv2.destroyAllWindows()
