"""Check that the current detector.py gives EXACTLY the same detections as an
older version, frame by frame (boxes, contact points, track ids, sources).

    git show <old_commit>:detector.py > detector_orig.py
    python3 verify_equivalence.py input.mp4 [max_frames]
"""
import sys
import cv2
import detector as new
import detector_orig as old


def main():
    path = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
    cap = cv2.VideoCapture(path)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    a = old.RoadObjectDetector((h, w, 3), old.Params())
    b = new.RoadObjectDetector((h, w, 3), new.Params())
    n = bad = 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        ta, fa, _, _ = a.process(frame)
        tb, fb, _, _ = b.process(frame)
        key = lambda ts: [(t.id, t.box, t.contact, t.hits, t.sources) for t in ts]
        if fa != fb or key(ta) != key(tb):
            bad += 1
            print("MISMATCH at frame", n)
    print(f"{n} frames compared, {bad} mismatching frames ->", "IDENTICAL" if bad == 0 else "DIFFERENT")


if __name__ == "__main__":
    main()
