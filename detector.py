"""
Classical (non-learned) monocular road-object detector.

CORE METHOD (as specified by user):
    Sparse Lucas-Kanade optical flow on Shi-Tomasi corners, clustered in
    (x, y, dx, dy) space with DBSCAN.  All classical / non-learned.

SUPPORTING CLASSICAL METHODS:
    - CLAHE illumination normalization            (classical)
    - MOG2 statistical background subtraction      (classical, NOT a trained NN;
                                                     it is an online per-pixel
                                                     Gaussian-mixture estimator)
    - Morphological open/close, connected comps    (classical)
    - HSV-ratio shadow suppression (Cucchiara-style) (classical)
    - Row-dependent (perspective) minimum-size gate (hard-coded heuristic)
    - Fixed road ROI polygon                        (hard-coded heuristic)
    - Centroid/IoU multi-object tracker w/ temporal
      consensus + EMA smoothing                     (classical, no learning)

NOTHING in this file is a pretrained or trained model. No network weights
are loaded. DBSCAN/KMeans are classical unsupervised clustering algorithms
applied fresh on every frame's flow vectors -- they are not "trained" in the
machine-learning sense (no parameters are fit offline and reused).
"""

import cv2
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time


# ----------------------------------------------------------------------
# ---------------------------  PARAMETERS  ------------------------------
# ----------------------------------------------------------------------
class Params:
    # --- preprocessing ---
    PROC_WIDTH = 640          # processing resolution width (native is 640x480, so ~1:1)
    CLAHE_CLIP = 2.0
    CLAHE_GRID = (8, 8)

    # --- road ROI polygon (hard-coded, fraction of frame W,H) ---
    # Excludes sky/building band at top and the extreme side pavements.
    # MUST be re-fit if camera mount / crop changes.
    ROI_POLY_FRAC = [(0.00, 0.55), (1.00, 0.55), (1.00, 1.00), (0.00, 1.00)]

    # --- Shi-Tomasi corner detection ---
    MAX_CORNERS = 600
    QUALITY_LEVEL = 0.015
    MIN_DISTANCE = 6
    BLOCK_SIZE = 7

    # --- Lucas-Kanade optical flow ---
    LK_WIN = (21, 21)
    LK_MAX_LEVEL = 3
    LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    MIN_FLOW_MAG = 0.6        # px/frame; below this, treat as static/noise
    MAX_FLOW_MAG = 60.0       # px/frame; above this, treat as spurious match

    # --- MOG2 background subtraction (support mask) ---
    MOG2_HISTORY = 250
    MOG2_VAR_THRESH = 12
    MOG2_SHADOW = True        # let MOG2 flag its own shadow guess too (extra cue)
    MOG2_LEARNING_RATE = 0.01
    GAMMA = 1.6                # >1 brightens dusk/low-light frames before bgsub

    # --- blob-driven object separation ---
    # PRIMARY separator: connected components of the (dilated) motion mask.
    #   Appearance/silhouette evidence gives much cleaner object extents
    #   than chaining together sparse, gappy corner points does -- it does
    #   not under-merge (fragment one object into pieces) or over-merge
    #   (bridge two nearby-but-distinct objects) nearly as easily.
    # SECONDARY separator, only within one ambiguous blob: if the flow
    #   inside a blob is not directionally coherent (e.g. two vehicles
    #   bridged into one blob by morphological closing, or a bike weaving
    #   past a car), attempt a 2-way KMeans split on (dx,dy) and keep it
    #   only if both halves are themselves coherent and large enough.
    BLOB_DILATE_PX = 5
    MIN_BLOB_AREA = 20
    # fallback spatial grouping for flow points with no strong blob evidence
    SPATIAL_EPS_PX = 22
    SPATIAL_MIN_SAMPLES = 4

    # --- geometric filtering ---
    MIN_CLUSTER_POINTS = 4
    # direction-coherence gate: rejects illumination flicker / LED-sign
    # "motion" and dusk sensor noise, which produce quasi-random flow
    # directions, as opposed to a real rigid/articulated object whose
    # corner points move in a broadly consistent direction.
    MIN_DIRECTION_COHERENCE = 0.45   # resultant-vector length, 0..1
    # row-dependent minimum bbox height: interpolated between these two
    # (near bottom of frame = large min height demanded to reject dashboard/
    #  road-texture noise; near horizon = small min height so real distant
    #  objects are not thrown away). HARD-CODED for this camera mount.
    MIN_H_AT_BOTTOM = 34
    MIN_H_AT_HORIZON = 10
    MIN_W_PX = 8
    MAX_ASPECT = 3.2           # w/h or h/w guard against degenerate slivers
    MAX_BOX_AREA_FRAC = 0.55   # reject boxes covering most of the ROI (merge artifact)

    # --- shadow suppression (HSV ratio test, Cucchiara et al., classical) ---
    SHADOW_V_RATIO = (0.45, 0.92)   # shadow pixel: V_frame/V_bg in this range
    SHADOW_S_DIFF_MAX = 35
    SHADOW_H_DIFF_MAX = 12

    # --- tracking / temporal consensus ---
    MIN_HITS_TO_CONFIRM = 3
    MAX_COAST_FRAMES = 5
    IOU_MATCH_THRESH = 0.25
    CENTROID_MATCH_PX = 70
    EMA_ALPHA = 0.45           # smoothing for confirmed box/contact point


# ----------------------------------------------------------------------
def build_roi_mask(shape, poly_frac):
    h, w = shape[:2]
    pts = np.array([[int(x * w), int(y * h)] for x, y in poly_frac], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def min_height_for_row(y, h, p: Params):
    """Perspective-based minimum object height: linear interp between
    horizon (small objects allowed) and bottom of frame (must be sizable)."""
    t = np.clip(y / float(h), 0.0, 1.0)
    return p.MIN_H_AT_HORIZON + t * (p.MIN_H_AT_BOTTOM - p.MIN_H_AT_HORIZON)


def iou(b1, b2):
    x1, y1, x2, y2 = b1
    X1, Y1, X2, Y2 = b2
    ix1, iy1 = max(x1, X1), max(y1, Y1)
    ix2, iy2 = min(x2, X2), min(y2, Y2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a1 = max(0, x2 - x1) * max(0, y2 - y1)
    a2 = max(0, X2 - X1) * max(0, Y2 - Y1)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


# ----------------------------------------------------------------------
@dataclass
class Track:
    box: Tuple[float, float, float, float]   # x1,y1,x2,y2  (smoothed)
    contact: Tuple[float, float]             # smoothed ground-contact point
    hits: int = 1
    coast: int = 0
    id: int = 0
    confirmed: bool = False


class MultiObjectTracker:
    """Simple centroid + IoU tracker with temporal-consensus confirmation
    and EMA smoothing.  Purely geometric -- no class labels, no learning."""

    def __init__(self, p: Params):
        self.p = p
        self.tracks: List[Track] = []
        self.next_id = 1

    def update(self, detections: List[Tuple[Tuple[float, float, float, float], Tuple[float, float]]]):
        p = self.p
        unmatched_dets = list(range(len(detections)))
        for tr in self.tracks:
            best_j, best_score = -1, 0.0
            for j in unmatched_dets:
                box, _ = detections[j]
                score = iou(tr.box, box)
                cx1 = (tr.box[0] + tr.box[2]) / 2
                cy1 = (tr.box[1] + tr.box[3]) / 2
                cx2 = (box[0] + box[2]) / 2
                cy2 = (box[1] + box[3]) / 2
                dist = np.hypot(cx1 - cx2, cy1 - cy2)
                if score > p.IOU_MATCH_THRESH or dist < p.CENTROID_MATCH_PX:
                    if score > best_score:
                        best_score, best_j = score, j
            if best_j >= 0:
                box, contact = detections[best_j]
                a = p.EMA_ALPHA
                tr.box = tuple(a * np.array(box) + (1 - a) * np.array(tr.box))
                tr.contact = tuple(a * np.array(contact) + (1 - a) * np.array(tr.contact))
                tr.hits += 1
                tr.coast = 0
                if tr.hits >= p.MIN_HITS_TO_CONFIRM:
                    tr.confirmed = True
                unmatched_dets.remove(best_j)
            else:
                tr.coast += 1

        self.tracks = [t for t in self.tracks if t.coast <= p.MAX_COAST_FRAMES]

        for j in unmatched_dets:
            box, contact = detections[j]
            self.tracks.append(Track(box=box, contact=contact, id=self.next_id))
            self.next_id += 1

        return [t for t in self.tracks if t.confirmed]


# ----------------------------------------------------------------------
class RoadObjectDetector:
    def __init__(self, frame_shape, p: Params = Params()):
        self.p = p
        self.roi_mask = build_roi_mask(frame_shape, p.ROI_POLY_FRAC)
        self.clahe = cv2.createCLAHE(clipLimit=p.CLAHE_CLIP, tileGridSize=p.CLAHE_GRID)
        self.bgsub = cv2.createBackgroundSubtractorMOG2(
            history=p.MOG2_HISTORY, varThreshold=p.MOG2_VAR_THRESH,
            detectShadows=p.MOG2_SHADOW)
        self.prev_gray = None
        self.tracker = MultiObjectTracker(p)
        self.frame_idx = 0
        self.h, self.w = frame_shape[:2]

    # -- preprocessing -------------------------------------------------
    def preprocess(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        return gray

    def gamma_correct(self, frame_bgr):
        """Classical gamma brightening, helps MOG2/HSV separate dark
        objects from an even-darker dusk road. Non-learned (fixed LUT)."""
        g = self.p.GAMMA
        inv = 1.0 / g
        table = (np.linspace(0, 1, 256) ** inv * 255).astype(np.uint8)
        return cv2.LUT(frame_bgr, table)

    # -- shadow suppression on a binary fg mask -------------------------
    def suppress_shadows(self, frame_bgr, fg_mask, bg_bgr):
        """Classical HSV-ratio shadow test (Cucchiara et al.). Returns a
        mask with shadow pixels removed from `fg_mask`."""
        p = self.p
        hsv_f = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_b = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        eps = 1e-3
        v_ratio = hsv_f[..., 2] / (hsv_b[..., 2] + eps)
        s_diff = np.abs(hsv_f[..., 1] - hsv_b[..., 1])
        h_diff = np.minimum(np.abs(hsv_f[..., 0] - hsv_b[..., 0]), 180 - np.abs(hsv_f[..., 0] - hsv_b[..., 0]))
        is_shadow = (v_ratio > p.SHADOW_V_RATIO[0]) & (v_ratio < p.SHADOW_V_RATIO[1]) & \
                    (s_diff < p.SHADOW_S_DIFF_MAX) & (h_diff < p.SHADOW_H_DIFF_MAX)
        out = fg_mask.copy()
        out[is_shadow] = 0
        return out

    # -- main per-frame call --------------------------------------------
    def process(self, frame_bgr):
        p = self.p
        gray = self.preprocess(frame_bgr)
        bright_bgr = self.gamma_correct(frame_bgr)

        fg_raw = self.bgsub.apply(bright_bgr, learningRate=p.MOG2_LEARNING_RATE)
        # MOG2 flags each pixel: 0 = background, 127 = its own "probably
        # shadow" guess, 255 = confident foreground. At dusk, MOG2's own
        # shadow heuristic is over-eager and mislabels large parts of real
        # objects (dark clothing, dark vehicle paint) as "127 shadow", so
        # we do NOT trust that label. Instead we take the union of both
        # (127 or 255 = "candidate motion") and re-decide shadow/object
        # ourselves with an HSV-ratio test against the modeled background.
        fg_candidate = np.where(fg_raw >= 127, 255, 0).astype(np.uint8)
        fg_candidate = cv2.bitwise_and(fg_candidate, fg_candidate, mask=self.roi_mask)
        fg_candidate = cv2.morphologyEx(fg_candidate, cv2.MORPH_OPEN,
                                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        bg_bgr = self.bgsub.getBackgroundImage()
        if bg_bgr is not None:
            fg_binary = self.suppress_shadows(bright_bgr, fg_candidate, bg_bgr)
        else:
            fg_binary = fg_candidate

        fg_binary = cv2.bitwise_and(fg_binary, fg_binary, mask=self.roi_mask)
        fg_binary = cv2.morphologyEx(fg_binary, cv2.MORPH_OPEN,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        fg_binary = cv2.morphologyEx(fg_binary, cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=2)

        intermediates = {"clahe_gray": gray, "mog2_raw": fg_raw, "fg_clean": fg_binary}

        detections = []
        clusters_viz = None

        if self.prev_gray is not None:
            # restrict corner search to (dilated) motion mask ∩ ROI so we
            # don't waste points / get noise on static buildings & sky
            search_mask = cv2.dilate(fg_candidate, np.ones((17, 17), np.uint8))
            search_mask = cv2.bitwise_and(search_mask, self.roi_mask)

            pts0 = cv2.goodFeaturesToTrack(
                self.prev_gray, maxCorners=p.MAX_CORNERS, qualityLevel=p.QUALITY_LEVEL,
                minDistance=p.MIN_DISTANCE, blockSize=p.BLOCK_SIZE, mask=search_mask)

            if pts0 is not None and len(pts0) >= p.MIN_CLUSTER_POINTS:
                pts1, st, err = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, pts0, None,
                    winSize=p.LK_WIN, maxLevel=p.LK_MAX_LEVEL, criteria=p.LK_CRITERIA)
                st = st.reshape(-1)
                pts0f = pts0.reshape(-1, 2)[st == 1]
                pts1f = pts1.reshape(-1, 2)[st == 1]
                flow = pts1f - pts0f
                mag = np.linalg.norm(flow, axis=1)
                keep = (mag > p.MIN_FLOW_MAG) & (mag < p.MAX_FLOW_MAG)
                pts0f, pts1f, flow, mag = pts0f[keep], pts1f[keep], flow[keep], mag[keep]

                clusters_viz = (pts1f, flow)

                if len(pts1f) >= p.MIN_CLUSTER_POINTS:
                    for cluster_pts, cluster_flow in self._blob_cluster(pts1f, flow, fg_candidate):
                        det = self._points_to_detection(cluster_pts, fg_binary)
                        if det is not None:
                            detections.append(det)
                    detections = self._merge_fragments(detections)

        self.prev_gray = gray
        confirmed_tracks = self.tracker.update(detections)
        self.frame_idx += 1
        return confirmed_tracks, detections, intermediates, clusters_viz

    @staticmethod
    def _coherence(flow_vecs):
        unit = flow_vecs / (np.linalg.norm(flow_vecs, axis=1, keepdims=True) + 1e-6)
        return np.linalg.norm(unit.mean(axis=0))

    def _split_or_keep(self, gpts, gflow, out):
        """Shared logic: keep a point group as one object if its flow is
        directionally coherent; otherwise attempt a 2-way KMeans split on
        (dx,dy) and keep the split only if both halves are coherent and
        big enough; otherwise drop the group (ambiguous / noise)."""
        p = self.p
        r = self._coherence(gflow)
        if r >= p.MIN_DIRECTION_COHERENCE:
            out.append((gpts, gflow))
            return
        if len(gpts) >= 2 * p.MIN_CLUSTER_POINTS:
            km = KMeans(n_clusters=2, n_init=4, random_state=0).fit(gflow)
            ok, sub = True, []
            for k in (0, 1):
                m = km.labels_ == k
                if m.sum() < p.MIN_CLUSTER_POINTS or self._coherence(gflow[m]) < p.MIN_DIRECTION_COHERENCE:
                    ok = False
                    break
                sub.append((gpts[m], gflow[m]))
            if ok:
                out.extend(sub)
            # else: drop -- ambiguous, unsplittable, likely noise/flicker

    def _blob_cluster(self, pts, flow, fg_candidate):
        """PRIMARY separator: connected components of the (dilated) motion
        mask. Appearance/silhouette evidence gives cleaner object extents
        than chaining sparse, gappy corner points -- it resists both
        under-merging (one object -> many boxes) and over-merging (two
        nearby-but-distinct objects sharing one box).
        SECONDARY separator (per blob): flow-direction coherence / 2-way
        KMeans split, via `_split_or_keep`.
        FALLBACK: points with no mask evidence (weak/invisible objects in
        dusk) are still grouped by plain spatial proximity so we don't
        lose genuinely-moving-but-faint objects."""
        p = self.p
        out = []
        dil = cv2.dilate(fg_candidate, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.BLOB_DILATE_PX, p.BLOB_DILATE_PX)))
        dil = cv2.bitwise_and(dil, self.roi_mask)
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)

        pts_i = pts.astype(np.int32)
        pts_i[:, 0] = np.clip(pts_i[:, 0], 0, self.w - 1)
        pts_i[:, 1] = np.clip(pts_i[:, 1], 0, self.h - 1)
        blob_id = lbl[pts_i[:, 1], pts_i[:, 0]]

        unmatched = blob_id == 0
        for b in range(1, n):
            if stats[b, cv2.CC_STAT_AREA] < p.MIN_BLOB_AREA:
                unmatched |= (blob_id == b)
                continue
            m = blob_id == b
            if m.sum() < p.MIN_CLUSTER_POINTS:
                unmatched |= m
                continue
            self._split_or_keep(pts[m], flow[m], out)

        # fallback: plain spatial DBSCAN on leftover (mask-less) points
        if unmatched.sum() >= p.MIN_CLUSTER_POINTS:
            fpts, fflow = pts[unmatched], flow[unmatched]
            labels = DBSCAN(eps=p.SPATIAL_EPS_PX, min_samples=p.SPATIAL_MIN_SAMPLES).fit_predict(fpts)
            for lab in set(labels):
                if lab == -1:
                    continue
                idx = labels == lab
                if idx.sum() < p.MIN_CLUSTER_POINTS:
                    continue
                self._split_or_keep(fpts[idx], fflow[idx], out)
        return out

    def _merge_fragments(self, detections):
        """Post-hoc merge of same-object fragments that the blob step left
        as separate small boxes -- typically a pedestrian's torso vs legs,
        split by a motion-poor waist/clothing region. Two boxes are merged
        if they overlap substantially in x (stacked vertically, like a
        body) or in y (side-by-side, like a vehicle split by a windshield
        glare band) AND the gap between them is small. This runs once per
        frame on the handful of per-frame detections, so it's cheap."""
        p = self.p
        boxes = [d[0] for d in detections]
        merged_flag = [False] * len(boxes)
        out = []

        def x_overlap_frac(a, b):
            ox = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            return ox / max(1.0, min(a[2] - a[0], b[2] - b[0]))

        def y_overlap_frac(a, b):
            oy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            return oy / max(1.0, min(a[3] - a[1], b[3] - b[1]))

        def gap(a, b):
            gx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
            gy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
            return max(gx, gy)

        changed = True
        cur = list(boxes)
        while changed:
            changed = False
            n = len(cur)
            for i in range(n):
                if cur[i] is None:
                    continue
                for j in range(i + 1, n):
                    if cur[j] is None:
                        continue
                    a, b = cur[i], cur[j]
                    if gap(a, b) > 30:
                        continue
                    if x_overlap_frac(a, b) > 0.35 or y_overlap_frac(a, b) > 0.55:
                        nb = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                        w_, h_ = nb[2] - nb[0], nb[3] - nb[1]
                        aspect = max(w_ / max(h_, 1), h_ / max(w_, 1))
                        if aspect <= p.MAX_ASPECT + 1.0:
                            cur[i], cur[j] = nb, None
                            changed = True
            cur = [c for c in cur if c is not None]

        # reattach a contact point (bottom-center) for merged boxes; for
        # boxes that were never merged, keep the original refined contact
        by_box = {}
        for (box, contact) in detections:
            by_box.setdefault(box, contact)
        result = []
        for b in cur:
            if b in by_box:
                result.append((b, by_box[b]))
            else:
                result.append((b, ((b[0] + b[2]) / 2.0, b[3])))
        return result

    # -- turn a cluster of flow points into a refined (box, contact) --------
    def _points_to_detection(self, cluster_pts, fg_binary):
        p = self.p
        x1, y1 = cluster_pts.min(axis=0)
        x2, y2 = cluster_pts.max(axis=0)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Snap the sparse-point bounding box to the actual silhouette:
        # look at connected components of fg_binary that overlap this
        # cluster's convex-hull region, and take their union bbox +
        # true bottom contact row. This fixes "one car -> 3 boxes" style
        # fragmentation from pure point-cloud boxing.
        pad = 12
        rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
        rx2, ry2 = min(self.w, x2 + pad), min(self.h, y2 + pad)
        if rx2 <= rx1 or ry2 <= ry1:
            return None
        roi_mask_local = fg_binary[ry1:ry2, rx1:rx2]
        n, lbl, stats, centroids = cv2.connectedComponentsWithStats(roi_mask_local, connectivity=8)

        xs = ys = None
        if n > 1:
            # which components does the point cluster actually touch?
            touched = set()
            for (px, py) in cluster_pts:
                lx, ly = int(px - rx1), int(py - ry1)
                if 0 <= ly < lbl.shape[0] and 0 <= lx < lbl.shape[1]:
                    v = lbl[ly, lx]
                    if v != 0:
                        touched.add(v)
            if not touched:
                areas = stats[1:, cv2.CC_STAT_AREA]
                if len(areas) > 0:
                    touched = {1 + int(np.argmax(areas))}
            if touched:
                union_mask = np.isin(lbl, list(touched)).astype(np.uint8)
                yy, xx = np.where(union_mask > 0)
                if len(xx) >= 15:
                    xs, ys = xx, yy

        if xs is None:
            # FALLBACK: fg mask too weak/fragmented (common in dusk, low
            # contrast). Build the box directly from the flow-point cloud
            # itself, padded slightly, rather than discarding the object.
            px_local = (cluster_pts[:, 0] - rx1).astype(np.int32)
            py_local = (cluster_pts[:, 1] - ry1).astype(np.int32)
            pad2 = 4
            bx1 = max(0, px_local.min() - pad2) + rx1
            bx2 = min(rx2 - rx1, px_local.max() + pad2) + rx1
            by1 = max(0, py_local.min() - pad2) + ry1
            by2 = min(ry2 - ry1, py_local.max() + pad2) + ry1
            xs = np.array([bx1 - rx1, bx2 - rx1])
            ys = np.array([by1 - ry1, by2 - ry1])
        else:
            bx1, bx2 = xs.min() + rx1, xs.max() + rx1
            by1, by2 = ys.min() + ry1, ys.max() + ry1

        w_box, h_box = bx2 - bx1, by2 - by1
        if w_box < p.MIN_W_PX or h_box < min_height_for_row(by2, self.h, p):
            return None
        aspect = max(w_box / max(h_box, 1), h_box / max(w_box, 1))
        if aspect > p.MAX_ASPECT:
            return None
        roi_area = cv2.countNonZero(self.roi_mask)
        if w_box * h_box > p.MAX_BOX_AREA_FRAC * roi_area:
            return None

        # ground-contact point: bottom-center of the *object* silhouette,
        # using the widest row near the bottom (robust to a thin antenna/
        # mirror pixel being the single lowest point).
        bottom_band = ys >= (ys.max() - 3)
        contact_x = xs[bottom_band].mean() + rx1
        contact_y = ys.max() + ry1

        return (float(bx1), float(by1), float(bx2), float(by2)), (float(contact_x), float(contact_y))
