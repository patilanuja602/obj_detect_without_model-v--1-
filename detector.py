"""
Phase 3: Classical (non-learned) monocular road-object detector.
Detects objects on the road WHETHER THEY ARE MOVING OR NOT.

Two independent, fused branches -- both purely classical, no trained model,
no network weights, CPU-only:

  BRANCH A -- MOTION (unchanged from Phase 2 "optical_based" detector)
      Sparse Lucas-Kanade optical flow + MOG2 background subtraction,
      clustered with DBSCAN/KMeans. Catches anything that moves relative
      to the road (cars, bikes, pedestrians, autos).

  BRANCH B -- STATIC APPEARANCE (NEW)
      Per-frame road-surface color model: sample a trusted "known road"
      seed patch near the bottom of the ROI (excluding any pixels the
      motion branch already flagged, so a car right in front doesn't
      poison the model), compute Lab-color statistics from that seed,
      then flag any pixel in the ROI whose Lab color deviates from the
      seed beyond a threshold as "not road". This has NO dependency on
      motion, so it also fires on parked vehicles, trees, poles, curbs,
      dropped cargo, etc. that never move relative to the camera.
      A light Sobel-edge cue is OR'd in to help solidify object
      silhouettes that are texture-distinct but color-similar to asphalt.

FUSION
      Both branches emit the same (box, contact_point) detection format
      per frame. They are merged with IoU-based de-duplication and fed
      into ONE shared tracker (identical tracker/confirmation logic as
      before), so a real object gets exactly one stable track whether
      it was caught by motion, appearance, or both.

PERFORMANCE NOTES (v2, output-identical to v1)
      Every optimisation here is exact -- the detections, boxes, contact
      points and track IDs are bit-for-bit the same as the original
      implementation; only redundant work was removed:
        * pixel-wise stages (gamma, MOG2, Lab, shadow test, static branch)
          run on the ROI rows only; morphology runs on a zero-padded ROI
          window so border behaviour matches the full-frame version
        * the static branch's Lab z-score uses 3 x 256-entry lookup tables
          instead of float32 maths over every pixel
        * shadow suppression only inspects foreground pixels
        * per-blob work uses the blob's bounding box, not the whole frame
        * the Sobel edge cue was removed: `dev | (edge & dev) == dev`, so it
          never affected the output
      Optical flow / corner detection still run on the full frame on purpose.
"""

import cv2
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from dataclasses import dataclass
from typing import List, Tuple
import os
import time
from concurrent.futures import ThreadPoolExecutor


# ----------------------------------------------------------------------
# ---------------------------  PARAMETERS  ------------------------------
# ----------------------------------------------------------------------
class Params:
    # --- preprocessing ---
    PROC_WIDTH = 640
    CLAHE_CLIP = 2.0
    CLAHE_GRID = (8, 8)

    # --- road ROI polygon (hard-coded, fraction of frame W,H) ---
    ROI_POLY_FRAC = [(0.00, 0.55), (1.00, 0.55), (1.00, 1.00), (0.00, 1.00)]

    # --- Shi-Tomasi / Lucas-Kanade (motion branch) ---
    MAX_CORNERS = 600
    QUALITY_LEVEL = 0.015
    MIN_DISTANCE = 6
    BLOCK_SIZE = 7
    LK_WIN = (21, 21)
    LK_MAX_LEVEL = 3
    LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    MIN_FLOW_MAG = 0.6
    MAX_FLOW_MAG = 60.0

    # --- MOG2 background subtraction (motion branch support mask) ---
    MOG2_HISTORY = 250
    MOG2_VAR_THRESH = 12
    MOG2_SHADOW = True
    MOG2_LEARNING_RATE = 0.01
    GAMMA = 1.6

    # --- blob-driven object separation (motion branch) ---
    BLOB_DILATE_PX = 5
    MIN_BLOB_AREA = 20
    SPATIAL_EPS_PX = 22
    SPATIAL_MIN_SAMPLES = 4

    # --- geometric filtering (shared by both branches) ---
    MIN_CLUSTER_POINTS = 4
    MIN_DIRECTION_COHERENCE = 0.45
    MIN_H_AT_BOTTOM = 34
    MIN_H_AT_HORIZON = 10
    MIN_W_PX = 8
    MAX_ASPECT = 3.2
    MAX_BOX_AREA_FRAC = 0.55

    # --- shadow suppression (HSV ratio test, motion branch) ---
    SHADOW_V_RATIO = (0.45, 0.92)
    SHADOW_S_DIFF_MAX = 35
    SHADOW_H_DIFF_MAX = 12

    # --- STATIC / appearance branch (NEW) ---
    STATIC_SEED_HEIGHT_FRAC = 0.16   # bottom slice of ROI used as "known road" seed
    STATIC_SEED_WIDTH_FRAC = 0.55    # center fraction of width sampled for the seed
    STATIC_SEED_MIN_PIXELS = 400     # if fewer valid (non-motion) seed px, widen seed
    STATIC_SEED_MAX_HEIGHT_FRAC = 0.35
    STATIC_Z_THRESH = 3.4            # Lab z-score beyond which a pixel is "not road"
    STATIC_MIN_STD = 4.0             # floor on per-channel std (avoid div-by-~0 on flat asphalt)
    STATIC_EDGE_THRESH = 55          # Sobel magnitude threshold (0-255 normalized)
    STATIC_MIN_BOX_AREA = 90
    STATIC_MAX_SINGLE_OBJ_WIDTH_FRAC = 0.5  # reject blobs wider than this fraction of ROI width

    # --- fusion (NEW) ---
    FUSE_IOU_THRESH = 0.3            # boxes above this IoU from the two branches = same object

    # --- performance only (no effect on results) ---
    # Run the static and motion branches concurrently on multi-core machines.
    # Automatically ignored when only one CPU is available.
    PARALLEL_BRANCHES = True

    # --- tracking / temporal consensus (shared) ---
    MIN_HITS_TO_CONFIRM = 3
    MAX_COAST_FRAMES = 5
    IOU_MATCH_THRESH = 0.25
    CENTROID_MATCH_PX = 70
    EMA_ALPHA = 0.45


# ----------------------------------------------------------------------
def build_roi_mask(shape, poly_frac):
    h, w = shape[:2]
    pts = np.array([[int(x * w), int(y * h)] for x, y in poly_frac], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def min_height_for_row(y, h, p: Params):
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
    box: Tuple[float, float, float, float]
    contact: Tuple[float, float]
    hits: int = 1
    coast: int = 0
    id: int = 0
    confirmed: bool = False
    sources: str = ""   # debug: "motion", "static", or "motion+static"


class MultiObjectTracker:
    """Simple centroid + IoU tracker with temporal-consensus confirmation
    and EMA smoothing. Purely geometric -- no class labels, no learning.
    Unchanged from Phase 2, except Track now carries a debug `sources` tag."""

    def __init__(self, p: Params):
        self.p = p
        self.tracks: List[Track] = []
        self.next_id = 1

    def update(self, detections):
        # detections: list of (box, contact, source_tag)
        p = self.p
        unmatched_dets = list(range(len(detections)))
        for tr in self.tracks:
            best_j, best_score = -1, 0.0
            for j in unmatched_dets:
                box, _, _ = detections[j]
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
                box, contact, src = detections[best_j]
                a = p.EMA_ALPHA
                tr.box = tuple(a * np.array(box) + (1 - a) * np.array(tr.box))
                tr.contact = tuple(a * np.array(contact) + (1 - a) * np.array(tr.contact))
                tr.hits += 1
                tr.coast = 0
                tr.sources = src
                if tr.hits >= p.MIN_HITS_TO_CONFIRM:
                    tr.confirmed = True
                unmatched_dets.remove(best_j)
            else:
                tr.coast += 1

        self.tracks = [t for t in self.tracks if t.coast <= p.MAX_COAST_FRAMES]

        for j in unmatched_dets:
            box, contact, src = detections[j]
            self.tracks.append(Track(box=box, contact=contact, id=self.next_id, sources=src))
            self.next_id += 1

        return [t for t in self.tracks if t.confirmed]


# ----------------------------------------------------------------------
class RoadObjectDetector:
    # Rows/cols of zero padding kept around the ROI when running morphology.
    # The original ran morphology on the full frame, where everything outside
    # the ROI is zero. A zero-padded window >= the deepest dilate->erode chain
    # (18-20 px here) reproduces that exactly. 32 gives headroom.
    PAD = 32

    def __init__(self, frame_shape, p: Params = Params(), collect_intermediates=False):
        self.p = p
        self.collect_intermediates = collect_intermediates
        self.h, self.w = frame_shape[:2]
        self.roi_mask = build_roi_mask(frame_shape, p.ROI_POLY_FRAC)
        ys, xs = np.where(self.roi_mask > 0)
        self.roi_y0, self.roi_y1 = int(ys.min()), int(ys.max())
        self.roi_x0, self.roi_x1 = int(xs.min()), int(xs.max())

        # ROI bounding box (end-exclusive) -- pixel-wise stages run only here
        self.ry0, self.ry1 = self.roi_y0, self.roi_y1 + 1
        self.rx0, self.rx1 = self.roi_x0, self.roi_x1 + 1
        self.rh, self.rw = self.ry1 - self.ry0, self.rx1 - self.rx0
        self.roi_c = np.ascontiguousarray(self.roi_mask[self.ry0:self.ry1, self.rx0:self.rx1])
        # For the default rectangular ROI every "AND with roi_mask" is a no-op
        self.roi_is_rect = bool(np.all(self.roi_c > 0))
        self.roi_area = cv2.countNonZero(self.roi_mask)

        # Zero-padded window around the ROI, used for morphology
        self.py0 = max(0, self.ry0 - self.PAD)
        self.py1 = min(self.h, self.ry1 + self.PAD)
        self.px0 = max(0, self.rx0 - self.PAD)
        self.px1 = min(self.w, self.rx1 + self.PAD)
        self.Hw, self.Ww = self.py1 - self.py0, self.px1 - self.px0
        self.oy, self.ox = self.ry0 - self.py0, self.rx0 - self.px0
        self._roi_in_win = (slice(self.oy, self.oy + self.rh), slice(self.ox, self.ox + self.rw))

        self.clahe = cv2.createCLAHE(clipLimit=p.CLAHE_CLIP, tileGridSize=p.CLAHE_GRID)
        self.bgsub = cv2.createBackgroundSubtractorMOG2(
            history=p.MOG2_HISTORY, varThreshold=p.MOG2_VAR_THRESH,
            detectShadows=p.MOG2_SHADOW)
        self.prev_gray = None
        self.tracker = MultiObjectTracker(p)
        self.frame_idx = 0

        # ---- constants hoisted out of the per-frame path ----
        inv = 1.0 / p.GAMMA
        self.gamma_table = (np.linspace(0, 1, 256) ** inv * 255).astype(np.uint8)
        E = cv2.getStructuringElement
        self.k_ell3 = E(cv2.MORPH_ELLIPSE, (3, 3))
        self.k_ell9 = E(cv2.MORPH_ELLIPSE, (9, 9))
        self.k_ell5 = E(cv2.MORPH_ELLIPSE, (5, 5))
        self.k_blob = E(cv2.MORPH_ELLIPSE, (p.BLOB_DILATE_PX, p.BLOB_DILATE_PX))
        self.k_ones17 = np.ones((17, 17), np.uint8)
        self.k_horiz = E(cv2.MORPH_RECT, (15, 3))
        self.k_vert = E(cv2.MORPH_RECT, (3, 15))
        self._v256 = np.arange(256, dtype=np.float32)
        self._z_thresh = np.float32(p.STATIC_Z_THRESH)

        # optional worker so the static branch can run alongside the motion branch
        try:
            n_cpu = len(os.sched_getaffinity(0))
        except AttributeError:
            n_cpu = os.cpu_count() or 1
        self._pool = ThreadPoolExecutor(max_workers=1) if (p.PARALLEL_BRANCHES and n_cpu > 1) else None

        # persistent full-frame buffers (zero outside the ROI, always)
        self._fg_bin_full = np.zeros((self.h, self.w), np.uint8)
        self._search_full = np.zeros((self.h, self.w), np.uint8)

    # -- preprocessing -------------------------------------------------
    def preprocess(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        return gray

    def gamma_correct(self, frame_bgr):
        return cv2.LUT(frame_bgr, self.gamma_table)

    def suppress_shadows(self, frame_bgr, fg_mask, bg_bgr):
        """Same HSV ratio test as before, but only evaluated at foreground
        pixels -- the original only ever zeroed pixels that were already
        foreground, so the result is identical."""
        p = self.p
        out = fg_mask.copy()
        ys, xs = np.nonzero(fg_mask)
        if ys.size == 0:
            return out
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        hsv_f = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        hsv_b = cv2.cvtColor(bg_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        ly, lx = ys - y0, xs - x0
        f = hsv_f[ly, lx].astype(np.float32)
        b = hsv_b[ly, lx].astype(np.float32)
        eps = 1e-3
        v_ratio = f[:, 2] / (b[:, 2] + eps)
        s_diff = np.abs(f[:, 1] - b[:, 1])
        h_abs = np.abs(f[:, 0] - b[:, 0])
        h_diff = np.minimum(h_abs, 180 - h_abs)
        is_shadow = (v_ratio > p.SHADOW_V_RATIO[0]) & (v_ratio < p.SHADOW_V_RATIO[1]) & \
                    (s_diff < p.SHADOW_S_DIFF_MAX) & (h_diff < p.SHADOW_H_DIFF_MAX)
        out[ys[is_shadow], xs[is_shadow]] = 0
        return out

    # ==================================================================
    # BRANCH B: static / appearance-based road-model detection
    # ==================================================================
    def _static_candidates(self, bright_roi, fg_c_roi):
        """Model the road surface's own Lab color from a trusted seed
        patch (excluding anything the motion branch already flagged),
        then flag any ROI pixel that deviates from that model. Runs
        every frame, independent of motion -- this is what catches
        parked/stationary objects that Branch A structurally cannot.

        `bright_roi` / `fg_c_roi` are the ROI-cropped gamma image and
        motion mask. Output coordinates are full-frame."""
        p = self.p
        lab = cv2.cvtColor(bright_roi, cv2.COLOR_BGR2LAB)     # uint8, ROI only

        # ---- grow the seed strip upward until we have enough clean pixels ----
        cx0 = int(self.w * (0.5 - p.STATIC_SEED_WIDTH_FRAC / 2))
        cx1 = int(self.w * (0.5 + p.STATIC_SEED_WIDTH_FRAC / 2))
        xa = max(cx0, self.rx0) - self.rx0
        xb = max(xa, min(cx1, self.rx1) - self.rx0)
        seed_h_frac = p.STATIC_SEED_HEIGHT_FRAC
        while True:
            sy0 = int(self.roi_y1 - self.h * seed_h_frac)
            sy0 = max(self.roi_y0, sy0)
            ya = sy0 - self.ry0
            yb = max(ya, self.roi_y1 - self.ry0)      # rows [sy0, roi_y1)
            cand = (fg_c_roi[ya:yb, xa:xb] == 0)
            if not self.roi_is_rect:
                cand &= (self.roi_c[ya:yb, xa:xb] > 0)
            if cand.sum() >= p.STATIC_SEED_MIN_PIXELS or seed_h_frac >= p.STATIC_SEED_MAX_HEIGHT_FRAC:
                break
            seed_h_frac += 0.06

        if cand.sum() < 50:
            # road immediately ahead is fully occluded (e.g. truck right in
            # front) -- nothing safe to model this frame, skip static branch
            return [], None

        seed_pixels = lab[ya:yb, xa:xb][cand].astype(np.float32)
        mean = seed_pixels.mean(axis=0)
        std = np.maximum(seed_pixels.std(axis=0), p.STATIC_MIN_STD)

        # ---- per-pixel Lab z-score distance from the road model ----
        # Lab channels are integers 0..255, so (v-mean)/std squared is a
        # 256-entry table per channel: same float32 arithmetic, one lookup
        # per pixel instead of a full float32 pass.
        v = self._v256
        lut0 = np.ascontiguousarray(((v - mean[0]) / std[0]) ** 2)
        lut1 = np.ascontiguousarray(((v - mean[1]) / std[1]) ** 2)
        lut2 = np.ascontiguousarray(((v - mean[2]) / std[2]) ** 2)
        Lp, Ap, Bp = cv2.split(lab)
        sq = cv2.LUT(Lp, lut0)
        sq2 = cv2.LUT(Ap, lut1)
        sq3 = cv2.LUT(Bp, lut2)
        z = np.sqrt(sq + (sq2 + sq3))
        deviation = (z > self._z_thresh).view(np.uint8) * np.uint8(255)
        if not self.roi_is_rect:
            deviation = cv2.bitwise_and(deviation, deviation, mask=self.roi_c)

        # NOTE: the original OR'd in a Sobel edge cue as
        #   combined = dev | (edge & dev)
        # which is identically `dev` (absorption law), so it is omitted.
        combined = deviation
        combined[ya:yb, xa:xb][cand] = 0   # never let the seed strip survive

        # ---- solidify into silhouettes (zero-padded window == full frame) ----
        win = np.zeros((self.Hw, self.Ww), np.uint8)
        win[self._roi_in_win] = combined
        solid = cv2.morphologyEx(win, cv2.MORPH_CLOSE, self.k_horiz)
        solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, self.k_vert)
        solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, self.k_ell5)

        n, lbl, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
        roi_w = self.roi_x1 - self.roi_x0
        max_w = p.STATIC_MAX_SINGLE_OBJ_WIDTH_FRAC * roi_w

        out = []
        for b in range(1, n):
            area = stats[b, cv2.CC_STAT_AREA]
            if area < p.STATIC_MIN_BOX_AREA:
                continue
            wx = stats[b, cv2.CC_STAT_LEFT]
            wy = stats[b, cv2.CC_STAT_TOP]
            w_ = stats[b, cv2.CC_STAT_WIDTH]
            h_ = stats[b, cv2.CC_STAT_HEIGHT]
            x = wx + self.px0
            y = wy + self.py0
            if w_ > max_w:
                continue
            if h_ < min_height_for_row(y + h_, self.h, p):
                continue
            aspect = max(w_ / max(h_, 1), h_ / max(w_, 1))
            if aspect > p.MAX_ASPECT:
                continue
            if w_ * h_ > p.MAX_BOX_AREA_FRAC * self.roi_area:
                continue
            # only look inside this blob's bounding box (not the whole frame)
            blob = (lbl[wy:wy + h_, wx:wx + w_] == b)
            ys_l, xs_l = np.where(blob)
            ys_ = ys_l + y
            xs_ = xs_l + x
            bottom_band = ys_ >= (ys_.max() - 3)
            contact_x = float(xs_[bottom_band].mean())
            contact_y = float(ys_.max())
            box = (float(x), float(y), float(x + w_), float(y + h_))
            out.append((box, (contact_x, contact_y), "static"))
        return out, solid

    # ==================================================================
    # BRANCH A: motion / optical-flow detection
    # ==================================================================
    def _motion_candidates(self, gray, fg_c_roi, fg_binary):
        p = self.p
        detections = []
        clusters_viz = None
        if self.prev_gray is not None:
            search_roi = cv2.dilate(fg_c_roi, self.k_ones17)
            if not self.roi_is_rect:
                search_roi = cv2.bitwise_and(search_roi, self.roi_c)
            self._search_full[self.ry0:self.ry1, self.rx0:self.rx1] = search_roi
            pts0 = cv2.goodFeaturesToTrack(
                self.prev_gray, maxCorners=p.MAX_CORNERS, qualityLevel=p.QUALITY_LEVEL,
                minDistance=p.MIN_DISTANCE, blockSize=p.BLOCK_SIZE, mask=self._search_full)
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
                    for cluster_pts, cluster_flow in self._blob_cluster(pts1f, flow, fg_c_roi):
                        det = self._points_to_detection(cluster_pts, fg_binary)
                        if det is not None:
                            box, contact = det
                            detections.append((box, contact, "motion"))
                    detections = self._merge_fragments(detections)
        return detections, clusters_viz

    @staticmethod
    def _coherence(flow_vecs):
        unit = flow_vecs / (np.linalg.norm(flow_vecs, axis=1, keepdims=True) + 1e-6)
        return np.linalg.norm(unit.mean(axis=0))

    def _split_or_keep(self, gpts, gflow, out):
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

    def _blob_cluster(self, pts, flow, fg_c_roi):
        p = self.p
        out = []
        dil = cv2.dilate(fg_c_roi, self.k_blob)
        if not self.roi_is_rect:
            dil = cv2.bitwise_and(dil, self.roi_c)
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)
        pts_i = pts.astype(np.int32)
        pts_i[:, 0] = np.clip(pts_i[:, 0], 0, self.w - 1)
        pts_i[:, 1] = np.clip(pts_i[:, 1], 0, self.h - 1)
        # labels only exist inside the ROI box; anything outside is label 0
        yy = pts_i[:, 1] - self.ry0
        xx = pts_i[:, 0] - self.rx0
        inside = (yy >= 0) & (yy < self.rh) & (xx >= 0) & (xx < self.rw)
        blob_id = np.zeros(len(pts_i), dtype=lbl.dtype)
        blob_id[inside] = lbl[yy[inside], xx[inside]]
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
        p = self.p
        boxes = [d[0] for d in detections]
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

        by_box = {}
        for (box, contact, src) in detections:
            by_box.setdefault(box, (contact, src))
        result = []
        for b in cur:
            if b in by_box:
                contact, src = by_box[b]
                result.append((b, contact, src))
            else:
                result.append((b, ((b[0] + b[2]) / 2.0, b[3]), "motion"))
        return result

    def _points_to_detection(self, cluster_pts, fg_binary):
        p = self.p
        x1, y1 = cluster_pts.min(axis=0)
        x2, y2 = cluster_pts.max(axis=0)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        pad = 12
        rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
        rx2, ry2 = min(self.w, x2 + pad), min(self.h, y2 + pad)
        if rx2 <= rx1 or ry2 <= ry1:
            return None
        roi_mask_local = fg_binary[ry1:ry2, rx1:rx2]
        n, lbl, stats, centroids = cv2.connectedComponentsWithStats(roi_mask_local, connectivity=8)
        xs = ys = None
        if n > 1:
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
        if w_box * h_box > p.MAX_BOX_AREA_FRAC * self.roi_area:
            return None
        bottom_band = ys >= (ys.max() - 3)
        contact_x = xs[bottom_band].mean() + rx1
        contact_y = ys.max() + ry1
        return (float(bx1), float(by1), float(bx2), float(by2)), (float(contact_x), float(contact_y))

    # ==================================================================
    # FUSION (NEW)
    # ==================================================================
    @staticmethod
    def _fuse(motion_dets, static_dets, thresh):
        """Merge the two branches' detections. If a motion box and a
        static box overlap strongly, they're the same physical object --
        keep the motion one (its contact point is usually better refined
        by the silhouette-snap step) but tag it 'motion+static'. Anything
        left in static-only survives as-is: that's exactly the case this
        phase exists for (an object with zero relative motion)."""
        used_static = set()
        fused = []
        for (mbox, mcontact, msrc) in motion_dets:
            merged_src = msrc
            for j, (sbox, scontact, ssrc) in enumerate(static_dets):
                if j in used_static:
                    continue
                if iou(mbox, sbox) > thresh:
                    used_static.add(j)
                    merged_src = "motion+static"
                    break
            fused.append((mbox, mcontact, merged_src))
        for j, (sbox, scontact, ssrc) in enumerate(static_dets):
            if j not in used_static:
                fused.append((sbox, scontact, ssrc))
        return fused

    # -- main per-frame call --------------------------------------------
    def process(self, frame_bgr):
        p = self.p
        rs = (slice(self.ry0, self.ry1), slice(self.rx0, self.rx1))
        in_win = self._roi_in_win

        gray = self.preprocess(frame_bgr)               # full frame (optical flow needs it)
        bright = self.gamma_correct(frame_bgr[rs])      # ROI only

        # MOG2 is strictly per-pixel, so running it on the ROI crop gives the
        # same result inside the ROI as running it on the full frame.
        fg_raw = self.bgsub.apply(bright, learningRate=p.MOG2_LEARNING_RATE)
        cand = cv2.threshold(fg_raw, 126, 255, cv2.THRESH_BINARY)[1]   # == (fg_raw >= 127)
        if not self.roi_is_rect:
            cand = cv2.bitwise_and(cand, cand, mask=self.roi_c)
        wcand = np.zeros((self.Hw, self.Ww), np.uint8)
        wcand[in_win] = cand
        wcand = cv2.morphologyEx(wcand, cv2.MORPH_OPEN, self.k_ell3)
        fg_c_roi = wcand[in_win]                        # == fg_candidate, ROI part

        if cv2.countNonZero(fg_c_roi):
            bg_bgr = self.bgsub.getBackgroundImage()
            fg_bin = self.suppress_shadows(bright, fg_c_roi, bg_bgr) if bg_bgr is not None else fg_c_roi.copy()
        else:
            fg_bin = fg_c_roi.copy()
        if not self.roi_is_rect:
            fg_bin = cv2.bitwise_and(fg_bin, fg_bin, mask=self.roi_c)
        wbin = np.zeros((self.Hw, self.Ww), np.uint8)
        wbin[in_win] = fg_bin
        wbin = cv2.morphologyEx(wbin, cv2.MORPH_OPEN, self.k_ell3)
        wbin = cv2.morphologyEx(wbin, cv2.MORPH_CLOSE, self.k_ell9, iterations=2)
        self._fg_bin_full[rs] = wbin[in_win]
        fg_binary = self._fg_bin_full

        if self._pool is not None:
            # both branches only read the shared masks, so order doesn't matter
            static_fut = self._pool.submit(self._static_candidates, bright, fg_c_roi)
            motion_dets, clusters_viz = self._motion_candidates(gray, fg_c_roi, fg_binary)
            static_dets, static_mask = static_fut.result()
        else:
            motion_dets, clusters_viz = self._motion_candidates(gray, fg_c_roi, fg_binary)
            static_dets, static_mask = self._static_candidates(bright, fg_c_roi)

        fused = self._fuse(motion_dets, static_dets, p.FUSE_IOU_THRESH)

        intermediates = {"clahe_gray": gray, "fg_clean": fg_binary}
        if self.collect_intermediates:
            full = np.zeros((self.h, self.w), np.uint8)
            full[rs] = fg_raw
            intermediates["mog2_raw"] = full
            sm = np.zeros((self.h, self.w), np.uint8)
            if static_mask is not None:
                sm[self.py0:self.py1, self.px0:self.px1] = static_mask
            intermediates["static_mask"] = sm

        self.prev_gray = gray
        confirmed_tracks = self.tracker.update(fused)
        self.frame_idx += 1
        return confirmed_tracks, fused, intermediates, clusters_viz
