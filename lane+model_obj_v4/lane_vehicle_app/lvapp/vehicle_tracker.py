"""
vehicle_tracker.py -- light multi-object tracker for the vehicle detections.  Pure NumPy, ~0.1 ms per frame.

What it adds on top of raw detections (every frame still runs the detector; the tracker never replaces a detection):
  * one-to-one association (IoU + normalised centre distance, so small distant boxes that barely overlap between frames still match);
  * confirmation instead of a global threshold: a candidate becomes a visible vehicle when
        score >= confirm_score                       (a clear detection: shown at once), or
        confirm_hits consecutive-ish hits AND mean score >= confirm_mean_score      (a weak, usually distant, vehicle seen repeatedly)
    so distant vehicles at conf 0.20-0.45 are kept only if they persist, while one-frame ghosts are not shown;
  * alpha-beta smoothing of the box (position uses the track's own velocity, so there is no fixed lag on approaching vehicles);
  * class voting with hysteresis (a class changes only if the new class has class_switch_ratio x the votes for class_switch_frames frames);
  * duplicate-track removal (two tracks with IoU >= dup_iou: the younger / weaker one is dropped);
  * geometric plausibility (aspect ratio, minimum height) -- rejects slivers, e.g. poles, barrier posts;
  * zone-dependent confirmation: a track whose ground-contact point was classified OUTSIDE the relevant road area by the lane
    association in the previous frame needs a higher score and more hits (roadside structures, parallel roads) -- distant vehicles
    in or beside the lane are NOT penalised;
  * short coasting: a confirmed vehicle that is missed for <= max_coast_draw frames keeps being drawn (flagged `coasted`, dashed
    box); a coasted box is NOT counted as a detection in the statistics and never feeds the lane mask.
The ground-contact point (bottom centre of the box) is kept for the raw detection and for the smoothed box.
"""
import numpy as np

CAR, BUS, TRUCK, BIKE, MOTO = 2, 5, 7, 1, 3


class Track:
    __slots__ = ("id", "box", "vel", "raw", "score", "score_ema", "hits", "misses", "age", "votes", "cls", "cls_run", "cand_cls",
                 "confirmed", "coasted", "det_frame", "zone", "zone_info", "gcp", "gcp_raw", "score_sum", "visible", "suppressed")

    def __init__(self, tid, box, score, cls, cls_scores, frame):
        self.id = tid
        self.box = box.astype(np.float32).copy()          # smoothed x1,y1,x2,y2 (original-frame px)
        self.raw = box.astype(np.float32).copy()          # last raw detection
        self.vel = np.zeros(4, np.float32)
        self.score = float(score)
        self.score_ema = float(score)
        self.score_sum = float(score)
        self.hits, self.misses, self.age = 1, 0, 1
        self.votes = {int(cls): float(score)}
        self.cls = int(cls)
        self.cand_cls, self.cls_run = None, 0
        self.confirmed = False
        self.coasted = False
        self.det_frame = frame
        self.zone = "UNCERTAIN"
        self.zone_info = ""
        self.gcp = self.ground_point(self.box)
        self.gcp_raw = self.ground_point(self.raw)
        self.visible = False
        self.suppressed = ""

    @staticmethod
    def ground_point(b):
        return (float((b[0] + b[2]) * 0.5), float(b[3]))   # bottom-centre = ground contact (original-frame px)


def _iou_dist(a, b):
    """a: (N,4) predicted track boxes, b: (M,4) detections -> IoU (N,M), centre distance / track diagonal (N,M)."""
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iou = inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-6)
    ca = np.stack([(a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2], 1)
    cb = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
    diag = np.hypot(a[:, 2] - a[:, 0], a[:, 3] - a[:, 1])
    dist = np.hypot(ca[:, None, 0] - cb[None, :, 0], ca[:, None, 1] - cb[None, :, 1]) / np.maximum(diag[:, None], 1.0)
    return iou, dist


class VehicleTracker:
    def __init__(self, cfg):
        self.c = cfg
        self.tracks = []
        self.next_id = 1
        self.frame = 0
        self.stats = dict(created=0, confirmed=0, dup_removed=0, implausible=0, class_switches=0, zone_suppressed=0)

    # ------------------------------------------------------------------ prediction (used for the lane mask prior)
    def predicted_boxes(self, min_hits, min_score):
        """Boxes of reliable tracks advanced by one frame with their own velocity (prior for the NEXT frame's lane edge mask)."""
        out = []
        for t in self.tracks:
            if t.confirmed and not t.coasted and t.hits >= min_hits and t.score_ema >= min_score:
                out.append(t.box + t.vel)
        return out

    # ------------------------------------------------------------------ plausibility
    def _plausible(self, b, cls):
        c = self.c
        w, h = float(b[2] - b[0]), float(b[3] - b[1])
        if h < c["min_box_h"] or w < 2:
            return False
        r = w / max(h, 1e-6)
        if cls in (BIKE, MOTO):
            return 0.2 <= r <= 2.5
        return c["min_aspect_wh"] <= r <= c["max_aspect_wh"]

    # ------------------------------------------------------------------ step / update
    def predict_step(self, frame_id):
        """Advances tracks by one frame using their estimated velocity when no new detection is ready."""
        self.frame = frame_id
        for t in self.tracks:
            t.age += 1
            t.box = (t.box + t.vel).astype(np.float32)
            t.box[0] = max(0.0, float(t.box[0]))
            t.box[1] = max(0.0, float(t.box[1]))
            t.gcp = Track.ground_point(t.box)
        return self.visible_tracks()

    def update_async(self, det, det_frame_id, curr_frame_id):
        """det: Detections from det_frame_id. Accounting for delay delta = curr_frame_id - det_frame_id."""
        c = self.c
        self.frame = curr_frame_id
        delta = max(0, curr_frame_id - det_frame_id)

        # --- association
        D = len(det.score)
        ok = np.array([self._plausible(det.boxes[i], int(det.cls[i])) for i in range(D)], bool) if D else np.zeros(0, bool)
        self.stats["implausible"] += int((~ok).sum())
        dets = np.nonzero(ok)[0]
        matched_t, matched_d = {}, set()
        if self.tracks and len(dets):
            # Compare track predicted position at det_frame_id with raw detection
            pred = np.stack([t.box - delta * t.vel for t in self.tracks])
            iou, dist = _iou_dist(pred, det.boxes[dets])
            # cost: prefer overlap, then closeness; gated
            cost = (1.0 - iou) + 0.5 * dist
            gate = (iou >= c["iou_gate"]) | (dist <= c["dist_gate"])
            # size gate
            tw = (pred[:, 2] - pred[:, 0])[:, None]
            dw = (det.boxes[dets][:, 2] - det.boxes[dets][:, 0])[None, :]
            gate &= (dw / np.maximum(tw, 1.0) < 2.2) & (tw / np.maximum(dw, 1.0) < 2.2)
            cost = np.where(gate, cost, 9.0)
            order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
            for ti, dj in order:
                if cost[ti, dj] >= 9.0:
                    break
                if ti in matched_t or dj in matched_d:
                    continue
                matched_t[int(ti)] = int(dets[dj])
                matched_d.add(int(dj))
        # --- update matched tracks
        for ti, di in matched_t.items():
            t = self.tracks[ti]
            b = det.boxes[di]
            dt_det = max(1, det_frame_id - t.det_frame)
            v_meas = (b - t.raw) / float(dt_det)
            if t.hits <= 1:
                t.vel = v_meas.astype(np.float32)
            else:
                t.vel = ((1 - c["vel_beta"]) * t.vel + c["vel_beta"] * v_meas).astype(np.float32)

            b_proj = (b + delta * t.vel).astype(np.float32)
            a = c["box_alpha"]
            new = (1 - a) * t.box + a * b_proj
            t.box = new.astype(np.float32)
            t.raw = b.astype(np.float32)
            s = float(det.score[di])
            t.score = s
            t.score_ema = 0.6 * t.score_ema + 0.4 * s
            t.score_sum += s
            t.hits += 1
            t.misses = 0
            t.coasted = False
            t.det_frame = det_frame_id
            self._vote(t, int(det.cls[di]), det.cls_scores[di], s)
            t.gcp = Track.ground_point(t.box)
            t.gcp_raw = Track.ground_point(b_proj)
        # --- unmatched tracks: missed in this detector run
        for ti, t in enumerate(self.tracks):
            if ti not in matched_t:
                t.misses += 1
                t.vel *= 0.7
                t.coasted = True
                t.gcp = Track.ground_point(t.box)
        # --- new tracks from unmatched detections
        for di in dets:
            if int(di) in matched_d:
                continue
            b = det.boxes[di]
            b_proj = (b + delta * np.zeros(4, np.float32)).astype(np.float32)
            tr = Track(self.next_id, b_proj, det.score[di], det.cls[di], det.cls_scores[di], curr_frame_id)
            tr.det_frame = det_frame_id
            tr.raw = b.astype(np.float32)
            self.next_id += 1
            self.stats["created"] += 1
            self.tracks.append(tr)
        # --- confirmation / deletion
        keep = []
        for t in self.tracks:
            if t.misses > c["max_age"] or (not t.confirmed and t.misses > 2):
                continue
            if not t.confirmed:
                need_score, need_hits = c["confirm_score"], c["confirm_hits"]
                if t.zone == "OUTSIDE":
                    need_score, need_hits = c["outside_confirm_score"], c["outside_confirm_hits"]
                    if t.score_ema < need_score and t.hits < need_hits:
                        t.suppressed = "outside-road"
                if t.score >= need_score and (t.zone != "OUTSIDE" or t.hits >= 2):
                    t.confirmed = True
                elif t.hits >= need_hits and (t.score_sum / t.hits) >= c["confirm_mean_score"] and (t.zone != "OUTSIDE" or t.score_ema >= need_score):
                    t.confirmed = True
                if t.confirmed:
                    self.stats["confirmed"] += 1
            keep.append(t)
        self.tracks = keep
        self._remove_duplicates()
        return self.visible_tracks()

    def update(self, det, frame_id):
        """det: Detections (original-frame coords). Synchronous update (det_frame_id == curr_frame_id)."""
        return self.update_async(det, frame_id, frame_id)

    def _vote(self, t, cls, cls_scores, score):
        # decay old votes a little so a long-held class can still be overturned by persistent evidence
        for k in list(t.votes):
            t.votes[k] *= 0.92
        for j, k in enumerate((1, 2, 3, 5, 7)):
            if cls_scores[j] >= 0.15:
                t.votes[k] = t.votes.get(k, 0.0) + float(cls_scores[j])
        best = max(t.votes, key=t.votes.get)
        if best == t.cls:
            t.cand_cls, t.cls_run = None, 0
            return
        if t.votes[best] >= self.c["class_switch_ratio"] * max(t.votes.get(t.cls, 1e-6), 1e-6):
            if t.cand_cls == best:
                t.cls_run += 1
            else:
                t.cand_cls, t.cls_run = best, 1
            if t.cls_run >= self.c["class_switch_frames"]:
                t.cls, t.cand_cls, t.cls_run = best, None, 0
                self.stats["class_switches"] += 1
        else:
            t.cand_cls, t.cls_run = None, 0

    def _remove_duplicates(self):
        conf = [t for t in self.tracks if t.confirmed]
        if len(conf) < 2:
            return
        B = np.stack([t.box for t in conf])
        iou, _ = _iou_dist(B, B)
        drop = set()
        for i in range(len(conf)):
            for j in range(i + 1, len(conf)):
                if iou[i, j] >= self.c["dup_iou"]:
                    a, b = conf[i], conf[j]
                    # keep the one with more evidence (hits, then score)
                    loser = b if (a.hits, a.score_ema) >= (b.hits, b.score_ema) else a
                    drop.add(loser.id)
        if drop:
            self.stats["dup_removed"] += len(drop)
            self.tracks = [t for t in self.tracks if t.id not in drop]

    def visible_tracks(self):
        c = self.c
        out = []
        for t in self.tracks:
            vis = t.confirmed and t.misses <= c["max_coast_draw"] and not (t.zone == "OUTSIDE" and t.score_ema < c["outside_confirm_score"] and t.hits < c["outside_confirm_hits"])
            if t.confirmed and t.zone == "OUTSIDE" and not vis:
                self.stats["zone_suppressed"] += 1
            t.visible = vis
            if vis:
                out.append(t)
        return out
