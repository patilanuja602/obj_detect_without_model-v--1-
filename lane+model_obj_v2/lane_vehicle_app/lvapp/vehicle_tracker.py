"""
vehicle_tracker.py -- light multi-object tracker that runs AFTER the detector on every frame (the detector is never skipped).

Purpose: stable boxes, stable class labels, fewer one-frame blips, without hiding distant vehicles.
  * Matching: greedy IoU (class-agnostic) between predicted track boxes and this frame's detections; very small boxes also match by centre distance.
  * Entry:  a detection starts a track if score >= enter_score (same 0.25 as the old display threshold).
            It is shown immediately when score >= confirm_score, otherwise after `confirm_hits` consecutive hits (low-confidence blips need 2 frames).
  * Keep:   an existing confirmed track is sustained by detections down to keep_score (< enter_score) -- hysteresis, so a distant car does not flicker.
  * Coast:  a confirmed track missed by the detector is carried along its velocity for up to max_coast frames and marked `coasting`.
  * Class:  decayed, score-weighted votes; the label only switches when another class leads by class_switch_ratio.
  * Box:    light exponential smoothing (box_alpha = weight of the new detection) -- low delay.
  * Ground contact point (bottom-centre of the box) is stored in every track as `ground`.
Coordinates are whatever the caller uses (the pipeline uses display pixels).
"""
import numpy as np

from .vehicle_detector import NAMES


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


class Track:
    __slots__ = ("id", "box", "vel", "score", "raw_score", "votes", "cls", "hits", "consec", "misses", "age", "confirmed", "coasting",
                 "zone", "zone_r", "zone_hist", "first_fid", "last_fid", "cls_changes")

    def __init__(self, tid, det, fid, confirmed):
        self.id, self.box, self.vel = tid, np.array(det[:4], float), np.zeros(4)
        self.score = self.raw_score = float(det[4])
        self.cls = int(det[5])
        self.votes = {self.cls: float(det[4])}
        self.hits, self.consec, self.misses, self.age = 1, 1, 0, 1
        self.confirmed, self.coasting = confirmed, False
        self.zone, self.zone_r, self.zone_hist = "uncertain", None, []
        self.first_fid = self.last_fid = fid
        self.cls_changes = 0

    @property
    def ground(self):                                                # bottom-centre ground-contact point
        return (0.5 * (self.box[0] + self.box[2]), float(self.box[3]))

    @property
    def name(self):
        return NAMES.get(self.cls, "?")


class VehicleTracker:
    def __init__(self, enter_score=0.25, confirm_score=0.45, confirm_hits=2, keep_score=0.20, max_coast=2, match_iou=0.30,
                 box_alpha=0.7, class_decay=0.85, class_switch_ratio=1.25, **_):
        self.enter, self.confirm_score, self.confirm_hits, self.keep = enter_score, confirm_score, confirm_hits, keep_score
        self.max_coast, self.match_iou, self.alpha = max_coast, match_iou, box_alpha
        self.decay, self.switch_ratio = class_decay, class_switch_ratio
        self.tracks = []
        self._next = 1

    def reset(self):
        self.tracks, self._next = [], 1

    # ------------------------------------------------------------------
    def update(self, dets, fid):
        dets = np.asarray(dets, float).reshape(-1, 6)
        preds = [t.box + t.vel for t in self.tracks]
        pairs = []
        for ti, pb in enumerate(preds):
            pw, ph = max(pb[2] - pb[0], 1.0), max(pb[3] - pb[1], 1.0)
            for di, d in enumerate(dets):
                iou = _iou(pb, d)
                dist = np.hypot((pb[0] + pb[2] - d[0] - d[2]) / 2, (pb[1] + pb[3] - d[1] - d[3]) / 2) / max(pw, ph)
                if iou >= self.match_iou or dist < 0.35:
                    pairs.append((iou - 0.5 * dist, ti, di))
        pairs.sort(reverse=True)
        used_t, used_d, match = set(), set(), {}
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            match[ti] = di
        for ti, t in enumerate(self.tracks):
            t.age += 1
            if ti in match:
                self._update_matched(t, dets[match[ti]], fid)
            else:
                t.misses += 1
                t.consec = 0
                if t.confirmed and t.misses <= self.max_coast:
                    t.box = t.box + t.vel
                    t.coasting = True
        self.tracks = [t for t in self.tracks if (t.confirmed and t.misses <= self.max_coast) or (not t.confirmed and t.misses == 0)]
        for di, d in enumerate(dets):
            if di in used_d or d[4] < self.enter:
                continue
            self.tracks.append(Track(self._next, d, fid, confirmed=d[4] >= self.confirm_score))
            self._next += 1
        return self.visible()

    def _update_matched(self, t, d, fid):
        a = self.alpha
        newbox = a * d[:4] + (1 - a) * (t.box + t.vel)
        t.vel = 0.5 * t.vel + 0.5 * (newbox - t.box)
        t.box = newbox
        t.raw_score = float(d[4])
        t.score = 0.6 * float(d[4]) + 0.4 * t.score
        for k in list(t.votes):
            t.votes[k] *= self.decay
        t.votes[int(d[5])] = t.votes.get(int(d[5]), 0.0) + float(d[4])
        lead = max(t.votes.items(), key=lambda kv: kv[1])
        if lead[0] != t.cls and lead[1] > self.switch_ratio * t.votes.get(t.cls, 0.0):
            t.cls = lead[0]
            t.cls_changes += 1
        t.hits += 1
        t.consec += 1
        t.misses = 0
        t.coasting = False
        t.last_fid = fid
        if not t.confirmed and (t.consec >= self.confirm_hits or t.raw_score >= self.confirm_score):
            t.confirmed = True

    # ------------------------------------------------------------------
    def visible(self):
        return [t for t in self.tracks if t.confirmed]

    def mask_boxes(self, min_score=0.45, min_hits=2):
        """Boxes (predicted one frame ahead) of reliable tracks -- used as the lane detector's occlusion prior."""
        return [t.box + t.vel for t in self.tracks if t.confirmed and t.hits >= min_hits and t.score >= min_score and not (t.coasting and t.misses > 1)]
