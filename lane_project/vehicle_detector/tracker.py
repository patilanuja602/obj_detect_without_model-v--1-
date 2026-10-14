"""Tiny IoU tracker: confirms a box after a few frames, smooths it, bridges short gaps."""
from dataclasses import dataclass


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


@dataclass
class Track:
    id: int
    box: list
    hits: int = 1
    missed: int = 0
    age: int = 0


class Tracker:
    def __init__(self, cfg):
        self.cfg, self.tracks, self._next = cfg, [], 1

    def update(self, dets, hold_only=False):
        cfg = self.cfg
        for t in self.tracks:
            t.age += 1
        used = set()
        pairs = sorted(((iou(t.box, d), ti, di) for ti, t in enumerate(self.tracks) for di, d in enumerate(dets)),
                       reverse=True)
        matched_t = set()
        for s, ti, di in pairs:
            if s < cfg.iou_match:
                break
            if ti in matched_t or di in used:
                continue
            t, d = self.tracks[ti], dets[di]
            a = cfg.smooth
            t.box = [(1 - a) * o + a * n for o, n in zip(t.box, d)]
            t.hits += 1
            t.missed = 0
            matched_t.add(ti)
            used.add(di)
        for ti, t in enumerate(self.tracks):
            if ti not in matched_t:
                t.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= cfg.max_missed]
        if not hold_only:
            for di, d in enumerate(dets):
                if di not in used:
                    self.tracks.append(Track(self._next, list(map(float, d))))
                    self._next += 1

    def confirmed(self, locate):
        from .detector import Vehicle
        out = []
        for t in self.tracks:
            if t.hits >= self.cfg.confirm_hits and t.missed <= 2:
                side, dist = locate(t.box)
                out.append(Vehicle(t.id, tuple(int(round(v)) for v in t.box), side, dist, t.hits))
        return out
