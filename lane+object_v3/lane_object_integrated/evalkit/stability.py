#!/usr/bin/env python3
"""Frame-to-frame jitter of tracked detections.  For every track with >= MINRUN consecutive OBSERVED frames, take the series of
box edges and the contact point (source px) and compute the RMS of the SECOND difference x[t+1]-2x[t]+x[t-1] (acceleration noise:
a smoothly moving vehicle has ~0, measurement jitter does not).  Lower = steadier.  Smoothness can be bought with lag, so the
mean absolute lag against the raw (unfiltered) per-frame measurement is not computed here; see README for the Kalman settings."""
import json, sys, numpy as np
def runs(frames, minrun):
    tr = {}
    for r in frames:
        for d in r["dets"]:
            if d["obs"]: tr.setdefault(d["id"], []).append((r["f"], d))
    for i, l in tr.items():
        seg = [l[0]]
        for a in l[1:]:
            if a[0] == seg[-1][0] + 1: seg.append(a)
            else:
                if len(seg) >= minrun: yield i, seg
                seg = [a]
        if len(seg) >= minrun: yield i, seg
def jitter(path, f0, f1, minrun=15):
    D = json.load(open(path)); fr = [r for r in D["frames"] if f0 <= r["f"] < f1]
    acc = {k: [] for k in ("x0", "x1", "y1", "cx", "cy")}; n = 0
    for i, seg in runs(fr, minrun):
        n += 1
        B = np.array([d["box"] for _, d in seg]); C = np.array([d["contact"] if d["contact"] else [np.nan, np.nan] for _, d in seg])
        for k, v in (("x0", B[:, 0]), ("x1", B[:, 2]), ("y1", B[:, 3]), ("cx", C[:, 0]), ("cy", C[:, 1])):
            s = v[2:] - 2 * v[1:-1] + v[:-2]; acc[k].extend(s[np.isfinite(s)])
    return n, {k: float(np.sqrt(np.mean(np.square(v)))) if v else None for k, v in acc.items()}
if __name__ == "__main__":
    for p in sys.argv[1:]:
        n, j = jitter(p, 0, 10**9); print(p.split("/")[-1], "tracks", n, {k: (None if v is None else round(v, 2)) for k, v in j.items()})
