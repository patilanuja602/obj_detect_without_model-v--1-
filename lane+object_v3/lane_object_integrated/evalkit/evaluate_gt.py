#!/usr/bin/env python3
"""Compare a detections dump (dump_detections.py) with hand annotations (gt_A.json).  All values in source pixels.
Match rule (top edge is often unannotated): GT and detection must overlap horizontally with IoU_x >= 0.4 and their bottom edges
must differ by <= 0.3*GT width.  One detection per GT (best IoU_x).  Reports per-vehicle and mean errors."""
import json, sys, numpy as np
def iou_x(a, b):
    i = max(0, min(a[2], b[2]) - max(a[0], b[0])); u = (a[2]-a[0]) + (b[2]-b[0]) - i; return i / u if u > 0 else 0
def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i; return i / u if u > 0 else 0
def evaluate(dump_path, gt_path, ids=None, observed_only=False):
    D = json.load(open(dump_path)); G = json.load(open(gt_path))["frames"]; rows = []
    for fi, vehs in G.items():
        if ids is not None and fi not in ids: continue
        rec = D["frames"][int(fi)]; assert rec["f"] == int(fi)
        dets = [d for d in rec["dets"] if (d["obs"] or not observed_only)]
        for v in vehs:
            gb = v["box"]; gw = gb[2] - gb[0]; best = None
            for d in dets:
                b = d["box"]
                if iou_x(gb, b) >= 0.4 and abs(gb[3] - b[3]) <= 0.3 * gw:
                    if best is None or iou_x(gb, b) > iou_x(gb, best["box"]): best = d
            r = dict(frame=fi, id=v["id"], w=gw, matched=best is not None)
            if best:
                b = best["box"]; r.update(e_x0=b[0]-gb[0], e_x1=b[2]-gb[2], e_y1=b[3]-gb[3], e_y0=None if gb[1] is None else b[1]-gb[1],
                    iou=None if gb[1] is None else box_iou(gb, b), cq=best["cq"], obs=best["obs"])
                gm = np.mean(v["contacts"], axis=0)
                if best["contact"] is not None:
                    c = np.array(best["contact"]); e = c - gm
                    r.update(c_dx=float(e[0]), c_dy=float(e[1]), c_err=float(np.hypot(*e)))
                else: r.update(c_dx=None, c_dy=None, c_err=None)
                # bottom-centre baseline-style reference for context
                r["bc_err"] = float(np.hypot((b[0]+b[2])/2 - gm[0], b[3] - gm[1]))
            rows.append(r)
    return rows, D
def summarize(rows):
    m = [r for r in rows if r["matched"]]; out = dict(n_gt=len(rows), n_matched=len(m))
    def st(k, absv=True):
        v = [r[k] for r in m if r.get(k) is not None]
        if not v: return None
        v = np.abs(v) if absv else np.array(v); return dict(mean=float(np.mean(v)), median=float(np.median(v)), p90=float(np.percentile(v, 90)), n=len(v))
    for k in ("e_x0", "e_x1", "e_y1", "e_y0", "c_dx", "c_dy", "c_err", "bc_err"): out[k] = st(k)
    out["iou"] = st("iou", absv=False)
    out["rel_bw_err_pct"] = float(np.mean([100*(abs(r["e_x0"])+abs(r["e_x1"]))/2/r["w"] for r in m])) if m else None
    return out
if __name__ == "__main__":
    rows, D = evaluate(sys.argv[1], sys.argv[2])
    for r in rows:
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()})
    print(json.dumps(summarize(rows), indent=1))
