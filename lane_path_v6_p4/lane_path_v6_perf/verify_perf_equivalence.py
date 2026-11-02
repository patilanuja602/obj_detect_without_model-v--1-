#!/usr/bin/env python3
"""verify_perf_equivalence.py -- proves the optimised lane_path_v6.py is behaviour-identical to the pre-optimisation version.

For every frame of several CONSECUTIVE segments of the real video it feeds two independent detectors (pre-opt and optimised) and checks:
  1. the processing frame given to process(): ROI rows/cols bit-identical (full resize vs ROI-only resize)
  2. det.edges and det.Lraw (Canny edge map / L channel that everything downstream is built from) are identical
  3. the complete internal result (result["lane"], ["corridor"], ["partial"], ["mode"], ["extent_y"], ...) is identical (deep compare)
  4. the DISPLAYED image from draw() is pixel-identical (debug off and on)
  5. draw() does not change the result
And separately: _blend_fill() against the original copy/fillPoly/addWeighted on random polygons, including polygons partly or fully
outside the image.

Usage: python3 verify_perf_equivalence.py VIDEO [--before lane_path_v6_pre_opt.py] [--after lane_path_v6.py]
                                           [--starts 0 60 120 200] [--frames 150]
Exit code 0 = all identical.
"""
import argparse, copy, importlib.util, sys
import cv2, numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, float) and a != a:
        return b != b
    return a == b


def snap(res):
    return copy.deepcopy(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--before", default="lane_path_v6_pre_opt.py")
    ap.add_argument("--after", default="lane_path_v6.py")
    ap.add_argument("--starts", type=float, nargs="+", default=[0, 60, 120, 200])
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--out-w", type=int, default=1280)
    a = ap.parse_args()
    old, new = load("v6_before", a.before), load("v6_after", a.after)
    bad = dict(roi_input=0, edges=0, Lraw=0, result=0, draw=0, draw_dbg=0, mutated=0)
    tot, states = 0, {}
    for st in a.starts:
        cap = cv2.VideoCapture(a.video)
        cap.set(cv2.CAP_PROP_POS_MSEC, st * 1000)
        ok, f = cap.read()
        if not ok:
            print("segment at %.0fs: cannot read, skipped" % st)
            continue
        sh, sw = f.shape[:2]
        cfg = dict(old.CFG)
        cfg.update(old.PROFILES["mountain"])
        pw = cfg["proc_w"]; ph = int(round(sh * pw / sw)); ow = a.out_w; oh = int(round(sh * ow / sw))
        d0 = old.PathDetectorV6(pw, ph, dict(cfg))
        d1 = new.PathDetectorV6(pw, ph, dict(cfg))
        rs = new._ProcResizer(d1, sw, sh, pw, ph)
        assert rs.self_check(f), "ROI resize fast path not active (check source/processing scale)"
        n = 0
        while ok and n < a.frames:
            full = cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA)
            small = rs(f)
            y0, y1, x0, x1 = d1.y0, d1.y1, d1.x0, d1.x1
            if not np.array_equal(full[y0:y1, x0:x1], small[y0:y1, x0:x1]):
                bad["roi_input"] += 1
            L0, R0 = d0.process(full)
            L1, R1 = d1.process(small)
            if not np.array_equal(d0.edges, d1.edges): bad["edges"] += 1
            if not np.array_equal(d0.Lraw, d1.Lraw): bad["Lraw"] += 1
            if not (same(L0, L1) and same(R0, R1) and same(snap(d0.result), snap(d1.result))): bad["result"] += 1
            states[d1.result["lane"]["state"]] = states.get(d1.result["lane"]["state"], 0) + 1
            disp = cv2.resize(f, (ow, oh), interpolation=cv2.INTER_AREA)
            s1 = snap(d1.result)
            o0 = d0.draw(disp.copy(), L0, R0, 30.0, False)
            o1 = d1.draw(disp.copy(), L1, R1, 30.0, False)
            if not np.array_equal(o0, o1): bad["draw"] += 1
            if not same(s1, snap(d1.result)): bad["mutated"] += 1
            o0 = d0.draw(disp.copy(), L0, R0, 30.0, True)
            o1 = d1.draw(disp.copy(), L1, R1, 30.0, True)
            if not np.array_equal(o0, o1): bad["draw_dbg"] += 1
            n += 1; tot += 1
            ok, f = cap.read()
        cap.release()
        print("segment @%5.0fs: %d consecutive frames checked" % (st, n))
    # ---- _blend_fill on random polygons (incl. outside the image) ----
    rng = np.random.default_rng(1)
    nb = 0
    for i in range(600):
        img = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
        k = int(rng.integers(3, 9))
        P = np.stack([rng.integers(-300, 940, k), rng.integers(-200, 560, k)], 1).astype(np.int32)
        ov = img.copy(); cv2.fillPoly(ov, [P], new.GREEN)
        ref = cv2.addWeighted(ov, 0.30, img, 0.70, 0)
        got = new._blend_fill(img.copy(), P, new.GREEN)
        nb += int(not np.array_equal(ref, got))
    print("_blend_fill vs original on 600 random polygons (some partly/fully outside the image): %d mismatches" % nb)
    print("frames %d | states %s" % (tot, states))
    print("mismatching frames:", bad, "| blend mismatches:", nb)
    allok = sum(bad.values()) + nb == 0
    print("RESULT:", "PASS (bit-identical)" if allok else "FAIL")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
