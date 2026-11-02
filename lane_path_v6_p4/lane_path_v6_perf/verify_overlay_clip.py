#!/usr/bin/env python3
"""Verify the display-only overlay clip on REAL video: original V6 (lane_path_v6_before.py) vs patched V6 (lane_path_v6.py).

For every processed frame it checks
  1. result["lane"] (state, extent_y, corridor, pts_L, pts_R, ...), result["corridor"], result["partial"], result["mode"] are IDENTICAL
     between the original detector and the patched detector (two independent detectors fed the same frames), before AND after draw();
  2. draw() with the clip disabled (overlay_top_y=0) is pixel-identical to the original draw();
  3. draw() with the clip: rows well below the clip row are identical to the original overlay, rows above it contain no overlay;
  4. timing of the two draw() paths (draw only, same frames; wall-clock on THIS machine).
Usage:  python3 verify_overlay_clip.py VIDEO [--start S] [--frames N] [--out-w 1280] [--clip-y PX | --clip-frac F] [--save-dir DIR]
"""
import argparse, copy, importlib.util, os, sys, time
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def same(a, b):
    if isinstance(a, dict): return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray): return isinstance(b, np.ndarray) and a.shape == b.shape and np.array_equal(a, b)
    return a == b


def snap(res):
    return copy.deepcopy({k: res.get(k) for k in ("lane", "corridor", "partial", "mode", "extent_y", "offset", "reason")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--start", type=float, default=0); ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--out-w", type=int, default=1280); ap.add_argument("--profile", default="mountain")
    ap.add_argument("--clip-y", type=float, default=None); ap.add_argument("--clip-frac", type=float, default=None)
    ap.add_argument("--before", default=os.path.join(HERE, "lane_path_v6_before.py")); ap.add_argument("--save-dir", default=None)
    a = ap.parse_args()
    old, new = load("v6_before", a.before), load("v6_after", os.path.join(HERE, "lane_path_v6.py"))
    cap = cv2.VideoCapture(a.video); fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    if a.start: cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
    ok_, f = cap.read(); assert ok_, "cannot read " + a.video
    sh, sw = f.shape[:2]; pw = 640; ph = int(round(sh * pw / sw)); ow = a.out_w; oh = int(round(sh * ow / sw))
    cfg = dict(old.CFG); cfg.update(old.PROFILES[a.profile])
    c6 = {}
    if a.clip_y is not None: c6["overlay_top_y"] = a.clip_y
    if a.clip_frac is not None: c6["overlay_top_frac"] = a.clip_frac
    d0, d1 = old.PathDetectorV6(pw, ph, cfg), new.PathDetectorV6(pw, ph, dict(cfg), cfg6=c6)
    yc = d1.c6["overlay_top_y"] if d1.c6["overlay_top_y"] is not None else d1.c6["overlay_top_frac"] * ph
    yc_disp = yc * ow / pw
    print("%s | processing %dx%d display %dx%d | clip row %.1f proc px = %.1f display px (%.0f%% of height)" % (os.path.basename(a.video), pw, ph, ow, oh, yc, yc_disp, 100 * yc / ph))
    stats = dict(frames=0, geom_diff=0, geom_mut=0, off_diff=0, below_diff=0, above_overlay=0)
    states = {}; t_old = []; t_new = []; worst_band = 0; saved = 0
    n = 0
    while ok_ and n < a.frames:
        small = cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA); disp = cv2.resize(f, (ow, oh), interpolation=cv2.INTER_AREA)
        d0.process(small); d1.process(small)
        s0 = snap(d0.result); s1 = snap(d1.result)
        if not same(s0, s1): stats["geom_diff"] += 1
        st = d1.result["lane"]["state"]; states[st] = states.get(st, 0) + 1
        t = time.perf_counter(); o0 = d0.draw(disp.copy(), None, None, None, False); t_old.append(time.perf_counter() - t)
        t = time.perf_counter(); o1 = d1.draw(disp.copy(), None, None, None, False); t_new.append(time.perf_counter() - t)
        if not same(s1, snap(d1.result)): stats["geom_mut"] += 1                       # draw() must not change the result
        keep = d1.c6["overlay_top_y"]; d1.c6["overlay_top_y"] = 0                      # clip disabled -> must equal the original overlay exactly
        if not np.array_equal(d1.draw(disp.copy(), None, None, None, False), o0): stats["off_diff"] += 1
        d1.c6["overlay_top_y"] = keep
        lo = int(yc_disp) + 14                                                           # one corridor sample step + line thickness
        if not np.array_equal(o0[lo:], o1[lo:]): stats["below_diff"] += 1
        hi = max(int(yc_disp) - 6, 34)
        if hi > 34 and not np.array_equal(o1[34:hi], disp[34:hi]): stats["above_overlay"] += 1   # nothing but the plain video above the clip row
        if a.save_dir and n % max(1, a.frames // 4) == 0:
            os.makedirs(a.save_dir, exist_ok=True); both = np.hstack([o0, o1]); cv2.imwrite(os.path.join(a.save_dir, "%s_%03d.jpg" % (os.path.splitext(os.path.basename(a.video))[0][:10], n)), cv2.resize(both, None, fx=0.5, fy=0.5)); saved += 1
        n += 1; stats["frames"] = n; ok_, f = cap.read()
    cap.release()
    print("frames %d | states %s" % (n, states))
    print("internal result differs original vs patched: %d frames | draw() mutated result: %d | clip OFF != original pixels: %d | below-clip pixels differ: %d | overlay above clip row: %d"
          % (stats["geom_diff"], stats["geom_mut"], stats["off_diff"], stats["below_diff"], stats["above_overlay"]))
    to, tn = np.array(t_old[3:]) * 1e6, np.array(t_new[3:]) * 1e6
    print("draw() only (this machine): original mean %.0f us (p95 %.0f) | patched mean %.0f us (p95 %.0f) | delta %+.0f us/frame" % (to.mean(), np.percentile(to, 95), tn.mean(), np.percentile(tn, 95), tn.mean() - to.mean()))
    bad = stats["geom_diff"] + stats["geom_mut"] + stats["off_diff"] + stats["below_diff"] + stats["above_overlay"]
    print("RESULT:", "PASS" if bad == 0 else "FAIL"); sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
