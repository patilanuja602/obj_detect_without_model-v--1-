#!/usr/bin/env python3
"""
bench_v6.py -- fair, reproducible benchmark + stage profiler for the lane_path_v6 family.

It loads ANY variant by file path (original V6, V6 + overlay clip, optimised) and drives it with a frame loop that is a
line-for-line copy of what lane_path_v6.main() does (read -> INTER_AREA resize to processing size -> process() ->
INTER_AREA resize to display size -> draw() -> optional VideoWriter.write), timing every step with perf_counter.

Two modes (run both, they answer different questions):

  --mode coarse   NO instrumentation inside the detector.  Per-frame wall time of: decode, resize_proc, process, resize_disp, draw,
                  write (only with --write), plus total.  This is the honest end-to-end number.
  --mode fine     Same loop, but detector methods and selected cv2 calls are wrapped by an EXCLUSIVE-time profiler
                  (a stage's time never includes the stages it calls -> nothing is double counted).  The wrappers cost a little,
                  so use 'fine' for the breakdown and 'coarse' for FPS claims.

Reported per stage: median and p95 in ms.  'algorithm' = process() only (what the on-screen "FPS (algo)" HUD shows).
'pipeline' = everything in the loop except decode and write.  'end-to-end FPS' = measured frames / wall time of the measured loop.

Example:
  python3 bench_v6.py VIDEO --variant lane_path_v6_before.py --mode coarse --start 20 --frames 300 --warmup 30
  python3 bench_v6.py VIDEO --variant lane_path_v6.py        --mode fine   --start 20 --frames 300 --warmup 30 --write /tmp/o.mp4
"""
import argparse, importlib.util, json, os, sys, time
from collections import defaultdict
import cv2
import numpy as np

pc = time.perf_counter


def load(path, name="v6mod"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------ exclusive-time profiler
class Prof:
    def __init__(self):
        self.stack = []                       # [name, t_start, child_time]
        self.cur = defaultdict(float)         # per-frame exclusive seconds
        self.rows = []                        # one dict per measured frame
        self.enabled = False

    def wrap(self, name, fn):
        prof = self

        def w(*a, **k):
            if not prof.enabled:
                return fn(*a, **k)
            fr = [name, pc(), 0.0]
            prof.stack.append(fr)
            try:
                return fn(*a, **k)
            finally:
                dt = pc() - fr[1]
                prof.stack.pop()
                prof.cur[name] += dt - fr[2]
                if prof.stack:
                    prof.stack[-1][2] += dt
        w.__name__ = getattr(fn, "__name__", name)
        return w

    def end_frame(self):
        self.rows.append(dict(self.cur))
        self.cur = defaultdict(float)


class CV2Proxy:
    """Stands in for the `cv2` name inside the detector module; selected callables are timed, the rest pass through."""
    def __init__(self, real, prof, names):
        self._real, self._prof, self._names = real, prof, set(names)
        self._cache = {}

    def __getattr__(self, k):
        if k in self._cache:
            return self._cache[k]
        v = getattr(self._real, k)
        if k in self._names and callable(v):
            v = self._prof.wrap("cv2." + k, v)
        self._cache[k] = v
        return v


class ClaheProxy:
    def __init__(self, real, prof):
        self._real = real
        self.apply = prof.wrap("cv2.CLAHE.apply", real.apply)

    def __getattr__(self, k):
        return getattr(self._real, k)


METHODS = ["_edges", "_segments", "_seed_candidates_both", "_seed_candidates", "_fit_side", "_evidence_batch", "_curve_evidence",
           "_sliding2", "_ridge_pairs", "_track", "_classify", "_corridor", "_fallback", "_btype", "_usable", "_describe",
           "_shape_dist", "_widens", "_clusters", "_paint_static"]
CV2_NAMES = ["cvtColor", "GaussianBlur", "Canny", "HoughLinesP", "bitwise_and", "dilate", "remap", "fillPoly", "addWeighted",
             "polylines", "line", "putText", "rectangle", "resize", "cvtColor", "erode", "copyTo"]


def instrument(mod, det, prof):
    mod.cv2 = CV2Proxy(cv2, prof, CV2_NAMES)
    for m in METHODS:
        if hasattr(det, m):
            setattr(det, m, prof.wrap(m, getattr(det, m)))
    det.clahe = ClaheProxy(det.clahe, prof)
    det.process = prof.wrap("process[own]", det.process)
    det.draw = prof.wrap("draw[own]", det.draw)


def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--variant", required=True, help="path of the lane_path_v6*.py variant to benchmark")
    ap.add_argument("--mode", choices=["coarse", "fine"], default="coarse")
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--frames", type=int, default=300, help="measured frames (after warm-up)")
    ap.add_argument("--warmup", type=int, default=30, help="frames processed but not measured")
    ap.add_argument("--out-w", type=int, default=1280)
    ap.add_argument("--profile", default="mountain")
    ap.add_argument("--no-roi-resize", action="store_true", help="optimised variants only: use the full-frame processing resize")
    ap.add_argument("--write", default=None, help="also write the annotated video here (mp4v, like main()); timed as 'write'")
    ap.add_argument("--show", action="store_true", help="also imshow+waitKey(1); timed as 'show'")
    ap.add_argument("--json", default=None, help="dump the raw result to this file")
    ap.add_argument("--dump-lane", default=None, help="pickle result['lane'] of every frame here (for equivalence checks)")
    a = ap.parse_args()

    mod = load(a.variant)
    cap = cv2.VideoCapture(a.video)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    if a.start:
        cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
    ok, f = cap.read()
    if not ok:
        raise SystemExit("cannot read " + a.video)
    sh, sw = f.shape[:2]
    cfg = dict(mod.CFG)
    cfg.update(mod.PROFILES[a.profile])
    pw = cfg["proc_w"]
    ph = int(round(sh * pw / sw))
    ow = a.out_w
    oh = int(round(sh * ow / sw))
    det = mod.PathDetectorV6(pw, ph, cfg)
    prof = Prof()
    rs = None
    if hasattr(mod, "_ProcResizer") and not a.no_roi_resize:           # optimised variant: same helper its main() uses
        rs = mod._ProcResizer(det, sw, sh, pw, ph)
        rs.self_check(f)
        print("ROI-only processing resize active:", rs.fast)
    if a.mode == "fine":
        instrument(mod, det, prof)
    vw = cv2.VideoWriter(a.write, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (ow, oh)) if a.write else None

    rows, lanes = [], []
    n = 0
    total = a.warmup + a.frames
    t_loop0 = None
    while ok and n < total:
        measured = n >= a.warmup
        if measured and t_loop0 is None:
            t_loop0 = pc()
        prof.enabled = measured and a.mode == "fine"
        t0 = pc()
        small = rs(f) if rs is not None else cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA)
        t1 = pc()
        L, R = det.process(small)
        t2 = pc()
        disp = cv2.resize(f, (ow, oh), interpolation=cv2.INTER_AREA)
        t3 = pc()
        fps_hud = 30.0   # constant HUD value so the text drawing cost is the same in every variant
        out = det.draw(disp, L, R, fps_hud, False)
        t4 = pc()
        if vw:
            vw.write(out)
        t5 = pc()
        if a.show:
            cv2.imshow("bench", out)
            cv2.waitKey(1)
        t6 = pc()
        if a.dump_lane is not None:
            lanes.append(det.result["lane"])
        ok, f = cap.read()
        t7 = pc()
        if measured:
            rows.append(dict(resize_proc=t1 - t0, process=t2 - t1, resize_disp=t3 - t2, draw=t4 - t3, write=t5 - t4, show=t6 - t5,
                             decode=t7 - t6, total=t7 - t0))
            if a.mode == "fine":
                prof.end_frame()
        n += 1
    t_loop = pc() - t_loop0 if t_loop0 is not None else float("nan")
    if vw:
        vw.release()
    cap.release()
    m = len(rows)
    if m == 0:
        raise SystemExit("no measured frames (video too short for --start/--warmup/--frames)")

    def col(k):
        return np.array([r[k] for r in rows]) * 1000.0

    print("variant: %s | mode: %s | video %dx%d -> proc %dx%d, display %dx%d | measured frames %d (+%d warm-up) | write=%s show=%s | cv2 threads=%d"
          % (os.path.basename(a.variant), a.mode, sw, sh, pw, ph, ow, oh, m, a.warmup, bool(a.write), a.show, cv2.getNumThreads()))
    print("%-22s %10s %10s" % ("stage (ms/frame)", "median", "p95"))
    res = {}
    for k in ["decode", "resize_proc", "process", "resize_disp", "draw", "write", "show"]:
        c = col(k)
        if k in ("write", "show") and c.max() == 0:
            continue
        res[k] = (float(np.median(c)), pct(c, 95))
        print("%-22s %10.2f %10.2f" % (k, *res[k]))
    pipe = col("resize_proc") + col("process") + col("resize_disp") + col("draw")
    res["pipeline(no decode/write)"] = (float(np.median(pipe)), pct(pipe, 95))
    print("%-22s %10.2f %10.2f   <- resize+process+resize+draw" % ("pipeline(no IO)", *res["pipeline(no decode/write)"]))
    tot = col("total")
    res["total"] = (float(np.median(tot)), pct(tot, 95))
    print("%-22s %10.2f %10.2f   <- everything incl. decode (+write/show if enabled)" % ("total/frame", *res["total"]))
    algo = col("process")
    print("algorithm-only FPS (1/median process) : %.1f   | pipeline FPS (1/median pipeline, no decode/write): %.1f"
          % (1000 / np.median(algo), 1000 / np.median(pipe)))
    print("END-TO-END FPS (frames / wall time of measured loop, incl. decode%s%s): %.1f   | 1/median(total)=%.1f   | 1/p95(total)=%.1f"
          % (", write" if a.write else "", ", show" if a.show else "", m / t_loop, 1000 / np.median(tot), 1000 / pct(tot, 95)))

    if a.mode == "fine":
        names = sorted({k for r in prof.rows for k in r})
        print("\nEXCLUSIVE time inside process()/draw() (each row excludes what it calls; instrumented => slightly inflated)")
        print("%-24s %9s %9s %9s %8s" % ("stage", "median", "p95", "mean", "calls/f"))
        tab = []
        for k in names:
            c = np.array([r.get(k, 0.0) for r in prof.rows]) * 1000.0
            tab.append((float(np.mean(c)), k, c))
        for mean, k, c in sorted(tab, reverse=True):
            print("%-24s %9.3f %9.3f %9.3f" % (k, np.median(c), pct(c, 95), mean))
        res["fine"] = {k: (float(np.median(c)), pct(c, 95), mean) for mean, k, c in tab}
    if a.json:
        json.dump(dict(args=vars(a), res=res, e2e_fps=m / t_loop), open(a.json, "w"), indent=1)
    if a.dump_lane is not None:
        import pickle
        pickle.dump(lanes, open(a.dump_lane, "wb"))


if __name__ == "__main__":
    main()
