"""
pipeline.py -- the application loop.

Per frame (every frame, nothing is skipped):
   decode (background thread) -> display resize (background thread, exactly as the V6 baseline main()) ->
   [ vehicle detector job (worker thread)  ||  lane detector (main thread) ]  -> join on the SAME frame id ->
   tracker -> road association -> drawing -> output.

Synchronisation: every detector job carries the frame id; the main thread asserts the returned id equals the current id before using the result.
There is no queue of old results, so a stale detection can never be drawn.  If the detector raises, that frame has no detections and the lane
pipeline continues unchanged (failures are counted).
Lane vehicle-awareness uses the tracker's boxes from the PREVIOUS frame (moved one frame ahead by their velocity) as an occlusion prior, so the lane
does not have to wait for the detector and the two still overlap in time.  `lane.mask_mode = same_frame` instead runs the detector first (serial)
and uses this frame's tracks; `off` disables vehicle awareness (baseline lane behaviour).
Frame integrity: a cheap fingerprint of every frame is recorded and verified (see benchmark.py): lane and detector must see the same frame,
and the processed frames must equal a plain sequential decode (no dropped / reordered frames).
"""
import os
import queue
import resource
import threading
import time

import cv2
import numpy as np
import psutil

from . import association, config, overlay
from .lane_engine import LaneEngine
from .vehicle_detector import VehicleDetector
from .vehicle_tracker import Track, VehicleTracker

KEYS = ["lane_prep", "lane", "det_pre", "det_inf", "det_post", "det_wait", "track", "assoc", "draw_lane", "draw_veh", "output", "wait_next", "total"]


def fp(a):
    return int(a[::64, ::64].sum(dtype=np.int64))


def stats(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return dict(mean=0.0, median=0.0, p95=0.0, min=0.0, max=0.0)
    return dict(mean=float(v.mean()), median=float(np.median(v)), p95=float(np.percentile(v, 95)), min=float(v.min()), max=float(v.max()))


class DetWorker:
    """Runs the detector for exactly one frame per submit(); result() blocks until THAT job is finished and returns its frame id."""

    def __init__(self, det):
        self.det, self.qi, self.qo = det, queue.Queue(1), queue.Queue(1)
        threading.Thread(target=self._run, name="vehicle-detector", daemon=True).start()

    def _run(self):
        while True:
            job = self.qi.get()
            if job is None:
                return
            fid, x, meta, fpv, t_pre = job
            try:
                t1 = time.perf_counter()
                o = self.det.infer(x)
                t2 = time.perf_counter()
                d = self.det.postprocess(o, meta)
                t3 = time.perf_counter()
                self.qo.put((fid, d, t_pre, (t2 - t1) * 1e3, (t3 - t2) * 1e3, None, fpv))
            except BaseException as e:                              # never let a detector failure reach the lane code
                self.qo.put((fid, np.zeros((0, 6), np.float32), t_pre, 0.0, 0.0, e, fpv))

    def submit(self, fid, x, meta, fpv, t_pre):
        self.qi.put((fid, x, meta, fpv, t_pre))

    def result(self):
        return self.qo.get()

    def close(self):
        self.qi.put(None)


def xat(pts, y):
    if pts is None or len(pts) < 2:
        return float("nan")
    ys, xs = pts[:, 1], pts[:, 0]
    if y > ys.max() or y < ys.min():
        return float("nan")
    o = np.argsort(ys)
    return float(np.interp(y, ys[o], xs[o]))


class Pipeline:
    def __init__(self, cfg, source, frames=None, warmup=0, output=None, out_file=None, log=False, quiet=False):
        self.cfg, self.source, self.frames, self.warmup = cfg, source, frames, warmup
        self.output = output or cfg.pipeline.output
        self.out_file = str(config.resolve(out_file or cfg.pipeline.out_file))
        self.log, self.quiet = log, quiet
        self.records, self.fps_main, self.fps_det, self.ids_ok = [], [], [], True

    def run(self):
        cfg = self.cfg
        if cfg.pipeline.cv_threads is not None:
            cv2.setNumThreads(int(cfg.pipeline.cv_threads))
        if self.output == "show" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise SystemExit("--output show needs a graphical session (no DISPLAY); use --output file or --output none")
        src = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(src)
        if cfg.source.start_sec and not isinstance(src, int):
            cap.set(cv2.CAP_PROP_POS_MSEC, cfg.source.start_sec * 1000)
        ok, f = cap.read()
        if not ok:
            raise SystemExit("cannot read source %s" % self.source)
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
        sh, sw = f.shape[:2]
        ow = int(cfg.pipeline.display_w)
        oh = int(round(sh * ow / sw))
        lane = LaneEngine(cfg, sw, sh)
        lane.self_check(f)
        lp = lane.lp
        k = ow / lane.pw                                              # display px per processing px
        vc = cfg.vehicle
        det = VehicleDetector(config.resolve(vc.model_file), band=vc.band, threads=vc.threads, conf=vc.conf, resize=vc.resize, nms_mode=vc.nms_mode,
                              nms_iou=vc.nms_iou, reject_nonvehicle_ratio=vc.reject_nonvehicle_ratio, min_height_px=vc.min_height_px, aspect_range=vc.aspect_range)
        use_disp = vc.input_source == "display"
        kd = 1.0 if use_disp else ow / sw                             # detector coords -> display coords
        tcfg = cfg.tracker
        tracker = VehicleTracker(**tcfg) if tcfg.enabled else None
        acfg = cfg.association
        assoc = association.RoadAssociator(**acfg) if acfg.enabled else None
        mask_mode = cfg.lane.mask_mode if tracker is not None else "off"
        parallel = bool(cfg.pipeline.parallel) and mask_mode != "same_frame"
        worker = DetWorker(det) if parallel else None
        dec_thr = bool(cfg.pipeline.decode_threads)

        interp = cv2.INTER_AREA
        prep_stage = bool(cfg.pipeline.preprocess_in_stage)

        def stage_fn(fr):
            """Runs in the background stage thread (like the baseline's display resize): display frame + the detector's input tensor for THIS frame,
            so the expensive preprocessing overlaps with the previous frame's processing instead of sitting on the critical path."""
            dsp = cv2.resize(fr, (ow, oh), interpolation=interp)
            if not prep_stage:
                return dsp, None
            x, meta, tp = det.preprocess(dsp if use_disp else fr)
            return dsp, (x, meta, tp, fp(fr))

        rd = st = None
        if dec_thr:
            rd = lp._Prefetch(cap)
            st = lp._Stage(rd.get, stage_fn)
        disp, prepared = stage_fn(f)
        vw = sink = None
        if self.output == "file":
            os.makedirs(os.path.dirname(self.out_file), exist_ok=True)
            vw = cv2.VideoWriter(self.out_file, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (ow, oh))
            sink = lp._AsyncWriter(vw) if dec_thr else vw

        T = {kk: [] for kk in KEYS}
        lane_times = []
        n = n_timed = n_lane = n_det = failures = 0
        total_frames = (self.warmup + self.frames) if self.frames else 10 ** 9
        proc = psutil.Process()
        c0 = t_loop = None
        sys_cpu = None
        nveh = []
        fps_e2e = 0.0
        t_hist = []
        try:
            while f is not None and n < total_frames:
                fid = n
                if n == self.warmup:
                    c0 = proc.cpu_times()
                    psutil.cpu_percent(percpu=True)
                    t_loop = time.perf_counter()
                rec = n >= self.warmup
                det_in = disp if use_disp else f
                fpf = fp(f)
                self.fps_main.append(fpf)
                ts = time.perf_counter()
                r = {kk: 0.0 for kk in KEYS}
                dets = np.zeros((0, 6), np.float32)

                def run_det_inline():
                    nonlocal dets, failures
                    try:
                        if prepared is not None:
                            x, meta, tp = prepared[0], prepared[1], prepared[2]
                        else:
                            x, meta, tp = det.preprocess(det_in)
                        t1 = time.perf_counter()
                        o = det.infer(x)
                        t2 = time.perf_counter()
                        dets = det.postprocess(o, meta)
                        t3 = time.perf_counter()
                        r["det_pre"], r["det_inf"], r["det_post"] = tp, (t2 - t1) * 1e3, (t3 - t2) * 1e3
                    except BaseException:
                        failures += 1
                        dets = np.zeros((0, 6), np.float32)
                    self.fps_det.append(fpf)

                tracks = []
                tracks_updated = False
                same_frame = mask_mode == "same_frame"

                def update_tracks():
                    """detections (detector-input coords) -> display coords -> tracker (or plain untracked tracks)."""
                    nonlocal dets, tracks
                    t0 = time.perf_counter()
                    d2 = dets.copy()
                    d2[:, :4] *= kd
                    dets = d2
                    if tracker is not None:
                        tracks = tracker.update(d2, fid)
                    else:
                        tracks = [Track(i, d, fid, True) for i, d in enumerate(d2)]      # plain boxes, as in the old system
                    r["track"] += (time.perf_counter() - t0) * 1e3

                if worker is not None:
                    if prepared is not None:
                        worker.submit(fid, prepared[0], prepared[1], prepared[3], prepared[2])      # tensor already built by the stage thread
                    else:
                        x_, meta_, tp_ = det.preprocess(det_in)                                    # (preprocess_in_stage=false: on the critical path)
                        worker.submit(fid, x_, meta_, fpf, tp_)
                elif same_frame:
                    run_det_inline()                                 # serial: detector first, then the lane sees THIS frame's tracks
                    update_tracks()
                    tracks_updated = True
                # ---------------- lane detector (main thread) ----------------
                if mask_mode != "off":
                    boxes = tracker.mask_boxes(cfg.lane.mask_min_conf, cfg.lane.mask_min_hits)
                    # near-field vehicles only: distant ones sit at the vanishing point where every lane line converges; masking them would delete
                    # the far-field evidence of both boundaries (measured regression) while they hide almost none of a marking.
                    lane.set_mask([b / k for b in boxes if (b[3] - b[1]) / k >= cfg.lane.mask_min_height_px])
                t0 = time.perf_counter()
                small = lane.prepare(f)
                t1 = time.perf_counter()
                L, R, lane_res = lane.run(small)
                t2 = time.perf_counter()
                r["lane_prep"], r["lane"] = (t1 - t0) * 1e3, (t2 - t1) * 1e3
                lane_times.append(t2 - t1)
                n_lane += 1
                # ---------------- join the detector on the SAME frame ----------------
                if worker is not None:
                    t0 = time.perf_counter()
                    rfid, dets, r["det_pre"], r["det_inf"], r["det_post"], err, fpd = worker.result()
                    r["det_wait"] = (time.perf_counter() - t0) * 1e3
                    if rfid != fid:
                        self.ids_ok = False
                    failures += err is not None
                    self.fps_det.append(fpd)
                elif not same_frame:
                    run_det_inline()
                n_det += 1
                if not tracks_updated:
                    update_tracks()
                # ---------------- association ----------------
                snap = None
                if assoc is not None:
                    t0 = time.perf_counter()
                    snap = association.snapshot(lane_res)
                    assoc.update(tracks, snap, k)
                    r["assoc"] = (time.perf_counter() - t0) * 1e3
                # ---------------- drawing ----------------
                t0 = time.perf_counter()
                saved = overlay.cut_regions(disp, tracks, cfg.lane.mask_min_conf) if cfg.lane.overlay_cut_vehicles else None
                out_frame = lane.draw(disp, L, R, 1.0 / np.mean(lane_times[-30:]))
                if saved:
                    overlay.restore_regions(out_frame, saved)
                t1 = time.perf_counter()
                overlay.draw_vehicles(out_frame, tracks, "zones" if assoc is not None else "plain", acfg.outside_policy)
                if len(t_hist) >= 2:
                    fps_e2e = (len(t_hist) - 1) / (t_hist[-1] - t_hist[0])
                overlay.hud_line(out_frame, "end-to-end %.1f FPS | lane %.1f ms | detector %.1f ms | vehicles %d%s" % (
                    fps_e2e, r["lane"], r["det_pre"] + r["det_inf"] + r["det_post"], len(tracks), " | lane-aware" if mask_mode != "off" else ""))
                t2 = time.perf_counter()
                r["draw_lane"], r["draw_veh"] = (t1 - t0) * 1e3, (t2 - t1) * 1e3
                t0 = time.perf_counter()
                if sink is not None:
                    sink.write(out_frame)
                if self.output == "show":
                    cv2.imshow("lane + vehicles (q to quit)", out_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                r["output"] = (time.perf_counter() - t0) * 1e3
                t_hist.append(time.perf_counter())
                del t_hist[:-31]
                if self.log:
                    self.records.append(dict(
                        fid=fid, state=lane_res["state"], mode=self.lane_mode(lane), extent=lane_res.get("extent_y"), mask_frac=lane.det.veh_mask_frac,
                        **{"%s%d" % (s, y): xat(lane_res["pts_" + s], y) for s in "LR" for y in (340, 300, 260)},
                        Ltype=None if lane_res.get("L") is None else lane_res["L"].get("type"), Rtype=None if lane_res.get("R") is None else lane_res["R"].get("type"),
                        raw=dets.copy(), tracks=[(t.id, t.box.copy(), t.cls, float(t.score), t.zone, bool(t.coasting), t.ground) for t in tracks]))
                n += 1
                t0 = time.perf_counter()
                if st is not None:
                    nx = st.get()
                    if nx is None:
                        f = disp = prepared = None
                    else:
                        f, (disp, prepared) = nx
                else:
                    ok, f = cap.read()
                    if ok:
                        disp, prepared = stage_fn(f)
                    else:
                        f = disp = prepared = None
                r["wait_next"] = (time.perf_counter() - t0) * 1e3
                r["total"] = (time.perf_counter() - ts) * 1e3
                if rec:
                    for kk in KEYS:
                        T[kk].append(r[kk])
                    nveh.append(len(tracks))
                    n_timed += 1
        finally:
            wall = (time.perf_counter() - t_loop) if t_loop else 0.0
            c1 = proc.cpu_times()
            if c0 is not None:
                sys_cpu = psutil.cpu_percent(percpu=True)
            if worker is not None:
                worker.close()
            if st is not None:
                st.close()
            if rd is not None:
                rd.close()
            if sink is not None and sink is not vw:
                sink.close()
            if vw is not None:
                vw.release()
            cap.release()
            if self.output == "show":
                cv2.destroyAllWindows()
        if n_timed == 0:
            raise SystemExit("source ended before the timed window started")
        S = {kk: stats(v) for kk, v in T.items()}
        detms = np.array(T["det_pre"]) + np.array(T["det_inf"]) + np.array(T["det_post"])
        cpu_cores = ((c1.user + c1.system) - (c0.user + c0.system)) / wall if wall else 0.0
        frames_ok = self.ids_ok and (self.fps_det == self.fps_main[:len(self.fps_det)]) and len(self.fps_det) == len(self.fps_main)
        return dict(
            frames_timed=n_timed, frames_total=n, lane_calls=n_lane, det_calls=n_det, dropped_frames=n - n_lane, detector_failures=int(failures),
            lane_and_detector_same_frame=bool(frames_ok), stages_ms=S, detector_ms=stats(detms),
            fps=dict(end_to_end_wall=n_timed / wall, from_mean_total=1000.0 / S["total"]["mean"], p95_frame=1000.0 / S["total"]["p95"],
                     lane_only_algo=1000.0 / (S["lane"]["mean"] + S["lane_prep"]["mean"]), detector_only=1000.0 / float(detms.mean())),
            cpu=dict(process_cores_used=cpu_cores, system_percent_mean=float(np.mean(sys_cpu)) if sys_cpu else None,
                     freq_mhz=(psutil.cpu_freq().current if psutil.cpu_freq() else None)),
            memory=dict(rss_MB=proc.memory_info().rss / 2 ** 20, peak_rss_MB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
            mean_vehicles_per_frame=float(np.mean(nveh)) if nveh else 0.0, detector_compile_s=det.compile_s, lane_file_sha256=lp.__sha256__,
            cv2_threads=cv2.getNumThreads(), vehicle_threads=det.threads, detector_input="%dx%d band=%s source=%s" % (det.w, det.h, vc.band, vc.input_source),
            mode=dict(parallel=parallel, lane_mask_mode=mask_mode, tracker=bool(tracker), association=bool(assoc)))

    @staticmethod
    def lane_mode(lane):
        return lane.det.result.get("mode")
