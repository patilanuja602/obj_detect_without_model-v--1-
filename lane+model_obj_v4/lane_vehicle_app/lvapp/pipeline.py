"""
pipeline.py -- per-frame decoupled engine for high-speed lane + vehicle detection.

Modes:
    mode "async" (DEFAULT): Decoupled tracking-by-detection.
                            Main thread runs lane detection + smooth vehicle tracking at 25-30+ FPS.
                            Background worker runs OpenVINO vehicle detector asynchronously.
                            When a detection finishes, tracks are updated with latency compensation;
                            on intermediate frames, tracks advance smoothly via motion prediction.
    mode "par":             Synchronous parallel worker (wait on every frame).
    mode "seq":             Sequential detector + lane (wait on every frame).
    mode "lane":            Lane detector only.
"""
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import association
from .lane_v7 import LP, PathDetectorV7
from .vehicle_detector import EMPTY, VehicleDetector
from .vehicle_tracker import VehicleTracker


def fingerprint(frame):
    """Cheap content fingerprint of a frame (audit only): CRC32 of a sparse sample of its pixels."""
    return zlib.crc32(np.ascontiguousarray(frame[::61, ::67]).tobytes())


class FrameResult:
    __slots__ = ("fid", "lane", "lane_result", "tracks", "raw_det", "t", "det_ok", "mask_boxes")


class FastProcResizer:
    """Fast linear ROI resizer for lane detection. Downsamples only the ROI region."""
    def __init__(self, det, sw, sh, pw, ph, interpolation=cv2.INTER_LINEAR):
        self.pw, self.ph = pw, ph
        self.interpolation = interpolation
        self.sw, self.sh = sw, sh
        self.roi = (det.y0, det.y1, det.x0, det.x1)
        self.buf = np.zeros((ph, pw, 3), np.uint8)
        self.k_y = sh / float(ph)
        self.k_x = sw / float(pw)
        self.sy0 = max(0, int(round(det.y0 * self.k_y)))
        self.sy1 = min(sh, int(round(det.y1 * self.k_y)))
        self.sx0 = max(0, int(round(det.x0 * self.k_x)))
        self.sx1 = min(sw, int(round(det.x1 * self.k_x)))
        self.target_size = (det.x1 - det.x0, det.y1 - det.y0)
        self.dy0, self.dy1 = det.y0, det.y1
        self.dx0, self.dx1 = det.x0, det.x1

    def __call__(self, f):
        if self.sy1 > self.sy0 and self.sx1 > self.sx0:
            self.buf[self.dy0:self.dy1, self.dx0:self.dx1] = cv2.resize(
                f[self.sy0:self.sy1, self.sx0:self.sx1], self.target_size, interpolation=self.interpolation)
        return self.buf

    def self_check(self, f):
        pass


class AsyncDetWorker:
    """Non-blocking background detector worker.
    Runs inference continuously on the latest submitted frame without accumulating queue backlog.
    """
    def __init__(self, vdet):
        self.vdet = vdet
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.pending_frame = None
        self.pending_fid = -1
        self.in_progress = False
        self.result = None
        self.has_result = False
        self.running = True
        self.thread = threading.Thread(target=self._run, name="async-det-worker", daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            with self.cond:
                while self.running and self.pending_frame is None:
                    self.cond.wait()
                if not self.running:
                    break
                frame = self.pending_frame
                fid = self.pending_fid
                self.pending_frame = None
                self.in_progress = True

            try:
                det, dt = self.vdet.detect(frame)
                ok = True
            except Exception as e:
                det, dt, ok = EMPTY, dict(pre=0, inf=0, post=0, total=0), False

            with self.cond:
                self.result = (fid, det, dt, ok)
                self.has_result = True
                self.in_progress = False

    def submit(self, frame, fid):
        """Submit a frame. If worker is idle, starts immediately. If busy, updates pending frame."""
        with self.cond:
            self.pending_frame = frame
            self.pending_fid = fid
            self.cond.notify()

    def poll_result(self):
        """Returns (fid, det, dt, ok) if a completed detection is ready, else None."""
        with self.cond:
            if self.has_result:
                self.has_result = False
                return self.result
            return None

    def is_idle(self):
        with self.cond:
            return not self.in_progress and self.pending_frame is None

    def close(self):
        with self.cond:
            self.running = False
            self.cond.notify()
        self.thread.join(timeout=1.0)


class Engine:
    def __init__(self, cfg, src_w, src_h, mode=None, with_vehicles=True, resizer=True):
        self.cfg = cfg
        self.mode = mode or cfg["pipeline"]["mode"]
        self.src_w, self.src_h = src_w, src_h
        lane_cfg = dict(LP.CFG)
        lane_cfg.update(LP.PROFILES[cfg["lane"]["profile"]])
        if "proc_w" in cfg["lane"]:
            lane_cfg["proc_w"] = cfg["lane"]["proc_w"]
        if "hough_max_gap" in cfg["lane"]:
            lane_cfg["hough_max_gap"] = cfg["lane"]["hough_max_gap"]
        if "hough_theta" in cfg["lane"]:
            lane_cfg["hough_theta"] = cfg["lane"]["hough_theta"]
        pw = lane_cfg["proc_w"]
        if "hough_min_len" in cfg["lane"]:
            lane_cfg["hough_min_len"] = cfg["lane"]["hough_min_len"]
        elif pw != 640:
            lane_cfg["hough_min_len"] = max(10, int(round(20.0 * pw / 640.0)))
        ph = int(round(src_h * pw / src_w))
        self.pw, self.ph = pw, ph
        self.scale = pw / float(src_w)                       # processing px / original px
        lv = cfg["lane_vehicle"]
        cfg2 = {}
        if "max_hold" in cfg["lane"]:
            cfg2["max_hold2"] = cfg["lane"]["max_hold"]
        cfg6 = dict(overlay_top_frac=cfg["lane"]["overlay_top_frac"])
        if "max_hold" in cfg["lane"]:
            cfg6["unsup_grace"] = max(10, cfg["lane"]["max_hold"])
        self.lane = PathDetectorV7(pw, ph, lane_cfg, cfg2=cfg2, cfg6=cfg6,
                                   mask_cfg=dict(mask_enabled=lv["mask_enabled"], mask_inset_x=lv["mask_inset_x"],
                                                 mask_bottom_trim=lv["mask_bottom_trim"], mask_min_h_px=lv["mask_min_h_px"]))
        if cfg["pipeline"].get("fast_resizer", True):
            self.rs = FastProcResizer(self.lane, src_w, src_h, pw, ph, cv2.INTER_LINEAR)
        else:
            self.rs = LP._ProcResizer(self.lane, src_w, src_h, pw, ph, enabled=resizer)
        self.vdet, self.det_error = None, None
        self.tracker = VehicleTracker(cfg["tracker"])
        if with_vehicles and self.mode != "lane":
            try:
                d = cfg["detector"]
                self.vdet = VehicleDetector(d["config"], d["threads"], d["conf_low"], d["nms_iou"], d["cross_class_iou"], d["resize"],
                                            d["hyperthreading"], d["ov_preprocess"])
            except Exception as e:                           # noqa: BLE001 -- lane detection must keep working without the detector
                self.det_error = "vehicle detector unavailable: %s" % e
                print("WARNING:", self.det_error)

        self.async_worker = None
        self.pool = None
        if self.vdet is not None:
            if self.mode == "async":
                self.async_worker = AsyncDetWorker(self.vdet)
            elif self.mode == "par":
                self.pool = ThreadPoolExecutor(1, thread_name_prefix="veh-det")

        self.n_lane = self.n_det = self.n_det_fail = 0
        self._consec_fail = 0
        self.audit = None                                    # set to a dict to record per-frame fingerprints (benchmark --audit)
        self.lane_times = []

    # ------------------------------------------------------------------ detector task (worker thread in "par" mode)
    def _detect(self, frame, fid):
        try:
            if self.audit is not None:
                self.audit.setdefault(fid, {})["det"] = fingerprint(frame)
            d, t = self.vdet.detect(frame)
            return fid, d, t, True
        except Exception as e:                               # noqa: BLE001
            self.det_error = "detector failed on frame %d: %s" % (fid, e)
            return fid, EMPTY, dict(pre=0, inf=0, post=0, total=0), False

    def warmup(self, frame):
        if self.vdet is not None:                            # first inference allocates; keep it out of the timing
            self.vdet.detect(frame)
        self.rs.self_check(frame)
        small = self.rs(frame)
        self.lane.process(small)

    # ------------------------------------------------------------------ one frame
    def process(self, frame, fid):
        pc = time.perf_counter
        T = {}
        t_start = pc()
        # lane mask prior from reliable tracks of EARLIER frames
        lv = self.cfg["lane_vehicle"]
        if self.vdet is not None and lv["mask_enabled"]:
            boxes = [b * self.scale for b in self.tracker.predicted_boxes(lv["mask_min_hits"], lv["mask_min_score"])]
        else:
            boxes = []
        self.lane.set_vehicle_boxes(boxes)
        if self.audit is not None:
            self.audit.setdefault(fid, {})["lane"] = fingerprint(frame)

        # ---------------- Lane Detection ----------------
        t0 = pc()
        small = self.rs(frame)
        t1 = pc()
        self.lane.process(small)
        t2 = pc()
        T["rs"], T["lane"] = t1 - t0, t2 - t1
        self.lane_times.append(T["lane"])
        self.n_lane += 1

        # ---------------- Vehicle Detection & Tracking ----------------
        det, dt, ok = EMPTY, dict(pre=0, inf=0, post=0, total=0), True

        if self.mode == "async" and self.async_worker is not None:
            res = self.async_worker.poll_result()
            t3 = pc()
            if res is not None:
                did, det, dt, ok = res
                self.n_det += 1
                if not ok:
                    self.n_det_fail += 1
                tracks = self.tracker.update_async(det, did, fid)
                # Immediately dispatch newest current frame to worker
                self.async_worker.submit(frame, fid)
            else:
                tracks = self.tracker.predict_step(fid)
                if self.async_worker.is_idle():
                    self.async_worker.submit(frame, fid)
            T["track"] = pc() - t3
            T["det_wait"] = 0.0
            T.update(det_pre=dt["pre"], det_inf=dt["inf"], det_post=dt["post"], det_total=dt["total"])
        elif self.mode == "par" and self.pool is not None:
            # Synchronous parallel mode
            fut = self.pool.submit(self._detect, frame, fid)
            tw = pc()
            did, det, dt, ok = fut.result()
            T["det_wait"] = pc() - tw
            assert did == fid, "frame-id mismatch: lane %d vs detector %d" % (fid, did)
            self.n_det += 1
            if not ok:
                self.n_det_fail += 1
            T.update(det_pre=dt["pre"], det_inf=dt["inf"], det_post=dt["post"], det_total=dt["total"])
            t3 = pc()
            tracks = self.tracker.update(det, fid)
            T["track"] = pc() - t3
        elif self.vdet is not None and self.mode != "lane":
            # Synchronous sequential mode
            did, det, dt, ok = self._detect(frame, fid)
            self.n_det += 1
            if not ok:
                self.n_det_fail += 1
            T["det_wait"] = 0.0
            T.update(det_pre=dt["pre"], det_inf=dt["inf"], det_post=dt["post"], det_total=dt["total"])
            t3 = pc()
            tracks = self.tracker.update(det, fid)
            T["track"] = pc() - t3
        else:
            tracks = []
            T["track"] = 0.0
            T["det_wait"] = 0.0
            T.update(det_pre=0, det_inf=0, det_post=0, total=0)

        # ---------------- Lane Association ----------------
        lane = self.lane.result["lane"]
        if tracks:
            association.associate(tracks, lane, self.scale, self.cfg["association"])
            pend = [t for t in self.tracker.tracks if not t.visible]
            association.associate(pend, lane, self.scale, self.cfg["association"])

        T["compute"] = pc() - t_start
        r = FrameResult()
        r.fid, r.lane, r.lane_result, r.tracks, r.raw_det, r.t, r.det_ok = fid, lane, self.lane.result, tracks, det, T, ok
        r.mask_boxes = self.lane.mask_boxes_used
        return r

    # ------------------------------------------------------------------ display
    def draw(self, disp, res, fps_text=None):
        """disp: display-size frame (modified in place).  Returns it."""
        from . import overlay
        lv = self.cfg["lane_vehicle"]
        s_disp = disp.shape[1] / float(self.src_w)
        patches = overlay.snapshot_bodies(disp, res.tracks, s_disp, lv["mask_inset_x"], lv["mask_bottom_trim"]) if (lv["restore_overlay"] and res.tracks) else []
        fps_lane = (1.0 / np.mean(self.lane_times[-30:])) if self.lane_times else None
        self.lane.draw(disp, None, None, fps_lane)
        if patches:
            overlay.restore_bodies(disp, patches)
        overlay.draw_vehicles(disp, res.tracks, s_disp)
        zc = {}
        for t in res.tracks:
            zc[t.zone] = zc.get(t.zone, 0) + 1
        hud = "%s  veh:%d  [%s]  frame %d" % (fps_text or "", len(res.tracks), " ".join("%s:%d" % (association.SHORT[k], v) for k, v in zc.items()), res.fid)
        overlay.draw_hud(disp, hud)
        if self.det_error:
            cv2.putText(disp, "vehicle detector: " + self.det_error[:80], (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return disp

    def close(self):
        if self.async_worker is not None:
            self.async_worker.close()
        if self.pool is not None:
            self.pool.shutdown(wait=True)
