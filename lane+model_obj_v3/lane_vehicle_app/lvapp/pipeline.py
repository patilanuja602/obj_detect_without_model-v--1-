"""
pipeline.py -- per-frame engine.

For every frame t (frame id = running counter) BOTH detectors run on that frame:

    mode "par":   worker thread: vehicle detector(frame t)          main thread: lane detector(frame t)      -> join -> combine
    mode "seq":   lane detector(frame t) then vehicle detector(frame t), one thread                            -> combine
    mode "lane":  lane detector only (baseline-equivalent loop, used for comparisons)

Frame-ID contract: the worker returns the id it was given; the engine asserts it equals the id of the lane result before anything is
combined, and counts frames processed by each detector (they must both equal the number of frames).  No frame skipping, no
alternating detection, no stale detections: a detection from frame t-1 is never presented as a detection of frame t.  (The tracker may
draw a vehicle that was missed for <= max_coast_draw frames; it is flagged `coasted`, drawn dashed, and counted separately.)

The lane detector's only use of vehicle information is the edge-mask PRIOR built from tracks of frames <= t-1 (see lane_v7.py);
the combine step (tracker update, lane association, occlusion-aware drawing) uses the frame-t detections.
"""
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


class Engine:
    def __init__(self, cfg, src_w, src_h, mode=None, with_vehicles=True, resizer=True):
        self.cfg = cfg
        self.mode = mode or cfg["pipeline"]["mode"]
        self.src_w, self.src_h = src_w, src_h
        lane_cfg = dict(LP.CFG)
        lane_cfg.update(LP.PROFILES[cfg["lane"]["profile"]])
        pw = lane_cfg["proc_w"]
        ph = int(round(src_h * pw / src_w))
        self.pw, self.ph = pw, ph
        self.scale = pw / float(src_w)                       # processing px / original px
        lv = cfg["lane_vehicle"]
        self.lane = PathDetectorV7(pw, ph, lane_cfg, cfg6=dict(overlay_top_frac=cfg["lane"]["overlay_top_frac"]),
                                   mask_cfg=dict(mask_enabled=lv["mask_enabled"], mask_inset_x=lv["mask_inset_x"],
                                                 mask_bottom_trim=lv["mask_bottom_trim"], mask_min_h_px=lv["mask_min_h_px"]))
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
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="veh-det") if (self.mode == "par" and self.vdet is not None) else None
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
        fut = None
        if self.pool is not None:
            fut = self.pool.submit(self._detect, frame, fid)
        t0 = pc()
        small = self.rs(frame)
        t1 = pc()
        self.lane.process(small)
        t2 = pc()
        T["rs"], T["lane"] = t1 - t0, t2 - t1
        self.lane_times.append(T["lane"])
        self.n_lane += 1
        # ---- vehicle result for the SAME frame
        det, dt, ok = EMPTY, dict(pre=0, inf=0, post=0, total=0), True
        if fut is not None:
            tw = pc()
            did, det, dt, ok = fut.result()
            T["det_wait"] = pc() - tw
            assert did == fid, "frame-id mismatch: lane %d vs detector %d" % (fid, did)
        elif self.vdet is not None:
            did, det, dt, ok = self._detect(frame, fid)
            T["det_wait"] = 0.0
        if self.vdet is not None:
            self.n_det += 1
            if not ok:
                self.n_det_fail += 1
                self._consec_fail += 1
            else:
                self._consec_fail = 0
        T.update(det_pre=dt["pre"], det_inf=dt["inf"], det_post=dt["post"], det_total=dt["total"])
        # ---- combine
        t3 = pc()
        tracks = self.tracker.update(det, fid) if self.vdet is not None else []
        lane = self.lane.result["lane"]
        association.associate(tracks, lane, self.scale, self.cfg["association"])
        # zone of non-visible (still tentative) tracks is refreshed too, so the confirmation rule sees the current road context
        pend = [t for t in self.tracker.tracks if not t.visible]
        association.associate(pend, lane, self.scale, self.cfg["association"])
        T["track"] = pc() - t3
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
        if self.pool is not None:
            self.pool.shutdown(wait=True)
