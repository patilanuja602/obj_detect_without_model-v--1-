"""
baseline_system.py -- the ORIGINAL system, reproduced from the untouched baseline files, for before/after comparisons:
  baseline/lane_path_v6_p3.py  (PathDetectorV6, unchanged)  +  baseline/vehicle_det_baseline.py  (Phase-2.5 detector, conf 0.25,
  class-aware NMS only, no tracking, no association, no lane mask, overlay painted over vehicles as in the original loop).
Run sequentially (lane then detector on the same frame) exactly like bench_pipeline.py --mode seq.
"""
import importlib.util
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .lane_v7 import LP, ROOT


def _load_vd():
    key = "vehicle_det_baseline"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, str(ROOT / "baseline" / "vehicle_det_baseline.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[key] = m
    spec.loader.exec_module(m)
    m.MODELS = ROOT / "models"                              # the baseline file resolves models relative to its own parent
    return m


VD = _load_vd()


class _Res:
    __slots__ = ("fid", "lane", "lane_result", "tracks", "raw_det", "t", "det_ok", "mask_boxes", "boxes")


class BaselineEngine:
    mode = "seq"

    def __init__(self, cfg, src_w, src_h, det_config="A2", threads=None, conf=0.25, mode=None, with_vehicles=True, resizer=True):
        lane_cfg = dict(LP.CFG)
        lane_cfg.update(LP.PROFILES[cfg["lane"]["profile"]])
        pw = lane_cfg["proc_w"]
        ph = int(round(src_h * pw / src_w))
        self.pw, self.ph, self.src_w, self.src_h = pw, ph, src_w, src_h
        self.scale = pw / float(src_w)
        self.lane = LP.PathDetectorV6(pw, ph, lane_cfg, cfg6=dict(overlay_top_frac=cfg["lane"]["overlay_top_frac"]))
        self.rs = LP._ProcResizer(self.lane, src_w, src_h, pw, ph, enabled=resizer)
        path, band = VD.resolve(det_config)
        self.vdet = VD.VehicleDetector(path, band, threads or cfg["detector"]["threads"], conf=conf, iou=0.5, resize="area") if with_vehicles else None
        self.n_lane = self.n_det = self.n_det_fail = 0
        self.det_error = None
        self.audit = None

    def warmup(self, frame):
        if self.vdet is not None:
            self.vdet.detect(frame)
        self.rs.self_check(frame)

    def process(self, frame, fid):
        pc = time.perf_counter
        T = {}
        if self.audit is not None:
            from .pipeline import fingerprint
            fp = fingerprint(frame)
            self.audit.setdefault(fid, {}).update(lane=fp, det=fp)
        t0 = pc()
        small = self.rs(frame)
        t1 = pc()
        self.lane.process(small)
        t2 = pc()
        T["rs"], T["lane"] = t1 - t0, t2 - t1
        self.n_lane += 1
        boxes = np.zeros((0, 6), np.float32)
        if self.vdet is not None:
            x, meta = self.vdet.preprocess(frame)
            t3 = pc()
            o = self.vdet.infer(x)
            t4 = pc()
            boxes = self.vdet.postprocess(o, meta)
            t5 = pc()
            T.update(det_pre=t3 - t2, det_inf=t4 - t3, det_post=t5 - t4, det_total=t5 - t2, det_wait=0.0)
            self.n_det += 1
        else:
            T.update(det_pre=0, det_inf=0, det_post=0, det_total=0, det_wait=0.0)
        T["track"] = 0.0
        T["compute"] = pc() - t0
        r = _Res()
        r.fid, r.lane, r.lane_result, r.tracks, r.raw_det, r.t, r.det_ok, r.mask_boxes, r.boxes = fid, self.lane.result["lane"], self.lane.result, [], None, T, True, 0, boxes
        return r

    def draw(self, disp, res, fps_text=None):
        self.lane.draw(disp, None, None, None)
        s = disp.shape[1] / float(self.src_w)
        VD.draw_boxes(disp, res.boxes, s)
        return disp

    def close(self):
        pass
