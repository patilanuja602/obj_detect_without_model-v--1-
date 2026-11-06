"""
vehicle_detector.py -- OpenVINO vehicle detector (YOLO26n ONNX, CPU).

Differences from baseline/vehicle_det_baseline.py (the Phase-2.5 detector):
  * candidates are returned at a LOW confidence (conf_low, default 0.20) so that distant vehicles survive; the tracker, not a global
    threshold, decides what is shown (see vehicle_tracker.py);
  * after class-aware NMS a second, stricter cross-class NMS removes "same vehicle, two classes" duplicates (car+truck, truck+bus);
    it only fires at IoU >= cross_class_iou (0.65), so two genuinely separate vehicles are not merged;
  * the per-class score vector of every surviving box is kept (class voting in the tracker);
  * preprocessing is done inside OpenVINO (u8 NHWC BGR -> RGB, /255) when ov_preprocess is on, which removes the numpy
    transpose/float conversion from the Python side; automatic fall-back to the numpy path if the PrePostProcessor is unavailable;
  * the letterbox canvas is only re-filled when the source size changes (grey bars never change).
Input is always the ORIGINAL decoded BGR frame.  Output boxes are in ORIGINAL-frame pixel coordinates.

detect() returns a Detections tuple:  boxes (N,4) float32 x1,y1,x2,y2 | score (N,) | cls (N,) COCO id | cls_scores (N,5)
"""
import os
import time
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

VEH = np.array([1, 2, 3, 5, 7])                       # COCO: bicycle, car, motorcycle, bus, truck
NAMES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
BAND = (0.35, 0.95)
BAND100 = (0.35, 1.0)
CONFIGS = {
    "A": ("yolo26n_960x320.onnx", BAND),
    "B": ("yolo26n_768x256.onnx", BAND),
    "A2": ("yolo26n_960x352.onnx", BAND100),
    "B2": ("yolo26n_768x288.onnx", BAND100),
    "F": ("yolo26n_960x544.onnx", None),
    "F2": ("yolo26n_640x384.onnx", None),
}
FOUR_WHEELED = (2, 5, 7)                              # classes that may be duplicates of each other

Detections = namedtuple("Detections", "boxes score cls cls_scores")
EMPTY = Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32), np.zeros((0, 5), np.float32))


def resolve(config):
    """'A'|'B'|'A2'|'B2'|'F'|'F2'|path.onnx[:y0,y1] -> (path, band)"""
    if config in CONFIGS:
        f, band = CONFIGS[config]
        return str(MODELS / f), band
    band = None
    if ":" in config and not os.path.exists(config):
        config, b = config.rsplit(":", 1)
        y0, y1 = b.split(",")
        band = (float(y0), float(y1))
    return config, band


def _iou_matrix(a, b):
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-6)


class VehicleDetector:
    def __init__(self, config="A2", threads=3, conf_low=0.20, nms_iou=0.50, cross_class_iou=0.65, resize="linear",
                 hyperthreading=None, ov_preprocess=True):
        import openvino as ov
        path, self.band = resolve(config)
        self.name, self.conf_low, self.nms_iou, self.cross_iou, self.resize = config, conf_low, nms_iou, cross_class_iou, resize
        core = ov.Core()
        cfg = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": int(threads), "NUM_STREAMS": 1}
        if hyperthreading is not None:
            cfg["ENABLE_HYPER_THREADING"] = bool(hyperthreading)
        t = time.perf_counter()
        model = core.read_model(str(path))
        self.ov_pre = False
        if ov_preprocess:
            try:
                from openvino import Layout, Type
                from openvino.preprocess import ColorFormat, PrePostProcessor
                ppp = PrePostProcessor(model)
                inp = ppp.input()
                inp.tensor().set_element_type(Type.u8).set_layout(Layout("NHWC")).set_color_format(ColorFormat.BGR)
                inp.preprocess().convert_element_type(Type.f32).convert_color(ColorFormat.RGB).scale(255.0)
                inp.model().set_layout(Layout("NCHW"))
                model = ppp.build()
                self.ov_pre = True
            except Exception as e:                                # noqa: BLE001 -- documented fall-back
                print("note: OpenVINO PrePostProcessor unavailable (%s) -> numpy preprocessing" % e)
        self.cm = core.compile_model(model, "CPU", cfg)
        self.compile_s = time.perf_counter() - t
        self.req = self.cm.create_infer_request()
        shp = [int(v) for v in self.cm.input(0).shape]
        if self.ov_pre:
            _, self.h, self.w, _ = shp                           # NHWC after preprocessing
            self.buf = np.full((1, self.h, self.w, 3), 114, np.uint8)
            self.canvas = self.buf[0]                            # the u8 canvas IS the input tensor (no extra copy)
        else:
            _, _, self.h, self.w = shp
            self.buf = np.zeros((1, 3, self.h, self.w), np.float32)
            self.canvas = np.full((self.h, self.w, 3), 114, np.uint8)
        self.threads = int(threads)
        self._geom_key = None
        self._geom = None

    # ---------------------------------------------------------------- stages
    def _geometry(self, shape):
        key = shape[:2]
        if key != self._geom_key:                                 # fixed for a given video: computed once
            H0, W0 = shape[:2]
            ya = int(self.band[0] * H0) if self.band else 0
            yb = int(self.band[1] * H0) if self.band else H0
            H, W = yb - ya, W0
            s = min(self.w / W, self.h / H)
            nw, nh = int(round(W * s)), int(round(H * s))
            ox, oy = (self.w - nw) // 2, (self.h - nh) // 2
            interp = cv2.INTER_AREA if (self.resize == "area" and s < 0.5) else cv2.INTER_LINEAR
            self._geom = (s, ox, oy, W, H, ya, yb, nw, nh, interp)
            self._geom_key = key
            self.canvas[:] = 114
        return self._geom

    def preprocess(self, frame):
        s, ox, oy, W, H, ya, yb, nw, nh, interp = self._geometry(frame.shape)
        crop = frame[ya:yb] if self.band else frame              # view, no copy
        small = cv2.resize(crop, (nw, nh), interpolation=interp)
        self.canvas[oy:oy + nh, ox:ox + nw] = small
        if not self.ov_pre:
            np.multiply(self.canvas[:, :, ::-1].transpose(2, 0, 1), np.float32(1.0 / 255.0), out=self.buf[0], casting="unsafe")
        return self.buf, (s, ox, oy, W, H, ya)

    def infer(self, x):
        return self.req.infer({0: x})[self.cm.output(0)]

    def postprocess(self, out, meta):
        s, ox, oy, W, H, ya = meta
        p = np.asarray(out)[0].T                                  # (N, 84)
        sc = p[:, 4 + VEH]
        sm = sc.max(1)
        m = sm >= self.conf_low
        if not m.any():
            return EMPTY
        p, sc, sm = p[m], sc[m], sm[m]
        ci = sc.argmax(1)
        cls = VEH[ci]
        cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        b = np.stack([(cx - bw / 2 - ox) / s, (cy - bh / 2 - oy) / s, (cx + bw / 2 - ox) / s, (cy + bh / 2 - oy) / s], 1).astype(np.float32)
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, W - 1)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, H - 1)
        # 1) class-aware NMS (same as baseline)
        off = cls[:, None].astype(np.float32) * 10000.0
        bx = b + off
        keep = cv2.dnn.NMSBoxes([[float(r[0]), float(r[1]), float(r[2] - r[0]), float(r[3] - r[1])] for r in bx], sm.tolist(), self.conf_low, self.nms_iou)
        keep = np.array(keep, int).reshape(-1)
        b, sm, cls, sc = b[keep], sm[keep], cls[keep], sc[keep]
        # 2) cross-class duplicate removal among four-wheeled classes (car / bus / truck): one object, one box
        if len(sm) > 1:
            fw = np.isin(cls, FOUR_WHEELED)
            idx = np.nonzero(fw)[0]
            if len(idx) > 1:
                iou = _iou_matrix(b[idx], b[idx])
                order = idx[np.argsort(-sm[idx])]
                alive = {int(i): True for i in idx}
                pos = {int(i): k for k, i in enumerate(idx)}
                for i in order:
                    if not alive[int(i)]:
                        continue
                    for j in idx:
                        if j != i and alive[int(j)] and cls[j] != cls[i] and iou[pos[int(i)], pos[int(j)]] >= self.cross_iou:
                            alive[int(j)] = False
                keepm = np.ones(len(sm), bool)
                for i, a in alive.items():
                    keepm[i] = a
                b, sm, cls, sc = b[keepm], sm[keepm], cls[keepm], sc[keepm]
        b[:, [1, 3]] += ya                                         # crop coords -> original-frame coords
        return Detections(b, sm.astype(np.float32), cls.astype(np.int32), sc.astype(np.float32))

    def detect(self, frame):
        """Returns (Detections, timing dict in seconds)."""
        t0 = time.perf_counter()
        x, meta = self.preprocess(frame)
        t1 = time.perf_counter()
        out = self.infer(x)
        t2 = time.perf_counter()
        d = self.postprocess(out, meta)
        t3 = time.perf_counter()
        return d, dict(pre=t1 - t0, inf=t2 - t1, post=t3 - t2, total=t3 - t0)
