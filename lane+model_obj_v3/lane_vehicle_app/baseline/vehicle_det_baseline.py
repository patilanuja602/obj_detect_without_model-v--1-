"""
vehicle_det.py -- portable OpenVINO vehicle detector for the benchmark package.

Takes the ORIGINAL decoded BGR frame (never the lane detector's resized / ROI-zeroed buffer), optionally crops a horizontal
band (a numpy view, no copy), letterboxes it into the network input, runs OpenVINO, decodes + class-aware NMS, and returns boxes
in ORIGINAL-frame pixel coordinates: float32 array (N, 6) = x1, y1, x2, y2, score, coco_class_id.

Vehicle classes (COCO ids): 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck.
Works with Ultralytics-style ONNX exports whose output is (1, 84, N)  (YOLOv8 / YOLO11 / YOLO26 non-end2end export).
"""
import os
import time
from pathlib import Path

import cv2
import numpy as np

PKG = Path(__file__).resolve().parent.parent                     # package root (portable; no absolute paths)
MODELS = PKG / "models"

VEH = np.array([1, 2, 3, 5, 7])
NAMES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
BAND = (0.35, 0.95)             # rows 35%-95%: near vehicles at the very bottom get their boxes cut at the 95% line
BAND100 = (0.35, 1.0)           # rows 35%-100%: keeps the bottom of the frame

# name -> (onnx file, band or None)
CONFIGS = {
    "A": ("yolo26n_960x320.onnx", BAND),      # YOLO26n, 960x320 band crop (rows 35%-95%)
    "B": ("yolo26n_768x256.onnx", BAND),      # YOLO26n, 768x256 band crop
    "A2": ("yolo26n_960x352.onnx", BAND100),  # YOLO26n, 960x352 band crop, rows 35%-100%  (A with the bottom kept)
    "B2": ("yolo26n_768x288.onnx", BAND100),  # YOLO26n, 768x288 band crop, rows 35%-100%  (B with the bottom kept)
    "F": ("yolo26n_960x544.onnx", None),      # YOLO26n, 960x544 FULL frame   (band-crop comparison)
    "F2": ("yolo26n_640x384.onnx", None),     # YOLO26n, 640x384 full frame   (cheap full-frame comparison)
}


def resolve(config):
    """'A' | 'B' | 'A2' | 'B2' | 'F' | 'F2' | path/to/model.onnx[:y0,y1]  ->  (path, band)"""
    if config in CONFIGS:
        f, band = CONFIGS[config]
        return str(MODELS / f), band
    band = None
    if ":" in config and not os.path.exists(config):
        config, b = config.rsplit(":", 1)
        y0, y1 = b.split(",")
        band = (float(y0), float(y1))
    return config, band


class VehicleDetector:
    def __init__(self, model_path, band=None, threads=1, conf=0.25, iou=0.5, resize="area", hyperthreading=None):
        import openvino as ov
        self.band, self.conf, self.iou = band, conf, iou
        self.resize = resize                                          # 'area' (Phase-2 behaviour, slower) | 'linear' (faster)
        core = ov.Core()
        cfg = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": int(threads), "NUM_STREAMS": 1}
        if hyperthreading is not None:
            cfg["ENABLE_HYPER_THREADING"] = bool(hyperthreading)
        t = time.perf_counter()
        self.cm = core.compile_model(core.read_model(str(model_path)), "CPU", cfg)
        self.compile_s = time.perf_counter() - t
        self.req = self.cm.create_infer_request()
        _, _, self.h, self.w = [int(v) for v in self.cm.input(0).shape]
        self.buf = np.zeros((1, 3, self.h, self.w), np.float32)
        self.canvas = np.full((self.h, self.w, 3), 114, np.uint8)
        self.threads = int(threads)

    # ------------------------------------------------------------------ stages (timed separately by the benchmark)
    def preprocess(self, frame):
        H0 = frame.shape[0]
        ya = 0
        if self.band:
            ya = int(self.band[0] * H0)
            frame = frame[ya:int(self.band[1] * H0)]              # view, no copy
        H, W = frame.shape[:2]
        s = min(self.w / W, self.h / H)
        nw, nh = int(round(W * s)), int(round(H * s))
        interp = cv2.INTER_AREA if (self.resize == "area" and s < 0.5) else cv2.INTER_LINEAR
        small = cv2.resize(frame, (nw, nh), interpolation=interp)
        ox, oy = (self.w - nw) // 2, (self.h - nh) // 2
        self.canvas[:] = 114
        self.canvas[oy:oy + nh, ox:ox + nw] = small
        np.multiply(self.canvas[:, :, ::-1].transpose(2, 0, 1), np.float32(1.0 / 255.0), out=self.buf[0], casting="unsafe")   # BGR->RGB, /255
        return self.buf, (s, ox, oy, W, H, ya)

    def infer(self, x):
        return self.req.infer({0: x})[self.cm.output(0)]

    def postprocess(self, out, meta, conf=None):
        conf = self.conf if conf is None else conf
        s, ox, oy, W, H, ya = meta
        p = np.asarray(out)[0].T                                   # (N, 84)
        sc = p[:, 4 + VEH]
        ci = sc.argmax(1)
        sm = sc[np.arange(len(sc)), ci]
        m = sm >= conf
        p, sm, cls = p[m], sm[m], VEH[ci[m]]
        if len(sm) == 0:
            return np.zeros((0, 6), np.float32)
        cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        b = np.stack([(cx - bw / 2 - ox) / s, (cy - bh / 2 - oy) / s, (cx + bw / 2 - ox) / s, (cy + bh / 2 - oy) / s], 1)
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, W - 1)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, H - 1)
        off = cls[:, None].astype(np.float32) * 10000.0             # class-aware NMS
        bx = b + off
        keep = cv2.dnn.NMSBoxes([[float(r[0]), float(r[1]), float(r[2] - r[0]), float(r[3] - r[1])] for r in bx], sm.tolist(), conf, self.iou)
        keep = np.array(keep, int).reshape(-1)
        b[:, [1, 3]] += ya                                           # crop coords -> original-frame coords
        return np.concatenate([b[keep], sm[keep, None], cls[keep, None].astype(np.float32)], 1)

    def detect(self, frame, conf=None):
        x, meta = self.preprocess(frame)
        return self.postprocess(self.infer(x), meta, conf)


def draw_boxes(img, boxes, scale=1.0):
    """Draw boxes (original-frame coords) onto img (possibly a resized display frame; scale = display/original)."""
    for x1, y1, x2, y2, sc, c in boxes:
        p1, p2 = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(img, p1, p2, (0, 200, 255), 2)
        lab = "%s %.2f" % (NAMES.get(int(c), "?"), sc)
        (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(p1[1], th + 4)
        cv2.rectangle(img, (p1[0], y - th - 4), (p1[0] + tw + 2, y), (0, 200, 255), -1)
        cv2.putText(img, lab, (p1[0] + 1, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img
