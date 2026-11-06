"""
vehicle_detector.py -- OpenVINO vehicle detector (YOLO26n ONNX, Ultralytics (1, 84, N) output).

Input frame: BGR uint8 (the pipeline passes either the 1280-wide display frame or the original frame).  Optional horizontal band crop (a
numpy view, no copy), letterbox, normalise into a preallocated tensor, infer, decode, merge duplicates.  Returns float32 (N, 6):
x1, y1, x2, y2, score, coco_class_id   in the coordinates of the frame that was passed in.

Vehicle classes (COCO ids): 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck.

Duplicate handling (the Phase-3 fix)
  nms_mode = "class_aware"      legacy behaviour: NMS per class, so one vehicle can be boxed as car + truck + bus   (BEFORE reference)
  nms_mode = "agnostic_merge"   one NMS across all vehicle classes; each suppressed box that overlaps a kept box by IoU >= nms_iou votes
                                for its class with its score; the kept box takes the score-weighted majority class and keeps its own score.
Two separate vehicles are only merged if their boxes overlap by IoU >= nms_iou, which real neighbouring vehicles rarely do.
Plausibility: boxes smaller than min_height_px or with an aspect ratio (w/h) outside aspect_range are dropped (flat dark strips etc.).
"""
import time

import cv2
import numpy as np

VEH = np.array([1, 2, 3, 5, 7])
NAMES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class VehicleDetector:
    def __init__(self, model_path, band=None, threads=2, conf=0.20, resize="linear", nms_mode="agnostic_merge", nms_iou=0.55,
                 reject_nonvehicle_ratio=0.0, min_height_px=6, aspect_range=(0.4, 4.0), hyperthreading=None):
        import openvino as ov
        self.band = tuple(band) if band else None
        self.conf, self.resize, self.nms_mode, self.nms_iou = conf, resize, nms_mode, nms_iou
        self.reject_ratio, self.min_h, self.ar = reject_nonvehicle_ratio, min_height_px, tuple(aspect_range)
        cfg = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": int(threads), "NUM_STREAMS": 1}
        if hyperthreading is not None:
            cfg["ENABLE_HYPER_THREADING"] = bool(hyperthreading)
        core = ov.Core()
        t = time.perf_counter()
        self.cm = core.compile_model(core.read_model(str(model_path)), "CPU", cfg)
        self.compile_s = time.perf_counter() - t
        self.req = self.cm.create_infer_request()
        _, _, self.h, self.w = [int(v) for v in self.cm.input(0).shape]
        self.threads = int(threads)

    # ------------------------------------------------------------------ stages (timed separately by the pipeline)
    def preprocess(self, frame):
        """-> (tensor, meta, ms).  Thread-safe: every call builds its own tensor, so it can run in the background stage thread for frame N+1
        while the worker is still inferring frame N."""
        t0 = time.perf_counter()
        H0 = frame.shape[0]
        ya = 0
        if self.band:
            ya = int(self.band[0] * H0)
            frame = frame[ya:int(self.band[1] * H0)]                 # view, no copy
        H, W = frame.shape[:2]
        s = min(self.w / W, self.h / H)
        nw, nh = int(round(W * s)), int(round(H * s))
        interp = cv2.INTER_AREA if (self.resize == "area" and s < 0.5) else cv2.INTER_LINEAR
        small = cv2.resize(frame, (nw, nh), interpolation=interp)
        ox, oy = (self.w - nw) // 2, (self.h - nh) // 2
        canvas = cv2.copyMakeBorder(small, oy, self.h - nh - oy, ox, self.w - nw - ox, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        buf = np.empty((1, 3, self.h, self.w), np.float32)
        np.multiply(canvas[:, :, ::-1].transpose(2, 0, 1), np.float32(1.0 / 255.0), out=buf[0], casting="unsafe")   # BGR->RGB, /255
        return buf, (s, ox, oy, W, H, ya), (time.perf_counter() - t0) * 1e3

    def infer(self, x):
        return self.req.infer({0: x})[self.cm.output(0)]

    def postprocess(self, out, meta):
        s, ox, oy, W, H, ya = meta
        p = np.asarray(out)[0].T                                     # (N, 84)
        allc = p[:, 4:]
        veh = allc[:, VEH]
        vi = veh.argmax(1)
        vs = veh[np.arange(len(veh)), vi]
        m = vs >= self.conf
        if self.reject_ratio > 0:                                    # optional: drop anchors where a NON-vehicle class clearly dominates
            non = allc.copy()
            non[:, VEH] = 0
            m &= ~(non.max(1) > vs * self.reject_ratio)
        if not m.any():
            return np.zeros((0, 6), np.float32)
        p, sc, cls = p[m], vs[m], VEH[vi[m]]
        cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        b = np.stack([(cx - bw / 2 - ox) / s, (cy - bh / 2 - oy) / s, (cx + bw / 2 - ox) / s, (cy + bh / 2 - oy) / s], 1)
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, W - 1)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, H - 1)
        w_, h_ = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        ok = (h_ >= self.min_h) & (w_ >= 3) & (w_ / np.maximum(h_, 1e-6) >= self.ar[0]) & (w_ / np.maximum(h_, 1e-6) <= self.ar[1])
        b, sc, cls = b[ok], sc[ok], cls[ok]
        if len(sc) == 0:
            return np.zeros((0, 6), np.float32)
        xywh = [[float(r[0]), float(r[1]), float(r[2] - r[0]), float(r[3] - r[1])] for r in b]
        if self.nms_mode == "class_aware":
            off = cls[:, None].astype(np.float32) * 10000.0
            bo = b + off
            keep = cv2.dnn.NMSBoxes([[float(r[0]), float(r[1]), float(r[2] - r[0]), float(r[3] - r[1])] for r in bo], sc.tolist(), self.conf, self.nms_iou)
            keep = np.array(keep, int).reshape(-1)
            res = np.concatenate([b[keep], sc[keep, None], cls[keep, None].astype(np.float32)], 1)
        else:
            keep = np.array(cv2.dnn.NMSBoxes(xywh, sc.tolist(), self.conf, self.nms_iou), int).reshape(-1)
            kept = set(keep.tolist())
            votes = {int(k): {int(cls[k]): float(sc[k])} for k in keep}
            for j in range(len(sc)):
                if j in kept:
                    continue
                best, bi = 0.0, -1
                for k in keep:
                    i = _iou(b[j], b[k])
                    if i > best:
                        best, bi = i, int(k)
                if bi >= 0 and best >= self.nms_iou:
                    votes[bi][int(cls[j])] = votes[bi].get(int(cls[j]), 0.0) + float(sc[j])
            res = np.concatenate([b[keep], sc[keep, None], cls[keep, None].astype(np.float32)], 1)
            for r, k in zip(res, keep):
                r[5] = max(votes[int(k)].items(), key=lambda kv: kv[1])[0]
        if len(res):
            res[:, [1, 3]] += ya                                     # crop coordinates -> input-frame coordinates
        return res.astype(np.float32)

    def detect(self, frame):
        x, meta, _ = self.preprocess(frame)
        return self.postprocess(self.infer(x), meta)


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def ground_point(box):
    """Bottom-centre ground-contact point (x, y) of a box [x1, y1, x2, y2, ...] -- retained for future use (distance, lateral position)."""
    return (0.5 * (box[0] + box[2]), float(box[3]))
