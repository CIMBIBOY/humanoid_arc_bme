"""YOLO instance segmentation (Ultralytics YOLOv8/YOLO11-seg ONNX export) behind the SemanticsModel Protocol.

Stdlib + numpy at import time; onnxruntime and cv2 are imported lazily. Frames: `rgb` is the image the detector sees
(pixels, origin top-left). Boxes are (x0, y0, x1, y1) integer pixels in the coordinates of the image the decoder was
given, x1 and y1 exclusive. Masks are bool arrays at the requested mask resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from humanoid_arc_bme.onboard.pipeline.interfaces import DepthFrame, Detection
from humanoid_arc_bme.onboard.validation import is_num

COCO80 = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

NUM_MASK_COEF = 32
PAD_VALUE = 114.0 / 255.0


@dataclass(frozen=True)
class YoloSegConfig:
    """Detector settings. `conf` and `iou` are the usual NMS thresholds in (0, 1); `mask_thresh` binarises the
    mask probability. `classes` restricts the labels returned (None = all classes)."""

    input_size: int = 640
    conf: float = 0.35
    iou: float = 0.5
    mask_thresh: float = 0.5
    max_det: int = 20
    class_names: tuple[str, ...] = COCO80
    classes: frozenset[str] | None = None

    def __post_init__(self):
        if not (isinstance(self.input_size, int) and not isinstance(self.input_size, bool) and self.input_size >= 32):
            raise ValueError("input_size: must be an int >= 32")
        if not (isinstance(self.max_det, int) and not isinstance(self.max_det, bool) and self.max_det >= 1):
            raise ValueError("max_det: must be an int >= 1")
        for name in ("conf", "iou", "mask_thresh"):
            v = getattr(self, name)
            if not (is_num(v) and 0.0 < v < 1.0):
                raise ValueError(f"{name}: must be a finite number in (0, 1)")
        names = tuple(self.class_names)
        if not names or not all(isinstance(n, str) and n for n in names):
            raise ValueError("class_names: expected a non-empty sequence of non-empty strings")
        object.__setattr__(self, "class_names", names)
        if self.classes is not None:
            allowed = frozenset(self.classes)
            if not allowed <= frozenset(names):
                raise ValueError("classes: must be a subset of class_names")
            object.__setattr__(self, "classes", allowed)


def _is_rgb8(rgb) -> bool:
    return isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.size > 0


@lru_cache(maxsize=1)
def _load_cv2():
    """cv2 module, or None when it is not installed (the numpy resize path is used then)."""
    try:
        import cv2
    except ImportError:
        return None
    return cv2


def _resize_bilinear(img: np.ndarray, nh: int, nw: int) -> np.ndarray:
    """Bilinear resize of a uint8 (H, W, 3) image without cv2. Pixel centres are aligned; returns uint8."""
    h, w = img.shape[:2]
    ys = np.clip((np.arange(nh, dtype=np.float32) + 0.5) * np.float32(h / nh) - 0.5, 0.0, h - 1)
    xs = np.clip((np.arange(nw, dtype=np.float32) + 0.5) * np.float32(w / nw) - 0.5, 0.0, w - 1)
    y0 = np.floor(ys).astype(np.intp)
    x0 = np.floor(xs).astype(np.intp)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    src = img.astype(np.float32)
    top = src[y0] * (1.0 - wy) + src[y1] * wy
    out = top[:, x0] * (1.0 - wx) + top[:, x1] * wx
    return np.rint(out).astype(np.uint8)


def _letterbox_into(rgb: np.ndarray, size: int, blob: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Write the letterboxed, normalised image into `blob` (1, 3, size, size) float32. Returns (scale, (pad_x, pad_y))."""
    h, w = rgb.shape[:2]
    scale = min(size / h, size / w)
    nh = min(size, max(1, round(h * scale)))
    nw = min(size, max(1, round(w * scale)))
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    cv2 = _load_cv2()
    if cv2 is not None:
        resized = cv2.resize(np.ascontiguousarray(rgb), (nw, nh), interpolation=cv2.INTER_LINEAR)
    else:
        resized = _resize_bilinear(rgb, nh, nw)
    blob.fill(np.float32(PAD_VALUE))
    blob[0, :, pad_y:pad_y + nh, pad_x:pad_x + nw] = resized.transpose(2, 0, 1) * np.float32(1.0 / 255.0)
    return scale, (pad_x, pad_y)


def letterbox(rgb: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Aspect-preserving resize into a size x size square, centred, padded with 114/255.

    Returns (blob, scale, (pad_x, pad_y)): blob is float32 (1, 3, size, size) in [0, 1], RGB, NCHW; a source pixel
    (u, v) lands at (u * scale + pad_x, v * scale + pad_y) in the blob. Raises ValueError for a non-uint8 or non-HxWx3
    image; the detector checks this before calling.
    """
    if not _is_rgb8(rgb):
        raise ValueError("rgb: expected a non-empty uint8 (H, W, 3) array")
    if not (isinstance(size, int) and not isinstance(size, bool) and size >= 32):
        raise ValueError("size: must be an int >= 32")
    blob = np.empty((1, 3, size, size), dtype=np.float32)
    scale, pad = _letterbox_into(rgb, size, blob)
    return blob, scale, pad


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, max_keep: int) -> np.ndarray:
    """Greedy NMS over (N, 4) x0y0x1y1 boxes with positive area. Returns kept indices, best score first."""
    order = np.argsort(-scores, kind="stable")
    b = boxes[order]
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    alive = np.ones(order.size, dtype=bool)
    keep: list[int] = []
    for k in range(order.size):
        if not alive[k]:
            continue
        keep.append(int(order[k]))
        if len(keep) >= max_keep or k + 1 == order.size:
            break
        tail = b[k + 1:]
        iw = np.clip(np.minimum(b[k, 2], tail[:, 2]) - np.maximum(b[k, 0], tail[:, 0]), 0.0, None)
        ih = np.clip(np.minimum(b[k, 3], tail[:, 3]) - np.maximum(b[k, 1], tail[:, 1]), 0.0, None)
        inter = iw * ih
        iou = inter / np.maximum(area[k] + area[k + 1:] - inter, 1e-12)
        alive[k + 1:] &= iou <= iou_thr
    return np.asarray(keep, dtype=np.intp)


def _hw(v, name: str) -> tuple[int, int]:
    h, w = int(v[0]), int(v[1])
    if h < 1 or w < 1:
        raise ValueError(f"{name}: expected positive (H, W)")
    return h, w


def decode_yolo_seg(out0, out1, cfg: YoloSegConfig, scale: float, pad: tuple[int, int], orig_hw, mask_hw=None) -> list[Detection]:
    """Decode the two outputs of an Ultralytics YOLO-seg ONNX model. Pure numpy.

    out0: (1, 4 + nc + 32, N): per anchor cx, cy, w, h in pixels of the letterboxed input, then nc class probabilities
    (already sigmoid), then 32 mask coefficients. out1: (1, 32, Hp, Wp) prototype masks. `scale` and `pad` are the
    letterbox transform from `letterbox`; `orig_hw` (H, W) is the image the boxes are returned in; masks are returned
    at `mask_hw` (H, W), defaulting to `orig_hw`. Returns detections best first. Raises ValueError on a shape mismatch
    (for example, len(cfg.class_names) != nc).
    """
    a0 = np.asarray(out0, dtype=np.float32)
    a1 = np.asarray(out1, dtype=np.float32)
    if a0.ndim != 3 or a0.shape[0] != 1 or a0.shape[1] < 4 + NUM_MASK_COEF + 1:
        raise ValueError("out0: expected shape (1, 4 + nc + 32, N)")
    nc = a0.shape[1] - 4 - NUM_MASK_COEF
    if len(cfg.class_names) != nc:
        raise ValueError(f"class_names: {len(cfg.class_names)} names for a model with {nc} classes")
    if a1.ndim != 4 or a1.shape[0] != 1 or a1.shape[1] != NUM_MASK_COEF:
        raise ValueError("out1: expected shape (1, 32, Hp, Wp)")
    s = float(scale)
    if not (math.isfinite(s) and s > 0.0):
        raise ValueError("scale: must be a positive finite number")
    px, py = int(pad[0]), int(pad[1])
    H, W = _hw(orig_hw, "orig_hw")
    mh, mw = _hw(mask_hw, "mask_hw") if mask_hw is not None else (H, W)
    size = cfg.input_size
    Hp, Wp = a1.shape[2], a1.shape[3]

    pred = a0[0]
    scores = pred[4:4 + nc]
    if cfg.classes is not None:
        allowed = np.fromiter((n in cfg.classes for n in cfg.class_names), dtype=bool, count=nc)
        scores = np.where(allowed[:, None], scores, np.float32(0.0))
    best = np.argmax(scores, axis=0)
    conf = scores[best, np.arange(pred.shape[1])]
    idx = np.nonzero((conf >= cfg.conf) & (conf <= 1.0))[0]
    if idx.size == 0:
        return []
    sub = pred[:, idx]
    ok = np.isfinite(sub).all(axis=0) & (sub[2] > 0) & (sub[3] > 0)
    sub = sub[:, ok]
    if sub.shape[1] == 0:
        return []
    cls = best[idx][ok]
    sc = conf[idx][ok]
    half_w, half_h = sub[2] / 2.0, sub[3] / 2.0
    boxes = np.stack((sub[0] - half_w, sub[1] - half_h, sub[0] + half_w, sub[1] + half_h), axis=1)
    coef = sub[4 + nc:].T

    # Class-wise NMS: offsetting each class by a disjoint coordinate range makes one greedy pass equivalent to per-class.
    offset = cls.astype(np.float32) * np.float32(size + 1)
    sel = _nms(boxes + offset[:, None], sc, cfg.iou, cfg.max_det)
    boxes, cls, sc, coef = boxes[sel], cls[sel], sc[sel], coef[sel]

    # Boxes back to the original image: undo the letterbox, then round outward and clip to the image.
    x0 = np.clip(np.floor((boxes[:, 0] - px) / s), 0, W).astype(np.intp)
    y0 = np.clip(np.floor((boxes[:, 1] - py) / s), 0, H).astype(np.intp)
    x1 = np.clip(np.ceil((boxes[:, 2] - px) / s), 0, W).astype(np.intp)
    y1 = np.clip(np.ceil((boxes[:, 3] - py) / s), 0, H).astype(np.intp)
    good = (x1 > x0) & (y1 > y0)
    if not good.any():
        return []
    x0, y0, x1, y1, cls, sc, coef = x0[good], y0[good], x1[good], y1[good], cls[good], sc[good], coef[good]

    # Masks: prototype logits combined with the coefficients, then sampled nearest-neighbour straight onto the mask grid
    # (letterbox coordinates -> prototype index), restricted to the box. No full-resolution upsample is materialised.
    logits = coef @ a1[0].reshape(NUM_MASK_COEF, Hp * Wp)
    np.clip(logits, -60.0, 60.0, out=logits)
    probs = (1.0 / (1.0 + np.exp(-logits))).reshape(-1, Hp, Wp)
    yl = ((np.arange(mh) + 0.5) * (H / mh)) * s + py
    xl = ((np.arange(mw) + 0.5) * (W / mw)) * s + px
    pyi = np.clip(np.floor(yl * (Hp / size)), 0, Hp - 1).astype(np.intp)
    pxi = np.clip(np.floor(xl * (Wp / size)), 0, Wp - 1).astype(np.intp)
    yc = (np.arange(mh) + 0.5) * (H / mh)
    xc = (np.arange(mw) + 0.5) * (W / mw)

    out: list[Detection] = []
    for m in range(probs.shape[0]):
        rows = np.nonzero((yc >= y0[m]) & (yc < y1[m]))[0]
        cols = np.nonzero((xc >= x0[m]) & (xc < x1[m]))[0]
        mask = np.zeros((mh, mw), dtype=bool)
        if rows.size and cols.size:
            mask[np.ix_(rows, cols)] = probs[m][np.ix_(pyi[rows], pxi[cols])] > cfg.mask_thresh
        if not mask.any():
            continue
        box = (int(x0[m]), int(y0[m]), int(x1[m]), int(y1[m]))
        out.append(Detection(cfg.class_names[int(cls[m])], float(sc[m]), mask, box))
    return out


class YoloSegDetector:
    """SemanticsModel backed by a YOLO-seg ONNX model. `detect` never raises: a model or runtime fault is recorded in
    `last_error` and yields [], which the caller reads as 'nothing seen'. Construction raises if the model cannot be
    opened (a configuration error that should stop start-up).

    `session` is any object with `.run(None, {name: array}) -> [out0, out1]` and `.get_inputs()[0].name`; when None an
    onnxruntime InferenceSession is created from `model_path`. Masks come back at the depth frame resolution; when the
    RGB image has another resolution the boxes are scaled to match.
    """

    def __init__(self, model_path: str | None = None, cfg: YoloSegConfig | None = None, providers: list[str] | None = None, session=None):
        self.cfg = cfg if cfg is not None else YoloSegConfig()
        self.last_error: str | None = None
        self._blob: np.ndarray | None = None
        if session is None:
            if model_path is None:
                raise ValueError("model_path or session: one is required")
            session = _open_session(str(model_path), providers)
        self._session = session
        self._input_name = session.get_inputs()[0].name

    def detect(self, rgb, depth: DepthFrame) -> list[Detection]:
        if not _is_rgb8(rgb):
            return []
        try:
            dets = self._run(rgb, depth)
        except Exception as exc:  # fail closed: any model or runtime fault means nothing was seen
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []
        self.last_error = None
        return dets

    def _run(self, rgb: np.ndarray, depth: DepthFrame) -> list[Detection]:
        size = self.cfg.input_size
        if self._blob is None or self._blob.shape != (1, 3, size, size):
            self._blob = np.empty((1, 3, size, size), dtype=np.float32)
        scale, pad = _letterbox_into(rgb, size, self._blob)
        out0, out1 = self._session.run(None, {self._input_name: self._blob})
        H, W = rgb.shape[:2]
        dh, dw = depth.depth.shape
        dets = decode_yolo_seg(out0, out1, self.cfg, scale, pad, (H, W), (dh, dw))
        if (H, W) == (dh, dw):
            return dets
        fx, fy = dw / W, dh / H
        rescaled = []
        for d in dets:
            x0, y0, x1, y1 = d.box
            box = (
                int(math.floor(x0 * fx)), int(math.floor(y0 * fy)),
                min(dw, int(math.ceil(x1 * fx))), min(dh, int(math.ceil(y1 * fy))),
            )
            rescaled.append(Detection(d.label, d.confidence, d.mask, box))
        return rescaled


def _open_session(model_path: str, providers: list[str] | None):
    import onnxruntime as ort

    if providers is None:
        available = set(ort.get_available_providers())
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 2
    return ort.InferenceSession(model_path, sess_options=opts, providers=providers)
