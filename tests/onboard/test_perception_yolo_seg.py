"""YOLO-seg: config validation, letterbox geometry, decode on hand-built tensors, detector fail-closed behaviour,
and a real onnxruntime session on a tiny synthetic graph."""

import subprocess
import sys

import numpy as np
import pytest

from humanoid_arc_bme.onboard.perception import yolo_seg
from humanoid_arc_bme.onboard.perception.yolo_seg import (
    COCO80, NUM_MASK_COEF, YoloSegConfig, YoloSegDetector, decode_yolo_seg, letterbox,
)
from humanoid_arc_bme.onboard.pipeline.interfaces import DepthFrame, Intrinsics, Pose

NAMES3 = ("a", "b", "c")


def cfg(**kw):
    kw.setdefault("class_names", NAMES3)
    return YoloSegConfig(**kw)


def tensors(boxes, nc=3, n=10, proto=(160, 160)):
    """boxes: list of (anchor, cx, cy, w, h, class, score). Protos and coefficients are all ones, so each detection's
    mask probability is ~1 everywhere, and the mask is exactly its box."""
    out0 = np.zeros((1, 4 + nc + NUM_MASK_COEF, n), dtype=np.float32)
    for a, cx, cy, w, h, c, s in boxes:
        out0[0, 0:4, a] = (cx, cy, w, h)
        out0[0, 4 + c, a] = s
        out0[0, 4 + nc:, a] = 1.0
    out1 = np.ones((1, NUM_MASK_COEF, *proto), dtype=np.float32)
    return out0, out1


def test_coco80_layout():
    assert len(COCO80) == 80 and len(set(COCO80)) == 80
    assert COCO80[39] == "bottle" and COCO80[41] == "cup" and COCO80[45] == "bowl"
    assert COCO80[0] == "person" and COCO80[-1] == "toothbrush"


@pytest.mark.parametrize(
    "kw",
    [
        {"input_size": 0}, {"input_size": 640.0}, {"conf": 0.0}, {"conf": 1.0}, {"iou": float("nan")},
        {"mask_thresh": 1.5}, {"max_det": 0}, {"class_names": ()}, {"classes": frozenset({"nope"})},
    ],
)
def test_config_validation(kw):
    with pytest.raises(ValueError):
        YoloSegConfig(**kw)


def test_config_normalises_classes():
    c = YoloSegConfig(classes={"cup"})
    assert c.classes == frozenset({"cup"})


def test_letterbox_geometry_identity_scale():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    blob, scale, (px, py) = letterbox(img, 640)
    assert blob.shape == (1, 3, 640, 640) and blob.dtype == np.float32
    assert scale == pytest.approx(1.0) and (px, py) == (0, 80)
    u, v = 100, 50
    assert np.allclose(blob[0, :, int(v * scale + py), int(u * scale + px)], img[v, u] / 255.0, atol=1e-6)
    assert np.allclose(blob[0, :, 0, 0], 114.0 / 255.0)


def test_letterbox_round_trip_with_scaling():
    img = np.zeros((300, 200, 3), dtype=np.uint8)
    img[100:150, 50:100] = 255  # bright block, rows 100..149, cols 50..99
    blob, scale, (px, py) = letterbox(img, 640)
    assert scale == pytest.approx(640 / 300)
    assert (px, py) == (106, 0)
    bright = blob[0, 0] > 0.5
    rows, cols = np.nonzero(bright)
    x0, x1 = (cols.min() - px) / scale, (cols.max() + 1 - px) / scale
    y0, y1 = (rows.min() - py) / scale, (rows.max() + 1 - py) / scale
    assert abs(x0 - 50) <= 1 and abs(x1 - 100) <= 1
    assert abs(y0 - 100) <= 1 and abs(y1 - 150) <= 1


def test_letterbox_numpy_fallback_matches_geometry(monkeypatch):
    monkeypatch.setattr(yolo_seg, "_load_cv2", lambda: None)
    img = np.zeros((300, 200, 3), dtype=np.uint8)
    img[100:150, 50:100] = 255
    blob, scale, (px, py) = letterbox(img, 640)
    rows, cols = np.nonzero(blob[0, 0] > 0.5)
    assert abs((cols.min() - px) / scale - 50) <= 1
    assert abs((rows.max() + 1 - py) / scale - 150) <= 1


def test_letterbox_rejects_bad_input():
    with pytest.raises(ValueError):
        letterbox(np.zeros((8, 8), dtype=np.uint8), 640)
    with pytest.raises(ValueError):
        letterbox(np.zeros((8, 8, 3), dtype=np.float32), 640)


def test_decode_single_box():
    out0, out1 = tensors([(0, 320, 320, 200, 200, 1, 0.9)])
    dets = decode_yolo_seg(out0, out1, cfg(), 1.0, (0, 0), (640, 640))
    assert len(dets) == 1
    d = dets[0]
    assert d.label == "b" and d.confidence == pytest.approx(0.9)
    assert d.box == (220, 220, 420, 420)
    assert d.mask.dtype == bool and d.mask.shape == (640, 640)
    assert d.mask.sum() == 200 * 200
    assert d.mask[320, 320] and not d.mask[0, 0]


def test_decode_nms_keeps_higher_score_of_overlapping_same_class():
    out0, out1 = tensors([(0, 320, 320, 200, 200, 0, 0.9), (1, 330, 330, 200, 200, 0, 0.8)])
    dets = decode_yolo_seg(out0, out1, cfg(), 1.0, (0, 0), (640, 640))
    assert [d.confidence for d in dets] == [pytest.approx(0.9)]


def test_decode_overlapping_different_classes_both_kept():
    out0, out1 = tensors([(0, 320, 320, 200, 200, 0, 0.9), (1, 320, 320, 200, 200, 2, 0.8)])
    dets = decode_yolo_seg(out0, out1, cfg(), 1.0, (0, 0), (640, 640))
    assert [d.label for d in dets] == ["a", "c"]


def test_decode_class_filter_and_sorting():
    out0, out1 = tensors([(0, 100, 100, 50, 50, 0, 0.6), (1, 400, 400, 50, 50, 1, 0.95), (2, 500, 100, 50, 50, 2, 0.8)])
    dets = decode_yolo_seg(out0, out1, cfg(classes=frozenset({"a", "c"})), 1.0, (0, 0), (640, 640))
    assert [d.label for d in dets] == ["c", "a"]
    assert all(d.confidence <= 1.0 for d in dets)
    assert [d.confidence for d in dets] == sorted((d.confidence for d in dets), reverse=True)


def test_decode_single_class_model():
    out0, out1 = tensors([(0, 320, 320, 100, 100, 0, 0.8)], nc=1)
    dets = decode_yolo_seg(out0, out1, YoloSegConfig(class_names=("thing",)), 1.0, (0, 0), (640, 640))
    assert [d.label for d in dets] == ["thing"]


def test_decode_wrong_class_names_length_raises():
    out0, out1 = tensors([(0, 320, 320, 100, 100, 0, 0.8)])
    with pytest.raises(ValueError):
        decode_yolo_seg(out0, out1, YoloSegConfig(class_names=("a", "b")), 1.0, (0, 0), (640, 640))


def test_decode_box_clipped_to_image():
    out0, out1 = tensors([(0, 10, 10, 100, 100, 0, 0.9)])
    (d,) = decode_yolo_seg(out0, out1, cfg(), 1.0, (0, 0), (640, 640))
    assert d.box == (0, 0, 60, 60)
    assert d.mask[:60, :60].all() and not d.mask[60:].any()


def test_decode_below_conf_and_degenerate_dropped():
    out0, out1 = tensors([(0, 320, 320, 200, 200, 0, 0.2), (1, 100, 100, 0.0, 50, 0, 0.9)])
    assert decode_yolo_seg(out0, out1, cfg(), 1.0, (0, 0), (640, 640)) == []


def test_decode_scale_and_pad_map_boxes_back():
    # Original 480x640 (H, W) letterboxed to 640 with pad_y=80 at scale 1: the centre pixel of the letterbox at
    # (320, 320) is original (320, 240).
    out0, out1 = tensors([(0, 320, 320, 160, 160, 0, 0.9)])
    (d,) = decode_yolo_seg(out0, out1, cfg(), 1.0, (0, 80), (480, 640))
    assert d.box == (240, 160, 400, 320)


def test_decode_masks_at_requested_resolution():
    out0, out1 = tensors([(0, 320, 320, 200, 200, 0, 0.9)])
    (d,) = decode_yolo_seg(out0, out1, cfg(), 1.0, (0, 0), (640, 640), (240, 320))
    assert d.mask.shape == (240, 320)
    assert d.box == (220, 220, 420, 420)  # boxes stay in the original image coordinates


def test_decode_shape_errors():
    out0, out1 = tensors([])
    with pytest.raises(ValueError):
        decode_yolo_seg(out0[:, :10], out1, cfg(), 1.0, (0, 0), (640, 640))
    with pytest.raises(ValueError):
        decode_yolo_seg(out0, out1[:, :8], cfg(), 1.0, (0, 0), (640, 640))
    with pytest.raises(ValueError):
        decode_yolo_seg(out0, out1, cfg(), 0.0, (0, 0), (640, 640))


def test_decode_nonfinite_rows_dropped():
    out0, out1 = tensors([(0, 320, 320, 200, 200, 0, 0.9), (1, 100, 100, 50, 50, 0, 0.8)])
    out0[0, 0, 1] = np.nan
    dets = decode_yolo_seg(out0, out1, cfg(), 1.0, (0, 0), (640, 640))
    assert [d.box for d in dets] == [(220, 220, 420, 420)]


class FakeSession:
    def __init__(self, out0, out1, error=None):
        self._out = (out0, out1)
        self._error = error
        self.feeds = []

    def get_inputs(self):
        return [type("I", (), {"name": "images"})()]

    def run(self, names, feeds):
        self.feeds.append(feeds)
        if self._error is not None:
            raise self._error
        return list(self._out)


def depth_frame(h=240, w=320, value=2.0):
    k = Intrinsics(200.0, 200.0, (w - 1) / 2, (h - 1) / 2, w, h)
    return DepthFrame(0.0, np.full((h, w), value, dtype=np.float32), k, Pose((0.0, 0.0, 0.0)))


def test_detector_fake_session_rescales_to_depth_resolution():
    out0, out1 = tensors([(0, 320, 320, 160, 160, 0, 0.9)])
    sess = FakeSession(out0, out1)
    det = YoloSegDetector(cfg=cfg(), session=sess)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = det.detect(rgb, depth_frame())
    assert det.last_error is None
    assert sess.feeds[0]["images"].shape == (1, 3, 640, 640)
    assert len(dets) == 1
    d = dets[0]
    assert d.mask.shape == (240, 320)
    assert d.mask.sum() == 80 * 80
    assert d.box == (120, 80, 200, 160)


def test_detector_reuses_blob_buffer():
    out0, out1 = tensors([])
    sess = FakeSession(out0, out1)
    det = YoloSegDetector(cfg=cfg(), session=sess)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    det.detect(rgb, depth_frame())
    det.detect(rgb, depth_frame())
    assert sess.feeds[0]["images"] is sess.feeds[1]["images"]


def test_detector_fail_closed_on_session_error():
    det = YoloSegDetector(cfg=cfg(), session=FakeSession(None, None, error=RuntimeError("boom")))
    assert det.detect(np.zeros((48, 64, 3), dtype=np.uint8), depth_frame()) == []
    assert det.last_error is not None and "boom" in det.last_error


def test_detector_none_or_wrong_dtype_rgb_returns_empty_without_error():
    out0, out1 = tensors([(0, 320, 320, 160, 160, 0, 0.9)])
    sess = FakeSession(out0, out1)
    det = YoloSegDetector(cfg=cfg(), session=sess)
    assert det.detect(None, depth_frame()) == []
    assert det.detect(np.zeros((48, 64, 3), dtype=np.float32), depth_frame()) == []
    assert det.last_error is None
    assert sess.feeds == []


def test_detector_requires_model_or_session():
    with pytest.raises(ValueError):
        YoloSegDetector()


def test_yolo_module_imports_are_lazy():
    code = (
        "import sys, humanoid_arc_bme.onboard.perception.yolo_seg as y; "
        "bad = [m for m in ('torch', 'isaaclab', 'rclpy', 'onnxruntime', 'cv2') if m in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_real_onnxruntime_session_end_to_end(tmp_path):
    pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    nc = 3
    out0_init = np.zeros((1, 4 + nc + NUM_MASK_COEF, 10), dtype=np.float32)
    out0_init[0, 0:4, 0] = (320, 320, 160, 160)  # letterbox centre, 160 px square
    out0_init[0, 4 + 1, 0] = 0.9  # class 'b'
    out0_init[0, 4 + nc:, 0] = 1.0
    out1_init = np.ones((1, NUM_MASK_COEF, 160, 160), dtype=np.float32)
    inits = [numpy_helper.from_array(out0_init, "c0"), numpy_helper.from_array(out1_init, "c1")]
    nodes = [helper.make_node("Identity", ["c0"], ["out0"]), helper.make_node("Identity", ["c1"], ["out1"])]
    graph = helper.make_graph(
        nodes,
        "yolo_stub",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [
            helper.make_tensor_value_info("out0", TensorProto.FLOAT, list(out0_init.shape)),
            helper.make_tensor_value_info("out1", TensorProto.FLOAT, list(out1_init.shape)),
        ],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    model_path = tmp_path / "stub_seg.onnx"
    onnx.save(model, str(model_path))

    det = YoloSegDetector(
        model_path=str(model_path),
        cfg=cfg(),
        providers=["CPUExecutionProvider"],
    )
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = det.detect(rgb, depth_frame(240, 320))
    assert det.last_error is None
    assert len(dets) == 1
    d = dets[0]
    assert d.label == "b" and d.confidence == pytest.approx(0.9, abs=1e-6)
    assert d.mask.shape == (240, 320)
    assert d.mask.sum() == 80 * 80
    assert d.box == (120, 80, 200, 160)
