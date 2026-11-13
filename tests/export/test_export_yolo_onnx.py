"""export_yolo_onnx: validate_yolo_seg_onnx accepts a YOLO-seg-shaped graph and refuses wrong ones; the export flow runs
against a stub ultralytics; without ultralytics the export fails with a clear message."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

import backends
import export_yolo_onnx as eyo

HAS_ULTRALYTICS = importlib.util.find_spec("ultralytics") is not None
needs_no_ultralytics = pytest.mark.skipif(HAS_ULTRALYTICS, reason="ultralytics is installed in this Python")


def test_accepts_yolo_seg_shaped_graph(tmp_path, yolo_onnx):
    facts = eyo.validate_yolo_seg_onnx(yolo_onnx(tmp_path / "m.onnx", imgsz=64, nc=3))
    assert facts["imgsz"] == 64 and facts["nc"] == 3 and facts["anchors"] == 64 + 16 + 4
    assert facts["outputs"] == {"output0": [1, 4 + 3 + 32, 84], "output1": [1, 32, 16, 16]}
    assert facts["input"]["shape"] == [1, 3, 64, 64]


def test_accepts_matching_imgsz_and_refuses_other_imgsz(tmp_path, yolo_onnx):
    path = yolo_onnx(tmp_path / "m.onnx", imgsz=64, nc=3)
    assert eyo.validate_yolo_seg_onnx(path, imgsz=64)["imgsz"] == 64
    with pytest.raises(ValueError, match="imgsz 128"):
        eyo.validate_yolo_seg_onnx(path, imgsz=128)


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"out1_shape": [1, 32, 8, 8]}, "output1"),
    ({"anchors": 83}, "output0"),
    ({"dynamic_batch": True}, "static"),
])
def test_refuses_wrong_output_or_input_shapes(tmp_path, yolo_onnx, kwargs, match):
    bad = yolo_onnx(tmp_path / "bad.onnx", imgsz=64, nc=3, **kwargs)
    with pytest.raises(ValueError, match=match):
        eyo.validate_yolo_seg_onnx(bad)


def test_refuses_a_file_onnxruntime_cannot_load(tmp_path):
    junk = tmp_path / "junk.onnx"
    junk.write_bytes(b"not an onnx model")
    with pytest.raises(ValueError, match="cannot load"):
        eyo.validate_yolo_seg_onnx(junk)


def install_stub_ultralytics(monkeypatch, yolo_onnx, src: Path, task: str = "segment") -> None:
    """A stand-in `ultralytics` whose YOLO.export writes a YOLO-seg-shaped ONNX next to the weights, as the real one does."""

    class FakeYOLO:
        def __init__(self, path: str) -> None:
            self.task = task
            self.names = {0: "person", 1: "bottle", 2: "cup"}

        def export(self, format: str, imgsz: int, opset: int, simplify: bool, dynamic: bool, half: bool) -> str:
            assert (format, dynamic, half) == ("onnx", False, False)
            return str(yolo_onnx(src / "yolo11n-seg.onnx", imgsz=imgsz, nc=3))

    stub = types.ModuleType("ultralytics")
    stub.YOLO = FakeYOLO
    stub.__version__ = "0.0-stub"
    monkeypatch.setitem(sys.modules, "ultralytics", stub)


def test_export_flow_with_stub_ultralytics(tmp_path, yolo_onnx, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    weights = src / "yolo11n-seg.pt"
    weights.write_bytes(b"stub weights")
    install_stub_ultralytics(monkeypatch, yolo_onnx, src)
    out = tmp_path / "out"
    info = eyo.export_yolo(weights, out, imgsz=64, opset=17)
    assert info["names"] == ["person", "bottle", "cup"] and info["nc"] == 3 and info["imgsz"] == 64
    assert (out / "model.onnx").is_file() and not (src / "yolo11n-seg.onnx").exists()
    meta = json.loads((out / "model.json").read_text())
    assert meta["sha256"] == backends.sha256(out / "model.onnx")
    assert meta["source_weights_sha256"] == backends.sha256(weights)
    assert "AGPL-3.0" in meta["licence_note"] and meta["licence_note"] == eyo.LICENCE_NOTE


def test_detection_weights_are_refused(tmp_path, yolo_onnx, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    weights = src / "yolo11n.pt"
    weights.write_bytes(b"stub weights")
    install_stub_ultralytics(monkeypatch, yolo_onnx, src, task="detect")
    with pytest.raises(ValueError, match="segmentation"):
        eyo.export_yolo(weights, tmp_path / "out")


@needs_no_ultralytics
def test_export_function_names_the_missing_package(tmp_path):
    weights = tmp_path / "yolo11n-seg.pt"
    weights.write_bytes(b"not a model")
    with pytest.raises(RuntimeError, match="ultralytics is not installed"):
        eyo.export_yolo(weights, tmp_path / "out")


@needs_no_ultralytics
def test_cli_without_ultralytics_exits_with_a_clear_message(tmp_path):
    weights = tmp_path / "yolo11n-seg.pt"
    weights.write_bytes(b"not a model")
    proc = subprocess.run(
        [sys.executable, str(Path(eyo.__file__)), "--weights", str(weights), "--out_dir", str(tmp_path / "out")],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2
    assert "ultralytics is not installed" in proc.stderr
    assert not (tmp_path / "out").exists()
