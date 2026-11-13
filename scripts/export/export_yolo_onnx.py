#!/usr/bin/env python3
"""Export a YOLO segmentation model to ONNX and validate the file. Needs ultralytics (AGPL-3.0) only for the export.

    python3 scripts/export/export_yolo_onnx.py --weights yolo11n-seg.pt --out_dir <dir> [--imgsz 640] [--opset 17]
        [--half] [--no-simplify]

Writes <out_dir>/model.onnx and <out_dir>/model.json {names, imgsz, nc, opset, sha256, source_weights_sha256, licence_note}.
The export is static (batch 1). validate_yolo_seg_onnx(path) checks any YOLO-seg ONNX with onnxruntime alone, so the
checks run where ultralytics is not installed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy" / "scripts"))

import backends  # noqa: E402

LICENCE_NOTE = ("Ultralytics YOLO weights/code are AGPL-3.0 unless you hold an enterprise licence; "
                "the exported file inherits that.")
BOX_CHANNELS = 4
MASK_CHANNELS = 32
STRIDES = (8, 16, 32)


def _import_ultralytics() -> Any:
    try:
        import ultralytics
    except ImportError as e:
        raise RuntimeError("ultralytics is not installed in this Python. Install it only where you export "
                           "(pip install ultralytics); it is AGPL-3.0, see the licence note in model.json") from e
    return ultralytics


def validate_yolo_seg_onnx(path: str | Path, imgsz: int | None = None) -> dict[str, Any]:
    """Check a YOLO segmentation ONNX file and return its facts; raise ValueError with the reason if it is wrong.

    Expected graph (Ultralytics export, static batch 1): input [1, 3, S, S] with S a multiple of 32 (units: pixels),
    output0 [1, 4 + nc + 32, N] (boxes, class scores, mask coefficients; N = sum over strides 8/16/32 of (S/stride)^2
    anchors), output1 [1, 32, S/4, S/4] (mask prototypes). A float16 input (--half) is accepted.
    `imgsz`, when given, must equal S. The graph is run once on zeros to confirm the outputs' runtime shapes.
    """
    p = Path(path)
    try:
        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    except Exception as e:  # noqa: BLE001 - onnxruntime raises its own Fail type for a bad file
        raise ValueError(f"{p}: onnxruntime cannot load it: {e}") from e
    ins, outs = sess.get_inputs(), sess.get_outputs()
    if len(ins) != 1 or len(outs) != 2:
        raise ValueError(f"{p}: YOLO-seg needs 1 input and 2 outputs, has {len(ins)} and {len(outs)}")
    x, (o0, o1) = ins[0], outs
    shape = list(x.shape)
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3 or not all(isinstance(d, int) for d in shape):
        raise ValueError(f"{p}: input {x.name} shape {shape} must be static [1, 3, S, S]")
    size = shape[2]
    if shape[3] != size or size % 32 or (imgsz is not None and size != imgsz):
        raise ValueError(f"{p}: input is {size}x{shape[3]}, needs a square size that is a multiple of 32"
                         + (f" equal to imgsz {imgsz}" if imgsz is not None else ""))
    if x.type not in ("tensor(float)", "tensor(float16)"):
        raise ValueError(f"{p}: input type {x.type} must be float32 or float16")
    anchors = sum((size // s) ** 2 for s in STRIDES)
    s0, s1 = list(o0.shape), list(o1.shape)
    if len(s0) != 3 or s0[0] != 1 or s0[2] != anchors or not isinstance(s0[1], int) or s0[1] <= BOX_CHANNELS + MASK_CHANNELS:
        raise ValueError(f"{p}: output0 {o0.name} shape {s0} must be [1, 4 + nc + 32, {anchors}] for imgsz {size}")
    nc = s0[1] - BOX_CHANNELS - MASK_CHANNELS
    want1 = [1, MASK_CHANNELS, size // 4, size // 4]
    if s1 != want1:
        raise ValueError(f"{p}: output1 {o1.name} shape {s1} must be {want1}")
    dtype = np.float16 if x.type == "tensor(float16)" else np.float32
    got = sess.run(None, {x.name: np.zeros((1, 3, size, size), dtype)})
    if list(got[0].shape) != s0 or list(got[1].shape) != want1:
        raise ValueError(f"{p}: runtime output shapes {got[0].shape}, {got[1].shape} differ from the declared ones")
    if not all(np.isfinite(g.astype(np.float32)).all() for g in got):
        raise ValueError(f"{p}: non-finite values on a zero input")
    return {"imgsz": size, "nc": nc, "anchors": anchors, "input": {"name": x.name, "shape": shape, "type": x.type},
            "outputs": {o0.name: s0, o1.name: s1}}


def export_yolo(weights: str | Path, out_dir: str | Path, imgsz: int = 640, opset: int = 17, half: bool = False,
                simplify: bool = True) -> dict[str, Any]:
    """Export a segmentation checkpoint to `out_dir`/model.onnx, validate it, write model.json, return its contents."""
    ultralytics = _import_ultralytics()
    src = Path(weights)
    if not src.is_file():
        raise FileNotFoundError(f"weights not found: {src}")
    model = ultralytics.YOLO(str(src))
    task = getattr(model, "task", None)
    if task != "segment":
        raise ValueError(f"{src} is a {task!r} model; the pack needs a segmentation model (e.g. yolo11n-seg.pt)")
    names = model.names
    names_list = [str(names[k]) for k in sorted(names)] if isinstance(names, dict) else [str(n) for n in names]
    exported = Path(model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=simplify, dynamic=False, half=half))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "model.onnx"
    shutil.move(str(exported), target)
    facts = validate_yolo_seg_onnx(target, imgsz=imgsz)
    if len(names_list) != facts["nc"]:
        raise ValueError(f"model has {len(names_list)} class names but its output has nc={facts['nc']}")
    info: dict[str, Any] = {
        "names": names_list,
        "imgsz": imgsz,
        "nc": facts["nc"],
        "opset": opset,
        "half": half,
        "simplify": simplify,
        "sha256": backends.sha256(target),
        "source_weights": src.name,
        "source_weights_sha256": backends.sha256(src),
        "ultralytics": getattr(ultralytics, "__version__", "unknown"),
        "licence_note": LICENCE_NOTE,
    }
    (out / "model.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True, help="a segmentation checkpoint, e.g. yolo11n-seg.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--half", action="store_true", help="FP16 export (needs a GPU in ultralytics)")
    ap.add_argument("--no-simplify", dest="simplify", action="store_false", help="skip the onnx simplifier")
    a = ap.parse_args(argv)
    try:
        info = export_yolo(a.weights, a.out_dir, a.imgsz, a.opset, a.half, a.simplify)
    except (RuntimeError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"wrote {Path(a.out_dir) / 'model.onnx'}: nc {info['nc']}, imgsz {info['imgsz']}, opset {info['opset']}, "
          f"sha256 {info['sha256'][:16]}...; model.json written")
    print(f"licence: {LICENCE_NOTE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
