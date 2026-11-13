"""Fixtures for the ONNX export tests: synthetic rsl-rl actor checkpoints and YOLO-shaped ONNX graphs.

Nothing here needs Isaac, rsl-rl or ultralytics. Everything is seeded and written only under pytest's tmp_path.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "export"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def make_actor_state_dict(
    obs_dim: int = 83,
    action_dim: int = 12,
    hidden: Sequence[int] = (512, 256, 128),
    *,
    normalizer: bool = True,
    mlp_prefix: str = "actor.",
    norm_prefix: str = "actor_obs_normalizer.",
    var_only: bool = False,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """An rsl-rl-style state dict: actor MLP (+ optional observation normaliser) plus the critic keys it carries.

    Weights are scaled by 1/sqrt(fan_in); the normaliser mean is N(0, 1) and the variance is in [0.5, 1.5)
    (units follow the observations). `var_only` stores only `_var`, as some rsl-rl versions do.
    """
    g = torch.Generator().manual_seed(seed)
    sizes = [obs_dim, *hidden, action_dim]
    sd: dict[str, torch.Tensor] = {}
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        sd[f"{mlp_prefix}{2 * i}.weight"] = torch.randn(fan_out, fan_in, generator=g) / fan_in**0.5
        sd[f"{mlp_prefix}{2 * i}.bias"] = 0.1 * torch.randn(fan_out, generator=g)
    sd["critic.0.weight"] = torch.randn(1, obs_dim, generator=g)
    sd["critic.0.bias"] = torch.zeros(1)
    sd["std"] = torch.ones(action_dim)
    if normalizer:
        var = torch.rand(1, obs_dim, generator=g) + 0.5
        sd[f"{norm_prefix}_mean"] = torch.randn(1, obs_dim, generator=g)
        sd[f"{norm_prefix}_var"] = var
        if not var_only:
            sd[f"{norm_prefix}_std"] = var.sqrt()
        sd[f"{norm_prefix}count"] = torch.tensor(1000, dtype=torch.long)
    return sd


def make_yolo_seg_onnx(
    path: Path,
    imgsz: int = 64,
    nc: int = 3,
    anchors: int | None = None,
    out1_shape: Sequence[int] | None = None,
    dynamic_batch: bool = False,
) -> Path:
    """A YOLO-seg-shaped ONNX graph with the right names and shapes, built from onnx.helper (no weights, no ultralytics).

    Input "images" [1, 3, S, S]; "output0" [1, 4 + nc + 32, anchors]; "output1" [1, 32, S/4, S/4]. Both outputs are the
    mean of the input expanded to their shape. `anchors`, `out1_shape` and `dynamic_batch` override the valid values
    to build wrong graphs.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    n = anchors if anchors is not None else sum((imgsz // s) ** 2 for s in (8, 16, 32))
    s0 = [1, 4 + nc + 32, n]
    s1 = list(out1_shape) if out1_shape is not None else [1, 32, imgsz // 4, imgsz // 4]
    batch = "batch" if dynamic_batch else 1
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["images"], ["mean"], keepdims=0),
            helper.make_node("Expand", ["mean", "shape0"], ["output0"]),
            helper.make_node("Expand", ["mean", "shape1"], ["output1"]),
        ],
        "yolo_seg_synthetic",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [batch, 3, imgsz, imgsz])],
        [
            helper.make_tensor_value_info("output0", TensorProto.FLOAT, s0),
            helper.make_tensor_value_info("output1", TensorProto.FLOAT, s1),
        ],
        initializer=[
            numpy_helper.from_array(np.array(s0, dtype=np.int64), "shape0"),
            numpy_helper.from_array(np.array(s1, dtype=np.int64), "shape1"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8  # the newest IR that onnxruntime 1.31 loads without complaint
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def corrupt_largest_initializer(path: Path, mode: str) -> None:
    """Rewrite the largest initializer of an ONNX file in place: mode 'flip' negates it, 'nan' sets it to NaN."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    init = max(model.graph.initializer, key=lambda t: int(np.prod(t.dims)))
    arr = numpy_helper.to_array(init).astype(np.float32)
    bad = -arr if mode == "flip" else np.full_like(arr, np.nan)
    init.CopyFrom(numpy_helper.from_array(bad, init.name))
    onnx.save(model, str(path))


@pytest.fixture
def actor_state_dict():
    """Factory for synthetic rsl-rl actor state dicts (see make_actor_state_dict)."""
    return make_actor_state_dict


@pytest.fixture
def yolo_onnx():
    """Factory for YOLO-shaped ONNX files (see make_yolo_seg_onnx)."""
    return make_yolo_seg_onnx


@pytest.fixture
def corrupt_onnx():
    """Callable that corrupts the largest initializer of an ONNX file (see corrupt_largest_initializer)."""
    return corrupt_largest_initializer


@pytest.fixture
def policy_export(tmp_path):
    """Build a policy export (obs 83, actions 12) under tmp_path; returns the directory."""
    import export_policy_onnx

    def build(name: str = "export", *, hidden: Sequence[int] = (128, 64), normalizer: bool = True) -> Path:
        out = tmp_path / name
        sd = make_actor_state_dict(83, 12, hidden, normalizer=normalizer)
        export_policy_onnx.export_actor(sd, 83, 12, list(hidden), "elu", out)
        return out

    return build
