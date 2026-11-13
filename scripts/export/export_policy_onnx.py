#!/usr/bin/env python3
"""Export an rsl-rl actor MLP, with its observation normaliser baked in, to ONNX. No Isaac needed.

    python3 scripts/export/export_policy_onnx.py --checkpoint <model_N.pt> --out_dir <dir>
        [--obs_dim N] [--action_dim N] [--hidden 512,256,128] [--activation elu] [--opset 17] [--layout layout.json]

Writes into <out_dir>:
    policy.onnx     input "obs" float32 [1, obs_dim] (raw observations), output "actions" float32 [1, action_dim]
    closedloop.npz  obs [256, obs_dim] (seeded, drawn around the normaliser statistics) and actions [256, action_dim]
                    (the torch policy's answers, for check_onnx.py)
    export.json     shapes, architecture, normaliser, opset, library versions, sha256 of checkpoint and policy.onnx
    layout.json     copied from --layout when given

Accepted actor layouts: rsl-rl 3/4 (`actor.0.weight`, `actor.2.weight`, ...) and rsl-rl 5 (`actor.mlp.0.weight`, ...).
Accepted normaliser statistics (`_mean`, `_std` or `_var`): `actor_obs_normalizer.*`, `actor.obs_normalizer.*` or
`obs_normalizer.*`. rsl-rl keeps the normaliser's epsilon out of the state dict, so it is a flag (default 0.01, rsl-rl's).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy" / "scripts"))

import backends  # noqa: E402

ONNX_INPUT = "obs"
ONNX_OUTPUT = "actions"
CLOSEDLOOP_SAMPLES = 256
CLOSEDLOOP_SEED = 0
DEFAULT_OBS_EPS = 1e-2  # rsl_rl.modules.normalization.EmpiricalNormalization default
ACTIVATIONS: dict[str, type[nn.Module]] = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}

_MLP_KEY = re.compile(r"^(actor\.(?:mlp\.)?)(\d+)\.(weight|bias)$")
_NORMALIZER_PREFIXES = ("actor_obs_normalizer.", "actor.obs_normalizer.", "obs_normalizer.")


class _ObsNormalize(nn.Module):
    """(obs - mean) / (std + eps): rsl-rl's EmpiricalNormalization forward, with its statistics as buffers.

    Shapes: obs [N, obs_dim] -> [N, obs_dim]. Units follow the observations.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float) -> None:
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("denom", torch.as_tensor(std + eps, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.mean) / self.denom


def _actor_layers(state_dict: Mapping[str, torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """[(weight [out, in], bias [out])] of the actor MLP, in layer order, as float32 tensors."""
    groups: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    for key, value in state_dict.items():
        m = _MLP_KEY.match(key)
        if m:
            groups.setdefault(m.group(1), {}).setdefault(int(m.group(2)), {})[m.group(3)] = value
    if not groups:
        actor_keys = sorted(k for k in state_dict if k.startswith("actor"))
        raise ValueError("no actor MLP weights in the checkpoint: expected keys like 'actor.0.weight' (rsl-rl 3/4) or "
                         f"'actor.mlp.0.weight' (rsl-rl 5); actor keys present: {actor_keys[:10]}")
    if len(groups) > 1:
        raise ValueError(f"both layouts present in the checkpoint: {sorted(groups)}; export one actor only")
    (prefix, layers), = groups.items()
    indices = sorted(layers)
    expected = list(range(0, 2 * len(indices), 2))
    if indices != expected:
        raise ValueError(f"actor MLP layers must be consecutive linear layers {expected} (activations in between), "
                         f"found {indices}")
    out = []
    for idx in indices:
        parts = layers[idx]
        missing = {"weight", "bias"} - set(parts)
        if missing:
            raise ValueError(f"{prefix}{idx} is missing {sorted(missing)}")
        w, b = parts["weight"].detach().to(torch.float32), parts["bias"].detach().to(torch.float32)
        if not (torch.isfinite(w).all() and torch.isfinite(b).all()):
            raise ValueError(f"{prefix}{idx} has non-finite values")
        out.append((w, b))
    return out


def _architecture(layers: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> tuple[int, list[int], int]:
    """(obs_dim, hidden dims, action_dim) implied by the layer shapes; raises if consecutive layers do not chain."""
    for i in range(1, len(layers)):
        fan_in, fan_out = layers[i][0].shape[1], layers[i - 1][0].shape[0]
        if fan_in != fan_out:
            raise ValueError(f"actor layer {2 * i} takes {fan_in} inputs but layer {2 * (i - 1)} produces {fan_out}")
    return layers[0][0].shape[1], [int(w.shape[0]) for w, _ in layers[:-1]], int(layers[-1][0].shape[0])


def _find_normalizer(state_dict: Mapping[str, torch.Tensor], obs_dim: int) -> tuple[np.ndarray, np.ndarray, str] | None:
    """(mean [obs_dim], std [obs_dim], key prefix without the trailing dot) of the first normaliser, or None."""
    for prefix in _NORMALIZER_PREFIXES:
        if f"{prefix}_mean" not in state_dict:
            continue
        mean = state_dict[f"{prefix}_mean"].detach().to(torch.float64).reshape(-1).numpy()
        if f"{prefix}_std" in state_dict:
            std = state_dict[f"{prefix}_std"].detach().to(torch.float64).reshape(-1).numpy()
        elif f"{prefix}_var" in state_dict:
            std = np.sqrt(state_dict[f"{prefix}_var"].detach().to(torch.float64).reshape(-1).numpy())
        else:
            raise ValueError(f"{prefix}_mean has neither {prefix}_std nor {prefix}_var")
        if mean.size != obs_dim or std.size != obs_dim:
            raise ValueError(f"{prefix} statistics have {mean.size} entries, the actor takes {obs_dim} observations")
        if not (np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()):
            raise ValueError(f"{prefix} statistics must be finite with positive std")
        return mean, std, prefix[:-1]
    return None


def build_actor(layers: Sequence[tuple[torch.Tensor, torch.Tensor]], normalizer: _ObsNormalize | None,
                activation: str) -> nn.Sequential:
    """Plain torch MLP: [normaliser] -> Linear -> act -> ... -> Linear (no output activation, as rsl-rl)."""
    act = ACTIVATIONS[activation]
    mods: list[nn.Module] = [normalizer] if normalizer is not None else []
    for i, (w, b) in enumerate(layers):
        lin = nn.Linear(w.shape[1], w.shape[0])
        with torch.no_grad():
            lin.weight.copy_(w)
            lin.bias.copy_(b)
        mods.append(lin)
        if i < len(layers) - 1:
            mods.append(act())
    return nn.Sequential(*mods).eval()


def _expect(name: str, given: Any, found: Any) -> None:
    if given is not None and given != found:
        raise ValueError(f"{name}={given} does not match the checkpoint's actor ({name}={found})")


def export_actor(
    state_dict: Mapping[str, torch.Tensor],
    obs_dim: int | None,
    action_dim: int | None,
    hidden: Sequence[int] | None,
    activation: str,
    out_dir: str | Path,
    opset: int = 17,
    *,
    obs_eps: float = DEFAULT_OBS_EPS,
    layout_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Export the actor of an rsl-rl state dict to `out_dir` and return the export.json contents.

    obs_dim, action_dim and hidden (units: count) are checked against the weights when given and inferred when None.
    The ONNX graph takes raw observations [1, obs_dim] float32 and returns actions [1, action_dim] float32.
    Raises ValueError with the reason for a wrong key layout, shape mismatch or non-finite weights.
    """
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation {activation!r} not in {sorted(ACTIVATIONS)}")
    layers = _actor_layers(state_dict)
    found_obs, found_hidden, found_action = _architecture(layers)
    _expect("obs_dim", obs_dim, found_obs)
    _expect("action_dim", action_dim, found_action)
    if hidden is not None and list(hidden) != found_hidden:
        raise ValueError(f"hidden={list(hidden)} does not match the checkpoint's actor (hidden={found_hidden})")
    norm = _find_normalizer(state_dict, found_obs)
    normalizer = _ObsNormalize(norm[0], norm[1], obs_eps) if norm else None
    model = build_actor(layers, normalizer, activation)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / "policy.onnx"
    torch.onnx.export(
        model,
        (torch.zeros(1, found_obs),),
        str(onnx_path),
        input_names=[ONNX_INPUT],
        output_names=[ONNX_OUTPUT],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )

    rng = np.random.default_rng(CLOSEDLOOP_SEED)
    mean, std = (norm[0], norm[1]) if norm else (np.zeros(found_obs), np.ones(found_obs))
    obs = (mean + std * rng.standard_normal((CLOSEDLOOP_SAMPLES, found_obs))).astype(np.float32)
    with torch.no_grad():
        actions = model(torch.from_numpy(obs)).numpy()
    if not np.isfinite(actions).all():
        raise ValueError("the torch actor produced non-finite actions on the closed-loop samples")
    np.savez(out / "closedloop.npz", obs=obs, actions=actions)

    if layout_path is not None:
        json.loads(Path(layout_path).read_text())  # refuse a layout that is not JSON
        shutil.copy2(layout_path, out / "layout.json")

    info: dict[str, Any] = {
        "obs_dim": found_obs,
        "action_dim": found_action,
        "hidden": found_hidden,
        "activation": activation,
        "opset": opset,
        "input": ONNX_INPUT,
        "output": ONNX_OUTPUT,
        "normalizer": {"source": norm[2], "eps": obs_eps} if norm else None,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": backends.sha256(checkpoint_path) if checkpoint_path else None,
        "onnx_sha256": backends.sha256(onnx_path),
        "layout_sha256": backends.sha256(out / "layout.json") if layout_path else None,
        "closedloop_samples": CLOSEDLOOP_SAMPLES,
        "closedloop_seed": CLOSEDLOOP_SEED,
        "torch": torch.__version__,
        "onnx": onnx.__version__,
    }
    (out / "export.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    return info


def _int_list(text: str) -> list[int]:
    try:
        vals = [int(x) for x in text.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"not a comma-separated list of integers: {text!r}") from e
    if not vals or min(vals) < 1:
        raise argparse.ArgumentTypeError(f"needs positive integers, got {text!r}")
    return vals


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True, help="rsl-rl model_N.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--obs_dim", type=int, help="checked against the weights; inferred when omitted")
    ap.add_argument("--action_dim", type=int, help="checked against the weights; inferred when omitted")
    ap.add_argument("--hidden", type=_int_list, help="e.g. 512,256,128; checked against the weights when given")
    ap.add_argument("--activation", default="elu", choices=sorted(ACTIVATIONS))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--obs_eps", type=float, default=DEFAULT_OBS_EPS, help="normaliser epsilon (rsl-rl default 0.01)")
    ap.add_argument("--layout", help="layout.json to copy next to the export")
    a = ap.parse_args(argv)
    try:
        ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001 - report any load failure with the reason
        print(f"error: cannot load {a.checkpoint} with weights_only=True: {e}", file=sys.stderr)
        return 2
    state = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else None
    if state is None:
        keys = sorted(ckpt) if isinstance(ckpt, dict) else type(ckpt).__name__
        print(f"error: {a.checkpoint} has no model_state_dict (top-level keys: {keys})", file=sys.stderr)
        return 2
    try:
        info = export_actor(state, a.obs_dim, a.action_dim, a.hidden, a.activation, a.out_dir, a.opset,
                            obs_eps=a.obs_eps, layout_path=a.layout, checkpoint_path=a.checkpoint)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    found = info["normalizer"]
    if found:
        print(f"normaliser: found {found['source']}.* (eps {found['eps']}), baked into the graph")
    else:
        print("normaliser: none found, the actor takes the observations as they are")
    print(f"actor: obs_dim {info['obs_dim']} -> hidden {info['hidden']} ({info['activation']}) -> action_dim "
          f"{info['action_dim']}")
    print(f"wrote {Path(a.out_dir) / 'policy.onnx'} (opset {info['opset']}, sha256 {info['onnx_sha256'][:16]}...), "
          f"closedloop.npz ({info['closedloop_samples']} samples), export.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
