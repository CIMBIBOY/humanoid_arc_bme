#!/usr/bin/env python3
"""Check an exported policy directory (export_policy_onnx.py output) with onnxruntime on CPU.

    python3 scripts/export/check_onnx.py <dir>               # policy.onnx, closedloop.npz, export.json in <dir>
    python3 scripts/export/check_onnx.py --engine-providers  # which onnxruntime providers this machine has

Checks: the input "obs" and output "actions" names, shapes [1, obs_dim] and [1, action_dim], float32 types, the policy.onnx
sha256 recorded in export.json, no NaN or inf, max |onnx - torch| over closedloop.npz below --atol (default 1e-4), and
batch-1 latency (median and p99 of 500 timed runs after 20 warm-up runs). Exit 0 when all checks pass, 1 when one fails,
2 on a usage or load error. Latency is always CPUExecutionProvider; the Jetson TensorRT path is timed by bench_policy.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy" / "scripts"))

import backends  # noqa: E402

ONNX_INPUT = "obs"
ONNX_OUTPUT = "actions"
RUNS = 500
WARMUP = 20


def _signature(sess: ort.InferenceSession, obs_dim: int, action_dim: int) -> tuple[dict[str, Any], list[str]]:
    """Describe the graph's single input and output, and the reasons it does not match the contract."""
    ins, outs = sess.get_inputs(), sess.get_outputs()
    fails: list[str] = []
    if len(ins) != 1 or len(outs) != 1:
        fails.append(f"signature: expected 1 input and 1 output, got {len(ins)} and {len(outs)}")
    desc: dict[str, Any] = {}
    for kind, tensors, name, dim in (("input", ins, ONNX_INPUT, obs_dim), ("output", outs, ONNX_OUTPUT, action_dim)):
        if len(tensors) != 1:
            continue
        t = tensors[0]
        shape = list(t.shape)
        desc[kind] = {"name": t.name, "shape": shape, "type": t.type}
        if t.name != name:
            fails.append(f"signature: {kind} is named {t.name!r}, expected {name!r}")
        if shape != [1, dim]:
            fails.append(f"signature: {kind} shape {shape}, expected [1, {dim}]")
        if t.type != "tensor(float)":
            fails.append(f"signature: {kind} type {t.type}, expected float32")
    return desc, fails


def _latency_ms(sess: ort.InferenceSession, x: np.ndarray) -> tuple[float, float]:
    """(median, p99) wall time in milliseconds of RUNS batch-1 runs after WARMUP runs."""
    for _ in range(WARMUP):
        sess.run([ONNX_OUTPUT], {ONNX_INPUT: x})
    times = np.empty(RUNS)
    for i in range(RUNS):
        t0 = time.perf_counter()
        sess.run([ONNX_OUTPUT], {ONNX_INPUT: x})
        times[i] = time.perf_counter() - t0
    ms = times * 1e3
    return float(np.median(ms)), float(np.percentile(ms, 99))


def check_policy_pack_dir(pack_dir: str | Path, atol: float = 1e-4) -> dict[str, Any]:
    """Check policy.onnx in `pack_dir` against closedloop.npz and export.json; return the results.

    The returned dict has `passed` (bool), `failures` (list of reasons, empty when passed), `max_abs_diff`,
    `median_ms`, `p99_ms` and the signature. Raises ValueError when the directory is not a policy export at all.
    """
    d = Path(pack_dir)
    try:
        info = json.loads((d / "export.json").read_text())
        obs_dim, action_dim = int(info["obs_dim"]), int(info["action_dim"])
    except (OSError, KeyError, ValueError) as e:
        raise ValueError(f"{d}: no usable export.json ({e})") from e
    try:
        sess = ort.InferenceSession(str(d / "policy.onnx"), providers=["CPUExecutionProvider"])
    except Exception as e:  # noqa: BLE001 - onnxruntime raises its own Fail type for a bad file
        raise ValueError(f"{d / 'policy.onnx'}: onnxruntime cannot load it: {e}") from e
    with np.load(d / "closedloop.npz") as z:
        obs, actions = z["obs"], z["actions"]
    if obs.ndim != 2 or obs.shape[1] != obs_dim or actions.shape != (obs.shape[0], action_dim):
        raise ValueError(f"closedloop.npz shapes {obs.shape} / {actions.shape} do not fit obs_dim {obs_dim}, "
                         f"action_dim {action_dim}")

    desc, fails = _signature(sess, obs_dim, action_dim)
    onnx_sha = backends.sha256(d / "policy.onnx")
    if onnx_sha != info.get("onnx_sha256"):
        fails.append("policy.onnx sha256 differs from export.json (file changed after export)")

    out = np.concatenate([sess.run([ONNX_OUTPUT], {ONNX_INPUT: x[None]})[0] for x in obs])
    finite = bool(np.isfinite(out).all())
    if not finite:
        fails.append("non-finite values in the ONNX output")
    max_diff = float(np.max(np.abs(out.astype(np.float64) - actions.astype(np.float64))))
    if not max_diff <= atol:  # written so that NaN fails too
        fails.append(f"max |onnx - torch| {max_diff:.3e} > atol {atol:.1e}")

    median_ms, p99_ms = _latency_ms(sess, obs[:1].astype(np.float32))
    return {
        "pack": str(d),
        "samples": int(obs.shape[0]),
        "signature": desc,
        "onnx_sha256": onnx_sha,
        "providers": sess.get_providers(),
        "finite": finite,
        "max_abs_diff": max_diff,
        "atol": atol,
        "runs": RUNS,
        "median_ms": median_ms,
        "p99_ms": p99_ms,
        "failures": fails,
        "passed": not fails,
    }


def print_table(res: dict[str, Any]) -> None:
    sig = res["signature"]
    rows = [
        ("signature", "ok" if not any(f.startswith("signature") for f in res["failures"]) else "FAIL",
         f"{sig.get('input', {}).get('shape')} {sig.get('input', {}).get('type')} -> "
         f"{sig.get('output', {}).get('shape')} {sig.get('output', {}).get('type')}"),
        ("policy.onnx sha256 == export.json", "FAIL" if any("sha256" in f for f in res["failures"]) else "ok",
         res["onnx_sha256"][:16] + "..."),
        ("finite outputs", "ok" if res["finite"] else "FAIL", ""),
        (f"max |onnx - torch| over {res['samples']} samples", "ok" if res["max_abs_diff"] <= res["atol"] else "FAIL",
         f"{res['max_abs_diff']:.2e} (atol {res['atol']:.0e})"),
        (f"latency, batch 1, {res['runs']} runs", "info",
         f"median {res['median_ms']:.3f} ms, p99 {res['p99_ms']:.3f} ms"),
    ]
    print(f"{res['pack']}  [{', '.join(res['providers'])}]")
    for name, status, detail in rows:
        print(f"  {name:<40} {status:<5} {detail}")
    for f in res["failures"]:
        print(f"  - {f}")
    print("RESULT: " + ("PASS" if res["passed"] else "FAIL"))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pack_dir", nargs="?", help="directory with policy.onnx, closedloop.npz and export.json")
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--engine-providers", action="store_true", help="list onnxruntime's providers and exit")
    a = ap.parse_args(argv)
    if a.engine_providers:
        print(f"onnxruntime {ort.__version__}: {', '.join(ort.get_available_providers())}")
        return 0
    if not a.pack_dir:
        ap.error("pack_dir is required (or pass --engine-providers)")
    try:
        res = check_policy_pack_dir(a.pack_dir, a.atol)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print_table(res)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
