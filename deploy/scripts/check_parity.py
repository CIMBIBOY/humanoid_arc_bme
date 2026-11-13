#!/usr/bin/env python3
"""Numerical parity of every inference backend found here against the torch actions recorded in the sim, with a verdict.

    python3 deploy/scripts/check_parity.py <policy dir or pack with closedloop.npz> [--engine e.fp32.engine ...]
        [--backends ort-cpu,ort-trt,trt] [--out parity.json]

Three checks (all on closedloop.npz from tools/training/export_lowerbody_onnx.py: 300 steps x 16 envs, torch-driven sim):
  1. builder: the observation rebuilt by the real chain (state_from_lowstate -> LowerBodyObsBuilder) from the recorded
     robot fields vs the recorded sim observation (max |diff|; this is the robot's input path, independent of backend).
  2. open loop: backend(recorded obs) vs recorded torch action, and vs ORT CPU, as raw action and as joint target (rad,
     after the S1 position clamp + decode). The recorded inputs were produced by torch, so this is teacher-forced.
  3. feedback: each backend runs the chain on the recorded states but feeds back ITS OWN clamped last action (5-frame
     history), against ORT CPU doing the same: shows whether small (FP16) errors grow through the action feedback.
Verdict per backend: PASS when open-loop max |raw - torch| <= tol (FP32 1e-4; FP16 2e-2 raw = 5 mrad target) and
the feedback target drift <= 10x tol*scale; FAIL otherwise. A PASS is a numerical statement only, not closed-loop
stability on the robot (that is the sim-as-robot run, W4 S4, repeated with the chosen backend).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import backends  # noqa: E402
from packdata import load_policy_dir, replay_inputs  # noqa: E402

from humanoid_arc_bme.onboard.runtime.io.unitree import state_from_lowstate  # noqa: E402
from humanoid_arc_bme.onboard.runtime.obs_builders import LowerBodyObsBuilder  # noqa: E402

TOL_FP32, TOL_FP16 = 1e-4, 2e-2


def decode(lay, raw):
    a = np.asarray(raw, np.float64)
    if lay.raw_lo is not None:
        a = np.clip(a, lay.raw_lo, lay.raw_hi)
    return lay.action_offset + lay.action_scale * a


def builder_check(lay, inputs) -> float:
    b = LowerBodyObsBuilder(lay)
    worst = 0.0
    for inp in inputs:
        if inp["done"]:
            b.reset()
        st, _ = state_from_lowstate(inp["lowstate"], 0.0)
        worst = max(worst, float(np.abs(b.build(st, inp["last_action"], inp["command"]) - inp["obs"]).max()))
    return worst


def feedback_run(runner, lay, inputs):
    b = LowerBodyObsBuilder(lay)
    last = np.zeros(len(lay.action_joint_names))
    out = []
    for inp in inputs:
        if inp["done"]:
            b.reset()
            last = np.zeros_like(last)
        st, _ = state_from_lowstate(inp["lowstate"], 0.0)
        raw = runner.run(b.build(st, last, inp["command"])[None])[0]
        last, tgt = b.decode(raw)
        out.append(tgt)
    return np.array(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_dir")
    ap.add_argument("--npz", help="default: <model_dir>/closedloop.npz")
    ap.add_argument("--engine", action="append", default=[], help="TensorRT engine(s); precision read from <engine>.json")
    ap.add_argument("--backends", help="comma list; default: all detected (ort-trt is run in FP32 and FP16)")
    ap.add_argument("--envs", type=int, default=16, help="recorded envs used (16 = all 4800 samples)")
    ap.add_argument("--out", default="parity.json")
    a = ap.parse_args(argv)

    raw_layout, lay, onnx_path = load_policy_dir(a.model_dir)
    npz = Path(a.npz or Path(a.model_dir) / "closedloop.npz")
    if not npz.exists():
        sys.exit(f"[parity] no recording {npz}: export with tools/training/export_lowerbody_onnx.py or pass --npz")
    inputs = replay_inputs(npz, raw_layout["all_joint_names"], envs=a.envs)
    X = np.stack([i["obs"] for i in inputs]).astype(np.float32)
    Y = np.stack([i["action"] for i in inputs]).astype(np.float64)
    Yt = np.stack([decode(lay, y) for y in Y])

    report = {"model_dir": str(a.model_dir), "onnx_sha256": backends.sha256(onnx_path), "npz": str(npz),
              "samples": len(X), "versions": backends.versions(), "builder_max_abs_obs_diff": builder_check(lay, inputs),
              "results": [], "skipped": {}}
    print(f"[parity] builder obs vs sim obs: max |diff| {report['builder_max_abs_obs_diff']:.2e} on {len(X)} samples")

    plan = []  # (label, backend name, kwargs, tol)
    names = a.backends.split(",") if a.backends else backends.detect(a.engine[0] if a.engine else None)
    for n in names:
        if n == "trt":
            for e in a.engine:
                meta = Path(e + ".json")
                prec = json.loads(meta.read_text()).get("precision", "fp32") if meta.exists() else "fp32"
                plan.append((f"trt:{Path(e).name}", "trt", {"engine": e}, TOL_FP16 if prec == "fp16" else TOL_FP32))
        elif n == "ort-trt":
            plan += [("ort-trt/fp32", n, {"fp16": False}, TOL_FP32), ("ort-trt/fp16", n, {"fp16": True}, TOL_FP16)]
        else:
            plan.append((n, n, {}, TOL_FP32))
    if "ort-cpu" not in [p[1] for p in plan]:
        plan.insert(0, ("ort-cpu", "ort-cpu", {}, TOL_FP32))

    ref_open = ref_fb = None
    for label, n, kw, tol in plan:
        try:
            r = backends.make_runner(n, onnx_path, **kw)
        except Exception as e:  # noqa: BLE001
            report["skipped"][label] = f"{type(e).__name__}: {e}"
            print(f"[parity] skip {label}: {e}")
            continue
        P = np.stack([r.run(x[None])[0] for x in X]).astype(np.float64)
        Pt = np.stack([decode(lay, p) for p in P])
        fb = feedback_run(r, lay, inputs)
        if label == "ort-cpu":
            ref_open, ref_fb = P, fb
        res = {"backend": label, "info": r.info, "tol_raw": tol,
               "raw_vs_torch_max": float(np.abs(P - Y).max()),
               "raw_vs_torch_p99": float(np.percentile(np.abs(P - Y), 99)),
               "target_rad_vs_torch_max": float(np.abs(Pt - Yt).max()),
               "raw_vs_ortcpu_max": float(np.abs(P - ref_open).max()) if ref_open is not None else None,
               "feedback_target_rad_vs_ortcpu_max": float(np.abs(fb - ref_fb).max()) if ref_fb is not None else None,
               "nonfinite": int((~np.isfinite(P)).sum())}
        fb_ok = res["feedback_target_rad_vs_ortcpu_max"] is None or \
            res["feedback_target_rad_vs_ortcpu_max"] <= 10 * tol * lay.action_scale
        res["verdict"] = "PASS" if (res["raw_vs_torch_max"] <= tol and fb_ok and res["nonfinite"] == 0) else "FAIL"
        report["results"].append(res)
        print(f"[parity] {label:22s} raw vs torch max {res['raw_vs_torch_max']:.2e} (tol {tol:.0e}) | target rad max "
              f"{res['target_rad_vs_torch_max']:.2e} | feedback drift vs ORT-CPU {res['feedback_target_rad_vs_ortcpu_max']:.2e} "
              f"rad | {res['verdict']}")
    report["verdict"] = "PASS" if report["results"] and all(r["verdict"] == "PASS" for r in report["results"]) and \
        report["builder_max_abs_obs_diff"] < 1e-4 else "FAIL"
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"[parity] overall {report['verdict']}; wrote {a.out}")
    return report


if __name__ == "__main__":
    rep = main()
    sys.exit(0 if rep["verdict"] == "PASS" else 1)
