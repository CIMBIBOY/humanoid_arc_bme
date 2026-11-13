#!/usr/bin/env python3
"""Refuse a policy pack unless its hashes, tensor shapes, action clamp and observation layout all check out.

    python3 deploy/scripts/verify_policy_pack.py <pack dir> [--limits config/g1_limits.yaml] [--engine <cached engine>]
        [--parity] [--allow-lin-vel]

Pack format (docs/plan/jetson-inference.md, "Policy pack"): one flat directory
    policy.onnx       exported actor (+ estimator), static [1, obs_dim] -> [1, n_action], float32
    layout.json       tools/training/export_lowerbody_onnx.py layout (loaded with humanoid_arc_bme.onboard.runtime.obs_builders.LowerBodyLayout)
    model-card.md     provenance; must contain the line `onnx_sha256: <hex>` of policy.onnx
    SHA256SUMS        `sha256sum` format, lists EVERY other file in the directory
    optional: *.engine + <engine>.json (prebuilt on the same device type by build_engine.sh), closedloop.npz (parity)
Exit 0 and "PACK OK" or exit 1 and "PACK REFUSED" with every reason found. `verify_pack()` is importable.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import backends  # noqa: E402
from packdata import DEFAULT_LIMITS  # noqa: E402

from humanoid_arc_bme.onboard.runtime.io.fake import G1_JOINTS  # noqa: E402
from humanoid_arc_bme.onboard.runtime.obs_builders import LowerBodyLayout, LowerBodyObsBuilder  # noqa: E402

REQUIRED = ("policy.onnx", "layout.json", "model-card.md")
CLAMP_TOL_RAD = 1e-5
CONTROL_DT = 0.02


def _parse_sums(path: Path) -> tuple[dict, list]:
    sums, errs = {}, []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        m = re.fullmatch(r"([0-9a-f]{64}) [ *]([^/\\]+)", line.strip())
        if not m:
            errs.append(f"SHA256SUMS line {i} malformed or not a flat file name: {line!r}")
            continue
        sums[m.group(2)] = m.group(1)
    return sums, errs


def _onnx_io(onnx_path: Path):
    """[(name, shape, elem)] for inputs and outputs, via onnx if present, else onnxruntime."""
    try:
        import onnx

        m = onnx.load(str(onnx_path), load_external_data=False)

        def io(vs):
            return [(v.name, [d.dim_value if d.HasField("dim_value") else d.dim_param for d in v.type.tensor_type.shape.dim],
                     "float32" if v.type.tensor_type.elem_type == onnx.TensorProto.FLOAT else str(v.type.tensor_type.elem_type))
                    for v in vs]

        return io(m.graph.input), io(m.graph.output)
    except ImportError:
        r = backends.make_runner("ort-cpu", onnx_path)
        conv = {"tensor(float)": "float32"}
        return ([(i.name, list(i.shape), conv.get(i.type, i.type)) for i in r.sess.get_inputs()],
                [(o.name, list(o.shape), conv.get(o.type, o.type)) for o in r.sess.get_outputs()])


def _check_engine_meta(meta_path: Path, onnx_sha: str, errs: list, label: str):
    if not meta_path.exists():
        errs.append(f"{label}: no metadata {meta_path.name}")
        return
    m = json.loads(meta_path.read_text())
    if m.get("onnx_sha256") != onnx_sha:
        errs.append(f"{label}: built from another ONNX (meta onnx_sha256 {str(m.get('onnx_sha256'))[:12]}...)")
    if not backends.trt_version_ok(str(m.get("tensorrt", ""))):
        errs.append(f"{label}: TensorRT {m.get('tensorrt')} (exactly {backends.REQUIRED_TRT}.x required)")
    eng = meta_path.with_suffix("")  # strip .json
    if m.get("engine_sha256") and eng.exists() and backends.sha256(eng) != m["engine_sha256"]:
        errs.append(f"{label}: engine file hash differs from its metadata")


def verify_pack(pack, limits_path=DEFAULT_LIMITS, engine=None, parity=False, allow_lin_vel=False) -> tuple[list, dict]:
    """(errors, facts). An empty error list means the pack may be loaded."""
    pack = Path(pack)
    errs, facts = [], {}
    if not pack.is_dir():
        return [f"{pack} is not a directory"], facts
    # 1. hashes: SHA256SUMS covers every file, every file matches, no subdirectories
    sums_p = pack / "SHA256SUMS"
    if not sums_p.exists():
        return ["no SHA256SUMS"], facts
    sums, e = _parse_sums(sums_p)
    errs += e
    present = {p.name for p in pack.iterdir() if p.name != "SHA256SUMS"}
    for p in pack.iterdir():
        if p.is_dir():
            errs.append(f"subdirectory {p.name}/ not allowed in a pack")
    for n in sorted(present - set(sums)):
        errs.append(f"{n} is not listed in SHA256SUMS")
    for n in REQUIRED:
        if n not in sums:
            errs.append(f"required file {n} missing from SHA256SUMS")
    for n, h in sums.items():
        f = pack / n
        if not f.is_file():
            errs.append(f"{n} listed in SHA256SUMS but missing")
        elif backends.sha256(f) != h:
            errs.append(f"{n}: sha256 mismatch")
    if any("sha256 mismatch" in x or "missing" in x for x in errs):
        return errs, facts  # do not parse files we cannot trust
    onnx_p = pack / "policy.onnx"
    onnx_sha = sums["policy.onnx"]
    facts["onnx_sha256"] = onnx_sha

    # 2. layout through the runtime's own loader
    try:
        raw = json.loads((pack / "layout.json").read_text())
        lay = LowerBodyLayout.from_json(pack / "layout.json")
        obs_dim = LowerBodyObsBuilder(lay).obs_dim
    except Exception as ex:  # noqa: BLE001 - any loader failure is a refusal
        return errs + [f"layout.json rejected by humanoid_arc_bme.onboard.runtime loader: {type(ex).__name__}: {ex}"], facts
    facts.update(obs_dim=obs_dim, n_action=len(lay.action_joint_names), hist=lay.hist)
    if set(lay.obs_joint_names) != set(G1_JOINTS) or len(lay.obs_joint_names) != len(G1_JOINTS):
        errs.append("layout obs_joint_names are not exactly the 29 G1 body joints the robot reports")
    if not set(lay.action_joint_names) <= set(lay.obs_joint_names):
        errs.append("action joints not all in the observation")
    if len(set(lay.action_joint_names)) != len(lay.action_joint_names):
        errs.append("duplicate action joint names")
    dflt = dict(zip(lay.obs_joint_names, lay.obs_default_pos))
    for n, o in zip(lay.action_joint_names, lay.action_offset):
        if n in dflt and abs(dflt[n] - o) > 1e-6:
            errs.append(f"action_offset of {n} ({o}) differs from its obs default pose ({dflt[n]})")
    if lay.lin_vel and not allow_lin_vel:
        errs.append("layout observes base_lin_vel, which the G1 does not measure (use a W4 deploy export)")
    dt = raw.get("control_dt")
    if dt is None or abs(float(dt) - CONTROL_DT) > 1e-9:
        errs.append(f"control_dt {dt} != {CONTROL_DT} (the runtime runs at 50 Hz)")
    for k in ("obs_default_pos", "action_offset", "raw_lo", "raw_hi"):
        v = raw.get(k)
        if v is not None and not all(math.isfinite(float(x)) for x in v):
            errs.append(f"layout {k} has non-finite values")
    if not (math.isfinite(lay.action_scale) and lay.action_scale > 0):
        errs.append("action_scale must be positive and finite")

    # 3. action clamp == S1 position limits (the policy was trained with exactly this clamp, W4)
    if lay.raw_lo is None or lay.raw_hi is None:
        errs.append("layout has no raw-action clamp (raw_lo/raw_hi): not a deployable export")
    else:
        from humanoid_arc_bme.onboard.api.safety import load_limits

        lim = load_limits(limits_path)
        facts["limits_validated_on_robot"] = lim.validated_on_robot
        worst = 0.0
        for n, o, lo, hi in zip(lay.action_joint_names, lay.action_offset, lay.raw_lo, lay.raw_hi):
            if not lo < hi:
                errs.append(f"clamp of {n}: raw_lo >= raw_hi")
                continue
            try:
                a, b = lim.joint_range(n)
            except Exception as ex:  # noqa: BLE001
                errs.append(f"S1 limits have no range for {n}: {ex}")
                continue
            d = max(abs(o + lay.action_scale * lo - a), abs(o + lay.action_scale * hi - b))
            worst = max(worst, d)
            if d > CLAMP_TOL_RAD:
                errs.append(f"clamp of {n} decodes to [{o + lay.action_scale * lo:.4f}, {o + lay.action_scale * hi:.4f}] "
                            f"rad, S1 range is [{a:.4f}, {b:.4f}] (limits file changed since training?)")
        facts["clamp_vs_s1_max_rad"] = worst

    # 4. tensor shapes
    try:
        ins, outs = _onnx_io(onnx_p)
        facts["onnx_io"] = {"inputs": ins, "outputs": outs}
        if len(ins) != 1 or len(outs) != 1:
            errs.append(f"ONNX must have 1 input and 1 output, has {len(ins)} / {len(outs)}")
        else:
            if ins[0][1] != [1, obs_dim] or ins[0][2] != "float32":
                errs.append(f"ONNX input {ins[0][1]} {ins[0][2]} != [1, {obs_dim}] float32 (layout)")
            if outs[0][1] != [1, len(lay.action_joint_names)] or outs[0][2] != "float32":
                errs.append(f"ONNX output {outs[0][1]} {outs[0][2]} != [1, {len(lay.action_joint_names)}] float32")
    except Exception as ex:  # noqa: BLE001
        errs.append(f"ONNX unreadable: {type(ex).__name__}: {ex}")

    # 5. model card names this exact model
    card = (pack / "model-card.md").read_text()
    m = re.search(r"onnx_sha256:\s*`?([0-9a-f]{64})", card)
    if not m or m.group(1) != onnx_sha:
        errs.append("model-card.md does not state `onnx_sha256: <hash of policy.onnx>`")

    # 6. engines: packaged ones and an external cached one
    for e in sorted(pack.glob("*.engine")):
        _check_engine_meta(pack / (e.name + ".json"), onnx_sha, errs, e.name)
    if engine:
        _check_engine_meta(Path(str(engine) + ".json"), onnx_sha, errs, f"engine {Path(engine).name}")

    # 7. it runs (and optionally matches the recording)
    if not errs:
        try:
            r = backends.make_runner("ort-cpu", onnx_p)
            y = r.run(np.zeros((1, obs_dim), np.float32))
            if y.shape != (1, len(lay.action_joint_names)) or not np.isfinite(y).all():
                errs.append(f"ORT smoke run gave shape {y.shape} / non-finite values")
            if parity and (pack / "closedloop.npz").exists():
                import check_parity

                rep = check_parity.main([str(pack), "--backends", "ort-cpu", "--envs", "4", "--out", "/dev/null"])
                facts["parity_raw_vs_torch_max"] = rep["results"][0]["raw_vs_torch_max"]
                if rep["verdict"] != "PASS":
                    errs.append("parity against closedloop.npz FAILED")
        except backends.BackendUnavailable as ex:
            facts["smoke"] = f"skipped: {ex}"
    return errs, facts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pack")
    ap.add_argument("--limits", default=str(DEFAULT_LIMITS))
    ap.add_argument("--engine", help="a cached engine outside the pack (checks its .json metadata)")
    ap.add_argument("--parity", action="store_true", help="also run ORT parity on the pack's closedloop.npz")
    ap.add_argument("--allow-lin-vel", action="store_true")
    a = ap.parse_args(argv)
    errs, facts = verify_pack(a.pack, a.limits, a.engine, a.parity, a.allow_lin_vel)
    print(json.dumps(facts, indent=1, default=str))
    if errs:
        print("PACK REFUSED:\n  - " + "\n  - ".join(errs))
        return 1
    print("PACK OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
