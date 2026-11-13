"""deploy/scripts: backend detection, TensorRT version rule, build_engine.sh refusals (with a stub trtexec), the policy-pack
verifier, and (where onnxruntime is installed) bench_policy.py / check_parity.py on the W4 deployE export.

    python3 -m pytest -q tests/runtime/test_bench_scripts.py        # ORT parts skip without onnxruntime
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "deploy" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import backends  # noqa: E402
import make_policy_pack  # noqa: E402
from verify_policy_pack import verify_pack  # noqa: E402

EXPORT = ROOT / "outputs" / "W4" / "onnx" / "deployE"
needs_export = pytest.mark.skipif(not (EXPORT / "policy.onnx").exists(), reason="no W4 deployE export")


def _ort():
    return pytest.importorskip("onnxruntime")


# ---- TensorRT version rule ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("v,ok", [("10.7.0.23", True), ("10.7", True), ("10.7.1", True), ("10.3.0.30", False),
                                  ("10.13.3.9", False), ("10.70.0", False), ("1.7.0", False), ("", False)])
def test_trt_version_rule(v, ok):
    assert backends.trt_version_ok(v) is ok


def test_trt_refusal_message():
    with pytest.raises(backends.BackendUnavailable, match="exactly 10.7"):
        backends.require_trt_version("10.3.0.30")


def test_detect_never_lists_trt_without_engine():
    assert "trt" not in backends.detect(None)


# ---- build_engine.sh with a stub trtexec ----------------------------------------------------------------------------

def _stub_trt(tmp_path, banner="100700", libver="10.7.0"):
    root = tmp_path / "TensorRT"
    (root / "bin").mkdir(parents=True)
    (root / "lib").mkdir()
    (root / "lib" / f"libnvinfer.so.{libver}").write_bytes(b"")
    exe = root / "bin" / "trtexec"
    exe.write_text(f"""#!/usr/bin/env bash
echo "&&&& RUNNING TensorRT.trtexec [TensorRT v{banner}] # trtexec $*"
for a in "$@"; do case $a in --saveEngine=*) echo engine-bytes > "${{a#--saveEngine=}}";; esac; done
echo "$*" >> "{tmp_path}/calls.txt"
""")
    exe.chmod(0o755)
    return root


def _run_build(tmp_path, root, prec="fp32", *extra):
    pack = tmp_path / "pack"
    pack.mkdir(exist_ok=True)
    (pack / "policy.onnx").write_bytes(b"not really onnx")
    env = {**os.environ, "TensorRT_ROOT": str(root), "ENGINE_CACHE": str(tmp_path / "cache"), "HOME": str(tmp_path)}
    env.pop("TRTEXEC", None)
    return subprocess.run(["bash", str(SCRIPTS / "build_engine.sh"), str(pack), prec, *extra], env=env,
                          capture_output=True, text=True, timeout=60)


def test_build_engine_refuses_without_trtexec(tmp_path):
    r = _run_build(tmp_path, tmp_path / "nowhere")
    assert r.returncode == 2 and "trtexec not found" in r.stderr


def test_build_engine_refuses_other_library_version(tmp_path):
    r = _run_build(tmp_path, _stub_trt(tmp_path, banner="100300", libver="10.3.0"))
    assert r.returncode == 2 and "10.7" in r.stderr and "libnvinfer.so.10.7" in r.stderr


def test_build_engine_refuses_other_banner_and_leaves_no_engine(tmp_path):
    r = _run_build(tmp_path, _stub_trt(tmp_path, banner="101303"))
    assert r.returncode == 2 and "10.13" in r.stderr
    assert not list((tmp_path / "cache").rglob("*.engine"))


def test_build_engine_flags_meta_and_cache(tmp_path):
    root = _stub_trt(tmp_path)
    r = _run_build(tmp_path, root, "fp32")
    assert r.returncode == 0, r.stderr
    eng = Path(r.stdout.strip().splitlines()[-1])
    assert eng.exists() and eng.name.startswith("policy.fp32.trt10.7.0.")
    meta = json.loads(Path(str(eng) + ".json").read_text())
    assert meta["precision"] == "fp32" and meta["tensorrt"] == "10.7.0" and meta["trtexec_banner"] == "v100700"
    assert meta["engine_sha256"] == backends.sha256(eng)
    call = (tmp_path / "calls.txt").read_text()
    assert "--noTF32" in call and "--fp16" not in call and "--skipInference" in call and "--shapes" not in call
    r2 = _run_build(tmp_path, root, "fp32")
    assert r2.returncode == 0 and "cached" in r2.stdout
    assert len((tmp_path / "calls.txt").read_text().splitlines()) == 1  # no second build
    r3 = _run_build(tmp_path, root, "fp16")
    assert r3.returncode == 0 and "--fp16" in (tmp_path / "calls.txt").read_text().splitlines()[-1]


# ---- policy pack ------------------------------------------------------------------------------------------------------

@pytest.fixture
def pack(tmp_path):
    if not (EXPORT / "policy.onnx").exists():
        pytest.skip("no W4 deployE export")
    out = tmp_path / "pack"
    assert make_policy_pack.main([str(EXPORT), str(out)]) == 0
    return out


def _resum(p):
    files = sorted(f.name for f in p.iterdir() if f.name != "SHA256SUMS")
    (p / "SHA256SUMS").write_text("".join(f"{backends.sha256(p / n)}  {n}\n" for n in files))


def test_pack_ok(pack):
    errs, facts = verify_pack(pack)
    assert errs == [] and facts["obs_dim"] == 384 and facts["n_action"] == 12
    assert facts["clamp_vs_s1_max_rad"] < 1e-5


def test_pack_refuses_hash_mismatch(pack):
    with open(pack / "policy.onnx", "ab") as f:
        f.write(b"\0")
    errs, _ = verify_pack(pack)
    assert any("policy.onnx: sha256 mismatch" in e for e in errs)


def test_pack_refuses_unlisted_file(pack):
    (pack / "extra.bin").write_bytes(b"x")
    assert any("extra.bin is not listed" in e for e in verify_pack(pack)[0])


def _edit_layout(pack, fn):
    lay = json.loads((pack / "layout.json").read_text())
    fn(lay)
    (pack / "layout.json").write_text(json.dumps(lay))
    _resum(pack)
    return verify_pack(pack)[0]


def test_pack_refuses_clamp_mismatch(pack):
    def widen(lay):
        lay["raw_hi"][10] += 0.4  # ankle roll clamp 0.1 rad past the S1 range
    errs = _edit_layout(pack, widen)
    assert any("clamp of left_ankle_roll_joint" in e for e in errs)


def test_pack_refuses_missing_clamp(pack):
    assert any("no raw-action clamp" in e for e in _edit_layout(pack, lambda lay: lay.update(raw_lo=None, raw_hi=None)))


def test_pack_refuses_obs_layout_mismatch(pack):
    # one fewer history frame: the humanoid_arc_bme.onboard.runtime loader rejects obs_dim, i.e. the ONNX input would not match
    errs = _edit_layout(pack, lambda lay: lay.update(hist=4))
    assert any("rejected by humanoid_arc_bme.onboard.runtime loader" in e for e in errs)


def test_pack_refuses_shape_mismatch(pack):
    def shrink(lay):  # a consistent layout for a model with 4 frames: loader passes, ONNX input shape does not
        lay["hist"], lay["obs_dim"] = 4, 4 + 4 * 76
    assert any("ONNX input [1, 384]" in e for e in _edit_layout(pack, shrink))


def test_pack_refuses_joint_order_swap(pack):
    def swap(lay):
        o = lay["action_offset"]
        o[6], o[8] = o[8], o[6]  # knee and ankle-pitch defaults swapped
    errs = _edit_layout(pack, swap)
    assert any("action_offset of left_knee_joint" in e for e in errs)


def test_pack_refuses_wrong_control_dt(pack):
    errs = _edit_layout(pack, lambda lay: lay.update(control_dt=0.005))
    assert any("control_dt" in e for e in errs)


def test_pack_refuses_card_for_another_model(pack):
    card = (pack / "model-card.md").read_text().replace(backends.sha256(pack / "policy.onnx"), "0" * 64)
    (pack / "model-card.md").write_text(card)
    _resum(pack)
    assert any("model-card.md" in e for e in verify_pack(pack)[0])


def test_pack_refuses_engine_from_other_onnx_or_trt(pack, tmp_path):
    eng = pack / "policy.fp32.engine"
    eng.write_bytes(b"e")
    (pack / "policy.fp32.engine.json").write_text(json.dumps({"onnx_sha256": "f" * 64, "tensorrt": "10.3.0",
                                                               "engine_sha256": backends.sha256(eng)}))
    _resum(pack)
    errs = verify_pack(pack)[0]
    assert any("built from another ONNX" in e for e in errs) and any("TensorRT 10.3.0" in e for e in errs)


# ---- benchmark and parity on the real export (onnxruntime only) -------------------------------------------------------

@needs_export
def test_bench_policy_writes_stats(tmp_path):
    _ort()
    import bench_policy

    out = tmp_path / "b.json"
    rep = bench_policy.main([str(EXPORT), "--calls", "300", "--warmup", "20", "--backends", "ort-cpu", "--out", str(out)])
    r = rep["results"][0]
    assert json.loads(out.read_text())["results"][0]["backend"] == "ort-cpu/t1"
    for k in ("convert", "obs", "infer", "decode", "s1", "total"):
        s = r["stages_us"][k]
        assert 0 < s["p50"] <= s["p95"] <= s["p99"] <= s["max"]
    assert r["stages_us"]["total"]["p50"] >= r["stages_us"]["infer"]["p50"]
    assert r["determinism"]["bitwise_identical"]
    assert rep["inputs"] == "closedloop.npz replay"


@needs_export
def test_bench_chain_equals_onnx_leg_policy():
    """The timed chain computes the same targets as the runtime's OnnxLegPolicy (no drift between bench and runtime)."""
    _ort()
    from humanoid_arc_bme.onboard.runtime.io.unitree import state_from_lowstate
    from humanoid_arc_bme.onboard.runtime.obs_builders import LowerBodyObsBuilder
    from humanoid_arc_bme.onboard.runtime.policies import OnnxLegPolicy
    from packdata import load_policy_dir, replay_inputs

    raw, lay, onnx_path = load_policy_dir(EXPORT)
    inputs = replay_inputs(EXPORT / "closedloop.npz", raw["all_joint_names"], envs=1)[:50]
    r = backends.make_runner("ort-cpu", onnx_path)
    b = LowerBodyObsBuilder(lay)
    pol = OnnxLegPolicy(EXPORT, command=lambda: None, session=r.sess)
    for inp in inputs:
        st, _ = state_from_lowstate(inp["lowstate"], 0.0)
        if inp["done"]:
            b.reset()
            pol.reset()
        pol.last_action = inp["last_action"]
        pol.command = lambda c=inp["command"]: c
        _, t = b.decode(r.run(b.build(st, inp["last_action"], inp["command"])[None])[0])
        assert np.array_equal(np.asarray(pol.act(st).positions), t)


@needs_export
def test_check_parity_pass_and_fail(tmp_path):
    _ort()
    import check_parity

    rep = check_parity.main([str(EXPORT), "--backends", "ort-cpu", "--envs", "2", "--out", str(tmp_path / "p.json")])
    assert rep["verdict"] == "PASS" and rep["results"][0]["raw_vs_torch_max"] < 1e-4
    assert rep["builder_max_abs_obs_diff"] < 1e-5
    # a corrupted recording must FAIL
    bad = tmp_path / "bad"
    bad.mkdir()
    for n in ("policy.onnx", "layout.json"):
        shutil.copy2(EXPORT / n, bad / n)
    with np.load(EXPORT / "closedloop.npz") as z:
        d = {k: z[k] for k in z.files}
    d["actions"] = d["actions"] + 1e-2
    np.savez(bad / "closedloop.npz", **d)
    rep = check_parity.main([str(bad), "--backends", "ort-cpu", "--envs", "1", "--out", str(tmp_path / "q.json")])
    assert rep["verdict"] == "FAIL"
