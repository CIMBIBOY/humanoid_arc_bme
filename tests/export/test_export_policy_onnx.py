"""export_policy_onnx: rsl-rl actor -> ONNX with the observation normaliser baked in, parity against torch."""

from __future__ import annotations

import json

import numpy as np
import onnxruntime as ort
import pytest
import torch
from torch import nn

import export_policy_onnx as epo

OBS_DIM, ACTION_DIM, HIDDEN = 83, 12, (512, 256, 128)
EPS = 1e-2


def reference_actor(sd, hidden, mlp_prefix="actor.", norm_prefix="actor_obs_normalizer.", normalize=True):
    """Independent torch reference, built straight from the state dict; returns f(raw_obs) -> actions."""
    layers = []
    for i in range(len(hidden) + 1):
        lin = nn.Linear(sd[f"{mlp_prefix}{2 * i}.weight"].shape[1], sd[f"{mlp_prefix}{2 * i}.weight"].shape[0])
        with torch.no_grad():
            lin.weight.copy_(sd[f"{mlp_prefix}{2 * i}.weight"])
            lin.bias.copy_(sd[f"{mlp_prefix}{2 * i}.bias"])
        layers.append(lin)
        if i < len(hidden):
            layers.append(nn.ELU())
    mlp = nn.Sequential(*layers).eval()
    mean = std = None
    if normalize:
        mean = sd[f"{norm_prefix}_mean"].reshape(-1)
        std = (sd[f"{norm_prefix}_std"] if f"{norm_prefix}_std" in sd else sd[f"{norm_prefix}_var"].sqrt()).reshape(-1)

    def run(raw: torch.Tensor) -> torch.Tensor:
        x = (raw - mean) / (std + EPS) if normalize else raw
        with torch.no_grad():
            return mlp(x)

    return run


def raw_observations(sd, n=50, seed=1, norm_prefix="actor_obs_normalizer."):
    """Observations drawn around the normaliser statistics, float32 [n, obs_dim]."""
    rng = np.random.default_rng(seed)
    mean = sd[f"{norm_prefix}_mean"].reshape(-1).numpy()
    std = (sd[f"{norm_prefix}_std"] if f"{norm_prefix}_std" in sd else sd[f"{norm_prefix}_var"].sqrt()).reshape(-1).numpy()
    return (mean + std * rng.standard_normal((n, mean.size))).astype(np.float32)


def run_onnx(path, obs):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return np.stack([sess.run(None, {"obs": x[None]})[0][0] for x in obs]), sess


def test_export_matches_torch_and_bakes_normalizer(tmp_path, actor_state_dict):
    sd = actor_state_dict(OBS_DIM, ACTION_DIM, HIDDEN)
    out = tmp_path / "pack"
    info = epo.export_actor(sd, OBS_DIM, ACTION_DIM, list(HIDDEN), "elu", out)

    assert {"policy.onnx", "closedloop.npz", "export.json"} <= {p.name for p in out.iterdir()}
    with np.load(out / "closedloop.npz") as z:
        assert z["obs"].shape == (256, OBS_DIM) and z["obs"].dtype == np.float32
        assert z["actions"].shape == (256, ACTION_DIM)
    meta = json.loads((out / "export.json").read_text())
    assert meta["obs_dim"] == OBS_DIM and meta["action_dim"] == ACTION_DIM and meta["hidden"] == list(HIDDEN)
    assert meta["normalizer"] == {"source": "actor_obs_normalizer", "eps": EPS}
    assert meta["onnx_sha256"] == info["onnx_sha256"]

    obs = raw_observations(sd)
    got, sess = run_onnx(out / "policy.onnx", obs)
    assert [i.name for i in sess.get_inputs()] == ["obs"], "the normaliser must be inside the graph, not an input"
    assert [i.shape for i in sess.get_inputs()] == [[1, OBS_DIM]]
    assert [o.name for o in sess.get_outputs()] == ["actions"]
    assert [o.shape for o in sess.get_outputs()] == [[1, ACTION_DIM]]
    want = reference_actor(sd, HIDDEN)(torch.from_numpy(obs)).numpy()
    assert np.abs(got - want).max() < 1e-5


def test_normalizer_is_applied_inside_the_graph(tmp_path, actor_state_dict):
    sd = actor_state_dict(OBS_DIM, ACTION_DIM, (128, 64))
    out = tmp_path / "pack"
    epo.export_actor(sd, OBS_DIM, ACTION_DIM, [128, 64], "elu", out)
    obs = raw_observations(sd, n=8)
    got, _ = run_onnx(out / "policy.onnx", obs)
    no_norm = reference_actor(sd, (128, 64), normalize=False)(torch.from_numpy(obs)).numpy()
    assert np.abs(got - no_norm).max() > 1e-2, "the raw observations were not normalised"


def test_export_without_normalizer(tmp_path, actor_state_dict):
    sd = actor_state_dict(OBS_DIM, ACTION_DIM, (128, 64), normalizer=False)
    out = tmp_path / "pack"
    info = epo.export_actor(sd, OBS_DIM, ACTION_DIM, None, "elu", out)
    assert info["normalizer"] is None and info["hidden"] == [128, 64]
    rng = np.random.default_rng(3)
    obs = rng.standard_normal((50, OBS_DIM)).astype(np.float32)
    got, _ = run_onnx(out / "policy.onnx", obs)
    want = reference_actor(sd, (128, 64), normalize=False)(torch.from_numpy(obs)).numpy()
    assert np.abs(got - want).max() < 1e-5


@pytest.mark.parametrize(("prefix", "norm_prefix", "var_only", "source"), [
    ("actor.mlp.", "actor.obs_normalizer.", True, "actor.obs_normalizer"),
    ("actor.", "obs_normalizer.", False, "obs_normalizer"),
])
def test_other_rsl_rl_layouts(tmp_path, actor_state_dict, prefix, norm_prefix, var_only, source):
    sd = actor_state_dict(OBS_DIM, ACTION_DIM, (128, 64), mlp_prefix=prefix, norm_prefix=norm_prefix, var_only=var_only)
    out = tmp_path / "pack"
    info = epo.export_actor(sd, OBS_DIM, ACTION_DIM, [128, 64], "elu", out)
    assert info["normalizer"]["source"] == source
    obs = raw_observations(sd, n=20, norm_prefix=norm_prefix)
    got, _ = run_onnx(out / "policy.onnx", obs)
    want = reference_actor(sd, (128, 64), mlp_prefix=prefix, norm_prefix=norm_prefix)(torch.from_numpy(obs)).numpy()
    assert np.abs(got - want).max() < 1e-5


def test_layout_and_checkpoint_hashes_recorded(tmp_path, actor_state_dict):
    sd = actor_state_dict(OBS_DIM, ACTION_DIM, (128, 64))
    layout = tmp_path / "layout.json"
    layout.write_text(json.dumps({"obs_dim": OBS_DIM}))
    ckpt = tmp_path / "model_10.pt"
    torch.save({"model_state_dict": sd}, ckpt)
    out = tmp_path / "pack"
    info = epo.export_actor(sd, OBS_DIM, ACTION_DIM, [128, 64], "elu", out, layout_path=layout, checkpoint_path=ckpt)
    assert (out / "layout.json").read_text() == layout.read_text()
    assert info["layout_sha256"] and info["checkpoint_sha256"]
    assert info["checkpoint"] == str(ckpt)


@pytest.mark.parametrize(("drop", "match"), [
    (["actor.2.weight"], "actor.2 is missing"),
    (["actor.4.weight", "actor.4.bias"], "consecutive"),
    (["actor.0.weight", "actor.0.bias", "actor.2.weight", "actor.2.bias", "actor.4.weight", "actor.4.bias",
      "actor.6.weight", "actor.6.bias"], "no actor MLP weights"),
])
def test_wrong_key_layout_is_refused(tmp_path, actor_state_dict, drop, match):
    sd = actor_state_dict(OBS_DIM, ACTION_DIM, HIDDEN)
    for k in drop:
        del sd[k]
    with pytest.raises(ValueError, match=match):
        epo.export_actor(sd, OBS_DIM, ACTION_DIM, None, "elu", tmp_path / "pack")
    assert not (tmp_path / "pack").exists()


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"obs_dim": 80}, "obs_dim=80"),
    ({"action_dim": 10}, "action_dim=10"),
    ({"hidden": [256, 256, 128]}, "hidden=\\[256, 256, 128\\]"),
])
def test_architecture_mismatch_is_refused(tmp_path, actor_state_dict, kwargs, match):
    sd = actor_state_dict(OBS_DIM, ACTION_DIM, HIDDEN)
    expected = {"obs_dim": OBS_DIM, "action_dim": ACTION_DIM, "hidden": list(HIDDEN), **kwargs}
    with pytest.raises(ValueError, match=match):
        epo.export_actor(sd, activation="elu", out_dir=tmp_path / "pack", **expected)


def test_non_finite_weights_are_refused(tmp_path, actor_state_dict):
    sd = actor_state_dict(OBS_DIM, ACTION_DIM, (128, 64))
    sd["actor.0.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        epo.export_actor(sd, OBS_DIM, ACTION_DIM, None, "elu", tmp_path / "pack")


def test_cli_writes_export_and_reports_missing_state_dict(tmp_path, actor_state_dict, capsys):
    sd = actor_state_dict(OBS_DIM, ACTION_DIM, (128, 64))
    ckpt = tmp_path / "model_5.pt"
    torch.save({"model_state_dict": sd, "iter": 5}, ckpt)
    out = tmp_path / "pack"
    assert epo.main(["--checkpoint", str(ckpt), "--out_dir", str(out), "--obs_dim", str(OBS_DIM)]) == 0
    printed = capsys.readouterr().out
    assert "found actor_obs_normalizer" in printed and (out / "policy.onnx").is_file()

    bad = tmp_path / "bad.pt"
    torch.save({"iter": 5}, bad)
    assert epo.main(["--checkpoint", str(bad), "--out_dir", str(tmp_path / "x")]) == 2
    assert "no model_state_dict" in capsys.readouterr().err
