"""Package layout, gym registration, base plug-in metadata, asset paths. No Isaac Sim needed."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import pytest

import humanoid_arc_bme.tasks.pick_place  # noqa: F401 - registers the gym ids
from humanoid_arc_bme.robots import BASES, get_base
from humanoid_arc_bme.robots import assets

UPPER_BODY_DIM = 28  # two wrist poses (2 x 7) + 14 hand joints

# Ids are stored in recorded datasets and robomimic checkpoints: they must not change.
RECORDED_IDS = ["HumanoidArc-G1-PickPlace-Demo-v0", "HumanoidArc-G1-PickPlace-Mimic-v0"]


@pytest.mark.parametrize("task", RECORDED_IDS + ["HumanoidArc-G1-PickPlace-FixedBase-Demo-v0"])
def test_gym_ids_registered_and_resolvable(task):
    spec = gym.spec(task)
    module, _, name = spec.kwargs["env_cfg_entry_point"].partition(":")
    assert module.startswith("humanoid_arc_bme.tasks.pick_place.")
    assert importlib.util.find_spec(module) is not None
    assert name


def test_robomimic_configs_exist_and_parse():
    kwargs = gym.spec("HumanoidArc-G1-PickPlace-Demo-v0").kwargs
    for key in ("robomimic_bc_obs_noise_cfg_entry_point", "robomimic_bc_obs_noise_h50_cfg_entry_point"):
        module, _, rel = kwargs[key].partition(":")
        path = Path(importlib.import_module(module).__file__).parent / rel
        cfg = json.loads(path.read_text())
        assert cfg["algo_name"] == "bc"


def test_base_plugins():
    assert set(BASES) == {"agile_legs", "fixed"}
    agile, fixed = get_base("agile_legs"), get_base("fixed")
    # The recorded datasets' layout: 28 upper body + 4 AGILE command, idle = stand still at 0.72 m.
    assert UPPER_BODY_DIM + agile.command_dim == 32
    assert agile.idle_command == (0.0, 0.0, 0.0, 0.72)
    assert fixed.command_dim == 0 and fixed.idle_command == ()
    for base in BASES.values():
        assert len(base.idle_command) == base.command_dim
    with pytest.raises(ValueError):
        get_base("wheels")


def test_ik_urdf_ships_with_package():
    assert assets.G1_IK_URDF.is_file()
    assert "right_wrist_yaw_link" in assets.G1_IK_URDF.read_text()


def test_asset_root_override():
    code = "from humanoid_arc_bme.robots import assets; print(assets.G1_29DOF_USD)"
    env = {**os.environ, "HUMANOID_ARC_ASSETS": "/opt/somewhere"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
    assert out.strip() == "/opt/somewhere/G1_29dof/g1.usd"
