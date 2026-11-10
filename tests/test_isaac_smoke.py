"""Builds the pick-place env on one base and steps the scripted skill. Slow: launches Isaac Sim.

    tools/launch.sh -m pytest tests/core --isaac                         # agile_legs
    tools/launch.sh -m pytest tests/core --isaac --robot_base fixed
"""

import pytest

pytestmark = pytest.mark.isaac

EXPECTED_ACTION_DIM = {"agile_legs": 32, "fixed": 28}


@pytest.mark.parametrize("robot_base", ["agile_legs", "fixed"])
def test_config_applies_base(isaac_app, robot_base):  # configs only: both bases in one process
    from humanoid_arc_bme.tasks.pick_place.env_cfg import G1PickPlaceDemoEnvCfg

    cfg = G1PickPlaceDemoEnvCfg(robot_base=robot_base)
    has_legs = cfg.actions.lower_body_joint_pos is not None
    assert has_legs == (robot_base == "agile_legs")
    assert cfg.scene.robot.spawn.articulation_props.fix_root_link == (robot_base == "fixed")


def test_env_steps_with_scripted_skill(isaac_app, robot_base):
    import torch
    from isaaclab.envs import ManagerBasedRLEnv

    from humanoid_arc_bme.skills.scripted_pick_place import ScriptedPickPlace
    from humanoid_arc_bme.tasks.pick_place.env_cfg import G1PickPlaceEnvCfg

    cfg = G1PickPlaceEnvCfg(robot_base=robot_base)
    cfg.scene.num_envs = 2
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        env.reset()
        assert env.action_manager.total_action_dim == EXPECTED_ACTION_DIM[robot_base]
        policy = ScriptedPickPlace(env)
        with torch.inference_mode():
            for _ in range(10):
                action = policy.compute()
                assert action.shape == (2, EXPECTED_ACTION_DIM[robot_base])
                env.step(action)
        assert torch.isfinite(env.scene["robot"].data.root_pos_w).all()
    finally:
        env.close()  # one env per process: see conftest.py
