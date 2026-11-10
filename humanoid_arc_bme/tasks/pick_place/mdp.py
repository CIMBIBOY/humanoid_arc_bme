"""Task-specific MDP terms for the pick-place task."""

import math

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg


def reset_joints_to(env, env_ids: torch.Tensor, joint_pos: dict[str, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Write `joint_pos` (by joint name) to the sim at reset, zero velocity.

    Unlike changing the articulation's init_state.joint_pos, this leaves its
    default joint positions alone: the Agile lower-body policy's observations
    and action offsets are relative to those, and moving them made the robot
    walk away backwards.
    """
    robot = env.scene[asset_cfg.name]
    target = robot.data.default_joint_pos[env_ids].clone()
    for name, value in joint_pos.items():
        target[:, robot.data.joint_names.index(name)] = value
    robot.write_joint_state_to_sim(target, torch.zeros_like(target), env_ids=env_ids)


def object_placed(
    env,
    goal: tuple[float, float, float],
    xy_tolerance: float = 0.03,
    z_tolerance: float = 0.02,
    max_speed: float = 0.05,
    max_tilt_deg: float = 15.0,
    min_hand_distance: float = 0.20,
    hand_link_name: str = "right_wrist_yaw_link",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The object rests upright at `goal` (env frame) and the hand has let go.

    Upright matters for Mimic, which counts a demo as successful if this held
    at any single step: without it, a box knocked over near the start and
    tumbling through the goal area counted (seen in generation).

    "Let go" is the right wrist being `min_hand_distance` from the object: the
    wrist sits about 0.13m from it while grasping (GRASP_OFFSET in
    skills/scripted_pick_place.py), so this only holds once the hand has pulled away.
    """
    obj = env.scene[object_cfg.name]
    robot = env.scene["robot"]
    obj_pos = obj.data.root_pos_w - env.scene.env_origins
    goal_t = torch.tensor(goal, device=env.device)
    wrist_pos = robot.data.body_pos_w[:, robot.data.body_names.index(hand_link_name)] - env.scene.env_origins

    at_goal = (obj_pos[:, :2] - goal_t[:2]).norm(dim=-1) < xy_tolerance
    at_goal &= (obj_pos[:, 2] - goal_t[2]).abs() < z_tolerance
    at_rest = obj.data.root_lin_vel_w.norm(dim=-1) < max_speed
    z_axis = torch.tensor((0.0, 0.0, 1.0), device=env.device).expand(env.num_envs, 3)
    upright = math_utils.quat_apply(obj.data.root_quat_w, z_axis)[:, 2] > math.cos(math.radians(max_tilt_deg))
    released = (wrist_pos - obj_pos).norm(dim=-1) > min_hand_distance
    return at_goal & at_rest & upright & released
