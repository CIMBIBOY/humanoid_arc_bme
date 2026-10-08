"""Evaluate an rsl_rl velocity-tracking checkpoint: falls, velocity-tracking
error, and pelvis height over N parallel envs with random velocity commands.

    scripts/launch.sh scripts/locomotion/eval_velocity.py --headless \
        --checkpoint assets/policies/g1_velocity_flat/checkpoint.pt
"""

import argparse
import functools
import importlib.metadata as metadata
import os

print = functools.partial(print, flush=True)  # noqa: A001 - Kit block-buffers piped stdout

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-Velocity-Flat-G1-Play-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=1000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

WARMUP_STEPS = 100


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=args_cli.device)

    robot = env.unwrapped.scene["robot"]
    command = env.unwrapped.command_manager.get_term("base_velocity")
    obs = env.get_observations()
    falls, errors, heights, speeds = 0, [], [], []
    with torch.inference_mode():
        for step in range(args_cli.steps):
            obs, _, dones, extras = env.step(policy(obs))
            policy.reset(dones)
            timeouts = extras.get("time_outs", torch.zeros_like(dones))
            falls += int((dones.bool() & ~timeouts.bool()).sum())
            if step >= WARMUP_STEPS:
                cmd = command.command[:, :2]
                errors.append((cmd - robot.data.root_lin_vel_b[:, :2]).norm(dim=-1).mean().item())
                speeds.append(cmd.norm(dim=-1).mean().item())
                heights.append(robot.data.root_pos_w[:, 2].mean().item())

    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    print(
        f"[EVAL] {args_cli.num_envs} envs x {args_cli.steps} steps | falls {falls}"
        f" | |cmd - vel| {mean(errors):.3f} m/s at |cmd| {mean(speeds):.3f} m/s | pelvis z {mean(heights):.3f}"
    )


if __name__ == "__main__":
    main()
    os._exit(0)
