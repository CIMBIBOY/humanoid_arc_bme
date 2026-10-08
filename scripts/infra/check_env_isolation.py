"""Check that parallel envs don't physically interact: stack every env's robot
at the same spot (env_spacing=0) and walk them with the pretrained G1
velocity policy. With inter-env collision filtering working, the result must
match a normally spaced run (no falls, same tracking error, no velocity
spikes); with it broken, overlapping robots push each other apart violently.

    scripts/launch.sh scripts/infra/check_env_isolation.py --headless                    # co-located, filtered
    scripts/launch.sh scripts/infra/check_env_isolation.py --headless --no_filter        # control: must FAIL
    scripts/launch.sh scripts/infra/check_env_isolation.py --headless --env_spacing 2.5  # baseline
"""

import argparse
import functools
import importlib.metadata as metadata
import os

print = functools.partial(print, flush=True)  # noqa: A001 - Kit block-buffers piped stdout

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--checkpoint", default="assets/policies/g1_velocity_flat/checkpoint.pt")
parser.add_argument("--task", default="Isaac-Velocity-Flat-G1-Play-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--env_spacing", type=float, default=0.0)
parser.add_argument("--no_filter", action="store_true", help="disable inter-env collision filtering (control run)")
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

WARMUP_STEPS = 50
# A walking G1 never exceeds ~2 m/s linear body speed; overlapping robots
# resolving interpenetration get launched far faster than that.
SPEED_SPIKE = 5.0


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.scene.env_spacing = args_cli.env_spacing
    env_cfg.scene.filter_collisions = not args_cli.no_filter
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=args_cli.device)

    robot = env.unwrapped.scene["robot"]
    command = env.unwrapped.command_manager.get_term("base_velocity")
    origins = env.unwrapped.scene.env_origins
    spread = (origins[:, :2] - origins[0, :2]).norm(dim=-1).max().item()
    print(f"[ISO] env_spacing {args_cli.env_spacing} (max origin distance {spread:.2f} m),"
          f" filter_collisions {env_cfg.scene.filter_collisions}, replicate_physics {env_cfg.scene.replicate_physics}")

    obs = env.get_observations()
    falls, errors, max_speed, min_z = 0, [], 0.0, float("inf")
    with torch.inference_mode():
        for step in range(args_cli.steps):
            obs, _, dones, extras = env.step(policy(obs))
            policy.reset(dones)
            timeouts = extras.get("time_outs", torch.zeros_like(dones))
            falls += int((dones.bool() & ~timeouts.bool()).sum())
            if step >= WARMUP_STEPS:
                errors.append((command.command[:, :2] - robot.data.root_lin_vel_b[:, :2]).norm(dim=-1).mean().item())
                max_speed = max(max_speed, robot.data.body_lin_vel_w.norm(dim=-1).max().item())
                min_z = min(min_z, robot.data.root_pos_w[:, 2].min().item())

    isolated = falls == 0 and max_speed < SPEED_SPIKE
    print(
        f"[ISO] {args_cli.num_envs} envs x {args_cli.steps} steps | falls {falls}"
        f" | |cmd - vel| {sum(errors) / len(errors):.3f} m/s | max body speed {max_speed:.2f} m/s"
        f" | min pelvis z {min_z:.3f} | {'ISOLATED' if isolated else 'INTERACTING'}"
    )


if __name__ == "__main__":
    main()
    os._exit(0)
