"""Evaluate a lower-body checkpoint on fixed commands: height error, velocity error, falls,
pelvis pitch swing and foot slip per (speed, height) block, arms scripted as in training.

    scripts/launch.sh scripts/lower_body/eval_lower_body.py --headless --checkpoint logs/rsl_rl/g1_lower_body/<run>/model_299.pt
    ... --disturb   also applies the training payloads and pushes (default: clean)
"""

import argparse
import functools
import importlib.metadata as metadata
import os

print = functools.partial(print, flush=True)  # noqa: A001 - Kit block-buffers piped stdout


import pinocchio  # noqa: E402,F401 - must be imported before Kit loads its own copy

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="HumanoidArc-G1-LowerBody-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--block_steps", type=int, default=250)
parser.add_argument("--heights", type=float, nargs="+", default=[0.62, 0.68, 0.72, 0.78])
parser.add_argument("--speeds", type=float, nargs="+", default=[0.0, 0.5])
parser.add_argument("--disturb", action="store_true")
parser.add_argument("--noise", action="store_true", help="Keep the training observation noise (default: off).")
parser.add_argument("--random_commands", action="store_true", help="Keep the training command sampling (one block).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import humanoid_arc_bme.envs.g1_lower_body  # noqa: E402,F401 - registers the gym ids

WARMUP = 100
FOOT_CONTACT_Z = 0.06


def pitch_deg(q):
    w, x, y, z = q.unbind(-1)
    return torch.rad2deg(torch.asin((2 * (w * y - z * x)).clamp(-1, 1)))


def main():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.observations.policy.enable_corruption = args_cli.noise
    if not args_cli.disturb:
        cfg.events.wrist_payload = None
        cfg.events.push_robot = None
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=args_cli.device)

    u = env.unwrapped
    robot, cmd = u.scene["robot"], u.command_manager.get_term("base_command")
    feet = robot.find_bodies(".*_ankle_roll_link")[0]
    obs = env.get_observations()
    print(f"[EVAL] {args_cli.num_envs} envs, {'with' if args_cli.disturb else 'no'} payloads/pushes, {args_cli.block_steps} steps/block")
    rows = []
    with torch.inference_mode():
        for vx in args_cli.speeds:
            for h in args_cli.heights:
                z, pitch = [], []
                slip = torch.zeros(u.num_envs, device=u.device)
                falls, prev = 0, None
                for step in range(args_cli.block_steps):
                    if not args_cli.random_commands:
                        cmd.is_standing_env[:] = False
                        cmd.vel_command_b[:] = torch.tensor([vx, 0.0, 0.0], device=u.device)
                        cmd.height_command[:] = h
                        cmd.time_left[:] = 1e9
                    obs, _, dones, extras = env.step(policy(obs))
                    policy.reset(dones)
                    falls += int((dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool()).sum())
                    if step >= WARMUP:
                        z.append(robot.data.root_pos_w[:, 2].clone())
                        pitch.append(pitch_deg(robot.data.root_quat_w))
                        fp = robot.data.body_pos_w[:, feet]
                        if prev is not None and vx == 0.0:
                            planted = (fp[..., 2] < FOOT_CONTACT_Z) & (prev[..., 2] < FOOT_CONTACT_Z)
                            slip += ((fp[..., :2] - prev[..., :2]).norm(dim=-1) * planted).sum(-1)
                        prev = fp.clone()
                z, pitch = torch.stack(z), torch.stack(pitch)
                herr = (z - h).abs().mean()
                swing = (pitch.max(0).values - pitch.min(0).values).mean()
                vel_err = (robot.data.root_lin_vel_b[:, 0] - vx).abs().mean()
                print(
                    f"[BLOCK] vx {vx:.1f} h {h:.2f}: pelvis z {z.mean():.3f} |err| {herr * 100:.1f}cm | "
                    f"pitch swing {swing:.1f}deg | slip {slip.mean() * 100:.1f}cm | |vx err| {vel_err:.2f} | falls {falls}"
                )
                rows.append((float(herr), falls))
    print(f"[SUMMARY] mean height error {sum(r[0] for r in rows) / len(rows) * 100:.1f}cm, falls {sum(r[1] for r in rows)}")


if __name__ == "__main__":
    main()
    os._exit(0)
