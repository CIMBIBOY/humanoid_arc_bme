"""Lower-body baseline: pelvis height, pitch swing and foot slip while the scripted arm pick-places.

Runs one ScriptedPickPlace episode per env on the demo env and reports, per env and overall,
pelvis z, pelvis pitch swing (max - min), and foot slip (horizontal foot travel while in contact),
plus falls and task success. Re-run after any lower-body change to compare against the backlog's
baseline (docs/backlog.md, problem A).

    scripts/launch.sh scripts/lower_body/baseline_lower_body.py --headless --num_envs 8 --speed 1.5
"""

import argparse
import functools
import os

print = functools.partial(print, flush=True)  # noqa: A001 - Kit block-buffers piped stdout


import pinocchio  # noqa: E402,F401 - must be imported before Kit loads its own copy

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", default="HumanoidArc-G1-PickPlace-Demo-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--speed", type=float, default=1.5)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher({**vars(args_cli), "enable_pinocchio": True}).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import humanoid_arc_core.tasks.pick_place  # noqa: E402,F401 - registers the gym ids
from humanoid_arc_core.tasks.pick_place.env_cfg import DEMO_OBJECT_POS, DEMO_PLACE_OFFSET  # noqa: E402
from humanoid_arc_core.skills.scripted_pick_place import PHASES, ScriptedPickPlace  # noqa: E402

FOOT_CONTACT_Z = 0.06  # ankle-roll link height below which the foot counts as planted


def pitch_deg(q):
    w, x, y, z = q.unbind(-1)
    return torch.rad2deg(torch.asin((2 * (w * y - z * x)).clamp(-1, 1)))


def main():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.seed = args_cli.seed
    cfg.terminations.success = None  # run the full script; timeouts/drops are what we count as failures
    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    env.reset()
    policy = ScriptedPickPlace(env, place_offset=DEMO_PLACE_OFFSET, object_home=DEMO_OBJECT_POS, speed=args_cli.speed)
    robot, obj, origins = env.scene["robot"], env.scene["object"], env.scene.env_origins
    feet = robot.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"])[0]
    n = env.num_envs
    steps = int(sum(p[1] for p in PHASES) / args_cli.speed / env.step_dt) + 1

    z, pitch = [], []
    slip = torch.zeros(n, device=env.device)
    fell = torch.zeros(n, dtype=torch.bool, device=env.device)
    prev_feet = None
    with torch.inference_mode():
        for _ in range(steps):
            _, _, terminated, truncated, _ = env.step(policy.compute())
            fell |= terminated
            z.append(robot.data.root_pos_w[:, 2].clone())
            pitch.append(pitch_deg(robot.data.root_quat_w))
            fp = robot.data.body_pos_w[:, feet]  # (n, 2, 3)
            if prev_feet is not None:
                planted = (fp[..., 2] < FOOT_CONTACT_Z) & (prev_feet[..., 2] < FOOT_CONTACT_Z)
                slip += ((fp[..., :2] - prev_feet[..., :2]).norm(dim=-1) * planted).sum(-1)
            prev_feet = fp.clone()
    z, pitch = torch.stack(z), torch.stack(pitch)
    err = (obj.data.root_pos_w - origins - policy.place_target).norm(dim=-1)
    swing = pitch.max(0).values - pitch.min(0).values

    for i in range(n):
        print(
            f"[ENV {i}] z {z[:, i].mean():.3f} (min {z[:, i].min():.3f}) pitch swing {swing[i]:.1f}deg "
            f"slip {slip[i] * 100:.1f}cm obj err {err[i] * 1000:.0f}mm{' FELL' if fell[i] else ''}"
        )
    print(
        f"[SUMMARY] speed {args_cli.speed} envs {n}: pelvis z mean {z.mean():.3f}, "
        f"pitch swing mean {swing.mean():.1f} max {swing.max():.1f}deg, foot slip mean {slip.mean() * 100:.1f}cm, "
        f"falls {int(fell.sum())}/{n}, placed within 50mm {int((err < 0.05).sum())}/{n}"
    )


if __name__ == "__main__":
    main()
    os._exit(0)
