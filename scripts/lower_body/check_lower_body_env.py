"""Check the lower-body training env with the Agile policy in the legs' place.

Agile shares the env's observation/action layout, so it should stand and walk here if the
layout is right; falls under the randomized arm motion and payloads are the baseline a
trained policy has to beat. Also reports the command, payload and arm-motion statistics.

    scripts/launch.sh scripts/lower_body/check_lower_body_env.py --headless --num_envs 256 --steps 1000
"""

import argparse
import functools
import os

print = functools.partial(print, flush=True)  # noqa: A001 - Kit block-buffers piped stdout


import pinocchio  # noqa: E402,F401 - must be imported before Kit loads its own copy

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", default="HumanoidArc-G1-LowerBody-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--policy", default="agile", help="'agile', 'zero' (default-pose targets) or a TorchScript file.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher({**vars(args_cli)}).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import humanoid_arc_bme.envs.g1_lower_body  # noqa: E402,F401 - registers the gym ids
from humanoid_arc_core.robots.assets import AGILE_LOCOMOTION_POLICY  # noqa: E402


def main():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    obs, _ = env.reset()
    obs = obs["policy"]
    print(f"[INFO] obs {tuple(obs.shape)}, action dim {env.action_manager.total_action_dim}")
    if args_cli.policy == "zero":
        policy = lambda o: torch.zeros(o.shape[0], 12, device=env.device)  # noqa: E731
    else:
        path = AGILE_LOCOMOTION_POLICY if args_cli.policy == "agile" else args_cli.policy
        policy = torch.jit.load(str(path), map_location=env.device)

    robot, arms = env.scene["robot"], env.action_manager.get_term("arms")
    wrist = robot.find_bodies(".*_wrist_yaw_link")[0]
    cmd_term = env.command_manager.get_term("base_command")
    upper_targets = []
    falls = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    fall_demo = fall_random = eps_demo = eps_random = 0
    z_err = []
    with torch.inference_mode():
        for step in range(args_cli.steps):
            action = policy(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            obs = obs["policy"]
            upper_targets.append(arms.processed_actions.clone())
            standing = cmd_term.command[:, :3].abs().sum(-1) < 1e-6
            z_err.append((robot.data.root_pos_w[:, 2] - cmd_term.height_command)[standing])
            done = terminated | truncated
            if done.any():
                d = arms._use_demo  # mode of the episode that just ended (reset happens inside step)
                falls += terminated
                fall_demo += int((terminated & d).sum())
                fall_random += int((terminated & ~d).sum())
                eps_demo += int((done & d).sum())
                eps_random += int((done & ~d).sum())

    masses = robot.root_physx_view.get_masses()[:, wrist].sum(-1).cpu()
    up = torch.stack(upper_targets)  # (T, N, U)
    z_err = torch.cat(z_err)
    print(f"[CHECK] obs dim {obs.shape[-1]} (expect 83), policy output {action.shape[-1]} (expect 12)")
    print(f"[CHECK] wrist mass, both hands: min {masses.min():.2f} mean {masses.mean():.2f} max {masses.max():.2f} kg")
    print(f"[CHECK] arm target std over time (rad): mean {up.std(0).mean():.3f}, max joint {up.std(0).max():.3f}")
    print(f"[CHECK] commands: vx {cmd_term.command[:,0].min():.2f}..{cmd_term.command[:,0].max():.2f}, "
          f"height {cmd_term.height_command.min():.2f}..{cmd_term.height_command.max():.2f}")
    print(f"[CHECK] standing pelvis z minus commanded height: mean {z_err.mean():+.3f} m (n={len(z_err)})")
    print(f"[RESULT] policy {args_cli.policy}: falls {int(falls.sum())} in {args_cli.steps * env.num_envs} env-steps; "
          f"demo-arm episodes ended {eps_demo} (falls {fall_demo}), random-arm {eps_random} (falls {fall_random})")


if __name__ == "__main__":
    main()
    os._exit(0)
