"""Side-by-side video: Agile lower body (left) vs our height-tracking policy (right).

Both robots get the same height/speed command schedule, with the arms replaying scripted pick-place
motion, and a fixed camera; the walking phases run on a 1 m radius circle so the robot stays in view.

    scripts/launch.sh scripts/lower_body/video_lower_body.py --headless --checkpoint logs/rsl_rl/g1_lower_body/<run>/model_898.pt
"""

import argparse
import functools
import importlib.metadata as metadata
import os
from pathlib import Path

print = functools.partial(print, flush=True)  # noqa: A001 - Kit block-buffers piped stdout


import pinocchio  # noqa: E402,F401 - must be imported before Kit loads its own copy

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", default="logs/lower_body_videos/agile_vs_ours.mp4")
parser.add_argument("--every", type=int, default=2, help="Record every Nth step (50Hz env: 2 -> 25fps).")
parser.add_argument("--max_steps", type=int, default=0, help="Stop early (0 = whole schedule).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import humanoid_arc_bme.envs.g1_lower_body  # noqa: E402,F401 - registers the gym ids
from humanoid_arc_core.robots.assets import AGILE_LOCOMOTION_POLICY  # noqa: E402

TASK = "HumanoidArc-G1-LowerBody-v0"
EYE = (0.6, -3.4, 1.3)  # fixed camera: per-step pose updates do not reach the renderer, so the walk is a circle
LOOK = (0.0, 0.5, 0.6)
TURN_RATE = 0.5  # rad/s while walking at 0.5 m/s -> 1 m radius circle
# (seconds, vx, height command)
SCHEDULE = [(3, 0, 0.78), (3, 0, 0.68), (3, 0, 0.62), (3, 0, 0.72), (6, 0.5, 0.72), (4, 0.5, 0.62), (4, 0.5, 0.78)]


def main():
    cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=2)
    cfg.observations.policy.enable_corruption = False
    cfg.events.wrist_payload = None
    cfg.events.push_robot = None
    cfg.events.reset_base.params["pose_range"] = {"x": (0, 0), "y": (0, 0), "yaw": (0, 0)}
    cfg.actions.arms.demo_prob = 1.0
    cfg.scene.env_spacing = 20.0
    cam_rot = math_utils.quat_from_matrix(
        math_utils.create_rotation_matrix_from_view(torch.tensor([EYE]), torch.tensor([LOOK]), "Z")
    )[0]
    cfg.scene.video_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/VideoCam",
        offset=TiledCameraCfg.OffsetCfg(pos=EYE, rot=tuple(cam_rot.tolist()), convention="opengl"),
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 40.0)),
    )
    agent_cfg = handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point"), metadata.version("rsl-rl-lib"))
    env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(args_cli.checkpoint)
    ours = runner.get_inference_policy(device=args_cli.device)
    agile = torch.jit.load(str(AGILE_LOCOMOTION_POLICY), map_location=args_cli.device)

    u = env.unwrapped
    robot, cam, cmd = u.scene["robot"], u.scene["video_cam"], u.command_manager.get_term("base_command")
    schedule = [(vx, h) for sec, vx, h in SCHEDULE for _ in range(int(sec / u.step_dt))]
    if args_cli.max_steps:
        schedule = schedule[: args_cli.max_steps]

    Path(args_cli.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args_cli.out, fps=round(1 / (u.step_dt * args_cli.every)), macro_block_size=1)
    labels = ["Agile (original)", "Ours (height-tracking)"]
    falls = [0, 0]
    obs = env.get_observations()
    with torch.inference_mode():
        for t, (vx, h) in enumerate(schedule):
            cmd.is_standing_env[:] = False
            cmd.vel_command_b[:] = torch.tensor([vx, 0.0, TURN_RATE if vx else 0.0], device=u.device)
            cmd.height_command[:] = h
            cmd.time_left[:] = 1e9
            act = ours(obs).clone()
            act[0] = agile(obs["policy"][0:1])[0]
            obs, _, dones, extras = env.step(act)
            if t % 20 == 0:
                print(f"[STEP] {t}")
            ours.reset(dones)
            fell = dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool()
            for i in range(2):
                falls[i] += int(fell[i])
            root = robot.data.root_pos_w
            if t % args_cli.every:
                continue
            rgb = cam.data.output["rgb"][..., :3].cpu().numpy()
            frames = []
            for i in range(2):
                img = Image.fromarray(rgb[i])
                d = ImageDraw.Draw(img)
                z = float(root[i, 2])
                d.rectangle([0, 0, 640, 58], fill=(0, 0, 0))
                d.text((8, 4), labels[i], fill=(255, 255, 255))
                d.text((8, 22), f"height cmd {h:.2f} m   pelvis z {z:.2f} m   speed cmd {vx:.1f} m/s", fill=(255, 220, 0))
                d.text((8, 40), f"falls {falls[i]}", fill=(255, 90, 90) if falls[i] else (160, 255, 160))
                frames.append(np.asarray(img))
            writer.append_data(np.concatenate(frames, axis=1))
    writer.close()
    print(f"[INFO] wrote {args_cli.out}; falls agile {falls[0]}, ours {falls[1]}")


if __name__ == "__main__":
    try:
        main()
    finally:
        os._exit(0)
