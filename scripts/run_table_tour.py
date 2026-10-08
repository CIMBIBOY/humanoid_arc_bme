"""Continuous 4-table pick-and-place tour: the G1 walks a square circuit,
picks a differently-colored box off each table, sets it down nearby, and
walks on to the next -- forever.

    scripts/launch.sh scripts/run_table_tour.py --headless
    scripts/launch.sh scripts/run_table_tour.py --headless --laps 1 --video logs/tour.mp4 --speed 1.5
"""

import argparse
import functools
import os
from pathlib import Path

print = functools.partial(print, flush=True)  # noqa: A001 - Kit block-buffers piped stdout


import pinocchio  # noqa: E402,F401 - must be imported before Kit loads its own copy

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Continuous G1 table-tour pick-and-place (IsaacLab native).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--laps", type=int, default=0, help="Stop after this many completed laps (0 = run forever).")
parser.add_argument("--steps", type=int, default=0, help="Stop after this many env steps (0 = no limit).")
parser.add_argument("--speed", type=float, default=1.0, help="Pick-place motion speed (1.0 = original timing).")
parser.add_argument("--walk_speed", type=float, default=0.5, help="Top walking speed [m/s] between tables.")
parser.add_argument("--video", default=None, help="Write an MP4 of env 0 to this path.")
parser.add_argument("--video_every", type=int, default=2, help="Record every Nth step (50Hz env: 2 -> 25fps).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
simulation_app = AppLauncher({**vars(args_cli), "enable_pinocchio": True}).app

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402

import humanoid_arc_bme.policies.navigator as navigator  # noqa: E402
from humanoid_arc_bme.envs.g1_table_tour.env_cfg import STOPS, G1TableTourEnvCfg  # noqa: E402
from humanoid_arc_bme.policies.table_tour import TableTourController  # noqa: E402

# A "landed in about the right spot" check, not a precision one: set when the
# tour placed 6-12 cm off (the place target was past the arm's reach, since
# fixed: now 7-31 mm, see results.md). Could be tightened to ~0.05.
PLACE_TOLERANCE = 0.15


def _fmt(t: torch.Tensor) -> str:
    return "(" + ", ".join(f"{v:+.3f}" for v in t.tolist()) + ")"


def main():
    cfg = G1TableTourEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    navigator.MAX_LINEAR_SPEED = args_cli.walk_speed
    if args_cli.video:
        # The viewer's view at env 0's origin, set as the spawn offset (see eval_bc.py).
        eye, lookat = torch.tensor([cfg.viewer.eye]), torch.tensor([cfg.viewer.lookat])
        rot = math_utils.quat_from_matrix(math_utils.create_rotation_matrix_from_view(eye, lookat, "Z"))[0]
        cfg.scene.video_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/VideoCam",
            offset=TiledCameraCfg.OffsetCfg(pos=cfg.viewer.eye, rot=tuple(rot.tolist()), convention="opengl"),
            height=600,
            width=800,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 30.0)),
        )
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()
    tour = TableTourController(env, speed=args_cli.speed)
    writer = None
    if args_cli.video:
        import imageio

        Path(args_cli.video).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args_cli.video, fps=round(1 / (env.step_dt * args_cli.video_every)), macro_block_size=1)
    steps = 0
    print(f"[INFO] action dim {env.action_manager.total_action_dim}, "
          f"{len(STOPS)} stops: {', '.join(s.name for s in STOPS)}")

    placed = attempts = drops = 0
    last_status = None
    with torch.inference_mode():
        while simulation_app.is_running():
            navigating_before = tour.navigating.clone()
            stop_before = tour.stop_idx.clone()
            _, _, terminated, truncated, _ = env.step(tour.compute())

            steps += 1
            if writer and steps % args_cli.video_every == 0:
                writer.append_data(env.scene["video_cam"].data.output["rgb"][0, ..., :3].cpu().numpy())
            status = tour.status(0)
            if status != last_status:
                pelvis = env.scene["robot"].data.root_pos_w[0]
                print(f"[STATUS] lap {int(tour.laps[0])} {status:24s} | pelvis {_fmt(pelvis)}")
                last_status = status

            # A pick-place just finished for env e the step tour.navigating[e]
            # flips False->True (see TableTourController.compute); check the
            # object it was just working on landed near its place target.
            resumed = tour.navigating & ~navigating_before
            if resumed.any():
                for e in resumed.nonzero().flatten().tolist():
                    stop = tour.stops[int(stop_before[e])]
                    obj_pos = stop.object.data.root_pos_w[e] - env.scene.env_origins[e]
                    err = float((obj_pos - stop.place_target[e]).norm())
                    if os.environ.get("DEBUG_PLACEMENT"):
                        print(f"[DEBUG] home={_fmt(stop._object_home[e])} pick={_fmt(stop._pick[e])} "
                              f"place_target={_fmt(stop.place_target[e])} obj={_fmt(obj_pos)}")
                    attempts += 1
                    placed += err < PLACE_TOLERANCE
                    if e == 0:
                        print(f"[RESULT] {STOPS[int(stop_before[e])].name:6s} placement error {err:.3f} m"
                              f" | running {placed}/{attempts}")

            done = (terminated | truncated).nonzero().flatten()
            if len(done):
                drops += int(terminated[done].sum())
                tour.reset(done)
                last_status = None
                print(f"[WARN] env(s) {done.tolist()} reset (object dropped or episode timeout)")

            if args_cli.laps and int(tour.laps.min()) >= args_cli.laps:
                break
            if args_cli.steps and steps >= args_cli.steps:
                break

    if writer:
        writer.close()
        print(f"[INFO] wrote {args_cli.video}")
    print(f"[SUMMARY] placed {placed}/{attempts}, {drops} drop-triggered resets, "
          f"min laps completed {int(tour.laps.min())}")


if __name__ == "__main__":
    main()
    # simulation_app.close() can hang on this install; exit hard instead.
    os._exit(0)
