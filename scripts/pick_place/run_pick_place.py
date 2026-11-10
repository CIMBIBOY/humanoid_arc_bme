"""Scripted pick-and-place at one table (IsaacLab native).

Pink IK drives the wrists and the right tri-hand grasps a box and moves it
across the table. --robot_base picks what carries the upper body (default
agile_legs: the G1 standing on the AGILE lower-body policy; fixed: fixed
pelvis). Loops forever (each episode resets the scene) unless --episodes.

    scripts/launch.sh scripts/pick_place/run_pick_place.py --headless --num_envs 4 --episodes 2
    scripts/launch.sh scripts/pick_place/run_pick_place.py --headless --robot_base fixed --episodes 1
"""

import argparse
import functools
import os

print = functools.partial(print, flush=True)  # noqa: A001 - Kit block-buffers piped stdout

import pinocchio  # noqa: E402,F401 - must be imported before Kit loads its own copy

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--robot_base", default="agile_legs", help="Base plug-in (humanoid_arc_bme/robots/bases.py).")
parser.add_argument("--episodes", type=int, default=0, help="Stop after this many episodes (0 = run forever).")
parser.add_argument(
    "--place_offset", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
    help="Place target relative to the object, pelvis frame at grasp time (default: ScriptedPickPlace's).",
)
parser.add_argument(
    "--skill", choices=("timer", "state"), default="timer",
    help="timer: open-loop scripted phases; state: closed-loop PickPlaceMachine (plan A4).",
)
parser.add_argument("--episode_s", type=float, default=None, help="Override the env's episode length [s].")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher({**vars(args_cli), "enable_pinocchio": True}).app

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from humanoid_arc_bme.tasks.pick_place.env_cfg import G1PickPlaceEnvCfg  # noqa: E402
from humanoid_arc_bme.skills.scripted_pick_place import PHASES, ScriptedPickPlace  # noqa: E402

PLACE_TOLERANCE = 0.03
STATE_PLACE_TOLERANCE = 0.015  # plan A4.2: placed within 15 mm
HOME_PHASE = [p[0] for p in PHASES].index("home")


def _fmt(t: torch.Tensor) -> str:
    return "(" + ", ".join(f"{v:+.3f}" for v in t.tolist()) + ")"


def main():
    cfg = G1PickPlaceEnvCfg(robot_base=args_cli.robot_base)
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    if args_cli.episode_s:
        cfg.episode_length_s = args_cli.episode_s
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()
    skill_cls = ScriptedPickPlace
    if args_cli.skill == "state":
        try:  # the closed-loop state-machine skill lives in the optional `arc` package, not in this repo
            from arc.sim.skills.state_runner import StatePickPlace as skill_cls  # noqa: PLC0415
        except ImportError as e:
            raise SystemExit(f"--skill state needs the optional `arc` package: {e}")
    policy = skill_cls(env, **({"place_offset": args_cli.place_offset} if args_cli.place_offset else {}))
    robot, obj, origins = env.scene["robot"], env.scene["object"], env.scene.env_origins
    right_wrist = robot.find_bodies("right_wrist_yaw_link")[0][0]
    print(f"[INFO] action dim {env.action_manager.total_action_dim}")

    placed = attempts = 0
    episodes, last_phase = 0, None
    reported = [False] * env.num_envs  # state skill: machine outcome already counted this episode
    with torch.inference_mode():
        while simulation_app.is_running():
            prev_phase = policy.phase.clone()
            _, _, terminated, truncated, _ = env.step(policy.compute())

            if args_cli.skill == "state":
                for i, m in enumerate(policy.machines):
                    if m.terminal and not reported[i]:
                        reported[i] = True
                        err = float((obj.data.root_pos_w[i] - origins[i] - policy.place_target[i]).norm())
                        ok = m.state.value == "DONE" and err < STATE_PLACE_TOLERANCE
                        attempts += 1
                        placed += int(ok)
                        print(f"[RESULT] env {i} {m.state.value} {m.reason or '-'} err {err * 1000:.1f} mm"
                              f" t {policy.clock[i]:.1f} s | placed {placed}/{attempts}")
                if all(reported):
                    episodes += 1
                    print(f"[INFO] episode {episodes} done (all machines terminal)")
                    if args_cli.episodes and episodes >= args_cli.episodes:
                        break
                    env.reset()
                    policy.reset(torch.arange(env.num_envs, device=env.device))
                    reported = [False] * env.num_envs
                    last_phase = None
                    continue

            finished = (policy.phase == HOME_PHASE) & (prev_phase != HOME_PHASE)
            if finished.any():
                err = (obj.data.root_pos_w[finished] - origins[finished] - policy.place_target[finished]).norm(dim=-1)
                attempts += int(finished.sum())
                placed += int((err < PLACE_TOLERANCE).sum())
                print(f"[RESULT] placed {placed}/{attempts} (this batch err: {_fmt(err)})")

            phase = policy.state_name(0) if args_cli.skill == "state" else policy.phase_name(0)
            if phase != last_phase:
                print(
                    f"[PHASE] ep {episodes} {phase:9s} | pelvis_z {robot.data.root_pos_w[0, 2]:.3f}"
                    f" | wrist {_fmt(robot.data.body_link_pos_w[0, right_wrist] - origins[0])}"
                    f" | object {_fmt(obj.data.root_pos_w[0] - origins[0])}"
                )
                last_phase = phase

            done = (terminated | truncated).nonzero().flatten()
            if len(done):
                if args_cli.skill == "state":  # timed out before its machine finished: count as a failure
                    for i in done.tolist():
                        if not reported[i]:
                            reported[i] = True
                            attempts += 1
                            print(f"[RESULT] env {i} EPISODE_END {policy.state_name(i)} | placed {placed}/{attempts}")
                policy.reset(done)
                if args_cli.skill == "state":
                    for i in done.tolist():
                        reported[i] = False
                last_phase = None
                if args_cli.skill != "state" and 0 in done.tolist():
                    episodes += 1
                    print(f"[INFO] episode {episodes} done ({'timeout' if truncated[0] else 'terminated'})")
                if args_cli.skill != "state" and args_cli.episodes and episodes >= args_cli.episodes:
                    break


if __name__ == "__main__":
    main()
    # simulation_app.close() can hang on this install; exit hard instead.
    os._exit(0)
