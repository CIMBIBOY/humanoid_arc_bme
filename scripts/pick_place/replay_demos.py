"""Replay a recorded demo dataset headless and check it reproduces.

Each episode is reset to its recorded initial state, then driven by its
recorded actions only (no policy). Reports, per episode: whether the env's
`success` term holds after the last action, and how far the object ends up
from where it ended in the recording. IsaacLab's own tools/replay_demos.py
does the same but needs a window (its pause/resume keyboard).

    scripts/launch.sh scripts/pick_place/replay_demos.py --headless --dataset_file datasets/g1_pick_place_scripted.hdf5
"""

import argparse
import functools
import os

print = functools.partial(print, flush=True)  # noqa: A001 - Kit block-buffers piped stdout

import pinocchio  # noqa: E402,F401 - must be imported before Kit loads its own copy

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--dataset_file", required=True)
parser.add_argument("--task", default=None, help="Gym id (default: the one stored in the dataset).")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher({**vars(args_cli), "enable_pinocchio": True}).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.utils.datasets import HDF5DatasetFileHandler  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import humanoid_arc_bme.tasks.pick_place  # noqa: E402,F401 - registers the gym ids


def main():
    data = HDF5DatasetFileHandler()
    data.open(args_cli.dataset_file)
    task = args_cli.task or data.get_env_name()
    names = sorted(data.get_episode_names(), key=lambda s: int(s.split("_")[-1]))
    n = min(args_cli.num_envs, len(names))

    cfg = parse_env_cfg(task, device=args_cli.device, num_envs=n)
    success_term = cfg.terminations.success
    cfg.terminations = {}  # no auto-resets mid-replay
    cfg.recorders = {}
    env = gym.make(task, cfg=cfg).unwrapped
    env.reset()
    obj, origins = env.scene["object"], env.scene.env_origins
    print(f"[INFO] replaying {len(names)} episodes of {task} on {n} envs")

    ok_count, drifts = 0, []
    with torch.inference_mode():
        for start in range(0, len(names), n):
            batch = [data.load_episode(name, env.device) for name in names[start : start + n]]
            env_ids = torch.arange(len(batch), device=env.device)
            for i, ep in enumerate(batch):
                env.reset_to(ep.get_initial_state(), env_ids[i : i + 1], is_relative=True)
            actions = [ep.data["actions"] for ep in batch]
            lengths = [len(a) for a in actions]
            finished = {}
            step_actions = torch.zeros(n, env.action_manager.total_action_dim, device=env.device)
            for t in range(max(lengths)):
                for i, a in enumerate(actions):
                    step_actions[i] = a[min(t, len(a) - 1)]  # hold the last action once an episode ends
                env.step(step_actions)
                success = success_term.func(env, **success_term.params)
                for i, length in enumerate(lengths):
                    if t == length - 1:
                        finished[i] = (bool(success[i]), (obj.data.root_pos_w[i] - origins[i]).clone())
            for i, ep in enumerate(batch):
                ok, final = finished[i]
                recorded = ep.data["obs"]["object_pos"][-1]
                drift = (final - recorded).norm().item()
                ok_count += ok
                drifts.append(drift)
                print(
                    f"[EPISODE] {names[start + i]}: {'success' if ok else 'FAIL'},"
                    f" final object {drift * 1000:.1f}mm from the recording"
                )
    print(
        f"[RESULT] {ok_count}/{len(names)} replayed successfully;"
        f" final-object drift vs recording max {max(drifts) * 1000:.1f}mm, mean {sum(drifts) / len(drifts) * 1000:.1f}mm"
    )


if __name__ == "__main__":
    main()
    # simulation_app.close() can hang on this install; exit hard instead.
    os._exit(0)
