"""Record scripted pick-place demos to an IsaacLab/Mimic HDF5 dataset.

Runs ScriptedPickPlace on HumanoidArc-G1-PickPlace-Demo-v0 (random object
start, fixed place goal) with IsaacLab's ActionStateRecorder, the same
recorder its teleop record_demos.py uses. Like that script, an episode ends
and is written once the env's `success` term has held for --success_steps
consecutive steps (ending on the first one leaves a replay exactly on the
threshold); timeouts and drops are discarded.

    scripts/launch.sh scripts/pick_place/record_demos.py --headless --num_envs 4 --num_demos 10
"""

import argparse
import functools
import os
import sys
from pathlib import Path

print = functools.partial(print, flush=True)  # noqa: A001 - Kit block-buffers piped stdout


import pinocchio  # noqa: E402,F401 - must be imported before Kit loads its own copy

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", default="HumanoidArc-G1-PickPlace-Demo-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_demos", type=int, default=10, help="Stop after this many successful demos.")
parser.add_argument("--max_attempts", type=int, default=0, help="Give up after this many episodes (0 = 3x num_demos).")
parser.add_argument("--dataset_file", required=True, help="New HDF5 file (an existing file is never overwritten).")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--success_steps", type=int, default=10, help="Consecutive success steps that end a demo.")
parser.add_argument("--speed", type=float, default=1.5, help="Scripted motion speed (1.0 = run_pick_place's timing).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher({**vars(args_cli), "enable_pinocchio": True}).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import humanoid_arc_bme.tasks.pick_place  # noqa: E402,F401 - registers the gym ids
from humanoid_arc_bme.tasks.pick_place.env_cfg import DEMO_OBJECT_POS, DEMO_PLACE_OFFSET  # noqa: E402
from humanoid_arc_bme.skills.scripted_pick_place import ScriptedPickPlace  # noqa: E402


def main():
    dataset = Path(args_cli.dataset_file).resolve()
    if dataset.exists():
        sys.exit(f"[ERROR] {dataset} exists; pick another --dataset_file or delete it.")
    max_attempts = args_cli.max_attempts or 3 * args_cli.num_demos

    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.seed = args_cli.seed
    cfg.env_name = args_cli.task  # stored in the dataset; replay/Mimic tools read it back
    cfg.recorders = ActionStateRecorderManagerCfg()
    cfg.recorders.dataset_export_dir_path = str(dataset.parent)
    cfg.recorders.dataset_filename = dataset.stem
    cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    success_term = cfg.terminations.success
    cfg.terminations.success = None  # checked here instead, so it can require a hold
    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    env.reset()
    policy = ScriptedPickPlace(env, place_offset=DEMO_PLACE_OFFSET, object_home=DEMO_OBJECT_POS, speed=args_cli.speed)
    obj, origins = env.scene["object"], env.scene.env_origins
    print(f"[INFO] action dim {env.action_manager.total_action_dim}, place goal {policy.place_target[0].tolist()}")

    attempts = 0
    held = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    with torch.inference_mode():
        while simulation_app.is_running():
            _, _, terminated, truncated, _ = env.step(policy.compute())
            ok = success_term.func(env, **success_term.params)
            held = torch.where(ok, held + 1, 0)
            succeeded = (held >= args_cli.success_steps).nonzero().flatten()
            if len(succeeded):
                # As scripts/pick_place/record_demos.py: record without exporting, mark success, export, then reset.
                err = (obj.data.root_pos_w[succeeded] - origins[succeeded] - policy.place_target[succeeded]).norm(dim=-1)
                env.recorder_manager.record_pre_reset(succeeded, force_export_or_skip=False)
                env.recorder_manager.set_success_to_episodes(
                    succeeded, torch.ones(len(succeeded), 1, dtype=torch.bool, device=env.device)
                )
                env.recorder_manager.export_episodes(succeeded)
                env.reset(env_ids=succeeded)
                for env_id, e in zip(succeeded.tolist(), err.tolist()):
                    print(f"[EPISODE] env {env_id}: success, object {e * 1000:.0f}mm from goal")
            # Timeouts/drops were auto-reset inside step() and discarded by the recorder.
            failed = (terminated | truncated).nonzero().flatten()
            for env_id in failed.tolist():
                print(f"[EPISODE] env {env_id}: {'timeout' if truncated[env_id] else 'dropped'}, discarded")
            done = torch.cat([succeeded, failed])
            if len(done) == 0:
                continue
            held[done] = 0
            policy.reset(done)
            attempts += len(done)
            demos = env.recorder_manager.exported_successful_episode_count
            print(f"[RESULT] {demos} demos written / {attempts} episodes")
            if demos >= args_cli.num_demos or attempts >= max_attempts:
                break
    env.recorder_manager.close()
    print(f"[INFO] wrote {dataset}")


if __name__ == "__main__":
    main()
    # simulation_app.close() can hang on this install; exit hard instead.
    os._exit(0)
