"""Evaluate a robomimic BC checkpoint on the pick-place demo env.

Like IsaacLab's robomimic/play.py (same observations, same `success` term),
but runs --num_envs rollouts at once and reports, per rollout, how far it
got: whether the object was lifted, where it ended up, and whether the hand
let go. play.py only prints success True/False.

--video adds a camera to every env and writes one MP4 per rollout to
--video_dir, named by outcome (env03_success.mp4, env05_fail.mp4).
--obs_noise / --action_noise perturb the policy every step (robustness).
--reference_dataset runs each policy rollout next to a replay of a Mimic
demo from the same initial state; with --video, one side-by-side clip each.

--backend picks how the policy runs:
  robomimic  (default) in this process, with IsaacLab's robomimic 0.4.0.
  server     in a separate python (--server_python, robomimic 0.5, see
             scripts/bc_chunked/README.md) speaking a small pipe protocol;
             used for robomimic 0.5 chunked (diffusion) policies.

    scripts/launch.sh scripts/bc/eval_bc.py --headless --num_envs 8 [--video] \
        --checkpoint logs/robomimic/HumanoidArc-G1-PickPlace-Demo-v0/<run>/models/model_epoch_500.pth
    scripts/launch.sh scripts/bc/eval_bc.py --headless --num_envs 24 --horizon 1000 --seed 303 \
        --backend server --server_python ~/envs/robomimic05/bin/python --checkpoint <0.5 checkpoint>
"""

import argparse
import functools
import os
import subprocess
from multiprocessing.connection import Listener
from pathlib import Path

print = functools.partial(print, flush=True)  # noqa: A001 - Kit block-buffers piped stdout

REPO_ROOT = Path(__file__).resolve().parents[2]

import pinocchio  # noqa: E402,F401 - must be imported before Kit loads its own copy

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="HumanoidArc-G1-PickPlace-Demo-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--horizon", type=int, default=900, help="Steps per rollout (generated demos are ~740).")
parser.add_argument("--seed", type=int, default=101)
parser.add_argument("--trace", default=None, help="Optional .npz path for per-step object/wrist positions.")
parser.add_argument("--video", action="store_true", help="Write one MP4 per rollout.")
parser.add_argument("--video_dir", default=str(REPO_ROOT / "logs" / "eval_bc_videos"))
parser.add_argument("--video_every", type=int, default=2, help="Record every Nth step (50Hz env: 2 -> 25fps).")
parser.add_argument("--obs_noise", type=float, default=0.0, help="Gaussian std added to every policy observation.")
parser.add_argument("--action_noise", type=float, default=0.0, help="Gaussian std [m] added to wrist position targets.")
parser.add_argument(
    "--reference_dataset",
    default=None,
    help="Pair each policy rollout with a replay of a demo from this dataset, from the same initial state"
    " (num_envs/2 pairs; with --video, one side-by-side clip per pair).",
)
parser.add_argument("--backend", choices=("robomimic", "server"), default="robomimic")
parser.add_argument(
    "--server_python",
    default=os.environ.get("ROBOMIMIC05_PYTHON"),
    help="Python with robomimic 0.5 for --backend server (default: $ROBOMIMIC05_PYTHON).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
simulation_app = AppLauncher({**vars(args_cli), "enable_pinocchio": True}).app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.utils.datasets import HDF5DatasetFileHandler  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import humanoid_arc_bme.tasks.pick_place  # noqa: E402,F401 - registers the gym ids
from humanoid_arc_bme.tasks.pick_place.env_cfg import OBJECT_SIZE, TABLE_SIZE  # noqa: E402

LIFTED = 0.03  # m above resting height
WRIST_POS_IDX = [0, 1, 2, 7, 8, 9]  # left and right wrist position targets in the action
POLICY_SERVER = REPO_ROOT / "scripts" / "bc_chunked" / "server.py"


class RobomimicPolicies:
    """One robomimic 0.4 policy per env, in this process (they keep per-env RNN state)."""

    def __init__(self, checkpoint: str, n: int, device):
        import robomimic.utils.file_utils as FileUtils

        self.device = device
        self.policies = [FileUtils.policy_from_checkpoint(ckpt_path=checkpoint, device=device)[0] for _ in range(n)]
        for p in self.policies:
            p.start_episode()
        self.obs_keys = list(self.policies[0].policy.global_config.observation.modalities.obs.low_dim)

    def act(self, obs: dict, actions: torch.Tensor, is_policy: torch.Tensor):
        """Fill the rows of `actions` where `is_policy`, with the requested noise."""
        for i, p in enumerate(self.policies):
            if not is_policy[i]:
                continue
            ob = {k: obs[k][i] for k in self.obs_keys}
            if args_cli.obs_noise > 0:
                ob = {k: v + args_cli.obs_noise * torch.randn_like(v) for k, v in ob.items()}
            actions[i] = torch.as_tensor(p(ob), device=self.device)
            if args_cli.action_noise > 0:
                actions[i, WRIST_POS_IDX] += args_cli.action_noise * torch.randn(len(WRIST_POS_IDX), device=self.device)

    def close(self):
        pass


class PolicyServer:
    """Policies in a separate python (scripts/bc_chunked/server.py), batched over envs.

    Isaac's python has robomimic 0.4.0; 0.5 checkpoints (chunked/diffusion
    policies) need 0.5, so they run in their own environment. The child gets a
    clean environment: Isaac's PYTHONPATH/LD_* settings would break it.
    """

    ISAAC_ENV_PREFIXES = ("PYTHON", "LD_", "ISAAC", "CARB", "OMNI", "EXP_PATH", "RESOURCE_NAME")

    def __init__(self, checkpoint: str, n: int, device, python: str, seed: int):
        if not python:
            raise SystemExit("--backend server needs --server_python (or $ROBOMIMIC05_PYTHON)")
        self.device = device
        listener = Listener(("127.0.0.1", 0), authkey=b"bc_chunked")
        cmd = [os.path.expanduser(python), str(POLICY_SERVER), "--checkpoint", checkpoint,
               "--port", str(listener.address[1]), "--seed", str(seed)]
        env = {k: v for k, v in os.environ.items() if not k.startswith(self.ISAAC_ENV_PREFIXES)}
        self.server = subprocess.Popen(cmd, env=env)
        self.conn = listener.accept()
        self.conn.send(("init", n))
        self.obs_keys = self.conn.recv()[1]
        self.conn.send(("reset",))
        self.conn.recv()

    def act(self, obs: dict, actions: torch.Tensor, is_policy: torch.Tensor):
        # The server steps every env (it keeps each env's observation history); rows that
        # are not policy-driven are overwritten by the caller.
        ob = {k: obs[k].cpu().numpy() for k in self.obs_keys}
        if args_cli.obs_noise > 0:
            ob = {k: v + args_cli.obs_noise * np.random.randn(*v.shape).astype(v.dtype) for k, v in ob.items()}
        self.conn.send(("act", ob))
        out = torch.as_tensor(self.conn.recv(), device=self.device, dtype=torch.float32)
        if args_cli.action_noise > 0:
            out[:, WRIST_POS_IDX] += args_cli.action_noise * torch.randn(len(out), len(WRIST_POS_IDX), device=self.device)
        actions[is_policy] = out[is_policy]

    def close(self):
        self.conn.send(("quit",))
        self.server.wait(timeout=30)


def label(frame: np.ndarray, text: str, success_step: int) -> np.ndarray:
    cv2.putText(frame, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    if success_step >= 0:
        cv2.putText(frame, f"placed (step {success_step})", (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 0), 2,
                    cv2.LINE_AA)
    return frame


def main():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.seed = args_cli.seed
    cfg.observations.policy.concatenate_terms = False
    success_term = cfg.terminations.success
    cfg.terminations = {}  # rollouts run the full horizon; outcomes are judged below
    cfg.recorders = {}
    if args_cli.video:
        # Same view as the env's viewer, at every env's origin. Set as the spawn offset:
        # set_world_poses_from_view() after creation left the camera at the env origin.
        eye, lookat = torch.tensor([cfg.viewer.eye]), torch.tensor([cfg.viewer.lookat])
        rot = math_utils.quat_from_matrix(math_utils.create_rotation_matrix_from_view(eye, lookat, "Z"))[0]
        cfg.scene.video_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/VideoCam",
            offset=TiledCameraCfg.OffsetCfg(pos=cfg.viewer.eye, rot=tuple(rot.tolist()), convention="opengl"),
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 20.0)),
        )
    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    n = env.num_envs
    device = env.device
    if args_cli.backend == "server":
        policy = PolicyServer(args_cli.checkpoint, n, device, args_cli.server_python, args_cli.seed)
    else:
        policy = RobomimicPolicies(args_cli.checkpoint, n, device)
    goal = torch.tensor(success_term.params["goal"], device=device)
    rest_z = TABLE_SIZE[2] + OBJECT_SIZE[2] / 2
    obj, origins = env.scene["object"], env.scene.env_origins

    obs, _ = env.reset()
    # Reference mode: env 2k replays demo k's recorded actions, env 2k+1 runs the policy,
    # both from demo k's recorded initial state.
    is_policy = torch.ones(n, dtype=torch.bool, device=device)
    ref_actions = None
    if args_cli.reference_dataset:
        data = HDF5DatasetFileHandler()
        data.open(args_cli.reference_dataset)
        demos = [data.load_episode(f"demo_{k}", device) for k in range(n // 2)]
        for k, ep in enumerate(demos):
            for env_id in (2 * k, 2 * k + 1):
                obs, _ = env.reset_to(ep.get_initial_state(), torch.tensor([env_id], device=device), is_relative=True)
        is_policy[0::2] = False
        ref_actions = [ep.data["actions"] for ep in demos]

    writers, labels = [], []
    if args_cli.video:
        cam = env.scene["video_cam"]
        video_dir = Path(args_cli.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        fps = round(1 / (env.step_dt * args_cli.video_every))
        if ref_actions is None:
            names = [f"env{i:02d}" for i in range(n)]
        else:
            names = [f"compare_demo{k:02d}" for k in range(n // 2)]
            labels = ["Mimic demo (IK replay)", "BC policy"]
        video_paths = [video_dir / f"{s}.mp4" for s in names]
        writers = [imageio.get_writer(p, fps=fps, macro_block_size=1) for p in video_paths]

    start = (obj.data.root_pos_w - origins).clone()
    max_lift = torch.zeros(n, device=device)
    first_success = torch.full((n,), -1, dtype=torch.long, device=device)
    trace = []
    actions = torch.zeros(n, env.action_manager.total_action_dim, device=device)
    for t in range(args_cli.horizon):
        policy.act(obs["policy"], actions, is_policy)
        for i in (~is_policy).nonzero().flatten().tolist():
            a = ref_actions[i // 2]
            actions[i] = a[min(t, len(a) - 1)]
        obs, *_ = env.step(actions)
        pos = obj.data.root_pos_w - origins
        max_lift = torch.maximum(max_lift, pos[:, 2] - rest_z)
        ok = success_term.func(env, **success_term.params)
        first_success = torch.where(ok & (first_success < 0), t, first_success)
        if args_cli.trace:
            trace.append(np.concatenate([pos.cpu().numpy(), obs["policy"]["right_eef_pos"].cpu().numpy()], axis=1))
        if writers and t % args_cli.video_every == 0:
            frames = cam.data.output["rgb"][..., :3].cpu().numpy()
            if ref_actions is None:
                for w, frame in zip(writers, frames):
                    w.append_data(frame)
            else:
                for k, w in enumerate(writers):
                    pair = [label(frames[i].copy(), labels[i % 2], first_success[i].item()) for i in (2 * k, 2 * k + 1)]
                    w.append_data(np.concatenate(pair, axis=1))

    end = obj.data.root_pos_w - origins
    wrist = obs["policy"]["right_eef_pos"]
    for i in range(n):
        err = (end[i, :2] - goal[:2]).norm().item()
        if first_success[i] >= 0:
            kind = f"SUCCESS at step {first_success[i].item()}"
        elif max_lift[i] < LIFTED:
            kind = "never lifted"
        elif end[i, 2] < rest_z - 0.05:
            kind = "dropped off table"
        elif err > success_term.params.get("xy_tolerance", 0.03):
            kind = "on table, off goal"
        else:
            kind = "at goal, hand not away / tilted / moving"
        print(
            f"[ROLLOUT] env {i}: start ({start[i, 0]:.3f},{start[i, 1]:.3f}) end ({end[i, 0]:.3f},{end[i, 1]:.3f},"
            f"{end[i, 2]:.3f}) goal err {err * 1000:.0f}mm max lift {max_lift[i] * 100:.1f}cm"
            f" hand-obj {(wrist[i] - end[i]).norm():.2f}m -> {kind}"
        )
    ok = first_success >= 0
    if ref_actions is None:
        print(f"[RESULT] {ok.sum().item()}/{n} successful")
    else:
        print(f"[RESULT] policy {ok[is_policy].sum().item()}/{n // 2}, demo replay {ok[~is_policy].sum().item()}/{n // 2}")
    for j, w in enumerate(writers):
        w.close()
        src = video_paths[j]
        if ref_actions is None:
            src.rename(src.with_name(f"{src.stem}_{'success' if ok[j] else 'fail'}.mp4"))
        else:
            src.rename(src.with_name(f"{src.stem}_bc_{'success' if ok[2 * j + 1] else 'fail'}.mp4"))
    if writers:
        print(f"[INFO] videos in {video_dir}")
    if args_cli.trace:
        np.savez(args_cli.trace, trace=np.stack(trace))
    policy.close()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
    # simulation_app.close() can hang on this install; exit hard instead.
    os._exit(0)
