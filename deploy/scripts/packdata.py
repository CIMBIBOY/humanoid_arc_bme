"""Inputs for the robot scripts: the policy directory/pack and realistic robot inputs replayed from closedloop.npz.

The replay builds `rt/lowstate`-like objects (motor_state[i].q/.dq in the G1 motor order of humanoid_arc_bme.onboard.runtime.io.fake.G1_JOINTS,
imu_state.quaternion/gyroscope/accelerometer) from the sim recording, so the timed chain starts at the same pure
converter the robot uses (humanoid_arc_bme.onboard.runtime.io.unitree.state_from_lowstate) and runs the real builder and S1 code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from humanoid_arc_bme.onboard.runtime.io.fake import G1_JOINTS  # noqa: E402
from humanoid_arc_bme.onboard.runtime.obs_builders import LowerBodyLayout, LowerBodyObsBuilder  # noqa: E402

DEFAULT_LIMITS = ROOT / "config" / "g1_limits.yaml"


def load_policy_dir(d):
    d = Path(d)
    raw = json.loads((d / "layout.json").read_text())
    lay = LowerBodyLayout.from_json(d / "layout.json")
    LowerBodyObsBuilder(lay)  # raises if the term sizes do not give obs_dim
    return raw, lay, d / "policy.onnx"


def _lowstate(q29, dq29, quat, gyro):
    motors = [SimpleNamespace(q=float(a), dq=float(b)) for a, b in zip(q29, dq29)]
    imu = SimpleNamespace(quaternion=[float(x) for x in quat], gyroscope=[float(x) for x in gyro],
                          accelerometer=[0.0, 0.0, 9.81])
    return SimpleNamespace(motor_state=motors, imu_state=imu, mode_machine=0)


def replay_inputs(npz_path, all_joint_names, envs=4):
    """List of dicts per (env, step): lowstate, last_action, command, done, obs (recorded), action (recorded torch)."""
    with np.load(npz_path) as f:
        z = {k: f[k] for k in f.files}  # NpzFile re-reads the zip member on every [key] access
    idx = {n: i for i, n in enumerate(all_joint_names)}
    sel = np.array([idx[n] for n in G1_JOINTS])
    out = []
    T, E = z["obs"].shape[:2]
    for e in range(min(envs, E)):
        for t in range(T):
            q = z["root_quat_w"][t, e].astype(np.float64)
            out.append({"lowstate": _lowstate(z["joint_pos"][t, e, sel], z["joint_vel"][t, e, sel], q / np.linalg.norm(q),
                                              z["root_ang_vel_b"][t, e]),
                        "last_action": z["last_action"][t, e].astype(np.float64),
                        "command": tuple(float(c) for c in z["command"][t, e]),
                        "done": bool(z["dones"][t, e]) or t == 0,
                        "obs": z["obs"][t, e], "action": z["actions"][t, e]})
    return out


def synthetic_inputs(lay: LowerBodyLayout, n=1000, seed=0):
    """Stand-in when no recording is available: default pose + noise, level IMU, zero command."""
    rng = np.random.default_rng(seed)
    dflt = dict(zip(lay.obs_joint_names, lay.obs_default_pos))
    q0 = np.array([dflt.get(j, 0.0) for j in G1_JOINTS])
    out = []
    for t in range(n):
        out.append({"lowstate": _lowstate(q0 + 0.05 * rng.standard_normal(29), 0.5 * rng.standard_normal(29),
                                          (1.0, 0.0, 0.0, 0.0), 0.1 * rng.standard_normal(3)),
                    "last_action": np.zeros(len(lay.action_joint_names)), "command": (0.5, 0.0, 0.0, 0.75),
                    "done": t == 0, "obs": None, "action": None})
    return out
