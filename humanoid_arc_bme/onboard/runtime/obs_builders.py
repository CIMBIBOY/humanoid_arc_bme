"""Policy observations rebuilt from the contract `RobotState` (+ last action + command), numpy only, no Isaac.

Each builder reproduces one exported policy's input exactly as Isaac Lab computes it (term order, per-term history
oldest first, reset fill, scales). Its numbers come from a `layout.json` written by tools/training/export_lowerbody_onnx.py next
to the ONNX file, so the builder never guesses joint order or default pose. Parity with the simulator is tested in
tests/runtime/test_obs_builders.py on recorded sim steps (|diff| < 1e-5).

Conventions (stated once, all checked by the parity test except where marked):
- `RobotState.imu.quat` is the pelvis orientation (w, x, y, z) in a gravity-aligned frame; projected gravity is
  R^T (0, 0, -1). On the real G1 the `rt/lowstate` IMU is the pelvis IMU: per Unitree docs, UNVERIFIED on our robot (B0).
- `RobotState.imu.gyro` is the pelvis angular velocity in the pelvis frame (Isaac `root_ang_vel_b`).
- joint arrays are looked up by NAME, so the robot's joint order does not matter.
- the leg action is decoded as target = default + scale * clamp(a, raw_lo, raw_hi) (the S1 position clamp, W4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROPRIO_TERMS = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions")


def quat_rotate_inverse(q_wxyz, v) -> np.ndarray:
    """R(q)^T v for a unit quaternion (w, x, y, z), the same formula as Isaac Lab's quat_apply_inverse."""
    w, x, y, z = (float(c) for c in q_wxyz)
    xyz = np.array([x, y, z])
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(xyz, v)
    return v - w * t + np.cross(xyz, t)


@dataclass(frozen=True)
class LowerBodyLayout:
    """What the builder needs from the sim; loaded from layout.json (see from_json)."""

    obs_joint_names: tuple  # joints in the observation, sim order (29)
    obs_default_pos: np.ndarray
    action_joint_names: tuple  # leg joints in the action, sim order (12)
    action_offset: np.ndarray  # default pose of the action joints (rad)
    action_scale: float
    raw_lo: np.ndarray | None  # raw-action clamp (None: no clamp)
    raw_hi: np.ndarray | None
    hist: int  # 1 = no history
    lin_vel: bool  # True: rough4 layout with base_lin_vel after the command (needs a velocity source)
    joint_vel_scale: float = 0.1
    obs_dim: int = 0

    @staticmethod
    def from_json(path) -> "LowerBodyLayout":
        d = json.loads(Path(path).read_text())
        lo, hi = d.get("raw_lo"), d.get("raw_hi")
        n = len(d["action_joint_names"])
        for label, v in (("raw_lo", lo), ("raw_hi", hi), ("action_offset", d["action_offset"])):
            if v is not None and np.shape(v) != (n,):  # a per-env array from an old exporter would silently broadcast
                raise ValueError(f"layout {label}: expected shape ({n},), got {np.shape(v)}")
        return LowerBodyLayout(
            obs_joint_names=tuple(d["obs_joint_names"]), obs_default_pos=np.asarray(d["obs_default_pos"], np.float64),
            action_joint_names=tuple(d["action_joint_names"]), action_offset=np.asarray(d["action_offset"], np.float64),
            action_scale=float(d["action_scale"]), raw_lo=None if lo is None else np.asarray(lo, np.float64),
            raw_hi=None if hi is None else np.asarray(hi, np.float64), hist=int(d["hist"]), lin_vel=bool(d["lin_vel"]),
            joint_vel_scale=float(d.get("joint_vel_scale", 0.1)), obs_dim=int(d["obs_dim"]))


class LowerBodyObsBuilder:
    """Stateful (history) builder for the lower-body actors: rough4 (83, lin vel, no history) and W4 deploy (384)."""

    def __init__(self, layout: LowerBodyLayout):
        self.lay = layout
        self._hist: dict[str, list] | None = None
        self._idx_cache: tuple | None = None
        n = 4 + (3 if layout.lin_vel else 0) + layout.hist * (3 + 3 + 2 * len(layout.obs_joint_names) + len(layout.action_joint_names))
        if layout.obs_dim and n != layout.obs_dim:
            raise ValueError(f"layout says obs_dim {layout.obs_dim}, terms give {n}")
        self.obs_dim = n

    def reset(self) -> None:
        """Next build() starts a new episode: its frame fills every history slot (Isaac Lab CircularBuffer)."""
        self._hist = None

    def _index(self, names: tuple) -> tuple:
        if self._idx_cache is None or self._idx_cache[0] != names:
            pos = {n: i for i, n in enumerate(names)}
            missing = [n for n in self.lay.obs_joint_names if n not in pos]
            if missing:
                raise ValueError(f"RobotState lacks joints {missing}")
            self._idx_cache = (names, np.array([pos[n] for n in self.lay.obs_joint_names]))
        return self._idx_cache[1]

    def frame(self, state, last_action) -> dict:
        idx = self._index(state.joint_names)
        q = np.asarray(state.joint_pos, np.float64)[idx]
        dq = np.asarray(state.joint_vel, np.float64)[idx]
        a = np.asarray(last_action, np.float64)
        if a.shape != (len(self.lay.action_joint_names),):
            raise ValueError(f"last_action: expected {len(self.lay.action_joint_names)} values")
        return {"base_ang_vel": np.asarray(state.imu.gyro, np.float64),
                "projected_gravity": quat_rotate_inverse(state.imu.quat, (0.0, 0.0, -1.0)),
                "joint_pos": q - self.lay.obs_default_pos,
                "joint_vel": dq * self.lay.joint_vel_scale,
                "actions": a}

    def build(self, state, last_action, command, base_lin_vel=None) -> np.ndarray:
        """One observation (float32, [obs_dim]); call once per control step, in order."""
        cmd = np.asarray(command, np.float64)
        if cmd.shape != (4,):
            raise ValueError("command: expected (vx, vy, wz, height)")
        f = self.frame(state, last_action)
        if self._hist is None:
            self._hist = {k: [f[k]] * self.lay.hist for k in PROPRIO_TERMS}
        else:
            for k in PROPRIO_TERMS:
                self._hist[k] = self._hist[k][1:] + [f[k]]
        parts = [cmd]
        if self.lay.lin_vel:
            if base_lin_vel is None:
                raise ValueError("this layout needs a base linear velocity (rough4): the G1 does not measure it")
            parts.append(np.asarray(base_lin_vel, np.float64))
        parts += [np.concatenate(self._hist[k]) for k in PROPRIO_TERMS]
        return np.concatenate(parts).astype(np.float32)

    def decode(self, raw_action) -> tuple[np.ndarray, np.ndarray]:
        """(clamped raw action = the next last_action, joint position targets in action_joint_names order)."""
        a = np.asarray(raw_action, np.float64).reshape(-1)
        if self.lay.raw_lo is not None:
            a = np.clip(a, self.lay.raw_lo, self.lay.raw_hi)
        return a, self.lay.action_offset + self.lay.action_scale * a
