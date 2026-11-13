"""Dex3 three-finger hand of the G1: the closure map of a power grasp and grasp evidence from the joint positions.

Closure r in [0, 1] is the fraction of the way from the open pose to the closed pose, joint by joint (a straight line in
joint space). The joint order is `onboard.g1_layout` (thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1).

ASSUMED, not validated on the real hand: the power-grasp preshape. Each finger flexes to its closing limit (the URDF
limit in the flexing direction: the left index and middle flex toward negative, the thumb toward positive; the right
hand is the mirror image). The closed pose is clipped to the safety range (`Limits.joint_range`), so the closed pose
sits one safety margin short of the URDF limit. The open pose is zero, clipped the same way (the index and middle
joints open to 0.05 rad short of their zero stop).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..api.safety import Limits
from ..g1_layout import LEFT_HAND_JOINTS, RIGHT_HAND_JOINTS
from .kinematics import SIDES, default_limits

# Closing targets: the URDF limits of the flexing direction (config/g1_limits.yaml, same order as g1_layout).
OPEN_POSE: dict[str, tuple[float, ...]] = {"left": (0.0,) * 7, "right": (0.0,) * 7}
CLOSED_POSE: dict[str, tuple[float, ...]] = {
    "left": (1.0471976, 1.0471976, 1.7453293, -1.5707963, -1.7453293, -1.5707963, -1.7453293),
    "right": (-1.0471976, -1.0471976, -1.7453293, 1.5707963, 1.7453293, 1.5707963, 1.7453293),
}
_STALL_COMMAND_MIN = 0.9  # a command at or above this closure is a request to close on an object


@dataclass(frozen=True)
class HandState:
    """Grasp evidence from one measurement. `closure` is the measured closure (NaN when the input is bad); `closed` and
    `open` are the thresholds of `grasp_state`; `stalled` means the fingers are blocked short of the command (an object is
    in the hand)."""

    closure: float
    closed: bool
    open: bool
    stalled: bool


def _as_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def _clip_closure(r) -> float:
    """Closure clipped to [0, 1]; NaN (or an unreadable value) is open."""
    x = _as_float(r)
    if math.isnan(x):
        return 0.0
    return min(max(x, 0.0), 1.0)


class Dex3Hand:
    """One Dex3 hand: closure -> joint targets, joint positions -> closure, and grasp evidence.

    `limits` supplies the joint ranges (default: config/g1_limits.yaml). Joint targets are always inside
    `limits.joint_range`, so the safety layer never clips them.
    """

    def __init__(self, side: str, limits: Limits | None = None):
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        self.side = side
        self.joint_names = LEFT_HAND_JOINTS if side == "left" else RIGHT_HAND_JOINTS
        limits = default_limits() if limits is None else limits
        lower = np.array([limits.joint_range(n)[0] for n in self.joint_names])
        upper = np.array([limits.joint_range(n)[1] for n in self.joint_names])
        self._lower, self._upper = lower, upper
        self.open_pose = np.clip(np.array(OPEN_POSE[side]), lower, upper)
        self.closed_pose = np.clip(np.array(CLOSED_POSE[side]), lower, upper)
        for pose in (self.open_pose, self.closed_pose):
            pose.flags.writeable = False
        self._delta = self.closed_pose - self.open_pose
        self._delta_sq = float(self._delta @ self._delta)

    def closure_to_joints(self, r: float) -> np.ndarray:
        """Seven joint targets (rad) for closure `r`: r clipped to [0, 1], NaN means open, inside the joint ranges."""
        return np.clip(self.open_pose + _clip_closure(r) * self._delta, self._lower, self._upper)

    def joints_to_closure(self, q) -> float:
        """Closure of seven measured joint positions: the least-squares projection onto the open-closed segment, clipped
        to [0, 1]. Returns NaN unless `q` is seven finite numbers."""
        try:
            arr = np.asarray(q, dtype=np.float64)
        except (TypeError, ValueError):
            return math.nan
        if arr.shape != (len(self.joint_names),) or not np.all(np.isfinite(arr)):
            return math.nan
        return min(max(float((arr - self.open_pose) @ self._delta) / self._delta_sq, 0.0), 1.0)

    def grasp_state(self, q, commanded_closure: float, *, closed_min: float = 0.95, open_max: float = 0.05,
                    stall_margin: float = 0.12) -> HandState:
        """Grasp evidence from measured joints `q` and the commanded closure. `stalled` is True when the command asks for
        a closed hand (>= 0.9) and the measured closure is more than `stall_margin` short of it. Bad input (NaN, wrong
        length) gives all flags False and a NaN closure."""
        measured = self.joints_to_closure(q)
        if math.isnan(measured):
            return HandState(math.nan, False, False, False)
        command = _as_float(commanded_closure)
        stalled = (not math.isnan(command)) and command >= _STALL_COMMAND_MIN and measured < command - stall_margin
        return HandState(measured, measured >= closed_min, measured <= open_max, stalled)
