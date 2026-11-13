"""SensorHub: one SensorFrame from the real sources (RGB-D camera, Livox, DLIO, joint state, Dex3 hand feed).

Pure assembly, no I/O of its own: every source is an object with the small method named below, so tests and the CPU
simulator plug in fakes. A frame is only produced when the essentials exist (a depth frame and a pose); a missing hand
feed leaves `joints` None, which makes the task layer refuse arm tasks with a reason instead of guessing.
"""

from __future__ import annotations

import math
import time
from typing import Protocol

from humanoid_arc_bme.onboard.g1_layout import LEFT_HAND_JOINTS, RIGHT_HAND_JOINTS
from humanoid_arc_bme.onboard.stack import SensorFrame


class HandFeed(Protocol):
    def latest(self) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
        """(left 7, right 7) Dex3 joint positions in g1_layout order, or None."""


def roll_pitch(quat) -> tuple[float, float]:
    """Roll and pitch (rad) of a unit quaternion (w, x, y, z): the attitude the gravity alignment needs."""
    w, x, y, z = quat
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    s = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return roll, math.asin(s)


class SensorHub:
    def __init__(self, camera, lidar=None, localizer=None, hands: HandFeed | None = None, clock=time.monotonic):
        """camera.latest() -> (rgb | None, DepthFrame | None); lidar.latest() -> LidarFrame | None;
        localizer.estimate(stamp) -> LocalizerEstimate | None; hands: HandFeed."""
        self.camera, self.lidar, self.localizer, self.hands, self.clock = camera, lidar, localizer, hands, clock
        self._prev: tuple[float, float, float] | None = None   # (stamp, x, y) of the last pose
        self._speed = 0.0

    def _joints(self, state) -> dict | None:
        if state is None:
            return None
        q = dict(zip(state.joint_names, state.joint_pos))
        feed = self.hands.latest() if self.hands is not None else None
        if feed is None:
            return None
        left, right = feed
        q.update(zip(LEFT_HAND_JOINTS, left))
        q.update(zip(RIGHT_HAND_JOINTS, right))
        return q

    def _measure_speed(self, est) -> float:
        """Planar speed from successive localizer poses (the G1 reports no base velocity), lightly low-passed."""
        if est is None or not est.healthy:
            return self._speed
        x, y = est.pose.position[0], est.pose.position[1]
        if self._prev is not None and est.stamp - self._prev[0] >= 0.05:
            v = math.hypot(x - self._prev[1], y - self._prev[2]) / (est.stamp - self._prev[0])
            self._speed = 0.5 * self._speed + 0.5 * v
            self._prev = (est.stamp, x, y)
        elif self._prev is None:
            self._prev = (est.stamp, x, y)
        return self._speed

    def frame(self, now: float, state, safe: bool = True) -> SensorFrame | None:
        """The newest consistent frame at `now`, or None when the camera or the state is missing."""
        rgb, depth = self.camera.latest()
        if depth is None or state is None:
            return None
        est = self.localizer.estimate(now) if self.localizer is not None else None
        roll, pitch = roll_pitch(state.imu.quat)
        lidar = self.lidar.latest() if self.lidar is not None else None
        return SensorFrame(now, depth, rgb, lidar, est, roll, pitch, self._joints(state), self._measure_speed(est), safe)
