"""Odometry behind one interface, with adapters per source.

`Odometry.pose(now)` returns the current planar pose estimate in the `odom` frame (or None if it has none / it is
stale); `Odometry.on_command(now, vx, vy, wz)` tells it what was commanded (dead reckoning uses it). Sources:

- `TruthOdometry`: the simulator's pose (tests, the CPU harness).
- `DeadReckoning`: integrates a body velocity, by default the COMMANDED one (the G1 measures no base velocity,
  runtime-design.md rule 3), with an optional yaw source (IMU quaternion yaw) and a speed scale calibrated on the
  floor (B4.3 measures it). Stage 1 of navstack.md.
- `ExternalPoseOdometry`: poses from outside (FAST-LIO / Point-LIO via ROS 2 `nav_msgs/Odometry`, the vendor's
  odometry topic if the G1 publishes one (UNKNOWN until B0), AprilTag fixes), with staleness and short
  extrapolation by the last command.
- `FusedOdometry`: dead reckoning between external fixes; each fix resets (or blends toward) the external pose.
"""

from __future__ import annotations

import math

from humanoid_arc_bme.onboard.geometry import yaw_from_quat  # noqa: F401 - re-exported

from .path import Pose2D, wrap


class Odometry:
    def pose(self, now: float) -> Pose2D | None:
        raise NotImplementedError

    def on_command(self, now: float, vx: float, vy: float, wz: float) -> None:
        pass

    def reset(self, pose: Pose2D = Pose2D(0.0, 0.0, 0.0), now: float = 0.0) -> None:
        pass


class TruthOdometry(Odometry):
    def __init__(self, get_pose):
        self.get_pose = get_pose

    def pose(self, now):
        return self.get_pose()


class DeadReckoning(Odometry):
    """Integrates (vx, vy, wz) in the body frame. `velocity="command"` uses on_command(); `velocity="measured"` uses
    `update_velocity()` (a base-velocity estimator). A yaw source (`set_yaw`) overrides the integrated yaw: IMU yaw
    drifts slowly, integrated commanded yaw drifts fast (turn-rate tracking of a gait is poor)."""

    def __init__(self, velocity: str = "command", scale: float = 1.0, yaw_scale: float = 1.0, max_dt: float = 0.2):
        if velocity not in ("command", "measured"):
            raise ValueError("velocity: 'command' or 'measured'")
        self.velocity, self.scale, self.yaw_scale, self.max_dt = velocity, scale, yaw_scale, max_dt
        self.reset()

    def reset(self, pose=Pose2D(0.0, 0.0, 0.0), now=0.0):
        self.x, self.y, self.yaw = pose.x, pose.y, pose.yaw
        self.t = now
        self.v = (0.0, 0.0, 0.0)
        self._yaw_offset = None  # IMU yaw at reset, so the odom frame starts at `pose`
        self._yaw0 = pose.yaw

    def _advance(self, now):
        dt = now - self.t
        if dt <= 0:
            return
        dt = min(dt, self.max_dt)  # a long gap is not integrated as if the robot kept walking
        vx, vy, wz = self.v
        ym = self.yaw + 0.5 * wz * self.yaw_scale * dt
        c, s = math.cos(ym), math.sin(ym)
        self.x += self.scale * (c * vx - s * vy) * dt
        self.y += self.scale * (s * vx + c * vy) * dt
        self.yaw = wrap(self.yaw + wz * self.yaw_scale * dt)
        self.t = now

    def on_command(self, now, vx, vy, wz):
        if self.velocity == "command":
            self._advance(now)
            self.v = (vx, vy, wz)

    def update_velocity(self, now, vx, vy, wz):
        if self.velocity == "measured":
            self._advance(now)
            self.v = (vx, vy, wz)

    def set_yaw(self, imu_yaw: float):
        """Absolute yaw from the IMU; the first call defines the offset to the odom frame."""
        if self._yaw_offset is None:
            self._yaw_offset = wrap(imu_yaw - self._yaw0)
        self.yaw = wrap(imu_yaw - self._yaw_offset)

    def pose(self, now):
        self._advance(now)
        return Pose2D(self.x, self.y, self.yaw)


class ExternalPoseOdometry(Odometry):
    """Latest external pose; extrapolated with the last command for at most `extrapolate_s`; None once older than
    `stale_s` (the follower then stops: no pose, no walking)."""

    def __init__(self, stale_s: float = 0.5, extrapolate_s: float = 0.2):
        self.stale_s, self.extrapolate_s = stale_s, extrapolate_s
        self.last = None  # (t, Pose2D)
        self.v = (0.0, 0.0, 0.0)

    def reset(self, pose=Pose2D(0.0, 0.0, 0.0), now=0.0):
        self.last = None

    def set_pose(self, t: float, pose: Pose2D):
        self.last = (t, pose)

    def on_command(self, now, vx, vy, wz):
        self.v = (vx, vy, wz)

    def pose(self, now):
        if self.last is None:
            return None
        t, p = self.last
        age = now - t
        if age > self.stale_s:
            return None
        dt = max(0.0, min(age, self.extrapolate_s))
        vx, vy, wz = self.v
        c, s = math.cos(p.yaw), math.sin(p.yaw)
        return Pose2D(p.x + (c * vx - s * vy) * dt, p.y + (s * vx + c * vy) * dt, wrap(p.yaw + wz * dt))


class FusedOdometry(Odometry):
    """Dead reckoning, corrected by external fixes. `blend` = 1 snaps to each fix; < 1 moves part way (a crude
    complementary filter; a proper EKF (robot_localization) replaces this in stage 3)."""

    def __init__(self, dr: DeadReckoning, blend: float = 1.0):
        if not 0 < blend <= 1:
            raise ValueError("blend in (0, 1]")
        self.dr, self.blend = dr, blend

    def reset(self, pose=Pose2D(0.0, 0.0, 0.0), now=0.0):
        self.dr.reset(pose, now)

    def on_command(self, now, vx, vy, wz):
        self.dr.on_command(now, vx, vy, wz)

    def fix(self, now: float, pose: Pose2D):
        cur = self.dr.pose(now)
        b = self.blend
        self.dr.x = cur.x + b * (pose.x - cur.x)
        self.dr.y = cur.y + b * (pose.y - cur.y)
        self.dr.yaw = wrap(cur.yaw + b * wrap(pose.yaw - cur.yaw))
        if self.dr._yaw_offset is not None:  # keep an IMU yaw source consistent with the corrected yaw
            self.dr._yaw_offset = wrap(self.dr._yaw_offset - b * wrap(pose.yaw - cur.yaw))

    def pose(self, now):
        return self.dr.pose(now)
