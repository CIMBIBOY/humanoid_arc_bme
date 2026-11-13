"""Body servo: drive the base to a goal in the map frame with the walking limits of the first-walk envelope.

Pure functions plus a small stateful slew limiter. Frames: map (x, y, yaw), body (x forward, y left); metres, radians,
seconds. Two rules shape every motion so it stays inside what the sensors cover:

* the base only travels FORWARD (the guard camera and lidar look forward; there is no rear sensor) and a lateral goal is
  reached by turning toward it first, so the travel direction always lies in the sensed corridor;
* the speed tapers as sqrt(2 a d) toward the goal, so the robot stops at the goal instead of overshooting it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from humanoid_arc_bme.onboard.validation import is_num


def wrap(a: float) -> float:
    """Angle in (-pi, pi]."""
    return math.remainder(a, 2.0 * math.pi) if a != math.pi else a


@dataclass(frozen=True)
class ServoLimits:
    vx_max: float = 0.2        # m/s   (config/g1_limits.yaml base_vel_max.vx)
    wz_max: float = 0.3        # rad/s (config/g1_limits.yaml base_vel_max.wz)
    decel: float = 0.15        # m/s^2 taper toward the goal
    k_pos: float = 1.0         # 1/s   proportional gain on the remaining distance
    k_yaw: float = 1.5         # 1/s   proportional gain on the heading error
    pos_tol: float = 0.04      # m     goal reached
    yaw_tol: float = 0.06      # rad   heading reached
    turn_first_rad: float = 0.35   # heading error above which translation is held at zero (turn in place first)
    slew_v: float = 0.3        # m/s^2 command acceleration limit
    slew_w: float = 0.8        # rad/s^2

    def __post_init__(self):
        for n in ("vx_max", "wz_max", "decel", "k_pos", "k_yaw", "pos_tol", "yaw_tol", "turn_first_rad", "slew_v", "slew_w"):
            v = getattr(self, n)
            if not is_num(v) or v <= 0:
                raise ValueError(f"{n}: must be a finite number > 0, got {v!r}")


def _taper(distance: float, lim: ServoLimits) -> float:
    return min(lim.vx_max, lim.k_pos * distance, math.sqrt(2.0 * lim.decel * max(distance, 0.0)))


def _sat(x: float, m: float) -> float:
    return max(-m, min(m, x))


def go_to_point(pose, goal_xy, lim: ServoLimits, hold_yaw: float | None = None) -> tuple[tuple[float, float, float], bool]:
    """(vx, vy, wz) and `done` toward the map point `goal_xy`. Turns toward the goal first when the bearing error exceeds
    `turn_first_rad`; after that walks forward with the bearing as a steering term. If `hold_yaw` is given the heading is
    held there instead of steering (straight-line moves). vy is always 0. Never raises on a finite pose."""
    x, y, yaw = pose
    dx, dy = goal_xy[0] - x, goal_xy[1] - y
    dist = math.hypot(dx, dy)
    if dist <= lim.pos_tol:
        return (0.0, 0.0, 0.0), True
    target_yaw = math.atan2(dy, dx) if hold_yaw is None else hold_yaw
    err = wrap(target_yaw - yaw)
    wz = _sat(lim.k_yaw * err, lim.wz_max)
    if abs(err) > lim.turn_first_rad:
        return (0.0, 0.0, wz), False
    along = dx * math.cos(yaw) + dy * math.sin(yaw)
    if along <= 0.0:                       # goal is behind or beside: never reverse; turn toward it
        return (0.0, 0.0, wz if abs(err) > 1e-3 else lim.wz_max), False
    # slow the forward speed while the heading is still being corrected
    k = max(0.0, 1.0 - abs(err) / lim.turn_first_rad)
    return (_taper(along, lim) * k, 0.0, wz), False


def turn_to(pose, goal_yaw: float, lim: ServoLimits) -> tuple[tuple[float, float, float], bool]:
    err = wrap(goal_yaw - pose[2])
    if abs(err) <= lim.yaw_tol:
        return (0.0, 0.0, 0.0), True
    wz = _sat(lim.k_yaw * err, lim.wz_max)
    if abs(wz) < 0.05:                    # deadband floor: the walking controller ignores tiny yaw commands
        wz = math.copysign(0.05, err)
    return (0.0, 0.0, wz), False


class Slew:
    """Acceleration limiter on (vx, vy, wz): a step change in the goal never becomes a step change in the command."""

    def __init__(self, lim: ServoLimits):
        self.lim = lim
        self.cmd = (0.0, 0.0, 0.0)

    def reset(self) -> None:
        self.cmd = (0.0, 0.0, 0.0)

    def __call__(self, want, dt: float) -> tuple[float, float, float]:
        if not (is_num(dt) and 0.0 <= dt <= 1.0):
            self.cmd = (0.0, 0.0, 0.0)
            return self.cmd
        dv, dw = self.lim.slew_v * dt, self.lim.slew_w * dt
        vx, vy, wz = self.cmd
        vx += _sat(want[0] - vx, dv)
        vy += _sat(want[1] - vy, dv)
        wz += _sat(want[2] - wz, dw)
        self.cmd = (vx, vy, wz)
        return self.cmd
