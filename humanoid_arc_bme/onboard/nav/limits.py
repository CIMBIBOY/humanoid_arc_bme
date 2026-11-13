"""Walking limits, the curvature-limited speed profile and the command shaper (ramps), shared by every controller.

These are the legged-robot adaptations that sit around any path-tracking law:
- separate forward / backward speed limits (a biped walks backward slower and less surely);
- a low yaw-rate limit coupled to the linear speed (v * |kappa| <= wz_max: slow down to keep the curve instead of
  cutting it);
- a lateral-acceleration limit (v^2 * |kappa| <= a_lat);
- linear and angular acceleration limits (speed ramps: the gait needs steps to change speed);
- a deadband floor: a learned or vendor gait may stand still for tiny commands (the AGILE policy did, see
  humanoid_arc_bme/policies/navigator.py), so commands between 0 and `v_min` are lifted to `v_min` while moving;
- stop-and-balance: the follower ramps to zero and holds zero for `settle_s` before it reports success.

Defaults are the S1 base limits of config/g1_limits.yaml (vx 0.2 m/s, wz 0.3 rad/s, NOT validated on the robot) and
`(proposed)` values for the rest. Every number here is a guess until B4.3 measures the robot's response.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NavLimits:
    v_fwd_max: float = 0.2  # m/s   (config/g1_limits.yaml base_vel_max.vx)
    v_back_max: float = 0.1  # m/s  (proposed: half of forward until measured)
    wz_max: float = 0.3  # rad/s    (config/g1_limits.yaml base_vel_max.wz)
    a_max: float = 0.2  # m/s^2 linear acceleration (proposed)
    d_max: float = 0.25  # m/s^2 linear deceleration (proposed)
    alpha_max: float = 0.6  # rad/s^2 yaw acceleration (proposed)
    a_lat_max: float = 0.15  # m/s^2 lateral (centripetal) acceleration (proposed)
    v_min: float = 0.0  # m/s deadband floor while tracking (0 = off; the vendor's deadband is UNKNOWN until B4)
    wz_min: float = 0.0  # rad/s floor for rotate-in-place (0 = off)
    goal_pos_tol: float = 0.10  # m
    goal_yaw_tol: float = 0.15  # rad
    settle_s: float = 0.6  # s of zero command before success (stop-and-balance)

    def __post_init__(self):
        for k in ("v_fwd_max", "v_back_max", "wz_max", "a_max", "d_max", "alpha_max", "a_lat_max", "goal_pos_tol",
                  "goal_yaw_tol"):
            v = getattr(self, k)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise ValueError(f"{k} must be a positive finite number, got {v!r}")
        for k in ("v_min", "wz_min", "settle_s"):
            v = getattr(self, k)
            if not (math.isfinite(v) and v >= 0):
                raise ValueError(f"{k} must be >= 0")
        if self.v_min >= min(self.v_fwd_max, self.v_back_max) or self.wz_min >= self.wz_max:
            raise ValueError("deadband floors must be below the maxima")

    def vmax(self, direction: int) -> float:
        return self.v_fwd_max if direction > 0 else self.v_back_max

    @classmethod
    def from_safety(cls, limits, **kw) -> "NavLimits":
        """Nav limits no looser than S1's base limits (humanoid_arc_bme.onboard.api.safety.Limits)."""
        vx, _, wz = limits.base_vel_max
        kw.setdefault("v_fwd_max", vx)
        kw.setdefault("v_back_max", min(vx, kw.get("v_back_max", vx / 2)))
        kw.setdefault("wz_max", wz)
        return cls(**kw)


def speed_profile(path, limits: NavLimits, v_start: float | None = None, v_end: float = 0.0, kappa_smooth: int = 3) -> list:
    """Time-optimal speed |v|(s) under the curvature, lateral-acceleration and accel/decel limits (forward-backward
    pass). `v_start=None` leaves the start free (the CommandShaper ramps up from standstill anyway; a zero start
    would make a profile-following law never start). Curvature is max-filtered over +-`kappa_smooth` samples so a sampled corner still slows the robot."""
    n = len(path)
    vmax = limits.vmax(path.direction)
    k = [abs(v) for v in path.kappa]
    if kappa_smooth > 0:
        k = [max(k[max(0, i - kappa_smooth):i + kappa_smooth + 1]) for i in range(n)]
    v = []
    for ki in k:
        cap = vmax
        if ki > 1e-9:
            cap = min(cap, limits.wz_max / ki, math.sqrt(limits.a_lat_max / ki))
        v.append(cap)
    v[0] = min(v[0], abs(v_start)) if v_start is not None else v[0]
    for i in range(1, n):  # acceleration pass
        ds = path.s[i] - path.s[i - 1]
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * limits.a_max * ds))
    v[-1] = min(v[-1], v_end)
    for i in range(n - 2, -1, -1):  # deceleration pass
        ds = path.s[i + 1] - path.s[i]
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * limits.d_max * ds))
    return v


def profile_time(path, v) -> list:
    """Arrival time at each sample for a profile (trapezoid in s; a zero-speed start uses the next sample's speed)."""
    t = [0.0]
    for i in range(1, len(path)):
        ds = path.s[i] - path.s[i - 1]
        vm = 0.5 * (v[i] + v[i - 1])
        t.append(t[-1] + (ds / vm if vm > 1e-6 else 0.0 if ds == 0 else ds / max(1e-3, v[i])))
    return t


def stop_speed(dist: float, limits: NavLimits, latency_s: float = 0.0) -> float:
    """Largest speed v from which the robot still stops within `dist`: it keeps going for `latency_s` (command delay +
    gait lag) and then decelerates at d_max:  v * T + v^2 / (2 a) <= d  ->  v = -aT + sqrt((aT)^2 + 2 a d)."""
    a, d = limits.d_max, max(0.0, dist)
    aT = a * max(0.0, latency_s)
    return -aT + math.sqrt(aT * aT + 2 * a * d)


class CommandShaper:
    """Turns a raw (vx, wz) into one the gait can follow: curvature-preserving saturation, deadband floor, and
    acceleration ramps. Stateful (remembers the last output)."""

    def __init__(self, limits: NavLimits):
        self.limits = limits
        self.vx = 0.0
        self.wz = 0.0

    def reset(self, vx: float = 0.0, wz: float = 0.0):
        self.vx, self.wz = vx, wz

    def saturate(self, vx: float, wz: float) -> tuple[float, float]:
        """Clamp to the walking limits, scaling vx and wz together so the commanded curvature is kept."""
        L = self.limits
        vcap = L.v_fwd_max if vx >= 0 else L.v_back_max
        s = 1.0
        if abs(vx) > vcap:
            s = min(s, vcap / abs(vx))
        if abs(wz) > L.wz_max:
            s = min(s, L.wz_max / abs(wz))
        return vx * s, wz * s

    def __call__(self, vx: float, wz: float, dt: float, floor: bool = True) -> tuple[float, float]:
        L = self.limits
        vx, wz = self.saturate(vx, wz)
        if floor and L.v_min > 0 and 0 < abs(vx) < L.v_min:
            k = L.v_min / abs(vx)
            vx, wz = vx * k, wz * k
            vx, wz = self.saturate(vx, wz)
        if floor and L.wz_min > 0 and abs(vx) < 1e-6 and 0 < abs(wz) < L.wz_min:
            wz = math.copysign(L.wz_min, wz)
        # ramps: speeding up uses a_max, slowing down (toward 0 or reversing) uses d_max
        dv = vx - self.vx
        lim = L.a_max if (abs(vx) > abs(self.vx) and vx * self.vx >= 0) else L.d_max
        vx = self.vx + max(-lim * dt, min(lim * dt, dv))
        dw = wz - self.wz
        wz = self.wz + max(-L.alpha_max * dt, min(L.alpha_max * dt, dw))
        self.vx, self.wz = vx, wz
        return vx, wz
