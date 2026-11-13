"""The path follower: state machine + shared walking rules around any controller.

    IDLE -> ALIGN (turn in place to the segment's body heading) -> TRACK (controller) -> SETTLE (zero command held
    for settle_s: stop-and-balance) -> next segment ... -> FINAL_ALIGN (goal yaw, if asked) -> SUCCEEDED
    any state -> FAILED(reason): no_pose (odometry missing/stale), off_path, blocked (obstacle cap 0 too long), timeout
    stop() -> STOPPED (zero command, ramped)

Per tick (`step(now)`): pose from odometry -> projection on the active segment (windowed, progress is monotone so a
closed loop never jumps to its start) -> controller (raw vx, wz) -> stop-at-goal cap sqrt(2 d_max d_remaining) ->
obstacle cap -> CommandShaper (saturation keeping curvature, deadband floor, accel ramps) -> BaseVelocity.

The output is a contract `BaseVelocity` (vy = 0); `to_whole_body` wraps it into a `WholeBodyCommand` with
`BaseVelocityHeight`. S1 (api/safety.py) still clamps it downstream: the follower's limits are tighter or equal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from humanoid_arc_bme.onboard.api.contract import BaseVelocity, BaseVelocityHeight, SkillStatus, WholeBodyCommand

from .controllers import Controller, TrackContext, make_controller
from .limits import CommandShaper, NavLimits, profile_time, speed_profile, stop_speed
from .obstacle import NoGuard
from .odometry import Odometry
from .path import Path, PathPlan, split_at_corners, wrap

STATES = ("IDLE", "ALIGN", "TRACK", "SETTLE", "FINAL_ALIGN", "SUCCEEDED", "FAILED", "STOPPED")


@dataclass
class FollowerConfig:
    align_enter: float = 0.6  # rad: heading error at a segment start that triggers turn-in-place
    align_exit: float = 0.15  # rad
    k_align: float = 1.5  # 1/s: rotate-in-place gain (saturated at wz_max)
    max_cte: float = 0.8  # m: further off the path than this -> FAILED(off_path)
    blocked_timeout_s: float = 10.0  # s at obstacle cap 0 -> FAILED(blocked)
    timeout_s: float | None = None  # None -> 3 x the profile time + 20 s
    window_m: float = 1.0  # m of path searched ahead of the progress point
    still_v: float = 0.01  # m/s: |command| below this counts as stopped
    still_w: float = 0.02  # rad/s
    stop_latency_s: float = 0.3  # s: command delay + gait lag assumed by the stop-at-goal cap (proposed; B4.3 measures)
    split_corners_rad: float = 0.8  # split polyline corners sharper than this into stop-turn-go segments (0 = off)
    stall_s: float = 1.5  # s without 1 cm of motion inside goal_pos_tol -> arrived
    v_creep: float = 0.03  # m/s: the stop-at-goal cap never goes below this, so the end is actually reached
    arrive_frac: float = 0.25  # SETTLE once within arrive_frac * goal_pos_tol of the end ...
    arrive_lead: float = 0.01  # ... or once the along-track position is within arrive_lead m of (or past) the end


class PathFollower:
    def __init__(self, odometry: Odometry, controller: Controller | str = "regulated_pure_pursuit",
                 limits: NavLimits = NavLimits(), guard=None, config: FollowerConfig = FollowerConfig()):
        self.odom = odometry
        self.controller = make_controller(controller) if isinstance(controller, str) else controller
        self.limits, self.cfg = limits, config
        self.guard = guard or NoGuard()
        self.shaper = CommandShaper(limits)
        self.plan: PathPlan | None = None
        self.state, self.reason = "IDLE", ""
        self.seg_i = 0
        self._progress_seg = 0
        self._settle_t = None
        self._blocked_t = None
        self._start_t = None
        self._last_t = None
        self._aligning = None
        self._stall = None
        self._timeout = math.inf
        self.last = {}  # diagnostics of the last tick (JSON friendly)

    # ------------------------------------------------------------ commands
    def follow(self, plan: PathPlan | Path, now: float) -> None:
        if isinstance(plan, Path):
            plan = PathPlan([plan])
        if self.cfg.split_corners_rad:
            plan = PathPlan([q for p in plan.segments for q in split_at_corners(p, self.cfg.split_corners_rad)])
        self.plan = plan
        self._stall = None
        self.seg_i = 0
        self._profiles = [speed_profile(p, self.limits) for p in plan.segments]
        self._enter_segment(now)
        self._start_t = now
        self.reason = ""
        self.controller.reset()
        if self.cfg.timeout_s is None:
            t = sum(profile_time(p, v)[-1] for p, v in zip(plan.segments, self._profiles))
            self._timeout = 3.0 * t + 20.0
        else:
            self._timeout = self.cfg.timeout_s

    def stop(self) -> None:
        if self.state not in ("SUCCEEDED", "FAILED"):
            self.state, self.reason = "STOPPED", "operator_stop"

    @property
    def segment(self) -> Path | None:
        return None if self.plan is None else self.plan.segments[self.seg_i]

    def status(self) -> SkillStatus:
        if self.state == "SUCCEEDED":
            return SkillStatus("succeeded")
        if self.state == "FAILED":
            return SkillStatus("failed", self.reason or "unknown")
        if self.state == "STOPPED":
            return SkillStatus("failed", "stopped")
        return SkillStatus("running")

    @property
    def done(self) -> bool:
        return self.state in ("SUCCEEDED", "FAILED", "STOPPED") and self.is_still

    @property
    def is_still(self) -> bool:
        return abs(self.shaper.vx) < self.cfg.still_v and abs(self.shaper.wz) < self.cfg.still_w

    # ------------------------------------------------------------ internals
    def _enter_segment(self, now):
        self._progress_seg = 0
        self._settle_t = None
        self._aligning = None
        self.state = "ALIGN"

    def _fail(self, reason):
        self.state, self.reason = "FAILED", reason

    def _ramp_to_zero(self, dt):
        return self.shaper(0.0, 0.0, dt, floor=False)

    def _emit(self, now, vx, wz, **diag):
        self.odom.on_command(now, vx, 0.0, wz)
        self.last = {"t": now, "state": self.state, "reason": self.reason, "seg": self.seg_i, "vx": vx, "wz": wz,
                     **diag}
        return BaseVelocity(vx, 0.0, wz)

    def step(self, now: float) -> BaseVelocity:
        dt = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        self._last_t = now
        if self.plan is None or self.state in ("IDLE", "SUCCEEDED", "FAILED", "STOPPED"):
            vx, wz = self._ramp_to_zero(dt)
            return self._emit(now, vx, wz)
        pose = self.odom.pose(now)
        if pose is None:
            self._fail("no_pose")
            vx, wz = self._ramp_to_zero(dt)
            return self._emit(now, vx, wz)
        if now - self._start_t > self._timeout:
            self._fail("timeout")
            vx, wz = self._ramp_to_zero(dt)
            return self._emit(now, vx, wz)
        path = self.segment
        prof = self._profiles[self.seg_i]
        L = self.limits
        ds = path.length / max(1, len(path) - 1)
        win = int(self.cfg.window_m / max(ds, 1e-6)) + 2
        ey, eyaw, s, seg = path.errors(pose, max(0, self._progress_seg - 2), win)
        self._progress_seg = max(self._progress_seg, seg)
        remaining = max(0.0, path.length - s)
        ex, ey_end = path.end
        d_end = math.hypot(pose.x - ex, pose.y - ey_end)
        diag = {"x": pose.x, "y": pose.y, "yaw": pose.yaw, "s": s, "length": path.length, "ey": ey, "eyaw": eyaw,
                "d_end": d_end}
        last_seg = self.seg_i == len(self.plan.segments) - 1
        if abs(ey) > self.cfg.max_cte and self.state in ("TRACK", "ALIGN"):
            self._fail("off_path")
            vx, wz = self._ramp_to_zero(dt)
            return self._emit(now, vx, wz, **diag)

        if self.state == "ALIGN":
            if self._aligning is None:  # decided once, at the segment start
                self._aligning = abs(eyaw) >= self.cfg.align_enter
            if self._aligning and abs(eyaw) > self.cfg.align_exit:
                w = max(-L.wz_max, min(L.wz_max, -self.cfg.k_align * eyaw))
                vx, wz = self.shaper(0.0, w, dt)
                return self._emit(now, vx, wz, **diag)
            self._aligning = None
            self.state = "TRACK"

        if self.state == "TRACK":
            # along-track position relative to the end, in the travel direction (> 0: past the goal)
            hx, hy = math.cos(path.heading[-1]), math.sin(path.heading[-1])
            past = (pose.x - ex) * hx + (pose.y - ey_end) * hy
            near_end = remaining < 2 * L.goal_pos_tol + 0.05  # in arc length: a closed loop starts near its end
            # stalled inside the tolerance (a gait deadband can eat the final creep command): accept the arrival
            if d_end < L.goal_pos_tol:
                if self._stall is None or math.hypot(pose.x - self._stall[1], pose.y - self._stall[2]) > 0.01:
                    self._stall = (now, pose.x, pose.y)
                stalled = now - self._stall[0] > self.cfg.stall_s
            else:
                self._stall, stalled = None, False
            if near_end and (d_end < self.cfg.arrive_frac * L.goal_pos_tol or past >= -self.cfg.arrive_lead or stalled):
                self.state = "SETTLE"
                self._settle_t = None
            else:
                ctx = TrackContext(pose, path, s, ey, eyaw, self.shaper.vx, self.shaper.wz, dt, L, prof)
                v, w = self.controller.compute(ctx)
                # stop-at-goal: never faster than what still stops at the end of the segment
                cap = max(self.cfg.v_creep, stop_speed(remaining, L, self.cfg.stop_latency_s))
                if abs(v) > cap:
                    k = cap / abs(v)
                    v, w = v * k, (w * k if abs(v) * k > 1e-3 else w)
                # obstacle cap (keeps curvature; at 0 the robot stops, no turning either)
                ocap = self.guard.speed_cap(now, path.direction)
                diag["obstacle_cap"] = ocap
                if ocap <= 1e-6:
                    v, w = 0.0, 0.0
                    self._blocked_t = self._blocked_t if self._blocked_t is not None else now
                    if now - self._blocked_t > self.cfg.blocked_timeout_s:
                        self._fail("blocked")
                elif abs(v) > ocap:
                    k = ocap / abs(v)
                    v, w = v * k, w * k
                    self._blocked_t = None
                else:
                    self._blocked_t = None
                vx, wz = self.shaper(v, w, dt, floor=ocap > 1e-6 and remaining > L.goal_pos_tol)
                return self._emit(now, vx, wz, **diag)

        if self.state == "SETTLE":
            vx, wz = self._ramp_to_zero(dt)
            if self.is_still:
                self._settle_t = now if self._settle_t is None else self._settle_t
                if now - self._settle_t >= L.settle_s:
                    if not last_seg:
                        self.seg_i += 1
                        self._enter_segment(now)
                    elif path.goal_yaw is not None and abs(wrap(pose.yaw - path.goal_yaw)) > L.goal_yaw_tol:
                        self.state = "FINAL_ALIGN"
                    else:
                        self.state = "SUCCEEDED"
            return self._emit(now, vx, wz, **diag)

        if self.state == "FINAL_ALIGN":
            e = wrap(pose.yaw - path.goal_yaw)
            if abs(e) <= 0.5 * L.goal_yaw_tol:
                self.state = "SETTLE"
                self._settle_t = None
                vx, wz = self._ramp_to_zero(dt)
            else:
                vx, wz = self.shaper(0.0, max(-L.wz_max, min(L.wz_max, -self.cfg.k_align * e)), dt)
            return self._emit(now, vx, wz, **diag)

        vx, wz = self._ramp_to_zero(dt)
        return self._emit(now, vx, wz, **diag)

    # ------------------------------------------------------------ contract output
    @staticmethod
    def to_whole_body(stamp: float, cmd: BaseVelocity, height: float) -> WholeBodyCommand:
        return WholeBodyCommand(stamp, velocity_height=BaseVelocityHeight(cmd.vx, cmd.vy, cmd.wz, height))
