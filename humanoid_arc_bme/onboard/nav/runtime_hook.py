"""Hook for the on-robot runtime and the web UI: one object, JSON-friendly in and out.

The web-UI / runtime agent owns humanoid_arc_bme/onboard/runtime/; this module is the only nav code it needs to call (see the Interface
section of docs/plan/navstack.md):

    from humanoid_arc_bme.onboard.nav.runtime_hook import NavSession
    nav = NavSession(controller="rpp")                       # limits default to the S1 base limits
    nav.load({"waypoints": [[0, 0], [1, 0], [1.5, 0.5]], "direction": 1, "smoothing": "corner", "corner_radius": 0.3})
    nav.start(now)
    cmd = nav.tick(now, state)    # BaseVelocity (vx, vy=0, wz); feed it through S1 like any base command
    nav.status()                  # dict: state, reason, progress, errors, pose, command
    nav.stop()

Two adapters for humanoid_arc_bme.onboard.runtime's two command paths:
- `NavPolicy(nav)`: the runtime `Policy` protocol (`act(state, frames) -> (BaseVelocity, [])`), for the vendor
  locomotion path (base command through `SafetyLayer.filter`).
- `nav.command_source(height)`: a callable returning (vx, vy, wz, height) for `OnnxLegPolicy(command=...)` (our leg
  policy path; the runtime already clamps it with S1's base limits).
`vendor_velocity_args(cmd, period_s)` gives the arguments of the vendor `LocoClient.SetVelocity(vx, vy, omega,
duration)` (unitree_sdk2_python, OPENED): duration = a few control periods, so a dead nav loop stops the robot
(the vendor's own example uses 1 s, and `Move(..., continous_move=True)` uses 864000 s: never use that here).

Odometry: by default dead reckoning on the COMMANDED velocity with the IMU yaw from `RobotState.imu.quat` (stage 1).
`nav.external_pose(t, x, y, yaw)` feeds LiDAR/vendor odometry (stage 2) and switches to the fused estimate.
"""

from __future__ import annotations

import math

from humanoid_arc_bme.onboard.api.contract import BaseVelocity

from .follower import FollowerConfig, PathFollower
from .limits import NavLimits
from .obstacle import NoGuard
from .odometry import DeadReckoning, FusedOdometry, yaw_from_quat
from .path import Path, PathPlan, Pose2D, catmull_rom, from_waypoints


def plan_from_spec(spec: dict) -> PathPlan:
    """JSON spec -> PathPlan. Keys: `waypoints` [[x, y], ...] (odom frame, metres) or `segments` [spec, ...];
    `direction` 1|-1; `smoothing` "none"|"corner"|"spline"; `corner_radius` m; `goal_yaw` rad or null; `ds` m;
    `frame` "odom" (default) or "start" (waypoints relative to the pose at start; converted by NavSession)."""
    if "segments" in spec:
        return PathPlan([p for s in spec["segments"] for p in plan_from_spec(s).segments])
    pts = spec.get("waypoints")
    if not pts or len(pts) < 2:
        raise ValueError("spec needs 'waypoints' with at least 2 points")
    direction = int(spec.get("direction", 1))
    ds = float(spec.get("ds", 0.05))
    goal_yaw = spec.get("goal_yaw")
    smoothing = spec.get("smoothing", "corner")
    if smoothing == "spline":
        p = catmull_rom(pts, ds, direction, goal_yaw=goal_yaw)
    elif smoothing == "corner":
        p = from_waypoints(pts, ds, direction, float(spec.get("corner_radius", 0.3)), goal_yaw)
    elif smoothing == "none":
        p = from_waypoints(pts, ds, direction, 0.0, goal_yaw)
    else:
        raise ValueError(f"smoothing: none | corner | spline, got {smoothing!r}")
    return PathPlan([p])


def _relative(spec: dict, origin: Pose2D) -> dict:
    c, s = math.cos(origin.yaw), math.sin(origin.yaw)
    out = dict(spec)
    if "segments" in spec:
        out["segments"] = [_relative(x, origin) for x in spec["segments"]]
        return out
    out["waypoints"] = [[origin.x + c * x - s * y, origin.y + s * x + c * y] for x, y in spec["waypoints"]]
    if spec.get("goal_yaw") is not None:
        out["goal_yaw"] = origin.yaw + spec["goal_yaw"]
    out["frame"] = "odom"
    return out


class NavSession:
    def __init__(self, controller: str = "regulated_pure_pursuit", limits: NavLimits | None = None, guard=None,
                 config: FollowerConfig = FollowerConfig(), use_imu_yaw: bool = True):
        self.limits = limits or NavLimits()
        self.dr = DeadReckoning("command")
        self.odom = FusedOdometry(self.dr)
        self.use_imu_yaw = use_imu_yaw
        self.follower = PathFollower(self.odom, controller, self.limits, guard or NoGuard(), config)
        self._spec = None
        self._plan = None
        self.last_cmd = BaseVelocity(0.0, 0.0, 0.0)
        self.external_fixes = 0

    # ---------------------------------------------------------------- operator API (JSON in)
    def load(self, spec: dict) -> dict:
        """Validate and store a path; it starts on `start()`. Returns a summary (length per segment)."""
        self._spec = spec
        self._plan = None if spec.get("frame", "odom") == "start" else plan_from_spec(spec)
        plan = self._plan or plan_from_spec(spec)
        return {"segments": [{"length_m": round(p.length, 3), "direction": p.direction} for p in plan.segments]}

    def load_plan(self, plan: PathPlan | Path):
        self._spec, self._plan = None, plan if isinstance(plan, PathPlan) else PathPlan([plan])

    def start(self, now: float) -> None:
        if self._plan is None and self._spec is None:
            raise ValueError("load a path first")
        plan = self._plan
        if plan is None:  # relative to the current pose
            plan = plan_from_spec(_relative(self._spec, self.odom.pose(now)))
        self.follower.follow(plan, now)

    def stop(self) -> None:
        self.follower.stop()

    def reset_odometry(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0, now: float = 0.0) -> None:
        self.odom.reset(Pose2D(x, y, yaw), now)

    def external_pose(self, t: float, x: float, y: float, yaw: float) -> None:
        """A pose in the odom frame from LiDAR odometry / the vendor / a marker fix."""
        self.odom.fix(t, Pose2D(x, y, yaw))
        self.external_fixes += 1

    # ---------------------------------------------------------------- loop
    def tick(self, now: float, state=None) -> BaseVelocity:
        if state is not None and self.use_imu_yaw:
            self.dr.set_yaw(yaw_from_quat(state.imu.quat))
        self.last_cmd = self.follower.step(now)
        return self.last_cmd

    def status(self) -> dict:
        f = self.follower
        st = f.status()
        d = dict(f.last)
        d.update({"state": f.state, "skill": st.state, "reason": st.reason,
                  "segment": f.seg_i, "segments": 0 if f.plan is None else len(f.plan.segments),
                  "controller": f.controller.name, "external_fixes": self.external_fixes,
                  "cmd": {"vx": self.last_cmd.vx, "vy": self.last_cmd.vy, "wz": self.last_cmd.wz}})
        return d

    def command_source(self, height: float, clock=None):
        """Callable for OnnxLegPolicy(command=...): returns (vx, vy, wz, height) of the latest tick, or ticks the
        follower itself when given a clock."""
        def cmd():
            c = self.tick(clock()) if clock is not None else self.last_cmd
            return (c.vx, c.vy, c.wz, height)
        return cmd


class NavPolicy:
    """humanoid_arc_bme.onboard.runtime Policy protocol adapter: base command only, no arms. The runtime passes it through S1."""

    def __init__(self, session: NavSession, clock):
        self.session, self.clock = session, clock

    def act(self, state, frames=None):
        return self.session.tick(self.clock(), state), []


def vendor_velocity_args(cmd: BaseVelocity, period_s: float, periods: float = 3.0) -> tuple:
    """(vx, vy, omega, duration) for LocoClient.SetVelocity: the command expires after `periods` control periods."""
    if period_s <= 0:
        raise ValueError("period_s must be positive")
    return cmd.vx, cmd.vy, cmd.wz, periods * period_s


class OpenLoopDriver:
    """Stage 0: open-loop velocity commands from the web UI (joystick / buttons / a timed script), ramped by the same
    CommandShaper and limits as the follower, with a dead-man timeout: no `set()` for `deadman_s` -> ramp to zero.

        drv = OpenLoopDriver()
        drv.set(now, vx=0.1, wz=0.0)                    # each UI message
        drv.play(now, [(0.1, 0.0, 5.0), (0.0, 0.0, 2.0), (-0.05, 0.0, 4.0), (0.1, 0.15, 8.0)])  # (vx, wz, seconds)
        cmd = drv.tick(now)                              # BaseVelocity at the control rate
    """

    def __init__(self, limits: NavLimits | None = None, deadman_s: float = 0.5):
        from .limits import CommandShaper

        self.limits = limits or NavLimits()
        self.shaper = CommandShaper(self.limits)
        self.deadman_s = deadman_s
        self.target = (0.0, 0.0)
        self._last_set = None
        self._script = None  # (t0, [(vx, wz, dur), ...])
        self._last_t = None

    def set(self, now: float, vx: float, wz: float) -> None:
        self._script = None
        self.target, self._last_set = (float(vx), float(wz)), now

    def play(self, now: float, sequence) -> float:
        """Run a timed list of (vx, wz, seconds); returns its total duration. Any set() or stop() cancels it."""
        seq = [(float(a), float(b), float(c)) for a, b, c in sequence]
        if any(c <= 0 for _, _, c in seq):
            raise ValueError("durations must be positive")
        self._script = (now, seq)
        return sum(c for _, _, c in seq)

    def stop(self) -> None:
        self._script, self.target, self._last_set = None, (0.0, 0.0), None

    def _target(self, now):
        if self._script is not None:
            t0, seq = self._script
            t = now - t0
            for vx, wz, dur in seq:
                if t < dur:
                    return vx, wz
                t -= dur
            self._script = None
            return 0.0, 0.0
        if self._last_set is None or now - self._last_set > self.deadman_s:
            return 0.0, 0.0
        return self.target

    def tick(self, now: float) -> BaseVelocity:
        dt = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        self._last_t = now
        vx, wz = self._target(now)
        vx, wz = self.shaper(vx, wz, dt, floor=False)
        return BaseVelocity(vx, 0.0, wz)
