"""S1, the deterministic safety envelope (docs/plan/safety-stack.md).

Every command passes through here before it reaches joints or the base, in simulation and on the robot
alike. The rules are simple on purpose: a limit is a limit, nothing here is learned, and a rule's effect
can be predicted by hand.

- `clamp_base`: planar base velocity limits.
- `clamp_arm`: wrist target inside a workspace box, and its change per second limited (position only;
  orientation is passed through unchanged, which is a known limitation).
- `clamp_joint_targets`: joint position limits (inside the URDF range by a margin) and joint speed limits,
  relative to the current joint positions. A joint name without limits is an error (fail closed).
- `Watchdog`: no command for `timeout_s` means hold. The timeout has no default: the caller must choose.
- `hold_commands`: the hold: base stopped, each arm keeps its current wrist pose.
- `SafetyLayer`: the three rules and the watchdog together; `filter_joints` is the path for joint-target policies
  (the W4 leg policy): position + speed clamp, and on a stale command stream the damping hold (targets = measured
  positions).

Every function that changes a command also returns the `Event`s that say which rule fired and what was
requested and applied. A rule that does not fire changes nothing (identical object values).

Imports only the standard library (PyYAML lazily, only in `load_limits`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

from .contract import ArmCommand, BaseVelocity, JointTargets, RobotState


@dataclass(frozen=True)
class Event:
    rule: str  # base_vel, workspace, wrist_speed, joint_pos, joint_speed, watchdog
    subject: str  # which component: "vx", "left", a joint name ...
    requested: object
    applied: object


@dataclass(frozen=True)
class Limits:
    joint_pos: dict  # joint name -> (lower, upper) in rad, as in the URDF
    pos_margin: float  # rad kept inside joint_pos
    joint_vel_max: float  # rad/s, for every commanded joint
    wrist_speed_max: float  # m/s
    ws_min: tuple  # workspace box (x, y, z), pelvis frame, metres
    ws_max: tuple
    base_vel_max: tuple  # (vx, vy, wz) absolute
    validated_on_robot: bool = False
    joint_vel_max_by_joint: dict = field(default_factory=dict)  # rad/s per joint, overrides joint_vel_max (legs)
    height_range: tuple | None = None  # (min, max) m for the leg policy's base-height command; None = finiteness only
    tracking_error_max: float | None = None  # rad, |target - measured| bound applied after the rate rule; None = off

    def __post_init__(self):
        for label, seq in (("ws_min", self.ws_min), ("ws_max", self.ws_max), ("base_vel_max", self.base_vel_max)):
            if len(seq) != 3:
                raise ValueError(f"{label}: expected 3 values, got {len(seq)}")
        nums = [self.pos_margin, self.joint_vel_max, self.wrist_speed_max, *self.ws_min, *self.ws_max, *self.base_vel_max]
        for lo_hi in self.joint_pos.values():
            nums += list(lo_hi)
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in nums):
            raise ValueError("limits must be finite numbers (NaN/inf/non-numeric would disable a clamp)")
        if min(self.joint_vel_max, self.wrist_speed_max, *self.base_vel_max) <= 0 or self.pos_margin < 0:
            raise ValueError("speed limits must be positive and the margin non-negative")
        if any(lo >= hi for lo, hi in zip(self.ws_min, self.ws_max)):
            raise ValueError("workspace box: min must be below max on every axis")
        for name, (lo, hi) in self.joint_pos.items():
            if hi - lo <= 2 * self.pos_margin:
                raise ValueError(f"joint {name}: range {lo}..{hi} is not wider than twice the margin")
        if self.height_range is not None and not (len(self.height_range) == 2 and all(map(math.isfinite, self.height_range))
                                                  and 0 < self.height_range[0] < self.height_range[1]):
            raise ValueError("height_range: expected finite (min, max) with 0 < min < max")
        if self.tracking_error_max is not None and not (math.isfinite(self.tracking_error_max) and self.tracking_error_max > 0):
            raise ValueError("tracking_error_max must be positive and finite")
        for name, v in self.joint_vel_max_by_joint.items():
            if name not in self.joint_pos or not (math.isfinite(v) and v > 0):
                raise ValueError(f"joint_vel_max for {name!r}: unknown joint or not a positive number")

    def joint_vel_limit(self, name: str) -> float:
        return self.joint_vel_max_by_joint.get(name, self.joint_vel_max)

    def joint_range(self, name: str) -> tuple[float, float]:
        if name not in self.joint_pos:
            raise ValueError(f"no limits for joint {name!r}")
        lo, hi = self.joint_pos[name]
        return lo + self.pos_margin, hi - self.pos_margin

    def require_validated(self) -> None:
        """Hardware steps call this first: placeholder limits must not drive a real robot."""
        if not self.validated_on_robot:
            raise RuntimeError("limits are placeholders (validated_on_robot: false); replace them at step B3.3")


def _num(x, label: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"limits: {label} must be a number, got {x!r}")
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"limits: {label} must be finite, got {x}")
    return x


def _bool(x, label: str) -> bool:
    if not isinstance(x, bool):  # SR-04: bool("false") is True; a quoted YAML value must not validate placeholder limits
        raise ValueError(f"limits: {label} must be true or false (unquoted), got {x!r}")
    return x


def limits_from_dict(d: dict) -> Limits:
    try:
        return _limits_from_dict(d)
    except (KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"limits: malformed ({type(e).__name__}: {e})") from e


def _limits_from_dict(d: dict) -> Limits:
    return Limits(
        joint_pos={k: (_num(v["lower"], f"{k}.lower"), _num(v["upper"], f"{k}.upper")) for k, v in d["joints"].items()},
        pos_margin=_num(d["pos_margin_rad"], "pos_margin_rad"),
        joint_vel_max=_num(d["joint_vel_max_default"], "joint_vel_max_default"),
        wrist_speed_max=_num(d["wrist_speed_max"], "wrist_speed_max"),
        ws_min=tuple(_num(x, "workspace.min") for x in d["workspace"]["min"]),
        ws_max=tuple(_num(x, "workspace.max") for x in d["workspace"]["max"]),
        base_vel_max=tuple(_num(d["base_vel_max"][k], f"base_vel_max.{k}") for k in ("vx", "vy", "wz")),
        validated_on_robot=_bool(d["validated_on_robot"], "validated_on_robot"),
        joint_vel_max_by_joint={k: _num(v, f"joint_vel_max.{k}") for k, v in (d.get("joint_vel_max") or {}).items()},
        height_range=(None if d.get("base_height") is None else
                      (_num(d["base_height"]["min"], "base_height.min"), _num(d["base_height"]["max"], "base_height.max"))),
        tracking_error_max=None if d.get("tracking_error_max_rad") is None else _num(d["tracking_error_max_rad"], "tracking_error_max_rad"),
    )


def load_limits(path) -> Limits:
    import yaml  # lazy: the rest of this module is standard library only

    return limits_from_dict(yaml.safe_load(Path(path).read_text()))


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def clamp_base(cmd: BaseVelocity, limits: Limits) -> tuple[BaseVelocity, list[Event]]:
    events, out = [], []
    for name, v, m in zip(("vx", "vy", "wz"), (cmd.vx, cmd.vy, cmd.wz), limits.base_vel_max):
        c = _clip(v, -m, m)
        if c != v:
            events.append(Event("base_vel", name, v, c))
        out.append(c)
    return (BaseVelocity(*out) if events else cmd), events


def clamp_height(h, limits: Limits) -> tuple[float, list[Event]]:
    """SR-06: the leg policy's base-height command. Non-finite raises (fail closed); outside `height_range` is clipped."""
    if isinstance(h, bool) or not isinstance(h, (int, float)) or not math.isfinite(h):
        raise ValueError(f"height command must be a finite number, got {h!r}")
    if limits.height_range is None:
        return float(h), []
    c = _clip(float(h), *limits.height_range)
    return c, ([Event("base_height", "h", h, c)] if c != h else [])


def clamp_arm(cmd: ArmCommand, ref_position, dt: float, limits: Limits) -> tuple[ArmCommand, list[Event]]:
    """Box first, then the speed limit relative to `ref_position` (the last applied target or the measured
    wrist position). With `ref_position` inside the box the result is inside the box and no more than
    `wrist_speed_max * dt` away from `ref_position`.
    """
    if not (math.isfinite(dt) and dt > 0):
        raise ValueError("dt must be positive and finite")
    if len(ref_position) != 3 or not all(map(math.isfinite, ref_position)):
        raise ValueError("ref_position: expected 3 finite numbers")
    events = []
    p = list(cmd.wrist_pose[:3])
    boxed = [_clip(x, lo, hi) for x, lo, hi in zip(p, limits.ws_min, limits.ws_max)]
    if boxed != p:
        events.append(Event("workspace", cmd.side, tuple(p), tuple(boxed)))
    step = [b - r for b, r in zip(boxed, ref_position)]
    dist = math.sqrt(sum(s * s for s in step))
    max_step = limits.wrist_speed_max * dt
    if dist > max_step * (1.0 + 1e-9):  # tolerance: re-clamping an already limited target must not fire again
        k = max_step / dist
        limited = [r + s * k for r, s in zip(ref_position, step)]
        events.append(Event("wrist_speed", cmd.side, tuple(boxed), tuple(limited)))
        boxed = limited
    if not events:
        return cmd, events
    return replace(cmd, wrist_pose=tuple(boxed) + cmd.wrist_pose[3:]), events


def clamp_joint_targets(names, targets, current, dt: float, limits: Limits) -> tuple[list[float], list[Event]]:
    """Position limit first (inside the URDF range by the margin), then the joint speed limit relative to
    `current`. A joint that is already outside its range moves toward it at the speed limit, not instantly.
    """
    if not (math.isfinite(dt) and dt > 0):
        raise ValueError("dt must be positive and finite")
    if not (len(names) == len(targets) == len(current)):
        raise ValueError("names, targets and current must have the same length")
    events, out = [], []
    for name, t, c in zip(names, targets, current):
        max_step = limits.joint_vel_limit(name) * dt
        if not (math.isfinite(t) and math.isfinite(c)):
            raise ValueError(f"joint {name}: non-finite value")
        lo, hi = limits.joint_range(name)
        a = _clip(t, lo, hi)
        if abs(a - t) > 1e-6:  # not an event for float32 rounding (a decoded target sitting exactly on the clamp bound)
            events.append(Event("joint_pos", name, t, a))
        d = a - c
        b = a if abs(d) <= max_step else c + math.copysign(max_step, d)  # exact `a` when in reach: no float drift
        if b != a:
            events.append(Event("joint_speed", name, a, b))
        out.append(b)
    return out, events


class Watchdog:
    """No command within `timeout_s` means hold. Time is passed in, never read from a clock."""

    def __init__(self, timeout_s: float):
        if not (math.isfinite(timeout_s) and timeout_s > 0):
            raise ValueError("timeout_s must be positive and finite")
        self.timeout_s = timeout_s
        self._last: float | None = None

    def feed(self, now: float) -> None:
        self._last = now

    def expired(self, now: float) -> bool:
        return self._last is None or (now - self._last) > self.timeout_s


def hold_commands(state: RobotState, wrist_pose_source: str = "fk") -> tuple[BaseVelocity, dict]:
    """Base stopped; each arm keeps its current wrist pose (pelvis frame). Hands keep their last command.
    With `wrist_pose_source == "none"` the wrist poses are placeholders: hold means no arm command."""
    if wrist_pose_source == "none":
        return BaseVelocity(0.0, 0.0, 0.0), {}
    arms = {side: ArmCommand(side, pose, "pelvis") for side, pose in state.wrist_poses.items()}
    return BaseVelocity(0.0, 0.0, 0.0), arms


@dataclass
class SafetyLayer:
    limits: Limits
    watchdog: Watchdog
    _applied: dict = field(default_factory=dict)  # side -> last applied wrist position
    # Joint speed rule reference: "previous_command" limits the change of the commanded target per tick (the rate the
    # robot is asked to move at; the first command after start/hold is limited against the measured position);
    # "measured" limits |target - measured position| (the PD error). W4/S4 measured that the second form fires on the
    # E2 walking gait in ~1 of 15 joint-ticks (ankle pitch pushed against contact) and costs 20-30x the falls.
    joint_rate_reference: str = "previous_command"
    _applied_joints: dict = field(default_factory=dict)  # joint -> last applied target
    _last_joint_t: float | None = None  # time of the last filter_joints call (SR-02: a gap resets the reference)

    def reset_joint_reference(self) -> None:
        """SR-01: the next joint command is rate-limited against the measured position. Call on arm, disarm, fault, and
        whenever filtered targets were not sent (disarmed ticks)."""
        self._applied_joints.clear()
        self._last_joint_t = None

    def command_received(self, now: float) -> None:
        self.watchdog.feed(now)

    def filter_base(self, base: BaseVelocity) -> tuple[BaseVelocity, list]:
        """Base-velocity clamp alone (no watchdog, no arms), for a velocity command that is not sent as a base command
        (the leg policy's observation input)."""
        return clamp_base(base, self.limits)

    def filter(self, now: float, dt: float, state: RobotState, base: BaseVelocity | None = None, arms=(),
               wrist_pose_source: str = "fk"):
        """Returns (base, {side: ArmCommand}, events). A stale command stream returns the hold. With
        `wrist_pose_source == "none"` (placeholder wrist poses) arm commands are refused and the hold has no arm part."""
        if self.watchdog.expired(now):
            hb, ha = hold_commands(state, wrist_pose_source)
            self._applied = {s: tuple(c.wrist_pose[:3]) for s, c in ha.items()}
            return hb, ha, [Event("watchdog", "all", None, "hold")]
        events, out_arms = [], {}
        out_base = None
        if base is not None:
            out_base, e = clamp_base(base, self.limits)
            events += e
        if wrist_pose_source == "none" and arms:
            events.append(Event("wrist_pose_unknown", "arms", len(list(arms)), 0))
            arms = ()
        for cmd in arms:
            if cmd.frame != "pelvis":
                raise ValueError("S1 limits are defined in the pelvis frame; transform the command first")
            ref = self._applied.get(cmd.side, tuple(state.wrist_poses[cmd.side][:3]))
            c, e = clamp_arm(cmd, ref, dt, self.limits)
            self._applied[cmd.side] = tuple(c.wrist_pose[:3])
            out_arms[cmd.side] = c
            events += e
        return out_base, out_arms, events

    def filter_joints(self, now: float, dt: float, state: RobotState, cmd: JointTargets):
        """Joint-target path. Returns (JointTargets, events). A stale command stream returns the damping hold: every
        commanded joint's target is its measured position (the PD then only damps; for legs this is NOT a balance
        controller, it is the stop). Position and speed limits as in `clamp_joint_targets`; the speed reference is the
        previous applied target (see `joint_rate_reference`)."""
        pos = dict(zip(state.joint_names, state.joint_pos))
        missing = [n for n in cmd.names if n not in pos]
        if missing:
            raise ValueError(f"state has no measurement for {missing}")
        current = [pos[n] for n in cmd.names]
        if self.watchdog.expired(now):
            self.reset_joint_reference()
            return JointTargets(cmd.names, tuple(current)), [Event("watchdog", "joints", None, "damping_hold")]
        if self._last_joint_t is not None and not (0.0 <= now - self._last_joint_t <= self.watchdog.timeout_s):
            self.reset_joint_reference()  # SR-02: a gap (or a clock step back) longer than the timeout: the old target is stale
        self._last_joint_t = now
        if self.joint_rate_reference not in ("previous_command", "measured"):
            raise ValueError(f"joint_rate_reference: {self.joint_rate_reference!r}")
        ref = current
        if self.joint_rate_reference == "previous_command" and all(n in self._applied_joints for n in cmd.names):
            ref = [self._applied_joints[n] for n in cmd.names]
        out, events = clamp_joint_targets(cmd.names, cmd.positions, ref, dt, self.limits)
        e_max = self.limits.tracking_error_max
        if e_max is not None:  # SR-03: the previous-command reference alone lets a blocked joint's PD error grow without bound
            for i, (n, c) in enumerate(zip(cmd.names, current)):
                b = _clip(out[i], c - e_max, c + e_max)
                if b != out[i]:
                    events.append(Event("tracking_error", n, out[i], b))
                    out[i] = b
        self._applied_joints.update(zip(cmd.names, out))
        return (cmd if not events else JointTargets(cmd.names, tuple(out))), events
