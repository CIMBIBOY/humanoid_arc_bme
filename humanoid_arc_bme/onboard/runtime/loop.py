"""The runtime loop: read state and cameras, optionally run a policy through S1, publish telemetry, count what went wrong.

Observer mode (no policy, the default) never commands the robot: the vendor controller keeps it upright and we only watch.
With a policy, its commands pass through `SafetyLayer` and then reach `RobotIO.send_joint_targets` only if the runtime is armed.
A policy returns either (base, arms) (skill level, not sent anywhere yet) or a `JointTargets` (leg / whole-body policy,
e.g. `policies.OnnxLegPolicy`), which S1 filters with `filter_joints`. `arm()` works only for a simulated RobotIO
(`io.simulated`): arming a real robot is step B4 (Hold) and is not implemented.
Time is injected (clock, sleep), so the loop is testable without waiting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from humanoid_arc_bme.onboard.api.contract import BaseVelocity, JointTargets
from humanoid_arc_bme.onboard.api.safety import SafetyLayer, clamp_height

from .camera import CameraSource
from .io.base import RobotIO
from .telemetry import TelemetrySink


class Policy(Protocol):
    def act(self, state, frames: dict) -> tuple | JointTargets | None:
        """Return (base: BaseVelocity | None, arms: list[ArmCommand]), a JointTargets, or None for 'no command this tick'."""


@dataclass
class Stats:
    ticks: int = 0
    no_state: int = 0
    stale_state: int = 0
    refused_commands: int = 0
    sent_commands: int = 0
    max_period_s: float = 0.0
    period_sum_s: float = 0.0
    max_state_age_s: float = 0.0
    events: dict = field(default_factory=dict)

    def summary(self) -> dict:
        n = max(1, self.ticks - 1)
        return {"ticks": self.ticks, "no_state": self.no_state, "stale_state": self.stale_state, "refused_commands": self.refused_commands,
                "sent_commands": self.sent_commands,
                "mean_period_ms": 1000 * self.period_sum_s / n, "max_period_ms": 1000 * self.max_period_s,
                "max_state_age_ms": 1000 * self.max_state_age_s, "events": dict(self.events)}


class Runtime:
    def __init__(self, io: RobotIO, sink: TelemetrySink, cameras: list[CameraSource] | None = None, policy: Policy | None = None,
                 safety: SafetyLayer | None = None, hz: float = 50.0, stale_after_s: float = 0.1, clock=time.monotonic, sleep=time.sleep):
        if policy is not None and safety is None:
            raise ValueError("a policy needs a SafetyLayer: every command passes S1")
        self.io, self.sink, self.cameras, self.policy, self.safety = io, sink, cameras or [], policy, safety
        self.dt, self.stale_after_s, self.clock, self.sleep = 1.0 / hz, stale_after_s, clock, sleep
        self.armed = False  # nothing here sets it; arming is a Hold step (B4)
        self.fault: str | None = None  # SR-05: latched; cleared only by reset_fault() (an operator action)
        self.future_tolerance_s = 0.005  # SR-08: a state stamped further in the future than this is not trusted
        self.stats = Stats()
        self._last_tick: float | None = None
        orig = getattr(policy, "command", None)
        if policy is not None and callable(orig):  # R-4: the leg policy's (vx, vy, wz, h) command passes S1's base clamp
            policy.command = lambda: self._clamped_command(orig())

    def arm(self) -> None:
        """Allow commands to reach the RobotIO. Only for a simulated robot (the sim-as-robot test, W4/S4)."""
        if not getattr(self.io, "simulated", False):
            raise PermissionError("arming a real robot is step B4 (Hold): not implemented")
        if self.fault is not None:
            raise PermissionError(f"latched fault {self.fault!r}: reset_fault() first")
        if self.safety is not None:
            self.safety.reset_joint_reference()  # SR-01: the first sent target is limited against the measured position
        self.armed = True

    def disarm(self, reason: str = "operator") -> None:
        """Stop sending policy commands. SR-01/SR-10: the reference is reset; the safe state itself (damping or vendor
        controller) is the adapter's `safe_stop()`, sent once here."""
        was = self.armed
        self.armed = False
        if self.safety is not None:
            self.safety.reset_joint_reference()
        if was:
            stop = getattr(self.io, "safe_stop", None)
            try:
                if callable(stop):
                    stop()
            finally:
                self._event("disarmed", {"reason": reason})

    def _trip(self, reason: str, exc: BaseException | None = None) -> None:
        """SR-05: any exception inside a tick latches a fault and disarms (safe_stop); the loop keeps running so state and
        telemetry continue and the operator sees why."""
        self.fault = reason
        self._event("fault", {"reason": reason, "error": repr(exc) if exc is not None else ""})
        try:
            self.disarm(reason)
        except Exception as e:  # noqa: BLE001 - nothing more we can do in-process; S0 (remote, power) remains
            self._event("safe_stop_failed", {"error": repr(e)})

    def reset_fault(self) -> None:
        self.fault = None

    def _event(self, kind: str, detail: dict | None = None):
        self.stats.events[kind] = self.stats.events.get(kind, 0) + 1
        self.sink.event(kind, detail or {})

    def _clamped_command(self, cmd):
        vx, vy, wz, *rest = cmd
        base, events = self.safety.filter_base(BaseVelocity(float(vx), float(vy), float(wz)))
        if rest:  # SR-06: the base height passes S1 too (finite, inside base_height when configured)
            h, e = clamp_height(rest[0], self.safety.limits)
            rest, events = [h, *rest[1:]], events + e
        for e in events:
            self._event("s1_" + e.rule, {"subject": e.subject, "requested": str(e.requested), "applied": str(e.applied)})
        return (base.vx, base.vy, base.wz, *rest)

    def tick(self) -> None:
        try:
            self._tick()
        except Exception as e:  # noqa: BLE001 - SR-05: fail to the safe state, never out of the loop with the robot armed
            self._trip(f"exception_in_tick:{type(e).__name__}", e)

    def _tick(self) -> None:
        now = self.clock()
        s = self.stats
        s.ticks += 1
        if self._last_tick is not None:
            p = now - self._last_tick
            s.period_sum_s, s.max_period_s = s.period_sum_s + p, max(s.max_period_s, p)
        self._last_tick = now
        got = self.io.read_state()
        frames = {}
        for cam in self.cameras:
            fr = cam.read()
            if fr is not None:
                frames[cam.name] = fr
                self.sink.frame(fr)
        if got is None:
            s.no_state += 1
            self._event("no_state")
            if self.policy is not None:
                self._event("commands_suspended_no_state")
            return
        st, meta = got
        age = now - st.stamp
        s.max_state_age_s = max(s.max_state_age_s, age)
        if age < -self.future_tolerance_s:  # SR-08: a future stamp is a clock fault, not a fresh state
            age = float("inf")
        if age > self.stale_after_s:
            s.stale_state += 1
            self._event("stale_state", {"age_s": age})
        self.sink.state(st, meta)
        if self.policy is None:
            return
        if age > self.stale_after_s:  # R-1: no fresh state, no policy call, no command, no watchdog feed
            self._event("commands_suspended_stale_state", {"age_s": age})
            return
        if self.fault is not None:
            return
        out = self.policy.act(st, frames)
        if out is not None:
            self.safety.command_received(now)
        if isinstance(out, JointTargets) or (out is None and getattr(self.policy, "joint_policy", False)):
            self._joint_path(now, st, out)
            return
        base, arms = out if out is not None else (None, [])
        base, arms, events = self.safety.filter(now, self.dt, st, base, arms, wrist_pose_source=meta.wrist_pose_source)
        for e in events:
            self._event("s1_" + e.rule, {"subject": e.subject, "requested": str(e.requested), "applied": str(e.applied)})
        if not self.armed and (base is not None or arms):
            s.refused_commands += 1
            self._event("disarmed_command_dropped")

    def _joint_path(self, now, st, cmd) -> None:
        if cmd is None:  # a joint policy that skipped a tick: the watchdog decides (damping hold once it expires)
            if not self.safety.watchdog.expired(now):
                return
            cmd = JointTargets(self.policy.joint_names, tuple(dict(zip(st.joint_names, st.joint_pos))[n] for n in self.policy.joint_names))
        out, events = self.safety.filter_joints(now, self.dt, st, cmd)
        for e in events:
            self._event("s1_" + e.rule, {"subject": e.subject, "requested": str(e.requested), "applied": str(e.applied)})
        if not self.armed:
            self.stats.refused_commands += 1
            self._event("disarmed_command_dropped")
            self.safety.reset_joint_reference()  # SR-01: nothing was sent, so nothing to be continuous with
            return
        self.io.send_joint_targets(out.names, out.positions)
        self.stats.sent_commands += 1

    def run(self, seconds: float) -> dict:
        end = self.clock() + seconds
        while self.clock() < end:
            t0 = self.clock()
            self.tick()
            self.sleep(max(0.0, self.dt - (self.clock() - t0)))
        return self.stats.summary()

    def close(self):
        for c in self.cameras:
            c.close()
        self.io.close()
        self.sink.close()
