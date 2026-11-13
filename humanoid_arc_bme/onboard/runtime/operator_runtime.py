"""The operator runtime: one control loop (thread) that the web UI and the ROS 2 wrapper drive through a small API.

Per tick (control_hz): read state -> camera at camera_hz -> supervisor checks (deadman, stale state, link loss, tilt,
controller fault) -> if armed: ramped command -> S1 base clamp -> controller.step; else controller.idle -> shadow policy
-> run log + live plots. Nothing is sent to the robot while disarmed (the one exception is the damping stream of the
low-level path after the operator explicitly released the vendor mode, which only resists motion).

API (thread-safe, all logged): heartbeat, request_arm, confirm_arm, disarm, stop, reset_stop, damp, select_controller,
release_vendor, status, plots, preflight. Time is injected (clock, wall) so the loop runs in tests without waiting.

Fail-safe rules (docs/audit/2026-10-30_safety_review.md, UI-01/UI-07, SR-05/SR-08/SR-14):
- Any exception inside a tick (policy NaN, run log "disk full", nav, controller, telemetry) is caught: the fault is
  latched (`fault`), STOP is latched (controller safe state: vendor SetVelocity(0) / policy damping), the cause is logged
  best effort. Arming needs Reset STOP, which also clears the fault.
- A supervisor thread, independent of the control thread, disarms with the controller's `emergency_stop` (which does
  not need the runtime lock) when the control loop has not completed a tick for `tick_deadline_s` (0.2 s) or the control
  thread died. While armed it also refreshes the free-disk value, so the tick never makes that syscall.
- While armed, a latched fault also follows: motor temperature >= motor_temp_max_c, motor error bits != 0, the
  lowstate `tick` not advancing over 3 new samples, free disk below the minimum, a state stamped in the future.
- status() / preflight() / plots() never take the control lock (they read snapshots), so UI polling cannot stall the loop.
"""

from __future__ import annotations

import collections
import math
import sys
import os
import shutil
import threading
import time
from dataclasses import dataclass, field

from humanoid_arc_bme.onboard.api.contract import BaseVelocity
from humanoid_arc_bme.onboard.api.safety import SafetyLayer

from .controllers import Controller
from .operator import ARMED, ZERO, OperatorLimits, Supervisor, finite_number, tilt_deg_from_quat


class ControlLoopDown(RuntimeError):
    """The control loop is not running: heartbeats are refused (the web server answers 503)."""


@dataclass
class RuntimeConfig:
    control_hz: float = 50.0
    camera_hz: float = 2.0
    plot_seconds: float = 60.0
    min_free_disk_gb: float = 1.0
    motor_temp_max_c: float = 75.0
    external_cmd_timeout_s: float = 0.5
    tick_deadline_s: float = 0.2  # UI-01: no completed tick for this long while armed -> the supervisor thread disarms
    future_stamp_tolerance_s: float = 0.005  # SR-08: a state stamped further in the future is not fresh
    tick_frozen_samples: int = 3  # SR-08 / T-STATE: lowstate `tick` unchanged over this many new samples -> fault
    extra: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict) -> "RuntimeConfig":
        keys = {"control_hz", "camera_hz", "plot_seconds", "min_free_disk_gb", "motor_temp_max_c", "external_cmd_timeout_s",
                "tick_deadline_s", "future_stamp_tolerance_s", "tick_frozen_samples"}
        return RuntimeConfig(**{k: float(v) for k, v in d.items() if k in keys}, extra={k: v for k, v in d.items() if k not in keys})


@dataclass
class Check:
    name: str
    status: str  # ok | fail | unknown
    detail: str
    required: bool = True

    def to_dict(self):
        return {"name": self.name, "status": self.status, "detail": self.detail, "required": self.required}


class OperatorRuntime:
    def __init__(self, io, controllers: dict, op_limits: OperatorLimits, safety: SafetyLayer, runlog, sink=None, cameras=None,
                 cfg: RuntimeConfig | None = None, shadow=None, nav=None, default_controller: str | None = None,
                 clock=time.monotonic, wall=time.time, sleep=time.sleep, disk_free=None, obstacle=None):
        """`obstacle`: an ObstacleStage (nav/obstacle_stage.py); default = no sensor, speed capped to the S1 limit, rear blocked.
        `controllers`: name -> Controller instance or zero-argument factory (built on first select)."""
        self.io, self.safety, self.runlog, self.sink = io, safety, runlog, sink
        self.cameras = cameras or []
        self.cfg = cfg or RuntimeConfig()
        self.lim = op_limits
        self.clock, self.wall, self.sleep = clock, wall, sleep
        self.dt = 1.0 / self.cfg.control_hz
        self.shadow, self.nav = shadow, nav
        from humanoid_arc_bme.onboard.nav.obstacle_stage import ObstacleStage
        self.obstacle = obstacle if obstacle is not None else ObstacleStage(no_sensor_speed=float(safety.limits.base_vel_max[0]))
        self._obstacle_key = None
        self.obstacle_info: dict | None = None if self.obstacle.has_sensor else {
            "distance": None, "cap": None, "scale": 1.0, "reason": "no_sensor", "direction": 0,
            "warning": "NO OBSTACLE SENSOR (forward capped to S1 limit, rear blocked)"}
        self._disk_free = disk_free or (lambda p: shutil.disk_usage(p).free)
        self.lock = threading.RLock()
        self.sup = Supervisor(op_limits, on_transition=self._on_transition)
        self._factories = dict(controllers)
        self._built: dict[str, Controller] = {}
        self.controller_name = default_controller or next(iter(self._factories))
        self.controller: Controller | None = None
        self._select_error: str | None = None
        self.owner: str | None = None  # client id that armed; only it may send heartbeats while armed
        self.mode = "teleop"  # teleop | curvature | waypoints | ros
        self._external: tuple | None = None  # (stamp, (vx, vy, wz)) from ROS cmd_vel
        self.state = None
        self.meta = None
        self.state_age: float | None = None
        self._last_state_stamp = None
        self._state_times = collections.deque(maxlen=400)
        self._tick_times = collections.deque(maxlen=400)
        self.max_period = 0.0
        self.ticks = 0
        self.last_cmd = (0.0, 0.0, 0.0)
        self.frames: dict = {}
        self._last_cam = -math.inf
        self._plot = collections.deque(maxlen=int(self.cfg.plot_seconds * 10))
        self._last_plot = -math.inf
        self._events = collections.deque(maxlen=200)
        self._thread = None
        self._sup_thread = None
        self._stop_evt = threading.Event()
        self.started_wall = wall()
        self.last_shadow: dict | None = None
        self.fault: str | None = None  # latched (UI-01); cleared only by reset_stop() while disarmed
        self.fault_detail: str | None = None
        self.tick_errors = 0
        self._last_tick_ok: float | None = None  # clock time of the last tick that completed without an exception
        self._pending_trip: str | None = None  # set by the supervisor thread when it could not take the lock
        self._disk_cache: tuple | None = None  # (clock time, free bytes or None, error)
        self._tick_seen: tuple | None = None  # (lowstate tick, repeats over new samples)
        self.select_controller(self.controller_name)

    # ---------------------------------------------------------------- logging
    def _log_event(self, kind: str, detail: dict | None = None) -> None:
        rec = {"t": self.clock(), "event": kind, **(detail or {})}
        self._events.append(rec)
        self.runlog.event(kind, detail or {})
        if self.sink is not None:
            self.sink.event(kind, detail or {})

    def _on_transition(self, rec: dict) -> None:
        """Never raises. The robot's safe state comes first; logging is best effort (the log may be what failed)."""
        c = self.controller
        now = rec["t"]
        if c is not None:
            try:
                if rec["reason"].startswith("stop:"):
                    c.stop(now)
                elif rec["from"] == ARMED and rec["to"] != ARMED:
                    c.on_disarm(now, rec["reason"])
            except Exception as e:  # noqa: BLE001 - try the lock-free path, then record the failure
                self.fault = self.fault or f"safe_state_failed:{type(e).__name__}"
                try:
                    c.emergency_stop(now)
                except Exception:  # noqa: BLE001 - S0 (remote, power) remains
                    pass
                self._safe_log("safe_state_failed", {"error": f"{type(e).__name__}: {e}", "reason": rec["reason"]})
        if self.nav is not None and rec["from"] == ARMED and rec["to"] != ARMED and hasattr(self.nav, "stop"):
            try:
                self.nav.stop()  # a disarm ends the path: re-arming never resumes motion on its own
            except Exception as e:  # noqa: BLE001
                self._safe_log("nav_stop_failed", {"error": f"{type(e).__name__}: {e}"})
        self._events.append({"event": "transition", **rec})
        try:
            self.runlog.transition(rec)
            if self.sink is not None:
                self.sink.event("transition", rec)
        except Exception as e:  # noqa: BLE001
            self._events.append({"t": now, "event": "log_failed", "error": f"{type(e).__name__}: {e}"})

    def _safe_log(self, kind: str, detail: dict | None = None) -> None:
        try:
            self._log_event(kind, detail)
        except Exception as e:  # noqa: BLE001 - the run log itself may have failed (disk full)
            self._events.append({"t": self.clock(), "event": kind, "log_failed": f"{type(e).__name__}: {e}", **(detail or {})})

    # ---------------------------------------------------------------- controllers
    def controller_names(self) -> list[str]:
        return list(self._factories)

    def select_controller(self, name: str) -> tuple[bool, str]:
        with self.lock:
            if self.sup.armed:
                return False, "disarm first"
            if name not in self._factories:
                return False, f"unknown controller {name!r}"
            if name not in self._built:
                f = self._factories[name]
                try:
                    self._built[name] = f() if not isinstance(f, Controller) else f
                except Exception as e:  # noqa: BLE001 - e.g. onnxruntime missing: report, keep the old one
                    self._select_error = f"{name}: {type(e).__name__}: {e}"
                    self._log_event("controller_select_failed", {"name": name, "error": self._select_error})
                    return False, self._select_error
            self.controller_name, self.controller, self._select_error = name, self._built[name], None
            self._log_event("controller_selected", {"name": name})
            return True, "ok"

    # ---------------------------------------------------------------- operator API
    def loop_age(self, now: float | None = None) -> float | None:
        """Seconds since the last tick that completed without an exception (None before the first)."""
        last = self._last_tick_ok
        return None if last is None else (self.clock() if now is None else now) - last

    def _locked(self, timeout: float = 0.1):
        """Context manager: take the control lock for at most `timeout` s, else ControlLoopDown (UI-06: a hung control
        thread must not pile up web request threads behind its lock)."""
        rt = self

        class _Ctx:
            def __enter__(self_):
                if not rt.lock.acquire(timeout=timeout):
                    raise ControlLoopDown("control loop busy or hung: request refused")
                return rt

            def __exit__(self_, *exc):
                rt.lock.release()
                return False
        return _Ctx()

    def heartbeat(self, payload: dict, client: str = "local") -> dict:
        """The UI calls this at >= 5 Hz. payload: {mode: teleop|curvature|waypoints|ros, held: bool, vx, vy, wz, v, kappa}.
        While armed only the client that armed (the owner) feeds the deadman; others get PermissionError.
        UI-04: everything is validated before anything changes; a malformed heartbeat does not feed the deadman.
        UI-03: motion only while `held` is true (hold-to-run), and after arming only once a released heartbeat arrived.
        UI-01: while armed and the control loop is down, heartbeats are refused (ControlLoopDown), so the UI shows it."""
        with self._locked():
            now = self.clock()
            if self.sup.armed and client != self.owner:
                raise PermissionError(f"armed by another client ({self.owner}); this one may only STOP")
            age = self.loop_age(now)
            if self.sup.armed and (age is None or not (0.0 <= age <= self.cfg.tick_deadline_s)):
                raise ControlLoopDown("no control loop: heartbeat refused")
            mode = payload.get("mode", "teleop")
            if mode not in ("teleop", "curvature", "waypoints", "ros"):
                raise ValueError(f"mode {mode!r}")
            if mode == "waypoints" and self.nav is None:
                raise ValueError("waypoints: no navigation module loaded")
            held = payload.get("held", False)
            if not isinstance(held, bool):
                raise ValueError("held must be true or false")
            if mode == "teleop":
                tgt = self.lim.validate(payload.get("vx", 0.0), payload.get("vy", 0.0), payload.get("wz", 0.0))
            elif mode == "curvature":
                tgt = self.lim.validate_curvature(payload.get("v", 0.0), payload.get("kappa", 0.0))
            else:  # waypoints / ros: the control loop recomputes the target every tick (and zeroes it unless held)
                tgt = ZERO
            if mode != self.mode:
                self._log_event("input_mode", {"from": self.mode, "to": mode})
                self.mode = mode
            applied = self.sup.heartbeat(now, tgt, mode, held=held, validated=True)
            return {"target": list(applied), "state": self.sup.state, "held": self.sup.held,
                    "need_neutral": self.sup.status(now)["need_neutral"]}

    def _source_target(self, now) -> tuple:
        if self.mode == "ros":
            if self._external is None or now - self._external[0] > self.cfg.external_cmd_timeout_s:
                return (0.0, 0.0, 0.0)
            return self._external[1]
        if self.mode == "waypoints" and self.nav is not None and self.state is not None:
            c = self.nav.command(self.state)
            return (0.0, 0.0, 0.0) if c is None else tuple(c)
        return (0.0, 0.0, 0.0)

    def set_external_cmd(self, vx: float, vy: float, wz: float) -> None:
        """ROS /cmd_vel (or another planner). Used only in mode 'ros', and only while the UI heartbeat keeps arriving
        with the HOLD input pressed. A non-finite value raises and clears the external command (zero)."""
        with self.lock:
            try:
                cmd = tuple(finite_number(float(x) if isinstance(x, int) and not isinstance(x, bool) else x, k)
                            for x, k in ((vx, "vx"), (vy, "vy"), (wz, "wz")))
            except ValueError:
                self._external = None
                raise
            self._external = (self.clock(), cmd)

    def request_arm(self) -> dict:
        with self._locked():
            now = self.clock()
            code = self.sup.request_arm(now, self.controller_name)
            return {"code": code, "expires_in_s": self.lim.arm_challenge_ttl_s, "type_this": f"ARM {code}"}

    def confirm_arm(self, text: str, client: str = "local") -> tuple[bool, str]:
        with self._locked():
            now = self.clock()
            checks = self.preflight(now)
            failed = [c.name for c in checks if c.required and c.status != "ok"]
            ok, why = self.sup.confirm_arm(now, text, self.controller_name, not failed, failed)
            if not ok:
                return ok, why
            self.owner = client
            try:
                self.controller.on_arm(now, self.state)
            except Exception as e:  # noqa: BLE001
                self.sup.disarm(now, f"on_arm_failed: {type(e).__name__}: {e}")
                return False, f"controller refused to arm: {e}"
            return True, "armed"

    def disarm(self, reason: str = "operator") -> bool:
        if not self.lock.acquire(timeout=0.1):  # loop hung: act on the robot side at once, like STOP
            self.stop(f"disarm:{reason}")
            return True
        try:
            return self.sup.disarm(self.clock(), reason)
        finally:
            self.lock.release()

    def stop(self, source: str = "ui") -> bool:
        """Always acts. Returns True when the STOP went through the supervisor (latched, controller safe state). If the
        control thread is hung inside the lock, the controller's lock-free emergency_stop runs at once, the STOP is
        applied when the loop runs again, and this returns False (the UI then says: not confirmed, use the remote)."""
        if self.lock.acquire(timeout=0.1):
            try:
                self.sup.stop(self.clock(), source)
                return True
            finally:
                self.lock.release()
        now = self.clock()
        try:
            if self.controller is not None:
                self.controller.emergency_stop(now)
        finally:
            self.fault = self.fault or f"stop_while_loop_hung:{source}"
            self._pending_trip = f"stop_while_loop_hung:{source}"
            self._events.append({"t": now, "event": "stop_without_lock", "source": source})
        return False

    def reset_stop(self) -> None:
        """Clears the STOP latch and a latched fault (an operator action; only while disarmed)."""
        with self.lock:
            if self.sup.armed:
                return
            if self.fault is not None:
                self._safe_log("fault_reset", {"fault": self.fault, "detail": self.fault_detail})
            self.fault, self.fault_detail, self._pending_trip = None, None, None
            self.sup.reset_stop(self.clock())

    def damp(self) -> str:
        with self.lock:
            now = self.clock()
            self.sup.stop(now, "damp")
            r = self.controller.damp(now)
            self._log_event("damp", {"controller": self.controller_name, "result": r})
            return r

    def release_vendor(self, text: str) -> tuple[bool, str]:
        """HOLD STEP. Typed confirmation 'RELEASE VENDOR MODE'. Only disarmed, only with the lowcmd path."""
        with self.lock:
            if self.sup.armed:
                return False, "disarm first"
            if text.strip() != "RELEASE VENDOR MODE":
                self._log_event("release_refused", {"why": "typed confirmation does not match"})
                return False, "type exactly: RELEASE VENDOR MODE"
            if not getattr(self.io, "lowcmd_enabled", False):
                return False, "low-level path not enabled (start with --allow-lowcmd)"
            ok = self.io.release_vendor_mode()
            self._log_event("vendor_mode_released" if ok else "vendor_release_failed", {})
            return ok, "released: the vendor controller no longer balances the robot" if ok else "release failed"

    # ---------------------------------------------------------------- loop
    def _trip(self, reason: str, exc: BaseException | None = None, now: float | None = None) -> None:
        """Latch a fault and STOP (controller safe state first, then the log, best effort). Never raises."""
        now = self.clock() if now is None else now
        first = self.fault is None
        self.fault = self.fault or reason
        if exc is not None:
            self.fault_detail = f"{type(exc).__name__}: {exc}"
        try:
            if self.sup.armed or not self.sup.stop_latched:
                self.sup.stop(now, f"fault:{reason}")  # -> _on_transition -> controller.stop (zero / damping)
        except Exception as e:  # noqa: BLE001 - the supervisor itself failed: force the robot side
            try:
                self.sup.state = "disarmed"
                self.sup.stop_latched = True
                if self.controller is not None:
                    self.controller.emergency_stop(now)
            except Exception:  # noqa: BLE001
                pass
            self.fault_detail = f"{self.fault_detail}; stop failed: {type(e).__name__}: {e}"
        if first or exc is not None and self.tick_errors <= 3:
            self._safe_log("fault", {"reason": reason, "error": self.fault_detail or ""})

    def tick(self) -> None:
        """One control tick. UI-01: an exception never leaves this method; it latches a fault and STOP."""
        with self.lock:
            try:
                self._tick()
            except Exception as e:  # noqa: BLE001
                self.tick_errors += 1
                self._trip(f"tick_exception:{type(e).__name__}", e)
                if self.tick_errors <= 3:
                    print(f"g1_runtime: tick exception (fault latched): {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            else:
                self._last_tick_ok = self.clock()

    def _runtime_fault(self, now) -> str | None:
        """Faults that end an armed run (latched). Fail closed on what the adapter reports."""
        if self._pending_trip:
            return self._pending_trip
        ex = self.meta.extra if self.meta is not None else {}
        temp = ex.get("motor_temp_max_c")
        if temp is not None and not (temp < self.cfg.motor_temp_max_c):
            return f"motor_temp:{temp:.0f}C"
        if ex.get("motor_error_any"):
            return "motor_error"  # T-MOTERR; meaning of the bits UNKNOWN: any bit set stops the run
        if self._tick_seen is not None and self._tick_seen[1] >= self.cfg.tick_frozen_samples:
            return "lowstate_tick_frozen"
        dc = self._disk_cache
        if dc is not None and dc[1] is not None and dc[1] / 1e9 < self.cfg.min_free_disk_gb:
            return "disk_low"
        return None

    def _tick(self) -> None:
        now = self.clock()
        self.ticks += 1
        if self._tick_times:
            self.max_period = max(self.max_period, now - self._tick_times[-1])
        self._tick_times.append(now)
        try:
            got = self.io.read_state()
        except Exception as e:  # noqa: BLE001 - a bad sample (mode_pr != 0, NaN, short array) is "no state", not a crash
            got = None
            self._safe_log("read_state_error", {"error": f"{type(e).__name__}: {e}"})
        if got is not None:
            st, meta = got
            new_sample = st.stamp != self._last_state_stamp
            self.state, self.meta = st, meta
            if new_sample:
                self._state_times.append(now)
                self._last_state_stamp = st.stamp
                tk = meta.extra.get("tick") if meta is not None else None
                if tk is not None:
                    prev = self._tick_seen
                    self._tick_seen = (tk, prev[1] + 1 if prev is not None and prev[0] == tk else 0)
            self.state_age = now - st.stamp
            self.runlog.state(st, meta)
            if self.sink is not None:
                self.sink.state(st, meta)
        else:
            self.state_age = None if self.state is None else now - self.state.stamp
        if self.state_age is not None and self.state_age < -self.cfg.future_stamp_tolerance_s:
            self.state_age = math.inf  # SR-08: a future stamp is a clock fault, never a fresh state
        elif self.state_age is not None and self.state_age < 0.0:
            self.state_age = 0.0  # a stamp taken microseconds after `now` (same clock): not a fault
        if self.cameras and now - self._last_cam >= 1.0 / self.cfg.camera_hz:
            self._last_cam = now
            for cam in self.cameras:
                try:
                    fr = cam.read()
                except Exception as e:  # noqa: BLE001
                    fr = None
                    self._safe_log("camera_error", {"name": cam.name, "error": str(e)})
                if fr is not None:
                    self.frames[cam.name] = fr
                    if self.sink is not None:
                        self.sink.frame(fr)
        tilt = self._tilt()
        if self.sup.armed:
            rf = self._runtime_fault(now)
            if rf:
                self._pending_trip = None
                self._trip(rf, now=now)
        if self.mode in ("ros", "waypoints") and self.sup.armed:
            self.sup.set_target(self._source_target(now), self.mode, now=now)
        self.sup.check(now, self.state_age, tilt, self.controller.fault() if self.controller else "no controller")
        cmd = self.sup.command(self.dt)
        if self.sup.armed:
            base, events = self.safety.filter_base(BaseVelocity(*cmd))
            for e in events:
                self._log_event("s1_" + e.rule, {"subject": e.subject, "requested": e.requested, "applied": e.applied})
            cmd = (base.vx, base.vy, base.wz)
            cmd = self._obstacle_stage(now, cmd)  # tightening only; a stop is an explicit zero step (streaming continues)
            self.last_cmd = cmd
            info = self.controller.step(now, self.dt, self.state, cmd)
            self.runlog.command({"t": now, "controller": self.controller_name, "target": list(self.sup.target),
                                 "cmd": list(cmd), "source": self.sup.source, "held": self.sup.held,
                                 "obstacle": self.obstacle_info, **(info or {})})
        else:
            self.last_cmd = cmd
            self.controller.idle(now, self.state)
        if self.shadow is not None and self.state is not None and self.state_age is not None and 0 <= self.state_age <= self.lim.state_stale_s:
            try:
                self.last_shadow = self.shadow.step(self.state, cmd)
            except Exception as e:  # noqa: BLE001 - the shadow never commands; its failure is not a robot fault
                self.last_shadow = {"error": str(e)}
        if now - self._last_plot >= 0.1 - 1e-9:
            self._last_plot = now
            self._plot.append(self._plot_row(now, tilt))

    def _obstacle_stage(self, now: float, cmd: tuple) -> tuple:
        try:
            res = self.obstacle.apply(cmd)
        except Exception as e:  # noqa: BLE001 - a broken stage fails closed
            from humanoid_arc_bme.onboard.nav.obstacle_stage import StageResult
            res = StageResult((0.0, 0.0, 0.0), 0.0, "bad_reading", None, 0.0, 0, f"stage error: {type(e).__name__}")
        self.obstacle_info = res.to_dict()
        key = (res.reason, res.warning, res.scale < 1.0)
        if key != self._obstacle_key:  # log state changes only, not every tick
            self._obstacle_key = key
            self._safe_log("obstacle_cap" if res.scale < 1.0 else "obstacle_state", {**self.obstacle_info, "requested": list(cmd)})
        out = res.cmd
        # defence in depth: whatever the stage returned, never exceed the input magnitude or flip a sign
        if not all(math.isfinite(o) and abs(o) <= abs(c) + 1e-12 and o * c >= 0 for o, c in zip(out, cmd)):
            out = (0.0, 0.0, 0.0)
        return tuple(out)

    # ---------------------------------------------------------------- the independent supervisor (UI-01)
    def supervise(self, now: float | None = None) -> str | None:
        """Runs in its own thread (start()), independent of the control thread; tests call it directly.
        Refreshes the free-disk cache, and while armed disarms when the control loop is dead or stalled."""
        now = self.clock() if now is None else now
        dc = self._disk_cache
        if dc is None or not (0.0 <= now - dc[0] < 2.0):
            try:
                self._disk_cache = (now, self._disk_free(str(self.runlog.dir)), None)
            except OSError as e:
                self._disk_cache = (now, None, str(e))
        if not self.sup.armed:
            return None
        dead = self._thread is not None and not self._thread.is_alive()
        age = self.loop_age(now)
        if not (dead or age is None or not (0.0 <= age <= self.cfg.tick_deadline_s)):
            return None
        reason = "control_thread_dead" if dead else f"control_loop_stalled:{'never' if age is None else f'{age:.3f}s'}"
        c = self.controller
        try:
            if c is not None:
                c.emergency_stop(now)  # robot side first, without the runtime lock (the control thread may hold it)
        except Exception as e:  # noqa: BLE001
            self.fault_detail = f"emergency_stop failed: {type(e).__name__}: {e}"
        if self.lock.acquire(timeout=0.05):
            try:
                self._trip(f"watchdog:{reason}", now=now)
            finally:
                self.lock.release()
        else:  # the control thread is stuck inside a tick: it applies the trip when (if) it runs again
            self.fault = self.fault or f"watchdog:{reason}"
            self._pending_trip = f"watchdog:{reason}"
            self._events.append({"t": now, "event": "watchdog_trip_pending", "reason": reason})
        return reason

    def _supervise_loop(self) -> None:
        while not self._stop_evt.wait(0.02):
            try:
                self.supervise()
            except Exception as e:  # noqa: BLE001 - the watchdog must survive
                self._events.append({"t": self.clock(), "event": "supervisor_error", "error": f"{type(e).__name__}: {e}"})

    def _tilt(self):
        return None if self.state is None else tilt_deg_from_quat(self.state.imu.quat)

    def _plot_row(self, now, tilt) -> dict:
        st = self.state
        return {"t": round(now - self._tick_times[0] if self._tick_times else 0.0, 3), "wall": self.wall(),
                "vx_target": self.sup.target[0], "vx": self.last_cmd[0], "vy": self.last_cmd[1], "wz": self.last_cmd[2],
                "state_age_ms": None if self.state_age is None else 1000 * self.state_age, "tilt_deg": tilt,
                "gyro_z": None if st is None else st.imu.gyro[2], "loop_hz": self._rate(self._tick_times),
                "armed": self.sup.armed, "obstacle_scale": (self.obstacle_info or {}).get("scale"), "temp_max_c": (self.meta.extra.get("motor_temp_max_c") if self.meta else None),
                "shadow_err": None if not self.last_shadow else self.last_shadow.get("max_target_minus_measured")}

    def _rate(self, times) -> float:
        times = tuple(times)  # a C-level copy: safe while the control thread appends (status() takes no lock)
        if len(times) < 2:
            return 0.0
        now = self.clock()
        return float(sum(1 for t in times if now - t <= 1.0))

    def run(self, seconds: float) -> None:
        """Blocking loop for `seconds` (tests / CLI). The server uses start()/close()."""
        end = self.clock() + seconds
        while self.clock() < end and not self._stop_evt.is_set():
            t0 = self.clock()
            self.tick()
            self.sleep(max(0.0, self.dt - (self.clock() - t0)))

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run, args=(float("inf"),), name="g1-control", daemon=True)
        self._thread.start()
        self._sup_thread = threading.Thread(target=self._supervise_loop, name="g1-supervisor", daemon=True)
        self._sup_thread.start()

    def close(self) -> None:
        got = self.lock.acquire(timeout=1.0)
        try:
            if self.sup.armed:
                self.sup.disarm(self.clock(), "shutdown")
        finally:
            if got:
                self.lock.release()
        if self.sup.armed and self.controller is not None:  # could not take the lock: the loop is hung
            self.controller.emergency_stop(self.clock())
        self._stop_evt.set()
        for t in (self._thread, self._sup_thread):
            if t:
                t.join(timeout=2.0)
        for c in self._built.values():
            c.close()
        for cam in self.cameras:
            cam.close()
        self.io.close()
        if self.sink is not None:
            self.sink.close()
        self.runlog.close()

    # ---------------------------------------------------------------- views
    def preflight(self, now: float | None = None) -> list[Check]:
        """Lock-free (UI-07): reads attributes the control thread updates; no syscall except a stale disk cache."""
        if True:
            now = self.clock() if now is None else now
            L = self.lim
            out = []
            out.append(Check("stop_not_latched", "fail" if self.sup.stop_latched else "ok",
                             "STOP is latched: press Reset STOP" if self.sup.stop_latched else "no latched STOP"))
            out.append(Check("no_fault", "fail" if self.fault else "ok",
                             f"latched fault {self.fault} ({self.fault_detail or ''}): fix the cause, then Reset STOP"
                             if self.fault else "no latched fault"))
            if self.state_age is None:
                out.append(Check("state_fresh", "fail", "no robot state received"))
            else:
                out.append(Check("state_fresh", "ok" if self.state_age <= L.state_stale_s else "fail",
                                 f"state age {1000 * self.state_age:.0f} ms (limit {1000 * L.state_stale_s:.0f} ms)"))
            sr = self._rate(self._state_times)
            out.append(Check("state_rate", "ok" if sr >= 0.8 * self.cfg.control_hz else "fail",
                             f"{sr:.0f} new states/s at the loop (need >= {0.8 * self.cfg.control_hz:.0f})"))
            lr = self._rate(self._tick_times)
            out.append(Check("loop_rate", "ok" if lr >= 0.9 * self.cfg.control_hz else "fail",
                             f"{lr:.0f} Hz (need >= {0.9 * self.cfg.control_hz:.0f}); max period {1000 * self.max_period:.0f} ms"))
            tilt = self._tilt()
            out.append(Check("upright", "unknown" if tilt is None else ("ok" if tilt <= L.max_tilt_deg else "fail"),
                             "no IMU yet" if tilt is None else f"pelvis tilt {tilt:.1f} deg (limit {L.max_tilt_deg:.0f})"))
            meta = self.meta
            ex = meta.extra if meta is not None else {}
            temp = ex.get("motor_temp_max_c")
            if temp is None:
                out.append(Check("motor_temp", "unknown", "not in this state source", required=False))
            else:
                out.append(Check("motor_temp", "ok" if temp < self.cfg.motor_temp_max_c else "fail",
                                 f"max {temp:.0f} C (limit {self.cfg.motor_temp_max_c:.0f}; checked every tick while armed)"))
            if "motor_error" in ex:
                bad = [i for i, b in enumerate(ex["motor_error"]) if b]
                out.append(Check("motor_error", "fail" if bad else "ok",
                                 f"motorstate != 0 on motors {bad} (bit meaning UNKNOWN: fail closed)" if bad else "all motorstate 0"))
            else:
                out.append(Check("motor_error", "unknown", "not in this state source", required=False))
            out.append(Check("battery", "unknown", "not read by this runtime (not in rt/lowstate); check the vendor app/remote",
                             required=False))
            dc = self._disk_cache
            if dc is None or not (0.0 <= now - dc[0] <= 5.0):
                try:
                    dc = (now, self._disk_free(str(self.runlog.dir)), None)
                except OSError as e:
                    dc = (now, None, str(e))
                self._disk_cache = dc
            if dc[1] is None:
                out.append(Check("disk", "fail", dc[2] or "unknown"))
            else:
                free = dc[1] / 1e9
                out.append(Check("disk", "ok" if free >= self.cfg.min_free_disk_gb else "fail",
                                 f"{free:.1f} GB free in {self.runlog.dir} (need {self.cfg.min_free_disk_gb})"))
            la = self.loop_age(now)
            out.append(Check("control_loop", "ok" if la is not None and 0.0 <= la <= self.cfg.tick_deadline_s else "fail",
                             "no tick yet" if la is None else f"last good tick {1000 * la:.0f} ms ago "
                             f"(limit {1000 * self.cfg.tick_deadline_s:.0f} ms; {self.tick_errors} tick errors)"))
            out.append(Check("run_log", "ok" if not self.runlog.closed else "fail", str(self.runlog.dir)))
            yr = time.gmtime(self.wall()).tm_year
            out.append(Check("wall_clock", "ok" if yr >= 2026 else "fail", f"year {yr} (log timestamps)", required=False))
            if self.controller is None:
                out.append(Check("controller", "fail", self._select_error or "none selected"))
            else:
                ok, why = self.controller.ready()
                out.append(Check("controller", "ok" if ok else "fail", f"{self.controller_name}: {why}"))
            out.append(Check("limits_validated", "ok" if self.safety.limits.validated_on_robot else "unknown",
                             "S1 limits validated on the robot" if self.safety.limits.validated_on_robot
                             else "placeholders (B3.3); required only for the policy controller on the robot", required=False))
            hb = self.sup.heartbeat_rate(now)
            out.append(Check("ui_heartbeat", "ok" if hb >= L.heartbeat_min_hz else "fail",
                             f"{hb:.0f} Hz (need >= {L.heartbeat_min_hz:.0f}); checked again at arming", required=False))
            return out

    def status(self) -> dict:
        """Lock-free snapshot (UI-07): never waits for the control loop."""
        if True:
            now = self.clock()
            checks = self.preflight(now)
            green = all(c.status == "ok" for c in checks if c.required)
            st = self.state
            return {
                "supervisor": self.sup.status(now), "owner": self.owner if self.sup.armed else None, "controller": self.controller_name, "controllers": self.controller_names(),
                "controller_status": self.controller.status() if self.controller else None, "mode": self.mode,
                "cmd": list(self.last_cmd), "preflight_green": green, "preflight": [c.to_dict() for c in checks],
                "state_age_ms": None if self.state_age is None else 1000 * self.state_age,
                "state_hz": self._rate(self._state_times), "loop_hz": self._rate(self._tick_times),
                "tilt_deg": self._tilt(), "imu": None if st is None else st.imu.to_dict(),
                "meta": None if self.meta is None else {"mode_machine": self.meta.mode_machine, **self.meta.extra},
                "limits": self.lim.to_dict(), "s1_base_vel_max": list(self.safety.limits.base_vel_max),
                "cameras": sorted(self.frames), "shadow": None if self.last_shadow is None else
                {k: v for k, v in self.last_shadow.items() if k != "targets"},
                "nav": self._nav_status(), "obstacle": self.obstacle_info,
                "fault": self.fault, "fault_detail": self.fault_detail, "tick_errors": self.tick_errors,
                "loop_age_ms": None if self.loop_age(now) is None else 1000 * self.loop_age(now),
                "lowcmd": {"enabled": bool(getattr(self.io, "lowcmd_enabled", False)), "vendor_released": bool(getattr(self.io, "vendor_released", False))},
                "events": list(tuple(self._events))[-30:], "run_dir": str(self.runlog.dir), "uptime_s": self.wall() - self.started_wall,
                "pid": os.getpid(),
            }

    def _nav_status(self):
        if self.nav is None:
            return None
        try:
            return self.nav.status()
        except Exception as e:  # noqa: BLE001 - read without the lock; a concurrent update is not worth a 500
            return {"error": f"{type(e).__name__}: {e}"}

    def plots(self, since_wall: float = 0.0) -> list[dict]:
        return [r for r in tuple(self._plot) if r["wall"] > since_wall]
