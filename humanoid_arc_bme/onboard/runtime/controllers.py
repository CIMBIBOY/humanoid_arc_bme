"""Controllers behind one interface: what turns the operator's (vx, vy, wz) into robot commands.

- `FakeController`: records calls (tests, UI dry runs).
- `VendorLocoController`: the Unitree high-level locomotion service through unitree_sdk2py's `LocoClient` (service
  "sport"); the vendor controller keeps the robot upright and walks at the commanded velocity.
  API (OPENED 2026-10-30, unitree_sdk2py/g1/loco/g1_loco_client.py + g1_loco_api.py + example/g1/high_level/
  g1_loco_client_example.py): `LocoClient()`, `.SetTimeout(s)`, `.Init()`; `SetVelocity(vx, vy, omega, duration=1.0)
  -> code` (API 7105, JSON {"velocity": [vx, vy, omega], "duration": d}); `Move(vx, vy, vyaw, continous_move=False)` =
  SetVelocity with duration 1 s (864000 s if continuous); `StopMove()` = SetVelocity(0, 0, 0); `Damp()` = SetFsmId(1);
  `GetFsmId() -> (code, id)`; also Start (FSM 500), Squat2StandUp (706), Sit (3), ZeroTorque (0), which this runtime
  never calls. We call SetVelocity with a SHORT duration (cmd_duration_s, default 0.4 s) at send_hz, so the vendor
  stops by itself if this process dies; disarm and STOP send SetVelocity(0, 0, 0). What the FSM ids mean on this
  robot's firmware and whether "sport" accepts velocity in the current mode: UNKNOWN until the robot (B4.0/B4.1).
- `PolicyController`: our exported ONNX walking policy (humanoid_arc_bme.onboard.runtime.policies.OnnxLegPolicy: obs builder from
  layout.json) -> S1 `filter_joints` -> `RobotIO.send_joint_targets`. On a real robot it needs the gated rt/lowcmd
  path (io/unitree.py) and validated S1 limits; `ready()` says what is missing.
- `ShadowPolicy`: the same ONNX policy run on the real state while another controller (the vendor) walks; logs what it
  would command, sends nothing.
"""

from __future__ import annotations

import math
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

from humanoid_arc_bme.onboard.api.safety import SafetyLayer, clamp_height


class Controller(ABC):
    name: str = "controller"
    moves_robot: bool = True

    def ready(self) -> tuple[bool, str]:
        return True, "ok"

    def on_arm(self, now: float, state) -> None:  # noqa: B027
        pass

    @abstractmethod
    def step(self, now: float, dt: float, state, cmd: tuple) -> dict:
        """One armed tick with the ramped, clamped (vx, vy, wz). Returns info for the log."""

    @abstractmethod
    def on_disarm(self, now: float, reason: str) -> None:
        """Bring the robot to this controller's safe state (called on every disarm: deadman, stale, operator...)."""

    def stop(self, now: float) -> None:
        """The STOP button. Default: the same as a disarm."""
        self.on_disarm(now, "stop")

    def emergency_stop(self, now: float) -> None:
        """Called by the runtime's supervisor thread when the control loop is dead or hung (UI-01), possibly while the
        control thread still holds the runtime lock. Must not need that lock. Default: the same as STOP."""
        self.stop(now)

    def damp(self, now: float) -> str:
        return "not supported by this controller"

    def idle(self, now: float, state) -> None:  # noqa: B027
        """Every tick while disarmed."""

    def fault(self) -> str | None:
        return None

    def status(self) -> dict:
        return {"name": self.name}

    def close(self) -> None:  # noqa: B027
        pass


class FakeController(Controller):
    name = "fake"
    moves_robot = False

    def __init__(self):
        self.calls: list = []
        self.fault_text: str | None = None

    def on_arm(self, now, state):
        self.calls.append(("arm", now))

    def step(self, now, dt, state, cmd):
        self.calls.append(("step", now, tuple(cmd)))
        return {}

    def on_disarm(self, now, reason):
        self.calls.append(("disarm", now, reason))

    def stop(self, now):
        self.calls.append(("stop", now))

    def emergency_stop(self, now):
        self.calls.append(("emergency_stop", now))

    def damp(self, now):
        self.calls.append(("damp", now))
        return "ok"

    def fault(self):
        return self.fault_text

    def status(self):
        steps = [c for c in self.calls if c[0] == "step"]
        return {"name": self.name, "steps": len(steps), "last": list(steps[-1][2]) if steps else None}


class FakeLocoClient:
    """Stands in for unitree_sdk2py's LocoClient in tests and in `serve --io fake --controller vendor` UI dry runs."""

    def __init__(self, codes=None):
        self.calls: list = []
        self.codes = list(codes or [])
        self.fsm = 500

    def SetVelocity(self, vx, vy, omega, duration=1.0):  # noqa: N802 - vendor API names
        self.calls.append(("SetVelocity", vx, vy, omega, duration))
        return self.codes.pop(0) if self.codes else 0

    def Damp(self):  # noqa: N802
        self.calls.append(("Damp",))
        self.fsm = 1

    def GetFsmId(self):  # noqa: N802
        return 0, self.fsm


def make_loco_client(rpc_timeout_s: float):
    """The real client. ChannelFactoryInitialize must already have run (UnitreeG1IO does it). Init only registers the
    RPC API ids: it sends no motion request."""
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    c = LocoClient()
    c.SetTimeout(rpc_timeout_s)
    c.Init()
    return c


class VendorLocoController(Controller):
    name = "vendor"

    def __init__(self, client, send_hz: float = 10.0, cmd_duration_s: float = 0.4, max_consecutive_errors: int = 3,
                 clock=time.monotonic, threaded: bool = True, step_deadline_s: float = 0.15):
        if not (send_hz > 0 and 0 < cmd_duration_s <= 1.0):
            raise ValueError("send_hz > 0 and 0 < cmd_duration_s <= 1 s (a long duration defeats the vendor-side stop)")
        if cmd_duration_s < 2.0 / send_hz:
            raise ValueError("cmd_duration_s must cover at least two send periods, or walking stutters")
        if not (0 < step_deadline_s <= cmd_duration_s):
            raise ValueError("step_deadline_s must be positive and not longer than cmd_duration_s")
        self.client, self.period, self.duration, self.max_err = client, 1.0 / send_hz, cmd_duration_s, max_consecutive_errors
        self.clock = clock
        self.step_deadline = float(step_deadline_s)  # UI-02: the sender's own deadline on the control loop's step()
        self._lock = threading.Lock()  # every client call, so a zero can never be overtaken by an older velocity
        self._latest: tuple | None = None  # the command to stream while armed; None = send nothing
        self._last_send = -math.inf
        self._last_step = -math.inf  # time (runtime clock) of the last step(): no fresh step, no renewal
        self.errors = 0
        self.last_code: int | None = None
        self.last_rpc_ms: float | None = None
        self.sent = 0
        self.fsm_id = None
        self._last_fsm_poll = -math.inf
        self._fault: str | None = None
        self._stop_evt = threading.Event()
        self._thread = None
        if threaded:
            self._thread = threading.Thread(target=self._run, name="vendor-sender", daemon=True)
            self._thread.start()

    def _call(self, fn, *args):
        t0 = time.perf_counter()
        try:
            code = fn(*args)
        except Exception as e:  # noqa: BLE001 - an SDK timeout or DDS error is a fault, never a crash of the loop
            code = f"{type(e).__name__}: {e}"
        self.last_rpc_ms = 1000 * (time.perf_counter() - t0)
        return code

    def _zero(self) -> None:
        code = self._call(self.client.SetVelocity, 0.0, 0.0, 0.0, 1.0)
        self.last_code = code

    def pump(self, now: float) -> None:
        """Send the latest command if a send period has passed. Called by the sender thread (or by tests)."""
        with self._lock:
            if self._latest is None:
                if now - self._last_fsm_poll >= 2.0 and hasattr(self.client, "GetFsmId"):
                    self._last_fsm_poll = now
                    r = self._call(self.client.GetFsmId)
                    self.fsm_id = r[1] if isinstance(r, tuple) and len(r) == 2 and r[0] == 0 else None
                return
            if not (0.0 <= now - self._last_step <= self.step_deadline):  # UI-02: loop stalled or dead (or clock fault)
                self._latest = None  # stop renewing the vendor's duration; the zero below is the last velocity call
                self._fault = f"control loop silent for {now - self._last_step:.3f} s (deadline {self.step_deadline} s)"
                self._zero()
                return
            if now - self._last_send < self.period - 1e-9:
                return
            vx, vy, wz = self._latest
            code = self._call(self.client.SetVelocity, float(vx), float(vy), float(wz), self.duration)
            self._last_send, self.last_code = now, code
            self.sent += 1
            self.errors = 0 if code == 0 else self.errors + 1
            if self.errors >= self.max_err:
                self._fault = f"SetVelocity failed {self.errors}x (last {code})"

    def _run(self):
        while not self._stop_evt.wait(min(self.period / 4, self.step_deadline / 3)):
            try:
                self.pump(self.clock())
            except Exception as e:  # noqa: BLE001 - the sender must survive; stop streaming and try a zero
                self._latest = None
                self._fault = f"sender error: {type(e).__name__}: {e}"
                try:
                    with self._lock:
                        self._zero()
                except Exception:  # noqa: BLE001
                    pass

    def ready(self):
        if self._fault:
            return False, self._fault
        return True, "LocoClient initialised"

    def on_arm(self, now, state):
        with self._lock:
            self._fault, self.errors = None, 0
            self._latest = (0.0, 0.0, 0.0)
            self._last_send = -math.inf
            self._last_step = now

    def step(self, now, dt, state, cmd):
        cmd = tuple(float(x) for x in cmd)
        if not all(math.isfinite(x) for x in cmd):
            raise ValueError("vendor step: non-finite command")
        with self._lock:
            if self._latest is not None:
                self._latest = cmd
                self._last_step = now
        if self._thread is None:
            self.pump(now)
        return {"sent": self.sent, "last_code": self.last_code}

    def on_disarm(self, now, reason):
        self._latest = None  # at once, without the lock: a pump in an RPC sends at most its old value, then the zero
        with self._lock:
            self._latest = None
            self._zero()

    def stop(self, now):
        self.on_disarm(now, "stop")

    def emergency_stop(self, now):
        self.on_disarm(now, "emergency")

    def damp(self, now):
        """Vendor damping (FSM 1): the robot goes limp. Only safe with the robot supported (gantry / rope)."""
        with self._lock:
            self._latest = None
            r = self._call(self.client.Damp)
        return f"Damp sent ({r})"

    def fault(self):
        return self._fault

    def status(self):
        return {"name": self.name, "sent": self.sent, "last_code": self.last_code, "errors": self.errors,
                "last_rpc_ms": self.last_rpc_ms, "fsm_id": self.fsm_id, "streaming": self._latest is not None,
                "cmd_duration_s": self.duration, "step_deadline_s": self.step_deadline, "fault": self._fault}

    def close(self):
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=1.0)


def list_policies(models_root) -> list[Path]:
    """Model directories (policy.onnx + layout.json) under models_root, or models_root itself if it is one."""
    root = Path(models_root).expanduser()
    if (root / "layout.json").is_file() and (root / "policy.onnx").is_file():
        return [root]
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if (p / "layout.json").is_file() and (p / "policy.onnx").is_file())


class PolicyController(Controller):
    """ONNX leg policy -> S1 filter_joints -> io.send_joint_targets (with the layout's kp/kd)."""

    def __init__(self, model_dir, io, safety: SafetyLayer, height: float = 0.78, session=None, tracking_abort_ticks: int = 5):
        from .policies import OnnxLegPolicy  # numpy; onnxruntime only when no session is injected

        self.model_dir = Path(model_dir)
        self.name = f"policy:{self.model_dir.name}"
        h, _ev = clamp_height(height, safety.limits)  # SR-06: finite, inside base_height when the limits file sets it
        if _ev:
            raise ValueError(f"policy height {height} outside the limits file's base_height range")
        self.io, self.safety, self.height = io, safety, h
        self.tracking_abort_ticks = int(tracking_abort_ticks)  # T-TRK: this many ticks in a row with the S1 clamp active
        self._trk_run = 0
        self._cmd = (0.0, 0.0, 0.0)
        self.policy = OnnxLegPolicy(self.model_dir, command=lambda: (*self._cmd, self.height), session=session)
        lay = __import__("json").loads((self.model_dir / "layout.json").read_text())
        self.kp, self.kd = lay.get("action_kp"), lay.get("action_kd")
        self.s1_events = 0
        self.sent = 0
        self.last_infer_ms: float | None = None
        self._fault: str | None = None

    def ready(self):
        if self._fault:
            return False, self._fault
        if getattr(self.io, "simulated", False):
            return True, "simulated robot"
        if not self.safety.limits.validated_on_robot:
            return False, "S1 limits are placeholders (config/g1_limits.yaml validated_on_robot: false, step B3.3)"
        if not getattr(self.io, "lowcmd_enabled", False):
            return False, "low-level command path not enabled (--allow-lowcmd)"
        if not getattr(self.io, "vendor_released", False):
            return False, "vendor motion mode not released (UI: release, Hold step)"
        return True, "lowcmd ready"

    def on_arm(self, now, state):
        self.policy.reset()
        self.safety.reset_joint_reference()  # SR-01: the first target is limited against the measured position
        self._trk_run = 0
        self.safety.command_received(now)
        if hasattr(self.io, "begin_commanding"):
            self.io.begin_commanding(state)

    def step(self, now, dt, state, cmd):
        self._cmd = tuple(float(x) for x in cmd)
        t0 = time.perf_counter()
        try:
            jt = self.policy.act(state)  # a NaN/inf action raises in JointTargets (T-NAN)
        except Exception as e:
            self._fault = f"policy: {type(e).__name__}: {e}"
            self.on_disarm(now, "policy_error")  # damping at once; the runtime latches the fault (UI-01)
            raise
        self.last_infer_ms = 1000 * (time.perf_counter() - t0)
        self.safety.command_received(now)
        out, events = self.safety.filter_joints(now, dt, state, jt)  # SR-02: a gap > the watchdog resets the reference
        self.s1_events += len(events)
        if any(e.rule == "tracking_error" for e in events):  # T-TRK (SR-03): the PD error hit the S1 bound
            self._trk_run += 1
            if self._trk_run >= self.tracking_abort_ticks:
                self._fault = f"T-TRK: tracking-error bound active {self._trk_run} ticks in a row"
        else:
            self._trk_run = 0
        try:
            self.io.send_joint_targets(out.names, out.positions, self.kp, self.kd)
            self.sent += 1
        except Exception as e:  # noqa: BLE001 - refused or failed: fault, the supervisor disarms
            self._fault = f"send failed: {e}"
        return {"s1_events": len(events), "infer_ms": self.last_infer_ms}

    def on_disarm(self, now, reason):
        self.safety.reset_joint_reference()  # SR-01
        if hasattr(self.io, "end_commanding"):
            self.io.end_commanding()
        if hasattr(self.io, "damp"):
            self.io.damp()

    def idle(self, now, state):
        if getattr(self.io, "vendor_released", False) and hasattr(self.io, "damp"):
            self.io.damp()  # keep the damping stream while the vendor mode is released and we are disarmed

    def damp(self, now):
        ok = self.io.damp() if hasattr(self.io, "damp") else False
        return "damping command sent" if ok else "no damping path (vendor mode not released)"

    def fault(self):
        return self._fault

    def status(self):
        return {"name": self.name, "sent": self.sent, "s1_events": self.s1_events, "infer_ms": self.last_infer_ms,
                "height": self.height, "fault": self._fault, "model_dir": str(self.model_dir)}


class ShadowPolicy:
    """Runs the policy on the real state with the operator's command; logs targets, never sends."""

    def __init__(self, model_dir, height: float = 0.78, session=None):
        from .policies import OnnxLegPolicy

        self.model_dir = Path(model_dir)
        self._cmd = (0.0, 0.0, 0.0)
        self.height = float(height)
        self.policy = OnnxLegPolicy(self.model_dir, command=lambda: (*self._cmd, self.height), session=session)
        self.last: dict = {}

    def step(self, state, cmd) -> dict:
        self._cmd = tuple(float(x) for x in cmd)
        t0 = time.perf_counter()
        jt = self.policy.act(state)
        ms = 1000 * (time.perf_counter() - t0)
        pos = dict(zip(state.joint_names, state.joint_pos))
        err = max(abs(t - pos[n]) for n, t in zip(jt.names, jt.positions))
        self.last = {"infer_ms": ms, "max_target_minus_measured": err, "targets": list(jt.positions)}
        return self.last
