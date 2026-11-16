"""AutonomyLoop: runs the RobotStack beside the operator runtime and hands its commands to it.

    sensors -> SensorHub -> RobotStack.step -> base command -> runtime.set_external_cmd   (mode "ros")
                                            -> arm / hand targets -> controller.set_upper_body

The runtime stays in charge: nothing moves unless the operator armed it and keeps the HOLD input pressed (the external
command is honoured only then), S1 clamps the base command, the supervisor's deadman / tilt / stale-state trips stop
everything. Autonomy adds one more rule: a disarm, a STOP or a trip cancels the running task, and re-arming never resumes it.
"""

from __future__ import annotations

import threading
import time

from humanoid_arc_bme.onboard.stack import RobotStack

ZERO = (0.0, 0.0, 0.0)


class AutonomyLoop:
    def __init__(self, runtime, stack: RobotStack, hub, controller=None, hz: float = 20.0, clock=time.monotonic):
        if not hz >= 5.0:
            raise ValueError("hz >= 5")
        self.rt, self.stack, self.hub, self.ctrl = runtime, stack, hub, controller
        self.period, self.clock = 1.0 / hz, clock
        self._last = None
        self.lock = threading.Lock()          # shared with the command gateway: held while the stack steps
        self.latest = None                    # newest SensorFrame (for the gateway; read without the lock)
        self._armed_before = False
        self.last_error = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def say(self, text: str):
        """Parse and submit one sentence (the same path as the gateway's confirm). Needs a fresh frame."""
        f = self.hub.frame(self.clock(), self.rt.state)
        if f is None:
            return None
        with self.lock:
            return self.stack.say(text, f)

    def tick(self) -> None:
        """One cycle. Never raises: a fault cancels the task and zeroes the external command."""
        now = self.clock()
        dt = self.period if self._last is None else max(1e-3, min(0.5, now - self._last))
        self._last = now
        try:
            armed = bool(self.rt.sup.armed)
            if self._armed_before and not armed:
                with self.lock:
                    self.stack.stop("disarmed")                 # a disarm ends the task; re-arming does not resume it
            if armed and not self._armed_before and hasattr(self.stack, "reset_perception"):
                with self.lock:
                    self.stack.reset_perception()               # re-armed: nothing seen before the disarm is trusted
            self._armed_before = armed
            frame = self.hub.frame(now, self.rt.state, safe=armed and not self.rt.fault)
            self.latest = frame
            if frame is None or not armed:
                self._push(ZERO)
                return
            with self.lock:
                out = self.stack.step(frame, dt)
            self._push(out.base)
            ctrl = self.ctrl if self.ctrl is not None else getattr(self.rt, "controller", None)
            if ctrl is not None and hasattr(ctrl, "set_upper_body"):
                if out.arm is not None:
                    ctrl.set_upper_body(out.arm.body, out.arm.left, out.arm.right)
                else:
                    ctrl.release_upper_body()
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            try:
                with self.lock:
                    self.stack.stop("autonomy fault")
                self._push(ZERO)
            except Exception:  # noqa: BLE001
                pass

    def _push(self, cmd) -> None:
        if getattr(self.rt, "mode", None) == "ros":
            self.rt.set_external_cmd(*cmd)

    def _run(self) -> None:
        while not self._stop.wait(self.period):
            self.tick()

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="autonomy", daemon=True)
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._push(ZERO)
