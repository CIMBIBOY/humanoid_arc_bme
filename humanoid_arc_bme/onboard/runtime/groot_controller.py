"""GR00T / GEAR-SONIC planner-mode adapter: S1-clamped (vx, vy, wz) -> the deploy's ZMQ `planner` topic.

Sources (OPENED 2026-11-06, see docs/plan/groot-behind-s1.md for the quotes):
- gear_sonic/utils/teleop/zmq/zmq_planner_sender.py: wire format `[topic bytes][1280-byte JSON header, NUL padded]
  [little-endian payload]`; header `{"v":1,"endian":"le","count":1,"fields":[{"name","dtype","shape"},...]}`;
  topic `command` = u8 start / u8 stop / u8 planner (+ optional f32 delta_heading); topic `planner` = i32 mode,
  f32[3] movement, f32[3] facing, f32 speed (-1 = mode default), f32 height (-1 = off).
- gear_sonic_deploy .../input_interface/zmq_manager.hpp (`--input-type zmq_manager`): subscribes to host:port (default
  5556); a planner message is used once; with no planner message for 1000 ms (PLANNER_TIMEOUT) the deploy resets the
  locomotion to IDLE by itself. `movement` and `facing` are unit direction vectors in the deploy's heading-reference
  frame (facing starts at (1,0,0); `I` re-initialises it), not body-frame velocities. Speed is `speed` in m/s.
- Idle is planner mode 0; "turn in place" is a facing change while idle or moving.
- `command` stop=1 is the deploy's "stop control" (the keyboard `O` emergency stop): NOT a clean stand. This class never
  sends it unless `allow_control_stop=True`; every disarm / STOP / emergency sends the IDLE planner message instead.

Our side: the publisher BINDS the port (the streamer in the vendor's own pico_manager_thread_server.py binds `tcp://*:5556`);
the deploy runs with `--input-type zmq_manager --zmq-host <this host> --zmq-port 5556`. The deploy takes over rt/lowcmd
(it releases the vendor motion service), so this controller and `VendorLocoController` are alternatives, never both.

Disarmed by default: `enabled=False` makes `ready()` false and `step()` a no-op, so nothing is published until the
operator constructs it enabled. NEVER RUN ON A ROBOT: untested against a real deploy (whether the planner honours speeds
as low as S1's 0.2 m/s, and the facing-frame convention under heading drift, are UNKNOWN).
"""

from __future__ import annotations

import json
import math
import struct
import threading
import time

from ..api.safety import Limits, clamp_joint_targets
from ..g1_layout import LEFT_HAND_JOINTS, RIGHT_HAND_JOINTS, UPPER_BODY_JOINTS
from .controllers import Controller

HEADER_SIZE = 1280  # zmq_planner_sender.py HEADER_SIZE; the deploy's ZMQPackedMessageSubscriber::HEADER_SIZE agrees

MODE_IDLE = 0  # planner ONNX reference / VR teleop doc: 0 idle, 1 slowWalk, 2 walk, 3 run
MODE_SLOW_WALK = 1
MODE_WALK = 2


def _header(fields: list) -> bytes:
    raw = json.dumps({"v": 1, "endian": "le", "count": 1, "fields": fields}, separators=(",", ":")).encode()
    if len(raw) > HEADER_SIZE:
        raise ValueError("header too large")
    return raw.ljust(HEADER_SIZE, b"\x00")


def build_command_message(start: bool, stop: bool, planner: bool, delta_heading: float | None = None) -> bytes:
    fields = [{"name": n, "dtype": "u8", "shape": [1]} for n in ("start", "stop", "planner")]
    payload = struct.pack("BBB", int(bool(start)), int(bool(stop)), int(bool(planner)))
    if delta_heading is not None:
        fields.append({"name": "delta_heading", "dtype": "f32", "shape": [1]})
        payload += struct.pack("<f", float(delta_heading))
    return b"command" + _header(fields) + payload


# Optional planner fields of the deploy (zmq_manager.hpp): name -> number of f32 values. Present only while the arms are
# commanded; a message without them leaves the upper body to the planner (and the deploy clears them after 1 s of silence).
OPTIONAL_PLANNER_FIELDS = {"upper_body_position": 17, "upper_body_velocity": 17, "left_hand_joints": 7, "right_hand_joints": 7}


def build_planner_message(mode: int, movement, facing, speed: float = -1.0, height: float = -1.0, **optional) -> bytes:
    """`optional` takes the OPTIONAL_PLANNER_FIELDS by name (sequences of finite floats of the stated length)."""
    if len(movement) != 3 or len(facing) != 3:
        raise ValueError("movement and facing need 3 components")
    fields = [{"name": "mode", "dtype": "i32", "shape": [1]},
              {"name": "movement", "dtype": "f32", "shape": [3]},
              {"name": "facing", "dtype": "f32", "shape": [3]},
              {"name": "speed", "dtype": "f32", "shape": [1]},
              {"name": "height", "dtype": "f32", "shape": [1]}]
    payload = (struct.pack("<i", int(mode)) + struct.pack("<3f", *map(float, movement))
               + struct.pack("<3f", *map(float, facing)) + struct.pack("<f", float(speed)) + struct.pack("<f", float(height)))
    for name, values in optional.items():
        if name not in OPTIONAL_PLANNER_FIELDS:
            raise ValueError(f"unknown planner field {name!r}")
        if values is None:
            continue
        n = OPTIONAL_PLANNER_FIELDS[name]
        vals = [float(v) for v in values]
        if len(vals) != n or not all(math.isfinite(v) for v in vals):
            raise ValueError(f"{name}: expected {n} finite values")
        fields.append({"name": name, "dtype": "f32", "shape": [n]})
        payload += struct.pack(f"<{n}f", *vals)
    return b"planner" + _header(fields) + payload


def decode_message(msg: bytes) -> tuple[str, dict]:
    """Inverse of the builders (tests and logs): (topic, {field: value})."""
    topic = "command" if msg.startswith(b"command") else "planner" if msg.startswith(b"planner") else None
    if topic is None:
        raise ValueError("unknown topic")
    body = msg[len(topic):]
    hdr = json.loads(body[:HEADER_SIZE].rstrip(b"\x00"))
    off, out = HEADER_SIZE, {}
    sizes = {"u8": ("B", 1), "i32": ("i", 4), "f32": ("f", 4)}
    for f in hdr["fields"]:
        code, n = sizes[f["dtype"]]
        cnt = math.prod(f["shape"])
        vals = struct.unpack_from("<" + str(cnt) + code, body, off)
        off += n * cnt
        out[f["name"]] = vals[0] if cnt == 1 else tuple(vals)
    return topic, out


class FakeTransport:
    """Records published messages (tests, UI dry runs)."""

    def __init__(self):
        self.sent: list[bytes] = []

    def send(self, msg: bytes) -> None:
        self.sent.append(msg)

    def close(self) -> None:
        pass

    def decoded(self) -> list:
        return [decode_message(m) for m in self.sent]


class ZmqPubTransport:
    """The real transport: a PUB socket that binds `endpoint` (the deploy's SUB connects to it). pyzmq is imported here only."""

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5556"):
        import zmq

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(endpoint)

    def send(self, msg: bytes) -> None:
        self._sock.send(msg)

    def close(self) -> None:
        self._sock.close(0)


class GrootPlannerController(Controller):
    name = "groot"

    def __init__(self, transport, send_hz: float = 20.0, step_deadline_s: float = 0.15, enabled: bool = False,
                 select_planner: bool = True, start_control: bool = False, allow_control_stop: bool = False,
                 walk_mode: int = MODE_SLOW_WALK, min_walk_speed: float = 0.0, move_eps: float = 0.02,
                 turn_eps: float = 0.02, clock=time.monotonic, threaded: bool = True, limits: Limits | None = None):
        if not (send_hz >= 5.0):
            raise ValueError("send_hz >= 5 (the deploy resets to IDLE after 1 s and uses each planner message once)")
        if not (0 < step_deadline_s <= 0.5):
            raise ValueError("0 < step_deadline_s <= 0.5 s (well under the deploy's 1 s PLANNER_TIMEOUT)")
        if walk_mode not in (MODE_SLOW_WALK, MODE_WALK):
            raise ValueError("walk_mode: 1 (slow walk) or 2 (walk)")
        self.tx, self.period, self.step_deadline = transport, 1.0 / send_hz, float(step_deadline_s)
        self.enabled, self.select_planner, self.start_control = bool(enabled), select_planner, start_control
        self.allow_control_stop, self.walk_mode = allow_control_stop, walk_mode
        self.min_walk_speed, self.move_eps, self.turn_eps = float(min_walk_speed), float(move_eps), float(turn_eps)
        self.clock = clock
        self.limits = limits                       # S1 joint limits for the arm/hand targets; without them the arms are refused
        self._ub: dict[str, list[float]] | None = None   # last applied arm/hand targets, streamed until released
        self._ub_goal: dict[str, list[float]] | None = None
        self._ub_dt = 1.0 / send_hz
        self._lock = threading.Lock()
        self.yaw = 0.0  # facing angle in the deploy's heading-reference frame, integrated from wz (open loop)
        self._cmd = (0.0, 0.0, 0.0)
        self._streaming = False
        self._last_send = -math.inf
        self._last_step = -math.inf
        self.sent = 0
        self._fault: str | None = None
        self._stop_evt = threading.Event()
        self._thread = None
        if threaded:
            self._thread = threading.Thread(target=self._run, name="groot-sender", daemon=True)
            self._thread.start()

    # ---- message construction
    def _facing(self) -> tuple:
        return (math.cos(self.yaw), math.sin(self.yaw), 0.0)

    def _optional(self) -> dict:
        ub = self._ub
        return {} if ub is None else {"upper_body_position": ub["body"], "left_hand_joints": ub["left"], "right_hand_joints": ub["right"]}

    def _planner_msg(self) -> bytes:
        vx, vy, _wz = self._cmd
        v = math.hypot(vx, vy)
        if v <= self.move_eps:  # idle: facing may still turn in place
            return build_planner_message(MODE_IDLE, (0.0, 0.0, 0.0), self._facing(), -1.0, -1.0, **self._optional())
        ux, uy = vx / v, vy / v
        c, s = math.cos(self.yaw), math.sin(self.yaw)  # body-frame direction -> heading-reference frame
        movement = (c * ux - s * uy, s * ux + c * uy, 0.0)
        return build_planner_message(self.walk_mode, movement, self._facing(), max(v, self.min_walk_speed), -1.0, **self._optional())

    def _send_idle(self) -> None:
        self.tx.send(build_planner_message(MODE_IDLE, (0.0, 0.0, 0.0), self._facing(), -1.0, -1.0))

    # ---- sender
    def pump(self, now: float) -> None:
        with self._lock:
            if not self._streaming:
                return
            if not (0.0 <= now - self._last_step <= self.step_deadline):  # loop stalled or dead (or clock fault)
                self._streaming = False
                self._fault = f"control loop silent for {now - self._last_step:.3f} s (deadline {self.step_deadline} s)"
                self._send_idle()
                return
            if now - self._last_send < self.period - 1e-9:
                return
            self.tx.send(self._planner_msg())
            self._last_send = now
            self.sent += 1

    def _run(self):
        while not self._stop_evt.wait(min(self.period / 4, self.step_deadline / 3)):
            try:
                self.pump(self.clock())
            except Exception as e:  # noqa: BLE001 - the sender must survive: stop streaming, try an IDLE message
                self._streaming = False
                self._fault = f"sender error: {type(e).__name__}: {e}"
                try:
                    with self._lock:
                        self._send_idle()
                except Exception:  # noqa: BLE001
                    pass

    # ---- arms and hands (planner-mode upper-body fields)
    def set_upper_body(self, body, left_hand, right_hand) -> None:
        """Command the upper body: 17 joint positions (g1_layout.UPPER_BODY_JOINTS, rad) and 7 Dex3 joints per hand
        (LEFT/RIGHT_HAND_JOINTS, rad). The targets are clamped to the S1 joint range and rate-limited in `step()`; they are
        streamed with every planner message until `release_upper_body()`. Raises ValueError on a wrong length or a
        non-finite value, and RuntimeError without S1 limits (fail closed)."""
        if self.limits is None:
            raise RuntimeError("arm commands need S1 joint limits: construct the controller with limits=")
        goal = {}
        for key, vals, n in (("body", body, 17), ("left", left_hand, 7), ("right", right_hand, 7)):
            v = [float(x) for x in vals]
            if len(v) != n or not all(math.isfinite(x) for x in v):
                raise ValueError(f"{key}: expected {n} finite values")
            goal[key] = v
        with self._lock:
            self._ub_goal = goal

    def release_upper_body(self) -> None:
        """Stop sending arm/hand fields: the planner's own upper-body motion resumes."""
        with self._lock:
            self._ub_goal = None
            self._ub = None

    def _advance_upper_body(self, state, dt: float) -> None:
        """Rate-limit and clamp the arm/hand goal against the last applied targets (the measured pose on the first tick)."""
        goal = self._ub_goal
        if goal is None:
            return
        names = {"body": UPPER_BODY_JOINTS, "left": LEFT_HAND_JOINTS, "right": RIGHT_HAND_JOINTS}
        measured = dict(zip(state.joint_names, state.joint_pos)) if state is not None else {}
        out = {}
        for key, jn in names.items():
            prev = self._ub[key] if self._ub is not None else [measured.get(n) for n in jn]
            if any(p is None for p in prev):
                raise ValueError(f"state does not report every {key} joint; refusing to move it")
            out[key], _events = clamp_joint_targets(jn, goal[key], prev, max(dt, 1e-3), self.limits)
        self._ub = out

    # ---- Controller interface
    def ready(self):
        if self._fault:
            return False, self._fault
        if not self.enabled:
            return False, "GR00T adapter disabled (construct with enabled=True; untested on a robot)"
        return True, "GR00T planner transport ready"

    def on_arm(self, now, state):
        if not self.enabled:
            return
        with self._lock:
            self._fault = None
            self._cmd, self._last_send, self._last_step = (0.0, 0.0, 0.0), -math.inf, now
            self.yaw = 0.0  # matches a freshly initialised deploy heading (facing (1,0,0)); drift is a known gap
            if self.select_planner or self.start_control:
                self.tx.send(build_command_message(self.start_control, False, True))
            self._streaming = True

    def step(self, now, dt, state, cmd):
        cmd = tuple(float(x) for x in cmd)
        if not all(math.isfinite(x) for x in cmd) or not (math.isfinite(dt) and dt >= 0):
            raise ValueError("groot step: non-finite command or dt")
        with self._lock:
            if self.enabled and self._streaming:
                self._cmd, self._last_step = cmd, now
                if abs(cmd[2]) > self.turn_eps:
                    self.yaw = math.remainder(self.yaw + cmd[2] * dt, 2 * math.pi)
                self._advance_upper_body(state, dt)
        if self._thread is None:
            self.pump(now)
        return {"sent": self.sent, "yaw": self.yaw}

    def on_disarm(self, now, reason):
        with self._lock:
            self._streaming = False
            self._cmd = (0.0, 0.0, 0.0)
            self._ub = self._ub_goal = None
            if self.enabled:
                self._send_idle()

    def stop(self, now):
        self.on_disarm(now, "stop")

    def emergency_stop(self, now):
        self.on_disarm(now, "emergency")

    def damp(self, now):
        if not self.allow_control_stop:
            return "not supported: the deploy's stop-control (command stop=1) is not a damping command"
        with self._lock:
            self._streaming = False
            self.tx.send(build_command_message(False, True, True))
        return "command stop sent (GR00T control loop exits; limp/hold state UNKNOWN)"

    def fault(self):
        return self._fault

    def status(self):
        return {"name": self.name, "enabled": self.enabled, "sent": self.sent, "streaming": self._streaming,
                "upper_body": self._ub is not None,
                "yaw": self.yaw, "cmd": list(self._cmd), "step_deadline_s": self.step_deadline, "fault": self._fault}

    def close(self):
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            self.tx.close()
        except Exception:  # noqa: BLE001
            pass
