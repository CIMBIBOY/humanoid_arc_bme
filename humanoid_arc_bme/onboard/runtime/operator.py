"""Operator safety core: the velocity envelope, the command ramp and the arming supervisor. Standard library only.

Rules (docs/plan/robot-deploy.md, "Safety by design"):
- The runtime starts DISARMED and nothing is sent to the robot while disarmed.
- Arming needs an explicit request (which returns a one-time code) AND the typed confirmation "ARM <code>", a green
  preflight, a fresh UI heartbeat at >= heartbeat_min_hz and no latched STOP.
- Deadman: while armed, a UI heartbeat older than deadman_timeout_s sets the command to zero and disarms.
- Hold-to-run (UI-03): a heartbeat keeps the link alive but never authorises motion by itself. A non-zero command needs
  `held=True` (the operator is pressing the pad, a drive key or the HOLD button), and after arming one heartbeat with
  the input RELEASED must arrive first (neutral first), so a key or button held across arming never starts motion.
- Validation first (UI-04): a non-finite, non-numeric or out-of-envelope command is rejected (ValueError) BEFORE it can
  feed the deadman, so a malformed client stops feeding the deadman and the robot disarms.
- Stale state (older than state_stale_s) or no state (link loss) while armed: zero and disarm.
- STOP: zero / damping (the controller decides what its safe state is), disarm, and latch; arming again needs a reset.
- Velocity limits come from config/g1_limits.yaml (`operator:`) intersected with S1's `base_vel_max`; nothing at
  runtime can widen them. A ramp limits the change of the command per second on every axis.
- Every state transition goes to `on_transition` (JSONL + MCAP in the runtime).
Time is always passed in (`now`), so every rule is testable with a fake clock.
"""

from __future__ import annotations

import math
import secrets
from dataclasses import asdict, dataclass

from humanoid_arc_bme.onboard.validation import finite_number  # noqa: F401 - re-exported: the web layer imports it from here

DISARMED, ARMED = "disarmed", "armed"
ZERO = (0.0, 0.0, 0.0)
ENVELOPE_TOL = 1e-6  # a command this far outside the envelope (float rounding in the browser) is clipped, beyond it rejected
KAPPA_MAX = 20.0  # 1/m: curvature sanity bound (the UI sends at most 2)


def _f(d: dict, key: str, label: str | None = None) -> float:
    v = d[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"operator limits: {label or key} must be a finite number, got {v!r}")
    return float(v)


@dataclass(frozen=True)
class OperatorLimits:
    vx_min: float
    vx_max: float
    vy_max: float
    wz_max: float
    ax: float
    ay: float
    awz: float
    deadman_timeout_s: float
    heartbeat_min_hz: float
    state_stale_s: float
    link_loss_s: float
    arm_challenge_ttl_s: float
    max_tilt_deg: float

    def __post_init__(self):
        if not (self.vx_min <= 0.0 <= self.vx_max) or self.vy_max < 0 or self.wz_max < 0:
            raise ValueError("operator limits: need vx_min <= 0 <= vx_max and non-negative vy_max, wz_max")
        for k in ("ax", "ay", "awz", "deadman_timeout_s", "heartbeat_min_hz", "state_stale_s", "link_loss_s", "arm_challenge_ttl_s", "max_tilt_deg"):
            if getattr(self, k) <= 0:
                raise ValueError(f"operator limits: {k} must be positive")
        if self.deadman_timeout_s > 1.0:
            raise ValueError("operator limits: deadman_timeout_s above 1 s is not a deadman")

    @staticmethod
    def from_dict(d: dict, base_vel_max: tuple | None = None) -> "OperatorLimits":
        """`base_vel_max` (S1, absolute vx, vy, wz) narrows the envelope: the tighter limit wins."""
        try:
            a = d["accel"]
            lim = dict(vx_min=_f(d, "vx_min"), vx_max=_f(d, "vx_max"), vy_max=_f(d, "vy_max"), wz_max=_f(d, "wz_max"),
                       ax=_f(a, "vx", "accel.vx"), ay=_f(a, "vy", "accel.vy"), awz=_f(a, "wz", "accel.wz"),
                       deadman_timeout_s=_f(d, "deadman_timeout_s"), heartbeat_min_hz=_f(d, "heartbeat_min_hz"),
                       state_stale_s=_f(d, "state_stale_s"), link_loss_s=_f(d, "link_loss_s"),
                       arm_challenge_ttl_s=_f(d, "arm_challenge_ttl_s"), max_tilt_deg=_f(d, "max_tilt_deg"))
        except (KeyError, TypeError) as e:
            raise ValueError(f"operator limits: malformed ({type(e).__name__}: {e})") from e
        if base_vel_max is not None:
            bx, by, bw = (float(x) for x in base_vel_max)
            lim.update(vx_min=max(lim["vx_min"], -bx), vx_max=min(lim["vx_max"], bx), vy_max=min(lim["vy_max"], by),
                       wz_max=min(lim["wz_max"], bw))
        return OperatorLimits(**lim)

    def clip(self, vx: float, vy: float, wz: float) -> tuple[float, float, float]:
        for v in (vx, vy, wz):
            if not math.isfinite(v):
                raise ValueError("velocity command must be finite")
        return (min(max(vx, self.vx_min), self.vx_max), min(max(vy, -self.vy_max), self.vy_max),
                min(max(wz, -self.wz_max), self.wz_max))

    def validate(self, vx, vy, wz) -> tuple[float, float, float]:
        """UI-04: refuse (ValueError) a non-finite or non-numeric value, or one outside the envelope by more than
        ENVELOPE_TOL; values within the tolerance are clipped. Nothing is changed when this raises."""
        vx, vy, wz = (finite_number(v, k) for v, k in ((vx, "vx"), (vy, "vy"), (wz, "wz")))
        t = ENVELOPE_TOL
        if not (self.vx_min - t <= vx <= self.vx_max + t and abs(vy) <= self.vy_max + t and abs(wz) <= self.wz_max + t):
            raise ValueError(f"command ({vx}, {vy}, {wz}) outside the envelope vx {self.vx_min}..{self.vx_max}, "
                             f"vy +-{self.vy_max}, wz +-{self.wz_max}")
        return self.clip(vx, vy, wz)

    def validate_curvature(self, v, kappa) -> tuple[float, float, float]:
        v, kappa = finite_number(v, "v"), finite_number(kappa, "kappa")
        if not (self.vx_min - ENVELOPE_TOL <= v <= self.vx_max + ENVELOPE_TOL) or abs(kappa) > KAPPA_MAX:
            raise ValueError(f"curvature command v={v}, kappa={kappa} outside vx {self.vx_min}..{self.vx_max}, |kappa| <= {KAPPA_MAX}")
        return self.clip_curvature(v, kappa)

    def clip_curvature(self, v: float, kappa: float) -> tuple[float, float, float]:
        """Path-following command: forward speed v along curvature kappa (1/m, left positive). If the yaw rate
        v * kappa would exceed wz_max, v is reduced so the path (the curvature) is kept and only the speed drops."""
        if not (math.isfinite(v) and math.isfinite(kappa)):
            raise ValueError("curvature command must be finite")
        vx = min(max(v, self.vx_min), self.vx_max)
        if abs(vx * kappa) > self.wz_max:
            vx = math.copysign(self.wz_max / abs(kappa), vx)
        return vx, 0.0, vx * kappa

    def to_dict(self) -> dict:
        return asdict(self)


def load_operator_limits(limits_yaml, base_vel_max=None) -> OperatorLimits:
    import yaml  # lazy, as in humanoid_arc_bme.onboard.api.safety
    from pathlib import Path

    d = yaml.safe_load(Path(limits_yaml).read_text())
    if "operator" not in d:
        raise ValueError(f"{limits_yaml}: no 'operator:' section")
    return OperatorLimits.from_dict(d["operator"], base_vel_max)


class RateLimiter:
    """Moves the output toward the target by at most accel * dt per axis and call."""

    def __init__(self, accel: tuple[float, float, float]):
        self.accel = tuple(float(a) for a in accel)
        self.value = (0.0, 0.0, 0.0)

    def reset(self) -> None:
        self.value = (0.0, 0.0, 0.0)

    def step(self, target, dt: float) -> tuple[float, float, float]:
        if not (math.isfinite(dt) and dt > 0):
            raise ValueError("dt must be positive and finite")
        out = []
        for cur, tgt, a in zip(self.value, target, self.accel):
            d = tgt - cur
            m = a * dt
            out.append(tgt if abs(d) <= m else cur + math.copysign(m, d))
        self.value = tuple(out)
        return self.value


class Supervisor:
    """Arming state machine + deadman + stale/link checks. `on_transition(record: dict)` is called for every change."""

    def __init__(self, limits: OperatorLimits, on_transition=None, rng=secrets):
        self.lim = limits
        self.state = DISARMED
        self.stop_latched = False
        self.reason = "startup"
        self.controller = None
        self.on_transition = on_transition or (lambda rec: None)
        self._rng = rng
        self._challenge: tuple[str, float, str] | None = None  # (code, expiry, controller)
        self._hb_last: float | None = None
        self._hb_times: list[float] = []
        self.held = False  # the latest heartbeat said an operator input (pad / key / HOLD button) is pressed
        self._need_neutral = False  # UI-03: set at arming; cleared by the first heartbeat with the input released
        self.target = (0.0, 0.0, 0.0)
        self.source = "teleop"
        self.ramp = RateLimiter((limits.ax, limits.ay, limits.awz))
        self.armed_since: float | None = None

    # ---------------------------------------------------------------- transitions
    def _transition(self, now: float, new_state: str, reason: str, **extra) -> None:
        old = self.state
        self.state, self.reason = new_state, reason
        if new_state != ARMED:
            self.ramp.reset()
            self.target = ZERO
            self.held = False
            self.armed_since = None
        else:
            self.armed_since = now
        self.on_transition({"t": now, "from": old, "to": new_state, "reason": reason, "controller": self.controller,
                            "stop_latched": self.stop_latched, **extra})

    @property
    def armed(self) -> bool:
        return self.state == ARMED

    # ---------------------------------------------------------------- heartbeat (the deadman)
    def heartbeat(self, now: float, target=ZERO, source: str = "teleop", held: bool = False,
                  validated: bool = False) -> tuple[float, float, float]:
        """Feed the deadman and set the target command. Returns the applied target.
        UI-04: the target is validated first; a malformed one raises and does NOT feed the deadman.
        UI-03: the target is zero unless `held` (an operator input is pressed) and, after arming, a released heartbeat
        has been seen (neutral first)."""
        tgt = tuple(target) if validated else self.lim.validate(*target)
        if not isinstance(held, bool):
            raise ValueError("held must be true or false")
        if self.armed and self._need_neutral:
            if not held:
                self._need_neutral = False  # released: from now on a NEW press may move the robot
            tgt = ZERO
        elif not held:
            tgt = ZERO  # the heartbeat alone keeps the link; it never authorises motion
        self._hb_last = now
        self._hb_times = [t for t in self._hb_times if now - t <= 1.0] + [now]
        self.held = held
        self.target = tgt
        self.source = source
        return self.target

    def motion_authorised(self, now: float) -> bool:
        """Armed, neutral seen after arming, the operator input held in the latest heartbeat, and that heartbeat fresh."""
        hb = self.heartbeat_age(now)
        return (self.armed and not self._need_neutral and self.held and hb is not None
                and 0.0 <= hb <= self.lim.deadman_timeout_s)

    def set_target(self, target, source: str, now: float | None = None) -> tuple[float, float, float]:
        """Target from a source the runtime recomputes every tick (waypoints, ROS cmd_vel). Does NOT feed the deadman:
        the operator's heartbeat must keep arriving, and (hold-to-run) its input must be held, else the target is zero.
        A non-finite target raises."""
        tgt = self.lim.clip(*(finite_number(x, "target") for x in target))
        if now is not None and not self.motion_authorised(now):
            tgt = ZERO
        self.target = tgt
        self.source = source
        return self.target

    def heartbeat_rate(self, now: float) -> float:
        """Heartbeats in the last second (0 when the last one is older than the deadman timeout)."""
        if self._hb_last is None or now - self._hb_last > self.lim.deadman_timeout_s:
            return 0.0
        return float(sum(1 for t in self._hb_times if now - t <= 1.0))

    def heartbeat_age(self, now: float) -> float | None:
        return None if self._hb_last is None else now - self._hb_last

    # ---------------------------------------------------------------- arming
    def request_arm(self, now: float, controller: str) -> str:
        if self.armed:
            raise PermissionError("already armed")
        if self.stop_latched:
            raise PermissionError("STOP is latched: reset it first")
        code = "".join(self._rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(4))
        self._challenge = (code, now + self.lim.arm_challenge_ttl_s, controller)
        self.on_transition({"t": now, "from": self.state, "to": self.state, "reason": "arm_requested", "controller": controller,
                            "stop_latched": False})
        return code

    def confirm_arm(self, now: float, text: str, controller: str, preflight_ok: bool, preflight_failed=()) -> tuple[bool, str]:
        """Arms only if every rule holds. Returns (armed, reason). A wrong code consumes the challenge."""
        ch, self._challenge = self._challenge, None
        why = None
        if self.armed:
            why = "already armed"
        elif self.stop_latched:
            why = "STOP is latched: reset it first"
        elif ch is None:
            why = "no arming request: press ARM first"
        elif now > ch[1]:
            why = "arming code expired"
        elif ch[2] != controller:
            why = "controller changed since the request"
        elif not isinstance(text, str) or text.strip() != f"ARM {ch[0]}":
            why = "typed confirmation does not match"
        elif not preflight_ok:
            why = "preflight not green: " + ", ".join(preflight_failed)
        elif self.heartbeat_rate(now) < self.lim.heartbeat_min_hz:
            why = f"UI heartbeat {self.heartbeat_rate(now):.0f} Hz < {self.lim.heartbeat_min_hz:.0f} Hz"
        if why:
            self.on_transition({"t": now, "from": self.state, "to": self.state, "reason": "arm_refused", "detail": why,
                                "controller": controller, "stop_latched": self.stop_latched})
            return False, why
        self.controller = controller
        self.target = ZERO  # arming never starts with motion: the operator moves the stick after
        self._need_neutral = True  # UI-03: and a key / button / pad held across arming is ignored until released once
        self.held = False
        self.ramp.reset()
        self._transition(now, ARMED, "operator_confirmed")
        return True, "armed"

    def disarm(self, now: float, reason: str = "operator") -> bool:
        if not self.armed:
            return False
        self._transition(now, DISARMED, reason)
        return True

    def stop(self, now: float, source: str = "operator") -> None:
        """Always succeeds; latches. Logged even when already disarmed."""
        self.stop_latched = True
        self._challenge = None
        self._transition(now, DISARMED, f"stop:{source}")

    def reset_stop(self, now: float) -> None:
        if self.stop_latched:
            self.stop_latched = False
            self._transition(now, DISARMED, "stop_reset")

    # ---------------------------------------------------------------- per tick
    def check(self, now: float, state_age: float | None, tilt_deg: float | None = None, controller_fault: str | None = None,
              runtime_fault: str | None = None) -> str | None:
        """Called every control tick. Returns the reason if it disarmed now, else None."""
        if not self.armed:
            return None
        why = None
        hb = self.heartbeat_age(now)
        if hb is None or not (0.0 <= hb <= self.lim.deadman_timeout_s):  # a heartbeat "from the future" is a clock fault
            why = "deadman_timeout"
        elif state_age is None or not math.isfinite(state_age) or state_age > self.lim.link_loss_s:
            why = "link_lost"
        elif state_age > self.lim.state_stale_s or state_age < 0.0:
            why = "state_stale"
        elif runtime_fault:
            why = f"runtime_fault:{runtime_fault}"
        elif tilt_deg is not None and tilt_deg > self.lim.max_tilt_deg:
            why = "tilt_limit"
        elif controller_fault:
            why = f"controller_fault:{controller_fault}"
        if why:
            self._transition(now, DISARMED, why, heartbeat_age=hb, state_age=state_age)
        return why

    def command(self, dt: float) -> tuple[float, float, float]:
        """The ramped command for this tick; zero (and the ramp reset) while disarmed."""
        if not self.armed:
            self.ramp.reset()
            return (0.0, 0.0, 0.0)
        return self.ramp.step(self.target, dt)

    def status(self, now: float) -> dict:
        ch = self._challenge
        return {"state": self.state, "reason": self.reason, "stop_latched": self.stop_latched, "controller": self.controller,
                "target": list(self.target), "ramped": list(self.ramp.value), "source": self.source,
                "held": self.held, "need_neutral": self._need_neutral and self.armed,
                "heartbeat_age_s": self.heartbeat_age(now), "heartbeat_hz": self.heartbeat_rate(now),
                "arm_pending": bool(ch and now <= ch[1]), "armed_for_s": None if self.armed_since is None else now - self.armed_since}


def tilt_deg_from_quat(q_wxyz) -> float:
    """Angle between the pelvis z axis and the world vertical, degrees."""
    w, x, y, z = (float(c) for c in q_wxyz)
    zz = 1.0 - 2.0 * (x * x + y * y)  # R[2,2]
    return math.degrees(math.acos(max(-1.0, min(1.0, zz))))
