"""Shared obstacle stage: runs in OperatorRuntime right after S1 `filter_base` and before `controller.step`, for every
control mode (teleop / curvature / waypoints / ros) and every controller (vendor, GR00T planner, ...).

Tightening only. Output (vx, vy, wz) is the input scaled by k in [0, 1] (same k for all three, so the curve is kept), or
zeros. It never raises a component or flips a sign. Rule: A2.6 speed-and-separation, v_max(d) = min(v_nom,
sqrt(2 a_stop max(0, d - d_margin))), applied to the translation speed hypot(vx, vy).

Direction of travel: +1 (forward corridor) when vx > 0 or (vx == 0 and vy != 0: lateral uses the forward corridor as the
best available proxy); -1 when vx < 0; none for a pure turn (vx = vy = 0). A pure turn passes only when the sensor is
healthy (or in the documented no-sensor state); it is zeroed when a sensor is configured but unhealthy.

Fail closed (reason in the result):
  - distance source returns None / raises / NaN / negative / non-number / bool  -> zero   ("no_reading", "bad_reading")
  - rear motion (-1) and the source gives None for the rear (no rear sensor)    -> zero   ("no_reading")
  - non-finite command components                                               -> zero   ("bad_command")
  - cap 0 (obstacle inside d_margin)                                            -> zero   ("blocked")
  - source returns +inf = "nothing in range" -> no limit ("clear"). Staleness is the source's job (DepthDistance returns
    None past `max_age_s`); the stage additionally treats a source that raises as no reading.

NO-SENSOR POLICY (distance=None): explicit state "no_sensor", never silent. Motion toward unknown space is NOT
free: forward/lateral translation is capped to `no_sensor_speed` (default: the S1 translation limit, i.e. no extra
slowdown because the robot is already at the first-walk envelope; set lower to slow it) and rear motion is blocked
unless `allow_rear_without_sensor` is set explicitly. The state carries warning="NO OBSTACLE SENSOR" which the runtime
shows in telemetry/status and logs as an event on every state change. Pure rotation passes. The stage is a planner-level
slow-down; it does NOT replace S1 or the vendor's protection.

Idempotent with PathFollower's own guard: both apply the same monotone cap, so a second application is a no-op.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

ZERO = (0.0, 0.0, 0.0)


@dataclass
class StageResult:
    cmd: tuple
    scale: float
    reason: str          # clear | capped | blocked | no_sensor | no_reading | bad_reading | bad_command | idle
    distance: float | None
    cap: float | None
    direction: int
    warning: str | None = None

    def to_dict(self) -> dict:
        return {"distance": self.distance, "cap": self.cap, "scale": self.scale, "reason": self.reason,
                "direction": self.direction, "warning": self.warning}


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


@dataclass
class ObstacleStage:
    distance: Callable[[int], float | None] | None = None
    a_stop: float = 0.3
    d_margin: float = 0.5          # >= 0.5 m at 0.2 m/s until G8 is measured
    v_nominal: float = math.inf
    no_sensor_speed: float = math.inf   # cap on translation speed when no source is configured (runtime sets the S1 limit)
    allow_rear_without_sensor: bool = False
    last: StageResult | None = field(default=None, init=False)

    def __post_init__(self):
        if self.a_stop <= 0 or self.d_margin < 0 or not self.no_sensor_speed > 0:
            raise ValueError("a_stop > 0, d_margin >= 0, no_sensor_speed > 0")

    @property
    def has_sensor(self) -> bool:
        return self.distance is not None

    def cap_for(self, d: float) -> float:
        return min(self.v_nominal, math.sqrt(2.0 * self.a_stop * max(0.0, d - self.d_margin)))

    def apply(self, cmd) -> StageResult:
        r = self._apply(cmd)
        self.last = r
        return r

    def _apply(self, cmd) -> StageResult:
        try:
            vx, vy, wz = cmd
        except (TypeError, ValueError):
            return StageResult(ZERO, 0.0, "bad_command", None, None, 0)
        if not (_num(vx) and _num(vy) and _num(wz)):
            return StageResult(ZERO, 0.0, "bad_command", None, None, 0)
        cmd = (float(vx), float(vy), float(wz))
        trans = math.hypot(vx, vy)
        direction = 0 if trans == 0.0 else (-1 if vx < 0 else 1)
        if self.distance is None:
            warn = "NO OBSTACLE SENSOR"
            if direction == 0:
                return StageResult(cmd, 1.0, "no_sensor", None, None, 0, warn)
            if direction < 0 and not self.allow_rear_without_sensor:
                return StageResult(ZERO, 0.0, "no_sensor", None, 0.0, direction, warn + ": rear motion blocked")
            return self._scale(cmd, trans, self.no_sensor_speed, "no_sensor", None, direction, warn)
        if direction == 0:
            return StageResult(cmd, 1.0, "clear", None, None, 0) if self._healthy() else \
                StageResult(ZERO, 0.0, "no_reading", None, 0.0, 0)
        try:
            d = self.distance(direction)
        except Exception:  # noqa: BLE001 - a crashing source is no reading
            d = None
        if d is None:
            return StageResult(ZERO, 0.0, "no_reading", None, 0.0, direction)
        if not isinstance(d, (int, float)) or isinstance(d, bool) or d != d or d < 0 or d == -math.inf:
            return StageResult(ZERO, 0.0, "bad_reading", None, 0.0, direction)
        if d == math.inf:
            return self._scale(cmd, trans, self.v_nominal, "clear", d, direction)
        return self._scale(cmd, trans, self.cap_for(float(d)), "capped", float(d), direction)

    def _healthy(self) -> bool:
        try:
            d = self.distance(1)
        except Exception:  # noqa: BLE001
            return False
        return isinstance(d, (int, float)) and not isinstance(d, bool) and d == d and d >= 0

    @staticmethod
    def _scale(cmd, trans, cap, reason, d, direction, warn=None) -> StageResult:
        if cap <= 1e-6:
            return StageResult(ZERO, 0.0, "blocked" if reason == "capped" else reason, d, 0.0, direction, warn)
        if trans <= cap:
            return StageResult(cmd, 1.0, reason if reason != "capped" else "clear", d, cap, direction, warn)
        k = cap / trans
        return StageResult((cmd[0] * k, cmd[1] * k, cmd[2] * k), k, "capped" if reason == "capped" else reason, d, cap,
                           direction, warn)
