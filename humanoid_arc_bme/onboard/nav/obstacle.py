"""Obstacle-stop hook: a speed cap from the nearest obstacle in the direction of travel.

The follower calls `guard.speed_cap(now, direction)` every tick and scales (vx, wz) together so |vx| <= cap
(the curve is kept, the robot slows, at cap 0 it stops and balances). The rule is A2.6's speed-and-separation:

    v_max(d) = min(v_nominal, sqrt(2 * a_stop * max(0, d - d_margin)))

`a_stop` is a placeholder until B4.3 measures the G1's deceleration; `d_margin` = 0.3 m (proposed in A2.6). The
distance comes from a callable (LiDAR / depth sector minimum computed elsewhere: ROS 2 node, nvblox ESDF, ...).
Fail closed: no distance reading for longer than `stale_s` -> cap 0 when `require_sensor` is set.
This is a planner-level slow-down; it does NOT replace S1 (api/safety.py) or the vendor's own protection.
"""

from __future__ import annotations

import math
import numbers
from typing import Callable


class NoGuard:
    def speed_cap(self, now: float, direction: int) -> float:
        return math.inf


class SpeedSeparationGuard:
    """Speed cap per direction (+1 forward, -1 backward) from a distance callable.

    Distance contract (per direction): a finite number >= 0 is a reading; `math.inf` is an explicit "nothing in
    range"; `None` means "no reading". NaN, -inf or a negative value is invalid evidence -> cap 0 immediately.
    Fail closed with `require_sensor` (default):
      * a direction that has NEVER produced a valid reading (e.g. no rear sensor) -> cap 0 immediately, so a
        command in that direction is stopped from the first call (not after `stale_s`);
      * a None reading right after a valid one in the same direction -> the cap is held at the last valid cap
        (never raised to v_nominal) for at most `stale_s`, then 0. The hold is never infinite: an infinite last cap
        (v_nominal = inf with an "inf" reading) is held as `hold_fallback` (m/s, PLACEHOLDER 0.1). The hold needs
        0 <= now - t_last <= stale_s; a clock that went backwards (or a non-finite one) gives cap 0.
      * an int too large for a float is converted safely: positive -> "inf" (nothing in range), else invalid -> 0.
    Validity is tracked per direction: a fresh forward reading never vouches for the rear.
    With `require_sensor=False` a None reading passes v_nominal (legacy, explicit opt-out); invalid numbers still
    give 0. Constructor knobs must be finite (a_stop > 0, d_margin >= 0, v_nominal > 0 (inf allowed), stale_s > 0).
    """

    def __init__(self, distance: Callable[[int], float | None], a_stop: float = 0.3, d_margin: float = 0.3,
                 v_nominal: float = math.inf, require_sensor: bool = True, stale_s: float = 0.3,
                 hold_fallback: float = 0.1):
        def num(v):
            return isinstance(v, numbers.Real) and not isinstance(v, bool)
        if not (num(a_stop) and math.isfinite(a_stop) and a_stop > 0):
            raise ValueError(f"a_stop: must be a finite number > 0, got {a_stop!r}")
        if not (num(d_margin) and math.isfinite(d_margin) and d_margin >= 0):
            raise ValueError(f"d_margin: must be a finite number >= 0, got {d_margin!r}")
        if not (num(v_nominal) and v_nominal == v_nominal and v_nominal > 0):   # +inf = "no nominal limit"
            raise ValueError(f"v_nominal: must be a number > 0 (inf allowed), got {v_nominal!r}")
        if not (num(stale_s) and math.isfinite(stale_s) and stale_s > 0):
            raise ValueError(f"stale_s: must be a finite number > 0, got {stale_s!r}")
        if not (num(hold_fallback) and math.isfinite(hold_fallback) and hold_fallback >= 0):
            raise ValueError(f"hold_fallback: must be a finite number >= 0, got {hold_fallback!r}")
        self.hold_fallback = float(hold_fallback)
        self.distance, self.a_stop, self.d_margin, self.v_nominal = distance, a_stop, d_margin, v_nominal
        self.require_sensor, self.stale_s = require_sensor, stale_s
        self._last_ok: dict[int, tuple[float, float]] = {}   # direction -> (time, cap) of the last valid reading
        self.last_distance = None

    def cap_for(self, d: float) -> float:
        return min(self.v_nominal, math.sqrt(2.0 * self.a_stop * max(0.0, d - self.d_margin)))

    def speed_cap(self, now: float, direction: int) -> float:
        d = self.distance(direction)
        self.last_distance = d
        if d is None:
            if not self.require_sensor:
                return self.v_nominal
            last = self._last_ok.get(direction)
            try:
                age = now - last[0] if last is not None else math.nan
                if not (0.0 <= age <= self.stale_s):   # never seen, stale, backwards/future/NaN clock
                    return 0.0
            except (TypeError, OverflowError):
                return 0.0
            return last[1] if math.isfinite(last[1]) else self.hold_fallback   # never an infinite hold
        if not isinstance(d, numbers.Real) or isinstance(d, bool) or d != d or d < 0:
            return 0.0   # NaN / negative / non-numeric: invalid evidence
        if not isinstance(d, float):
            try:
                d = float(d)
            except OverflowError:   # a huge positive int (negatives were rejected above)
                d = math.inf
        cap = self.v_nominal if d == math.inf else self.cap_for(d)
        self._last_ok[direction] = (now, cap)
        return cap


def sector_min_distance(points_xy, direction: int, half_width: float = 0.35, max_range: float = 3.0,
                        min_range: float = 0.15) -> float:
    """Nearest obstacle distance in a corridor of +-half_width ahead (direction +1) or behind (-1) the body, from
    body-frame points (x forward, y left), e.g. a LiDAR scan already cropped to the obstacle height band. Returns inf
    when nothing is in range. Points closer than `min_range` are ignored (self-hits; the MID-360's 0.1-0.2 m band is
    "reference only" per Livox)."""
    best = math.inf
    for x, y in points_xy:
        ax = x * direction
        if ax < min_range or ax > max_range or abs(y) > half_width:
            continue
        best = min(best, ax)
    return best
