"""Pose gate (A3): fail-closed accept/reject of localizer estimates before the pipeline uses them.

Stdlib only. Units: metres, radians, seconds. A rejected estimate never updates the held (last accepted) one, so a
consumer that keeps using `last_accepted` after rejects must also check its age; `consecutive_rejects` (alias
`hold_count`, the consumer API: "stop after N rejects in a row") lets it stop after N rejects.
Reason codes: ok, none, unhealthy, stale, variance_too_high, out_of_range, jump, nonmonotonic.

Restart / relocalisation: the jump limit uses at most `max_age_s` of elapsed time and the yaw jump is capped below
pi, so after a LONGER outage a legitimately moved robot is rejected as 'jump'. Any restart, deliberate
relocalisation or recovery from a long outage MUST go through `reset()` (an explicit, logged act of the
operator/supervisor). The first estimate after `reset()` has no reference; it is only checked for health, age,
variance and `max_abs_position`. A single plausible-but-wrong first fix is NOT detected here (v1: kept simple, no
two-fix confirmation); the supervisor that calls reset() owns that decision.

Validation done by `LocalizerEstimate`/`Pose` (finite stamp, bool healthy, finite non-negative variance, finite
position, unit quaternion) is not repeated; anything that is not a `LocalizerEstimate` is rejected as 'none'.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from humanoid_arc_bme.onboard.nav.path import wrap  # shared angle wrap (same signature/range as the old local copy)
from humanoid_arc_bme.onboard.pipeline.interfaces import LocalizerEstimate

REASONS = ("ok", "none", "unhealthy", "stale", "variance_too_high", "out_of_range", "jump", "nonmonotonic")
MAX_YAW_JUMP_CAP = math.pi / 2  # a single accepted step may never turn more than 90 deg, however long the gap


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reason: str
    estimate: LocalizerEstimate | None = None  # the estimate that was judged (None when there was none)


class PoseGate:
    """`max_age_s`: oldest tolerated estimate (now - stamp). `max_pos_sigma` / `max_yaw_sigma`: largest tolerated
    1-sigma (m / rad) from the reported variance. A jump is a change between consecutive ACCEPTED estimates larger
    than `max_jump_m + max_speed * dt_eff` (yaw: `min(max_jump_yaw + max_yaw_rate * dt_eff, pi/2)`) with
    `dt_eff = min(dt, max_age_s)`: the signature of a relocalization or LIO failure; `max_jump_*` are the floors that
    absorb jitter. `max_abs_position` (m, default 1e4: far beyond any site map, far below garbage like 1e6) bounds
    every coordinate of every estimate, including the first."""

    def __init__(self, max_age_s: float, max_pos_sigma: float, max_yaw_sigma: float, max_jump_m: float,
                 max_jump_yaw: float, max_speed: float = 1.0, max_yaw_rate: float = 1.5,
                 max_abs_position: float = 1e4):
        for n, v in (("max_age_s", max_age_s), ("max_pos_sigma", max_pos_sigma), ("max_yaw_sigma", max_yaw_sigma),
                     ("max_jump_m", max_jump_m), ("max_jump_yaw", max_jump_yaw), ("max_speed", max_speed),
                     ("max_yaw_rate", max_yaw_rate), ("max_abs_position", max_abs_position)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not (math.isfinite(v) and v > 0):
                raise ValueError(f"{n}: must be a finite number > 0")
        self.max_age_s, self.max_pos_sigma, self.max_yaw_sigma = float(max_age_s), float(max_pos_sigma), float(max_yaw_sigma)
        self.max_jump_m, self.max_jump_yaw = float(max_jump_m), float(max_jump_yaw)
        self.max_speed, self.max_yaw_rate = float(max_speed), float(max_yaw_rate)
        self.max_abs_position = float(max_abs_position)
        self.reset()

    def reset(self) -> None:
        """Forget the held estimate and counters. Required for any restart/relocalisation (see module docstring)."""
        self.last_accepted: LocalizerEstimate | None = None
        self.consecutive_rejects = 0
        self.counts = {r: 0 for r in REASONS}

    @property
    def hold_count(self) -> int:
        """Alias of `consecutive_rejects`: how many estimates in a row have been rejected."""
        return self.consecutive_rejects

    def _judge(self, est, now: float) -> str:
        if not isinstance(est, LocalizerEstimate):  # None or a wrong type: nothing trustworthy to judge
            return "none"
        if not isinstance(now, (int, float)) or isinstance(now, bool) or not math.isfinite(now):
            return "stale"
        stamp = est.stamp
        if est.healthy is not True:
            return "unhealthy"
        if now - stamp > self.max_age_s:
            return "stale"
        if stamp > now:  # from the future: clock mismatch
            return "nonmonotonic"
        var = est.variance
        if math.sqrt(max(var[0], var[1])) > self.max_pos_sigma or math.sqrt(var[5]) > self.max_yaw_sigma:
            return "variance_too_high"
        p = est.pose.position
        if not all(abs(c) <= self.max_abs_position for c in p):
            return "out_of_range"
        last = self.last_accepted
        if last is not None:
            if stamp < last.stamp:
                return "nonmonotonic"
            dt = min(stamp - last.stamp, self.max_age_s)  # a long outage does not widen the allowance
            q = last.pose.position
            d = math.hypot(p[0] - q[0], p[1] - q[1])
            dyaw = abs(wrap(est.pose.yaw() - last.pose.yaw()))
            yaw_allow = min(self.max_jump_yaw + self.max_yaw_rate * dt, MAX_YAW_JUMP_CAP)
            if d > self.max_jump_m + self.max_speed * dt or dyaw > yaw_allow:
                return "jump"
        return "ok"

    def check(self, est: LocalizerEstimate | None, now: float) -> GateDecision:
        reason = self._judge(est, now)
        self.counts[reason] += 1
        if reason == "ok":
            self.last_accepted = est
            self.consecutive_rejects = 0
            return GateDecision(True, reason, est)
        self.consecutive_rejects += 1
        return GateDecision(False, reason, est)

    def fresh_pose(self, now: float) -> LocalizerEstimate | None:
        """The last accepted estimate if it is still younger than `max_age_s`, else None (fail closed)."""
        e = self.last_accepted
        if e is None or not isinstance(now, (int, float)) or isinstance(now, bool) or not math.isfinite(now):
            return None
        if not (0.0 <= now - e.stamp <= self.max_age_s):  # also rejects a clock that went backwards
            return None
        return e
