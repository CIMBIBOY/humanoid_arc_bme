"""PipelineRunner (G1): CPU end-to-end integration of pose gate, depth guard, costmap, grasp branch and scheduler.

Stdlib + numpy only. One `tick(TickInput) -> TickOutput` per control step. The runner only OUTPUTS a base velocity
command (never actuates) plus grasp candidates. Fail closed everywhere: any missing/invalid input or exception gives a
zero command and a reason code, never an unhandled exception out of `tick`.

Order: (1) pose gate, (2) forward depth -> points -> gravity align -> ground fit -> corridor clearance, costmap,
SpeedSeparationGuard cap, tighten-only command, (3) SafetyStatus (stop wins, sticky `hold_ticks`), (4) isolated grasp
branch, (5) bounded event log, (6) phase scheduler hook, (7) per-stage timing.

Pose policy: only "none" and "stale" (dropout) are tolerated until `fresh_pose` expires (the gate's max age); an
unhealthy, high-variance, jumping or non-monotonic estimate stops immediately (it is evidence the estimate is wrong,
not merely absent).
Tick stamps must be finite, strictly increasing and may not jump forward more than `max_stamp_jump_s` past the last
accepted stamp; anything else is a hard stop `bad_stamp` and LATCHES (the runner stays stopped) until an explicit
`reset()`. This is deliberate fail-closed behaviour: a forward glitch would otherwise expire the obstacle memory TTL
and restore speed. A relocalisation, a localizer restart or a clock restart MUST call `reset()`, which clears the
obstacle memory, costmap, pose gate (`gate.reset()` alone is not enough), sticky hold, pending events and stamp
history. A non-numeric/NaN stamp is not time evidence: it stops that tick but does not latch.
Guard camera: `cfg.guard_camera` selects which depth frame the obstacle/ground analysis uses. "forward" (default) is the neck
camera. "down" runs the same analysis on the down-tilted camera, for a robot that has no forward camera: the floor the
camera cannot see out to `cfg.max_range` is reported as unexplained missing ground, i.e. a drop edge, so the speed cap
shrinks to what the camera really covers. Use it with `require_lidar=True` so the Livox covers the rest of the corridor.
Depth frames must be fresh: |tick stamp - frame stamp| <= `max_depth_age_s` (forward: hard stop `depth_stale`;
down camera: the grasp branch is skipped with an event, no safety stop).
Speed envelope: the output is clamped to `max_vx/max_vy/max_wz` (defaults = config/g1_limits.yaml base_vel_max) and
the obstacle cap applies to the planar speed hypot(vx, vy) by scaling vx, vy and wz together. With a zero cap
(obstacle stop) the whole command, including wz, is zero, EXCEPT for the bounded turn-in-place exception below.
Reverse motion (vx < 0) is stopped by the runner (`reverse_unsensed`) as defence in depth: there is no rear sensor.
`SpeedSeparationGuard` alone also fails closed for a direction that never had a reading (cap 0 immediately).

Turn-in-place exception (user decision 2026-11-10). A stop whose ONLY reason is `obstacle_stop` (the soft stop from
the corridor / costmap / obstacle memory) leaves the robot unable to rotate away. When, and only when, all hold:
the pose was accepted by the gate THIS tick (no dropout hold), ground_ok, depth fresh, costmap healthy, scheduler
fine, stamps fine, no hard fault this tick and no sticky hold left over from a hard fault, and no inflated-occupied
costmap cell has a CORNER within `turn_clearance_m` of the body centre, the output is (0, 0, clamp(wz, +-turn_wz_max)), the
reason `turn_only` is added and the status stays `stop` (translation is zero). vx and vy are always 0 in this mode;
any hard fault gives all zeros incl. wz. `turn_clearance_m` (default 0.45 m) is a PLACEHOLDER: body half-width ~0.2 m
plus ~0.25 m swept-arm margin; `turn_wz_max` defaults to max_wz / 2.
RAW vs RAW (no double counting): rotating about the body centre sweeps a circle of radius ~turn_clearance_m, and the
translation inflation (`inflation_radius` = robot radius) is a different margin. So the check uses the costmap's
UN-inflated `occupied()` cells (nearest-corner distance, bounded as cell-centre distance minus half the cell
diagonal, so a cell whose corner is inside the clearance blocks the turn even if its centre is outside) and the obstacle memory's RAW remembered points (body-frame position
of the depth return, any bearing, not only the forward corridor), both measured as Euclidean distance from the body
centre. The guard distance is NOT used: it is measured to the inflated edge and includes the stop margin. An
obstacle straight ahead at raw 0.6 m stops the robot (inflated edge at 0.3 m = d_margin) yet allows the turn; one at
raw 0.35 m, or a remembered one at 0.35 m (e.g. in the camera blind zone), does not. Memory with an unusable pose
blocks the turn.

Lidar fusion (optional `TickInput.lidar`, a body-frame `LidarFrame`). The forward camera has a ~1.1 m near-field blind
zone (a 0.3 m object nearer than ~0.7 m is invisible). A fresh lidar (|stamp - frame stamp| <= `max_lidar_age_s`) adds:
its clearance is min-combined with the camera's, its points join the camera's in the SAME costmap update (so the turn
rule and the stop use the fused map), and its nearest obstacle (raw body xy) feeds the obstacle memory when nearer than
the camera's. It only TIGHTENS: ground_ok, max_blind, depth freshness and every camera requirement are unchanged.
A lidar that is present but stale/invalid is "unavailable" (event `lidar_unavailable`); with `require_lidar` that, or
a missing lidar, is a hard stop `lidar_stale`. The lidar must also OBSERVE the near field: floor returns in the corridor
at x in (0.3, 1.2) >= `lidar_min_near_ground_points` and bin coverage >= `lidar_min_near_coverage`; otherwise the
near field is unobserved (event `lidar_near_unobserved`, `TickOutput.lidar.near_observed` False) and the runner
behaves exactly as camera-only (the pre-lidar behaviour); with `require_lidar` it is a hard stop
`lidar_near_unobserved`. RESIDUAL RISK: with `require_lidar` False and a lidar that does not observe the near field, a
short obstacle inside the camera blind zone is NOT detected (the lidar never vouches "clear", it simply adds nothing).
`require_lidar=True` is the setting for hardware with a downward-looking lidar.
Self-hits: lidar returns inside a body exclusion cylinder (`lidar_self_radius_m`, default 0.25 m, heights 0.05..1.5 m
above the ground; PLACEHOLDERS) are masked before both the costmap update and `lidar_clearance`, so the robot's own
body never causes a permanent `obstacle_stop`. An obstacle that close would already be a collision.
The lidar does NOT replace negative-obstacle (drop/hole) detection, which stays with the camera.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections import deque
from typing import Callable

import numpy as np

from humanoid_arc_bme.onboard.nav.obstacle import SpeedSeparationGuard
from humanoid_arc_bme.onboard.pipeline.perception.costmap import CostmapConfig, LocalCostmap
from humanoid_arc_bme.onboard.pipeline.perception.ground import analyze_corridor, fit_ground, gravity_align
from humanoid_arc_bme.onboard.pipeline.perception.lidar import LidarFrame, lidar_clearance, self_exclusion_keep
from humanoid_arc_bme.onboard.pipeline.interfaces import DepthFrame, GraspCandidate
from humanoid_arc_bme.onboard.validation import is_num as _is_num
from humanoid_arc_bme.onboard.pipeline.perception.pose_gate import PoseGate
from humanoid_arc_bme.onboard.pipeline.perception.obstacle_memory import ObstacleMemory, ObstacleMemoryConfig
from humanoid_arc_bme.onboard.pipeline.perception.sensing import depth_to_points
from humanoid_arc_bme.onboard.pipeline.stubs import LatestObjectMemory, NullSemantics

from humanoid_arc_bme.onboard.pipeline.runner_config import (  # noqa: F401 - re-exported: this module is the public entry point
    OK, SLOW, STOP, STAGES, ZERO, RunnerConfig, SafetyStatus, TickInput, TickOutput, _validate_cfg)


class PipelineRunner:
    def __init__(self, cfg: RunnerConfig | None = None, pose_gate: PoseGate | None = None,
                 costmap: LocalCostmap | None = None, semantics=None, memory=None, proposer=None,
                 scheduler=None, clock: Callable[[], float] = time.perf_counter):
        self.cfg = cfg or RunnerConfig()
        _validate_cfg(self.cfg)
        self.gate = pose_gate or PoseGate(max_age_s=0.5, max_pos_sigma=0.3, max_yaw_sigma=0.3, max_jump_m=0.3,
                                          max_jump_yaw=0.3)
        self.costmap = costmap or LocalCostmap(CostmapConfig())
        self.semantics = semantics if semantics is not None else NullSemantics()
        self.memory = memory if memory is not None else LatestObjectMemory()
        self.proposer = proposer
        self.scheduler = scheduler
        self._clock = clock
        self._dist = math.inf
        self._mem = ObstacleMemory(ObstacleMemoryConfig(ttl_s=self.cfg.memory_ttl_s, corridor_half_width=self.cfg.corridor_half_width, lateral_margin=self.cfg.memory_lateral_margin))
        self.guard = self._make_guard()
        self.grasp_request: str | None = None
        self._sched_fault = False
        self._hold_left = 0
        self._hard_hold_left = 0     # ticks left of a sticky hold caused by a HARD fault (blocks the turn exception)
        self._pose_ok_now = False
        self._turn_pose = None       # (x, y, yaw) of the pose accepted this tick
        self._tick = 0
        self._approach = False
        self._plane_mem = None
        self._floor_seen = None
        self._events: deque[dict] = deque(maxlen=self.cfg.max_events)
        self._timing = {s: deque(maxlen=self.cfg.timing_window) for s in STAGES}
        self._counts = {OK: 0, SLOW: 0, STOP: 0}
        self._reason_counts: dict[str, int] = {}
        self._pending: deque[dict] = deque(maxlen=self.cfg.max_events)   # events raised outside tick (scheduler hook)
        self._last_stamp: float | None = None   # last ACCEPTED tick stamp
        self._stamp_fault = False               # latched by a bad finite stamp; cleared only by reset()
        self._ev: list[dict] = []               # events of the tick in progress
        self._lidar_res = None

    def _make_guard(self):
        return SpeedSeparationGuard(lambda d: self._dist if d > 0 else None, a_stop=self.cfg.a_stop,
                                    d_margin=self.cfg.d_margin, require_sensor=True)

    def clearance_ahead(self, now: float, max_age_s: float = 0.5) -> float | None:
        """Distance (m) to the nearest obstacle ahead as of the last accepted tick, for a consumer outside the runner (the runtime's
        obstacle stage on manual driving). inf = nothing in range; 0.0 = a fail-closed condition (no ground, no pose, bad frame);
        None = no reading: the runner has not ticked yet, its stamps are faulted, or the last tick is older than `max_age_s` at `now`."""
        last = self._last_stamp
        if last is None or self._stamp_fault or not (0.0 <= now - last <= max_age_s):
            return None
        return self._dist

    def reset(self) -> None:
        """Supported entry after a relocalisation, localizer/clock restart or a latched `bad_stamp`. Clears the
        obstacle memory, costmap, pose gate, guard distance, sticky hold, pending events and the stamp history, so
        the next tick starts from "nothing known" and the first accepted stamp is unconstrained. Deliberately
        fail-closed: nothing is restored automatically (gate.reset() alone leaves memory/costmap/stamps stale).
        The first ticks after reset stop until pose, depth and costmap are healthy again."""
        self._mem.reset()
        self.costmap = type(self.costmap)(self.costmap.cfg)
        self.gate.reset()
        self._dist = math.inf
        self.guard = self._make_guard()
        self._hold_left = 0
        self._hard_hold_left = 0
        self._pending.clear()
        self._lidar_res = None
        self._last_stamp = None
        self._stamp_fault = False
        self._approach = False
        self._plane_mem = None
        self._floor_seen = None

    # ------------------------------------------------------------------ scheduler hook
    def _switch(self, phase: str) -> bool:
        if self.scheduler is None:
            return True
        try:
            self.scheduler.switch(phase)
            self._sched_fault = False
            return True
        except Exception as exc:  # noqa: BLE001 - any scheduler failure is a stop
            self._sched_fault = True
            self._pending.append({"event": "scheduler_failed", "phase": phase, "error": type(exc).__name__})
            return False

    def set_approach(self, on: bool) -> None:
        """Approach mode (the executor sets it only while walking up to an object it is about to pick): the obstacle stop
        uses the raw nearest-obstacle distance with `approach_clearance_m` and the speed is capped at
        `approach_max_speed`. It is cleared by `reset()` and whenever the caller says so; nothing else changes."""
        self._approach = bool(on)

    def start_grasp(self, label: str) -> bool:
        """Begin a grasp request: switch the scheduler to 'grasp'. On failure the request is not set and the
        runner stops (reason scheduler_failed) until a later switch succeeds."""
        if not isinstance(label, str) or not label:
            raise ValueError("label: expected a non-empty string")
        if not self._switch("grasp"):
            return False
        self.grasp_request = label
        return True

    def end_grasp(self) -> bool:
        """Clear the request and switch back to 'walk'. By design the request is cleared even when the switch
        fails; the failure persists as `scheduler_failed` (stop) on every tick until a later switch succeeds
        (call end_grasp()/start_grasp() again). There is no automatic retry."""
        self.grasp_request = None
        return self._switch("walk")

    # ------------------------------------------------------------------ tick
    def tick(self, inp) -> TickOutput:
        t_start = self._clock()
        times: dict[str, float] = {}
        events: list[dict] = list(self._pending)
        self._pending.clear()
        reasons: list[str] = []
        clearance = None
        cands: list[GraspCandidate] = []
        slow = False
        cmd = ZERO
        cmd_in = None
        stamp = math.nan
        self._pose_ok_now = False
        self._turn_pose = None
        self._ev = events
        self._lidar_res = None
        prior_hard_hold = self._hard_hold_left > 0
        try:
            stamp, cmd_in = self._validate(inp, reasons)
            pose_est = self._stage(times, "pose", self._pose_stage, inp, stamp, reasons) if stamp == stamp else None
            cap = math.inf
            if stamp == stamp:
                clearance, cap = self._stage(times, "depth", self._depth_stage, inp, stamp, reasons, times, pose_est)
                if cmd_in is not None and not reasons:
                    tight = self._limit(cmd_in, cap)
                    if cmd_in[0] < 0.0:
                        reasons.append("reverse_unsensed")
                    elif tight != tuple(cmd_in):
                        slow = True
                        reasons.append("speed_capped")
                    cmd = tight
            if self._sched_fault:
                reasons.append("scheduler_failed")
            if self.grasp_request is not None:
                cands = self._stage(times, "grasp", self._grasp_stage, inp, pose_est, events)
        except Exception as exc:  # noqa: BLE001 - last line of defence: fail closed
            reasons.append(f"exception:{type(exc).__name__}")
            cmd = ZERO
        hard = [r for r in reasons if r != "speed_capped"]
        hard_fault = [r for r in hard if r != "obstacle_stop"]
        if hard:
            cmd = ZERO
        turn = None
        if hard and not hard_fault and not prior_hard_hold and cmd_in is not None:
            turn = self._turn_wz(cmd_in[2], stamp)
        if hard_fault:
            self._hard_hold_left = self.cfg.hold_ticks
        elif self._hard_hold_left > 0:
            self._hard_hold_left -= 1
        # stop wins; sticky after a stop (the current command is also zeroed during the hold)
        if hard:
            level = STOP
            self._hold_left = self.cfg.hold_ticks
            if turn is not None:
                cmd = (0.0, 0.0, turn)
                reasons.append("turn_only")
        elif self._hold_left > 0:
            level = STOP
            reasons.append("hold")
            self._hold_left -= 1
            cmd = ZERO
        else:
            level = SLOW if slow else OK
        cmd = tuple(float(c) for c in cmd)
        times["total"] = (self._clock() - t_start) * 1e3
        for k, v in times.items():
            self._timing[k].append(v)
        self._counts[level] += 1
        for r in reasons:
            self._reason_counts[r] = self._reason_counts.get(r, 0) + 1
        rec = {"tick": self._tick, "stamp": float(stamp) if stamp == stamp else None, "status": level,
               "reasons": list(reasons), "timing_ms": {k: round(float(v), 3) for k, v in times.items()}}
        events.append(rec)
        for e in events:
            self._events.append(e)
        self._tick += 1
        return TickOutput(cmd, SafetyStatus(level, tuple(reasons)), clearance, cands, events, self._lidar_res)

    def _turn_wz(self, wz: float, stamp: float) -> float | None:
        """wz for the turn-in-place exception, or None when it is not allowed. Called only when `obstacle_stop` is
        the sole stop reason. Re-checks health itself (defence in depth) and never raises."""
        try:
            if not (self._pose_ok_now and not self._sched_fault and not self._stamp_fault and math.isfinite(wz)):
                return None
            if not self.costmap.healthy(stamp):
                return None
            c = self.cfg
            cm = self.costmap
            half_diag = 0.5 * math.sqrt(2.0) * cm.cfg.resolution   # centre distance - half diagonal <= nearest corner
            near = (np.hypot(cm._cx[:, None], cm._cy[None, :]) - half_diag) <= c.turn_clearance_m
            if bool((cm.occupied(stamp) & near).any()):     # RAW occupancy, not the inflated grid
                return None
            if len(self._mem):                               # RAW remembered obstacles, any bearing, TTL respected
                pose = self._turn_pose
                if pose is None:
                    return None
                for it in self._mem._items:
                    if stamp - it[2] > c.memory_ttl_s:
                        continue
                    xb, yb = self._mem._body(it, pose)
                    if math.hypot(xb, yb) <= c.turn_clearance_m:
                        return None
            lim = c.turn_wz
            return float(max(-lim, min(lim, wz)))
        except Exception:  # noqa: BLE001 - fail closed
            return None

    def _limit(self, cmd, cap):
        """Clamp to the absolute envelope, then apply the obstacle cap to the planar speed hypot(vx, vy) by scaling
        vx, vy and wz together. Never increases a component; invalid cap -> zeros; cap 0 -> zeros (incl. wz)."""
        c = self.cfg
        vx = max(-c.max_vx, min(c.max_vx, cmd[0]))
        vy = max(-c.max_vy, min(c.max_vy, cmd[1]))
        wz = max(-c.max_wz, min(c.max_wz, cmd[2]))
        if not (cap == cap) or cap <= 0.0:   # NaN, zero or negative cap
            return ZERO
        sp = math.hypot(vx, vy)
        if sp > cap:
            k = cap / sp
            vx, vy, wz = vx * k, vy * k, wz * k
        return (vx, vy, wz)

    def _stage(self, times, name, fn, *args):
        t0 = self._clock()
        try:
            return fn(*args)
        finally:
            times[name] = (self._clock() - t0) * 1e3

    # ------------------------------------------------------------------ stages
    def _validate(self, inp, reasons):
        if not isinstance(inp, TickInput):
            reasons.append("bad_input")
            return math.nan, None
        if not _is_num(inp.stamp):
            reasons.append("bad_stamp")
            return math.nan, None
        st = float(inp.stamp)
        last = self._last_stamp
        if self._stamp_fault or (last is not None and (st <= last or st - last > self.cfg.max_stamp_jump_s)):
            self._stamp_fault = True      # latched: only reset() clears it; the bad stamp is NOT recorded
            reasons.append("bad_stamp")
            return math.nan, None
        self._last_stamp = st
        c = inp.command
        try:
            ok = len(c) == 3 and all(_is_num(v) for v in c)
        except TypeError:
            ok = False
        if not ok:
            reasons.append("bad_command")
            return float(inp.stamp), None
        return float(inp.stamp), tuple(float(v) for v in c)

    def _pose_stage(self, inp, stamp, reasons):
        dec = self.gate.check(inp.pose, stamp)
        if dec.accepted:
            self._pose_ok_now = True
            px, py, _ = dec.estimate.pose.position
            self._turn_pose = (px, py, dec.estimate.pose.yaw())
            return dec.estimate
        fresh = self.gate.fresh_pose(stamp)
        if dec.reason in ("none", "stale") and fresh is not None:
            return fresh                    # dropout inside the max age: hold the last accepted pose
        reasons.append(f"pose_{dec.reason}")
        return None

    def _depth_stage(self, inp, stamp, reasons, times, pose_est):
        """Returns (ClearanceResult | None, speed cap). Appends a reason on every failure (cap 0)."""
        cam = self.cfg.guard_camera
        fd = inp.forward_depth if cam == "forward" else inp.down_depth
        if fd is None:
            reasons.append(f"no_{cam}_depth")
            return None, 0.0
        if not isinstance(fd, DepthFrame):
            reasons.append("bad_depth")
            return None, 0.0
        if not (_is_num(inp.roll) and _is_num(inp.pitch)):
            reasons.append("bad_attitude")
            return None, 0.0
        c = self.cfg
        if abs(stamp - fd.stamp) > c.max_depth_age_s:
            reasons.append("depth_stale")
            return None, 0.0
        d = fd.depth
        valid_fraction = float(np.count_nonzero(~np.isnan(d) & (d > 0))) / d.size
        pts = depth_to_points(fd, max_range=c.points_max_range, stride=c.points_stride)
        aligned = gravity_align(pts, inp.roll, inp.pitch)
        plane = fit_ground(aligned, min_inlier_frac=c.approach_ground_frac) if self._approach else fit_ground(aligned)
        plane = self._plane_with_memory(plane, stamp, pose_est)
        if c.camera_self_radius_m > 0.0:                    # the robot's own feet, legs and hands in a down-tilted view (see RunnerConfig)
            aligned = aligned[self_exclusion_keep(aligned, plane, c.camera_self_radius_m)]
        cl = analyze_corridor(aligned, plane, corridor_half_width=c.corridor_half_width, max_range=c.max_range,
                              valid_fraction=valid_fraction, min_valid_fraction=c.min_valid_fraction)
        lres, laligned = self._lidar_stage(inp, stamp, reasons, plane)
        self._lidar_res = lres
        lidar_d = math.inf if lres is None else lres.distance
        # costmap: same points (gravity-aligned so z is height above the floor), plus the lidar's, stop if unhealthy
        t0 = self._clock()
        try:
            self.costmap.update(aligned if laligned is None else np.vstack([aligned, laligned]), stamp)
        except Exception:  # noqa: BLE001
            reasons.append("costmap_update_failed")
        healthy = self.costmap.healthy(stamp)
        near = self.costmap.nearest_obstacle(c.corridor_half_width, c.max_range) if healthy else 0.0
        if self._approach and healthy:
            near = None             # approach mode: the inflated costmap distance is replaced by the raw distance below
        times["costmap"] = (self._clock() - t0) * 1e3
        if not healthy and "costmap_update_failed" not in reasons:
            reasons.append("costmap_unhealthy")
        if not cl.ground_ok:
            reasons.append("ground_not_ok")
        # distance: min of positive/negative clearance and the costmap's nearest cell; NaN -> 0
        # NaN is checked explicitly: min() with a NaN is order dependent and could drop it.
        dneg = self._blind_floor_ok(cl, stamp, pose_est)
        geo_parts = (cl.distance_positive, dneg)
        geo = 0.0 if any(v != v for v in geo_parts) else min(geo_parts)
        dist = min(geo, math.inf if near is None else near)
        dist = 0.0 if lidar_d != lidar_d else min(dist, lidar_d)   # fused: lidar only tightens (min), NaN -> 0
        dist = 0.0 if dist != dist else float(dist)
        self._dist = self._remember(dist, geo, cl, stamp, pose_est, lres) if (cl.ground_ok and pose_est is not None) else dist
        if self._approach:
            # stop at raw clearance `approach_clearance_m` instead of robot radius + d_margin: the guard's own formula then
            # gives cap 0 exactly there. Drops (distance_negative), the lidar and a failed costmap still stop the robot.
            self._dist = max(0.0, self._dist - (c.approach_clearance_m - c.d_margin))
        t0 = self._clock()
        cap = self.guard.speed_cap(stamp, 1)
        times["guard"] = (self._clock() - t0) * 1e3
        if cl.ground_ok and cap <= 0.0:
            reasons.append("obstacle_stop")
        if self._approach:
            cap = min(cap, c.approach_max_speed)
        return cl, (0.0 if reasons else cap)

    def _blind_floor_ok(self, cl, stamp, pose_est) -> float:
        """`distance_negative`, except in approach mode when the floor up to the nearest obstacle was SEEN earlier in this
        approach: the robot only creeps forward, so a floor strip that is now inside the camera's near-field blind zone
        was visible from farther back. Without such a sighting (or without a pose) the conservative value stands."""
        c = self.cfg
        dneg, dpos = cl.distance_negative, cl.distance_positive
        xy = None if pose_est is None else pose_est.pose.position[:2]
        if not self._approach:
            self._floor_seen = None
            return dneg
        if math.isfinite(dpos) and dneg > dpos and xy is not None:
            self._floor_seen = (stamp, xy)                     # floor verified all the way to the obstacle
            return dneg
        seen = self._floor_seen
        if (dneg == 0.0 and cl.ground_ok and seen is not None and xy is not None and math.isfinite(dpos)
                and 0.0 <= stamp - seen[0] <= c.approach_plane_age_s and math.dist(xy, seen[1]) <= c.approach_plane_travel_m):
            return math.inf
        return dneg

    def _plane_with_memory(self, plane, stamp, pose_est):
        """Remember every good floor fit with the pose it was made at. In approach mode only, a failed fit falls back to
        that plane while it is younger than `approach_plane_age_s` and the robot is within `approach_plane_travel_m` of it:
        next to a table the floor is too small a share of the view to fit, but it has not changed. Never used outside
        approach mode, never without a pose, and cleared by reset()."""
        c = self.cfg
        xy = None if pose_est is None else pose_est.pose.position[:2]
        if plane is not None:
            self._plane_mem = (plane, stamp, xy)
            return plane
        mem = self._plane_mem
        if not (self._approach and mem is not None and xy is not None and mem[2] is not None):
            return None
        if 0.0 <= stamp - mem[1] <= c.approach_plane_age_s and math.dist(xy, mem[2]) <= c.approach_plane_travel_m:
            return mem[0]
        return None

    def _lidar_stage(self, inp, stamp, reasons, plane):
        """Returns (LidarClearance | None, gravity-aligned lidar points | None). None = no usable lidar: an event, and
        with `require_lidar` the hard stop `lidar_stale`. With `require_lidar` an unobserved near field (too few
        floor returns) is also a hard stop `lidar_near_unobserved`; without it the camera-only behaviour (known
        blind zone) applies unchanged and the event `lidar_near_unobserved` is logged. Never raises."""
        c = self.cfg
        lf = inp.lidar
        if lf is None:
            if c.require_lidar:
                reasons.append("lidar_stale")
            return None, None
        why = None
        res = aligned = None
        if not isinstance(lf, LidarFrame):
            why = "invalid"
        elif abs(stamp - lf.stamp) > c.max_lidar_age_s:
            why = "stale"
        else:
            try:
                aligned = gravity_align(lf.points, inp.roll, inp.pitch)
                aligned = aligned[self_exclusion_keep(aligned, plane, c.lidar_self_radius_m)]   # own body, before use
                res = lidar_clearance(aligned, plane, c.corridor_half_width, c.max_range, h_min=c.lidar_h_min,
                                      min_near_points=c.lidar_min_near_ground_points,
                                      min_near_coverage=c.lidar_min_near_coverage)
                if not res.ok:
                    why = "invalid"
            except Exception:  # noqa: BLE001 - fail closed: treated as unavailable
                why = "invalid"
        if why is not None:
            self._ev.append({"event": "lidar_unavailable", "why": why})
            if c.require_lidar:
                reasons.append("lidar_stale")
            return None, None
        if not res.near_observed:
            self._ev.append({"event": "lidar_near_unobserved", "ground_points_near": res.ground_points_near,
                             "near_coverage": round(res.near_coverage, 3)})
            if c.require_lidar:
                reasons.append("lidar_near_unobserved")
        return res, aligned

    def _remember(self, meas: float, geo: float, cl, stamp: float, pose_est, lres=None) -> float:
        """Obstacle memory (fed by the camera's nearest obstacle, or the lidar's if nearer). The camera has a near-field blind zone (docs: ground.ClearanceResult.blind_until): a low
        obstacle that was seen and then leaves the view would otherwise read as "clear" and the robot would walk
        into it. The nearest obstacle seen is kept as a map-frame point and re-projected with the (gated) pose; it is
        dropped when its place is visible again and empty, when it is behind the robot or off the corridor, or after
        `memory_ttl_s`. Returns the distance to use (min of measured and remembered)."""
        c = self.cfg
        px, py, _ = pose_est.pose.position
        pose = (px, py, pose_est.pose.yaw())
        near = (geo, 0.0) if (geo < c.max_range) else None   # only depth geometry is remembered, not costmap cells
        if lres is not None and lres.nearest_xy is not None and lres.distance < min(geo, c.max_range):
            near = lres.nearest_xy                           # the lidar saw something nearer than the camera: raw body xy
        self._mem.observe(stamp, pose, near, geo, blind_until=cl.blind_until)
        remembered = self._mem.blocked_distance(stamp, pose)
        if self._approach and math.isfinite(cl.distance_positive):
            return meas      # approach mode: the camera MEASURES the obstacle (a table edge): the coarse one-point memory defers
        return min(meas, remembered)

    def _grasp_stage(self, inp, pose_est, events):
        """Isolated: never raises, never touches the safety result."""
        try:
            return self._grasp(inp, pose_est, events)
        except Exception as exc:  # noqa: BLE001
            events.append({"event": "grasp_error", "error": type(exc).__name__, "msg": str(exc)[:120]})
            return []

    def _grasp(self, inp, pose_est, events):
        dd = inp.down_depth
        if not isinstance(dd, DepthFrame):
            events.append({"event": "grasp_skipped", "reason": "no_down_depth"})
            return []
        if pose_est is None:
            events.append({"event": "grasp_skipped", "reason": "no_pose"})
            return []
        if abs(inp.stamp - dd.stamp) > self.cfg.max_depth_age_s:
            events.append({"event": "grasp_skipped", "reason": "down_depth_stale"})
            return []
        dets = self.semantics.detect(inp.rgb, dd)
        self.memory.update(dets, dd, pose_est)
        match = [d for d in dets if d.label == self.grasp_request and d.mask is not None]
        if not match:
            events.append({"event": "grasp_skipped", "reason": "target_not_detected"})
            return []
        mask = max(match, key=lambda d: d.confidence).mask
        if mask.shape != dd.depth.shape:
            events.append({"event": "grasp_skipped", "reason": "mask_shape"})
            return []
        r = self.cfg.grasp_max_range
        tgt = depth_to_points(dataclasses.replace(dd, depth=np.where(mask, dd.depth, np.nan).astype(dd.depth.dtype)),
                              max_range=r, min_range=0.05)
        rest = depth_to_points(dataclasses.replace(dd, depth=np.where(mask, np.nan, dd.depth).astype(dd.depth.dtype)),
                               max_range=r, min_range=0.05, stride=2)
        if self.proposer is None:
            events.append({"event": "grasp_skipped", "reason": "no_proposer"})
            return []
        out = list(self.proposer.propose(tgt, rest))
        events.append({"event": "grasp", "n_target_points": int(len(tgt)), "n_candidates": len(out)})
        return out

    # ------------------------------------------------------------------ report
    def report(self) -> dict:
        """JSON-serialisable summary: counts, reason counts, per-stage p50/p95/max (ms), bounded recent events."""
        timing = {}
        for k, v in self._timing.items():
            if v:
                a = np.asarray(v, dtype=np.float64)
                timing[k] = {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
                             "max": float(a.max()), "n": int(a.size)}
        return {"ticks": self._tick, "status_counts": dict(self._counts), "reason_counts": dict(self._reason_counts),
                "timing_ms": timing, "events": list(self._events), "grasp_request": self.grasp_request}
