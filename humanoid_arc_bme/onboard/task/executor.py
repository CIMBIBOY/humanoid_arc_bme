"""TaskExecutor: one typed operator command in, base velocity and arm targets out, tick by tick.

    Move / Turn   body servo to a goal fixed in the map frame at submission (DLIO / localizer pose)
    Pick          resolve the named object in the CURRENT scene -> turn toward it and walk to a stand-off ->
                  stand still -> choose a grasp -> ArmPickPlace (pick skill on the skill contract)
    Place         ArmPickPlace (place skill) at a point in front / left / right of the body, then the arm returns to rest
    Stop          always accepted, cancels everything, base zero; a held object stays held
    Status        describes what is visible; changes nothing

The executor only OUTPUTS a desired base command: the pipeline runner tightens it (obstacle caps, stops) and S1 clamps it.
Rules that keep it safe by construction: the base never reverses (no rear sensor) and a lateral goal is reached by turning
toward it; a command is refused while another runs (say stop first); a missing pose, a safety stop that lasts, a lost
target or a failed skill ends the command as FAILED with a reason and leaves the base at zero; the arm holds its last
targets on any failure (opening the hand would drop a held object).
"""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass, field
from typing import Callable, Sequence

from humanoid_arc_bme.onboard.command.resolve import resolve_pick, summarize
from humanoid_arc_bme.onboard.command.types import Command, Move, Pick, Place, Reject, Status, Stop, Turn
from humanoid_arc_bme.onboard.g1_layout import LEFT_HAND_JOINTS, RIGHT_HAND_JOINTS, UPPER_BODY_JOINTS
from humanoid_arc_bme.onboard.pipeline.interfaces import GraspCandidate, Pose
from humanoid_arc_bme.onboard.task.arm_task import ArmPickPlace, ArmTargets
from humanoid_arc_bme.onboard.task.motion import ServoLimits, Slew, go_to_point, turn_to, wrap
from humanoid_arc_bme.onboard.validation import is_num

# Longest a single phase may take before the command fails with a reason (the arm phases have their own skill timeouts).
PHASE_TIMEOUT_S = {"turn": 60.0, "go": 240.0, "approach": 240.0, "settle": 20.0, "plan": 15.0}
IDLE, RUNNING, SUCCEEDED, FAILED = "idle", "running", "succeeded", "failed"
ZERO = (0.0, 0.0, 0.0)
ZERO_CMD = ZERO


@dataclass(frozen=True)
class ExecutorConfig:
    servo: ServoLimits = field(default_factory=ServoLimits)
    # Where the object's centre must stand in the body frame before the arm starts: inside the arm's measured reachable
    # workspace (IK sweep of the URDF: at pelvis height the tool reaches ~0.30 m ahead, y 0..0.30 on the arm's side). The
    # y sign follows the arm side. UNVERIFIED on the robot: a crouch (GR00T height command) would extend the reach.
    reach_x_m: float = 0.25
    reach_y_m: float = 0.05
    max_lateral_m: float = 0.12         # docking: sideways error above this at the final spot fails the pick
    dock_entry_m: float = 0.6           # docking: square up this far before the final spot, in free space, then drive straight in
    reach_tol_m: float = 0.04
    reach_z_range: tuple = (-0.10, 0.12)  # object centre height relative to the pelvis that a standing pick can reach
    object_radius_m: float = 0.04       # the camera sees the near surface; the grasp point is this much deeper
    still_s: float = 0.6                # base must stand still this long before the arm moves
    still_speed_mps: float = 0.03
    grasp_wait_s: float = 2.0           # wait this long for grasp candidates, then use the geometric default
    lost_target_s: float = 4.0          # approach: object unseen this long -> failed
    blocked_s: float = 10.0             # runner stop this long -> failed
    no_pose_s: float = 2.0
    match_radius_m: float = 0.15        # re-detections within this of the target's predicted position are "the same object"
    place_side_step_m: float = 0.11     # "left" / "right" shift the place point this far sideways from the pick spot (inside the arm's reach)
    max_standoff_walk_m: float = 3.0
    palm_to_pinch_m: float = 0.06       # grasp candidate (hand base) -> pinch point along the approach axis (HandSpec.palm_standoff)
    grasp_gate_m: float = 0.08          # a candidate whose pinch point is farther than this from the object centre is ignored
    pelvis_height_m: float = 0.78       # pelvis above the ground while standing (UNMEASURED): perception is ground-referenced, the arm pelvis-referenced

    def __post_init__(self):
        for n in ("reach_x_m", "reach_y_m", "dock_entry_m", "max_lateral_m", "reach_tol_m", "object_radius_m", "still_s", "still_speed_mps", "grasp_wait_s",
                  "lost_target_s", "blocked_s", "no_pose_s", "match_radius_m", "place_side_step_m", "max_standoff_walk_m", "pelvis_height_m", "palm_to_pinch_m", "grasp_gate_m"):
            if not is_num(getattr(self, n)) or getattr(self, n) <= 0:
                raise ValueError(f"{n}: must be a finite number > 0")


@dataclass(frozen=True)
class TaskObservation:
    stamp: float
    pose: tuple[float, float, float] | None      # (x, y, yaw) of the body in the map frame; None = no usable pose
    scene: Sequence[object] = ()                 # perception.SceneObject-like: label, color, position_body, range_m, confidence
    joints: dict | None = None                   # measured joints by name (waist, arms, hands); None if unknown
    base_speed: float = 0.0                      # measured or commanded planar speed, m/s
    safety_stop: bool = False                    # the pipeline runner's stop level this tick
    safe: bool = True                            # estop clear and safety ok (False: no arm motion)
    grasp_candidates: Sequence[GraspCandidate] = ()


@dataclass(frozen=True)
class TaskStatus:
    state: str = IDLE            # idle | running | succeeded | failed
    phase: str = ""
    message: str = ""
    command: str = ""


@dataclass(frozen=True)
class TaskOutput:
    base: tuple[float, float, float]
    arm: ArmTargets | None               # None: the arm is not being commanded (planner default)
    status: TaskStatus
    want_grasp: str | None = None        # label the pipeline runner should run its grasp branch for
    approach: bool = False               # a pick or place is running next to its table: the runner uses its close-range mode
    info: str = ""                       # answer to a Status command / event text


def _missing_joints(obs) -> str:
    """Reason text when the measured arm / hand joints are incomplete (an arm task needs every one), else ''."""
    if obs.joints is None:
        return "joint feedback unavailable: no arm motion"
    missing = [n for n in UPPER_BODY_JOINTS + LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS if n not in obs.joints]
    return f"joint feedback incomplete (missing {missing[0]} and {len(missing) - 1} more): no arm motion" if missing else ""


def default_grasp(obj, cfg: ExecutorConfig) -> Pose:
    """Geometric fallback: a horizontal approach along body +x to the object centre (near surface + radius), hand
    orientation = identity (fingers forward). The scripted baseline, used when the proposer gives nothing executable."""
    x, y, z = obj.position_body
    d = math.hypot(x, y) or 1.0
    return Pose((x + cfg.object_radius_m * x / d, y + cfg.object_radius_m * y / d, z))


def choose_grasp(candidates: Sequence[GraspCandidate], centre_body, palm_to_pinch_m: float, gate_m: float) -> Pose | None:
    """Best-scoring candidate whose pinch point lies within `gate_m` of the believed object centre (`centre_body`),
    returned as the TOOL pose in the body frame. The proposer's pose is the hand base; the IK tool point is the pinch
    point, `palm_to_pinch_m` further along the approach axis (HandSpec.palm_standoff; the real Dex3 value is an open
    calibration item)."""
    best_pose = None
    for c in sorted(candidates, key=lambda c: -c.score):
        a = c.pose.rotation_matrix()[:, 0]
        tip = tuple(float(p + palm_to_pinch_m * ai) for p, ai in zip(c.pose.position, a))
        if math.dist(tip, centre_body) <= gate_m:
            best_pose = Pose(tip, c.pose.quat)
            break
    return best_pose


class TaskExecutor:
    def __init__(self, arm: ArmPickPlace | None = None, cfg: ExecutorConfig | None = None, clock: Callable[[], float] | None = None):
        self.cfg = cfg or ExecutorConfig()
        self.arm = arm
        self.slew = Slew(self.cfg.servo)
        self._clear()
        self.status = TaskStatus()
        self._arm_targets: ArmTargets | None = None   # streamed while the arm holds something or moves
        self._pick_z = math.nan                       # height of the last grasp: objects are put back at this height
        self._info = ""
        self.last_error = ""
        self._last_pose = None
        self._site = None                              # (x, y, yaw) where the last pick/place ended: close-range mode persists nearby

    # ------------------------------------------------------------------ submit
    def _clear(self) -> None:
        self._steps: list[tuple] = []
        self._phase = ""
        self._t_phase = 0.0
        self._target = None            # (label, color, map position (x, y, z))
        self._grasp = None
        self._blocked_since = None
        self._nopose_since = None
        self._still_since = None
        self._lost_since = None
        self._want_grasp = None
        self._cmd_text = ""
        self._arm_started = False
        self._place_side = "front"
        self._side = "left"
        self._dock = "entry"
        self._dock_bearing = None

    def busy(self) -> bool:
        return self.status.state == RUNNING

    def submit(self, cmd: Command, obs: TaskObservation) -> TaskStatus:
        """Accept or refuse `cmd`. Stop is always accepted. Everything else is refused while busy."""
        if isinstance(cmd, Stop):
            self.stop("operator stop")
            return self.status
        if isinstance(cmd, Reject):
            return self._refuse(cmd.reason, "reject")
        if isinstance(cmd, Status):
            self._info = summarize(list(obs.scene))
            return self.status
        if self.busy():
            return self._refuse(f"busy ({self.status.phase}): say stop first", type(cmd).__name__)
        if obs.pose is None and isinstance(cmd, (Move, Turn, Pick)):
            return self._refuse("no pose estimate yet", type(cmd).__name__)
        self._clear()
        self.slew.reset()
        name = type(cmd).__name__
        try:
            if isinstance(cmd, Move):
                return self._start_move(cmd, obs)
            if isinstance(cmd, Turn):
                x, y, yaw = obs.pose
                self._steps = [("turn", wrap(yaw + cmd.angle_rad))]
                return self._begin(name, "turn")
            if isinstance(cmd, Pick):
                return self._start_pick(cmd, obs)
            if isinstance(cmd, Place):
                return self._start_place(cmd, obs)
        except Exception as exc:  # noqa: BLE001 - a bad internal state must never crash the control loop
            self.stop(f"internal error: {type(exc).__name__}")
            return self._finish(FAILED, f"internal error: {type(exc).__name__}: {exc}")
        return self._refuse("unsupported command", name)

    def _refuse(self, why: str, name: str) -> TaskStatus:
        if not self.busy():
            self.status = TaskStatus(FAILED, "", why, name)
        return TaskStatus(FAILED, "", why, name)

    def _begin(self, name: str, phase: str) -> TaskStatus:
        self._cmd_text = name
        self._phase = phase
        self.status = TaskStatus(RUNNING, phase, "", name)
        return self.status

    def _start_move(self, cmd: Move, obs) -> TaskStatus:
        if cmd.forward_m < 0:
            return self._refuse("backward motion is disabled: there is no rear sensor", "Move")
        x, y, yaw = obs.pose
        if cmd.forward_m > 0 and cmd.left_m != 0:
            return self._refuse("one direction at a time", "Move")
        if cmd.forward_m > 0:
            goal = (x + cmd.forward_m * math.cos(yaw), y + cmd.forward_m * math.sin(yaw))
            self._steps = [("go", goal, yaw)]
        else:   # lateral: turn toward the side, then walk forward (the travel direction stays inside the sensed corridor)
            side = 1.0 if cmd.left_m > 0 else -1.0
            yaw2 = wrap(yaw + side * math.pi / 2)
            d = abs(cmd.left_m)
            goal = (x + d * math.cos(yaw2), y + d * math.sin(yaw2))
            self._steps = [("turn", yaw2), ("go", goal, yaw2)]
        return self._begin("Move", self._steps[0][0])

    def _start_pick(self, cmd: Pick, obs) -> TaskStatus:
        if self.arm is None:
            return self._refuse("this robot has no arm configured", "Pick")
        why = _missing_joints(obs)
        if why:
            return self._refuse(why, "Pick")
        if self.arm.holding:
            return self._refuse("already holding an object: place it first", "Pick")
        hit = resolve_pick(cmd, list(obs.scene))
        if isinstance(hit, Reject):
            return self._refuse(hit.reason, "Pick")
        zc = hit.position_body[2] - self.cfg.pelvis_height_m
        lo, hi = self.cfg.reach_z_range
        if not lo <= zc <= hi:
            return self._refuse(f"the {hit.color} {hit.label} is {zc:+.2f} m from pelvis height: out of reach for a standing pick", "Pick")
        self._side = "left" if hit.position_body[1] >= 0.0 else "right"
        self._dock = "entry"
        self._dock_bearing = None
        self._target = (hit.label, hit.color, self._centre_map(obs.pose, hit))
        self._steps = [("approach",)]
        return self._begin("Pick", "approach")

    def _start_place(self, cmd: Place, obs) -> TaskStatus:
        if self.arm is None or not self.arm.holding:
            return self._refuse("nothing is held", "Place")
        why = _missing_joints(obs)
        if why:
            return self._refuse(why, "Place")
        self._place_side = cmd.side
        self._steps = [("settle",), ("place",), ("rest",)]
        return self._begin("Place", "settle")

    def stop(self, why: str = "stop") -> None:
        """Cancel the running command. The base goes to zero at once; the arm holds its last targets."""
        self._want_grasp = None
        if self.arm is not None and getattr(self.arm, "_runner", None) is not None:
            self.arm._runner.abort()
        was_running = self.status.state == RUNNING
        cmd = self._cmd_text or self.status.command
        self._clear()
        self.slew.reset()
        if was_running:
            self.status = TaskStatus(FAILED, "", why, cmd)

    def _finish(self, state: str, msg: str) -> TaskStatus:
        if self._cmd_text in ("Pick", "Place") and self._last_pose is not None:
            self._site = self._last_pose
        self.status = TaskStatus(state, "", msg, self._cmd_text or self.status.command)
        self._want_grasp = None
        self.slew.reset()
        return self.status

    # ------------------------------------------------------------------ tick
    @staticmethod
    def _to_map(pose, p):
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        return (x + c * p[0] - s * p[1], y + s * p[0] + c * p[1], p[2])

    def _centre_map(self, pose, obj):
        """Map position of the object's CENTRE: the camera sees the near surface, the centre is `object_radius_m` deeper
        along the horizontal line of sight."""
        x, y, z = obj.position_body
        d = math.hypot(x, y) or 1.0
        r = self.cfg.object_radius_m
        return self._to_map(pose, (x + r * x / d, y + r * y / d, z))

    @staticmethod
    def _to_body(pose, p):
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = p[0] - x, p[1] - y
        return (c * dx + s * dy, -s * dx + c * dy, p[2])

    def tick(self, obs: TaskObservation, dt: float) -> TaskOutput:
        """Advance one control step. Never raises: an internal error ends the command as FAILED with base zero."""
        try:
            out = self._tick(obs, dt)
        except Exception as exc:  # noqa: BLE001
            self.last_error = traceback.format_exc()      # kept for the run log; the operator sees the one-line status
            self.stop("internal error")
            self._finish(FAILED, f"internal error: {type(exc).__name__}: {exc}")
            out = TaskOutput(ZERO, self._arm_hold(obs), self.status)
        return out

    def _close_range(self) -> bool:
        """Close-range guard mode: while a pick or place runs, and afterwards until the robot has walked 0.5 m or turned
        ~70 deg away from the table it worked at (so it can leave it)."""
        picking = self.busy() and self._cmd_text in ("Pick", "Place") and (self._phase != "approach" or self._dock == "dock")
        return picking or self._site is not None

    def _arm_hold(self, obs) -> ArmTargets | None:
        return self._arm_targets if (self.arm is not None and (self.arm.holding or self._arm_targets is not None)) else None

    def _tick(self, obs: TaskObservation, dt: float) -> TaskOutput:
        info, self._info = self._info, ""
        if obs.pose is not None:
            self._last_pose = obs.pose
            if self._site is not None and (math.hypot(obs.pose[0] - self._site[0], obs.pose[1] - self._site[1]) > 0.5
                                           or abs(wrap(obs.pose[2] - self._site[2])) > 1.2):
                self._site = None                    # walked or turned away from the table: back to the normal guard
        if not self.busy():
            return TaskOutput(ZERO, self._arm_hold(obs), self.status, None, approach=self._close_range(), info=info)
        cfg, now = self.cfg, obs.stamp
        if self._t_phase == 0.0:
            self._t_phase = now
        # pose supervision (every motion phase needs it)
        if obs.pose is None:
            self._nopose_since = self._nopose_since if self._nopose_since is not None else now
            if now - self._nopose_since > cfg.no_pose_s:
                self._finish(FAILED, "pose estimate lost")
            return TaskOutput(ZERO, self._arm_hold(obs), self.status, self._want_grasp, info=info)
        self._nopose_since = None
        # a lasting runner stop ends the command
        if obs.safety_stop:
            self._blocked_since = self._blocked_since if self._blocked_since is not None else now
            if now - self._blocked_since > cfg.blocked_s and self._phase in ("turn", "go", "approach", "walk"):
                self._finish(FAILED, "blocked: the safety layer has stopped the robot")
        else:
            self._blocked_since = None
        if not self.busy():
            return TaskOutput(ZERO, self._arm_hold(obs), self.status, None, info=info)

        limit = PHASE_TIMEOUT_S.get(self._phase)
        if limit is not None and now - self._t_phase > limit:
            self._finish(FAILED, f"timed out in {self._phase} after {limit:.0f} s")
            return TaskOutput(ZERO, self._arm_hold(obs), self.status, None, info=info)
        handler = getattr(self, f"_p_{self._phase}")
        want, arm = handler(obs, dt)
        base = self.slew(want, dt) if self.busy() else ZERO
        if not self.busy():
            self.slew.reset()
        return TaskOutput(base, arm if arm is not None else self._arm_hold(obs), self.status, self._want_grasp,
                          approach=self._close_range(), info=info)

    # ------------------------------------------------------------------ phases
    def _next(self, obs) -> None:
        """Pop the finished step; start the next or finish the command."""
        self._steps.pop(0)
        self._t_phase = obs.stamp
        self._still_since = None
        if not self._steps:
            self._finish(SUCCEEDED, self._done_message())
            return
        self._phase = self._steps[0][0]
        self.status = TaskStatus(RUNNING, self._phase, "", self._cmd_text)

    def _done_message(self) -> str:
        c = self._cmd_text
        if c == "Pick" and self._target:
            return f"picked the {self._target[1]} {self._target[0]}"
        return {"Move": "arrived", "Turn": "turned", "Place": "placed"}.get(c, "done")

    def _p_turn(self, obs, dt):
        want, done = turn_to(obs.pose, self._steps[0][1], self.cfg.servo)
        if done:
            self._next(obs)
        return want, None

    def _p_go(self, obs, dt):
        _, goal, hold = self._steps[0]
        want, done = go_to_point(obs.pose, goal, self.cfg.servo, hold_yaw=hold)
        if done:
            self._next(obs)
        return want, None

    def _find_target(self, obs):
        """The scene object that is the target: same label and colour, nearest to the target's predicted position."""
        label, color, pos = self._target
        pred = self._to_body(obs.pose, pos)
        best, bd = None, self.cfg.match_radius_m
        for o in obs.scene:
            if o.label == label and (color is None or o.color == color):
                d = math.dist(o.position_body[:2], pred[:2])
                if d <= bd:
                    best, bd = o, d
        return best

    def _p_approach(self, obs, dt):
        cfg = self.cfg
        seen = self._find_target(obs)
        if seen is not None:
            self._lost_since = None
            label, color, _ = self._target
            self._target = (label, color, self._centre_map(obs.pose, seen))   # refresh from the camera
        else:
            self._lost_since = self._lost_since if self._lost_since is not None else obs.stamp
            if obs.stamp - self._lost_since > cfg.lost_target_s:
                self._finish(FAILED, "lost sight of the object")
                return ZERO, None
        tx, ty, _ = self._target[2]
        x, y, yaw = obs.pose
        dist = math.hypot(tx - x, ty - y)
        if dist > cfg.max_standoff_walk_m:
            self._finish(FAILED, f"the object is {dist:.1f} m away: too far")
            return ZERO, None
        if self._dock_bearing is None:
            self._dock_bearing = math.atan2(ty - y, tx - x)      # the docking axis is fixed by the first line of sight
        bearing = self._dock_bearing
        # Final spot P: the object's centre at (reach_x, +-reach_y) in the body frame, i.e. distance `rho` from the object
        # along the line of sight and a heading turned by the offset angle toward the arm side. Docking in three steps so
        # the robot never rotates next to the table: walk to an entry point `dock_entry_m` short of P along the final
        # heading, square up there in free space, then drive straight in (close-range guard mode).
        sgn = 1.0 if self._side == "left" else -1.0
        rho = math.hypot(cfg.reach_x_m, cfg.reach_y_m)
        yaw_goal = wrap(bearing - sgn * math.atan2(cfg.reach_y_m, cfg.reach_x_m))
        p = (tx - rho * math.cos(bearing), ty - rho * math.sin(bearing))
        u = (math.cos(yaw_goal), math.sin(yaw_goal))
        entry = (p[0] - cfg.dock_entry_m * u[0], p[1] - cfg.dock_entry_m * u[1])
        if self._dock == "entry":
            if math.dist((x, y), entry) <= 2 * cfg.servo.pos_tol or math.dist((x, y), p) <= cfg.dock_entry_m:
                self._dock = "align"
            else:
                want, _ = go_to_point(obs.pose, entry, cfg.servo)
                return want, None
        if self._dock == "align":
            want, done = turn_to(obs.pose, yaw_goal, cfg.servo)
            if done:
                self._dock = "dock"
            return want, None
        # straight-in: judge progress ALONG the final heading; the sideways error cannot be corrected without vy, so it is
        # only checked (the arm's reach tolerates a few centimetres, not more)
        dx, dy = p[0] - x, p[1] - y
        along = dx * u[0] + dy * u[1]
        lateral = -dx * u[1] + dy * u[0]
        if along <= cfg.reach_tol_m:
            if abs(lateral) > cfg.max_lateral_m:
                self._finish(FAILED, f"could not line up with the object ({lateral:+.2f} m sideways)")
                return ZERO_CMD, None
            self._steps = [("settle",), ("plan",), ("grasp",)]
            self._phase, self._t_phase, self._still_since = "settle", obs.stamp, None
            self.status = TaskStatus(RUNNING, "settle", "", self._cmd_text)
            return ZERO_CMD, None
        ahead = (x + along * u[0], y + along * u[1])
        want, _ = go_to_point(obs.pose, ahead, cfg.servo, hold_yaw=yaw_goal)
        return want, None

    def _p_settle(self, obs, dt):
        """Stand still for `still_s` before the arm moves."""
        if obs.base_speed <= self.cfg.still_speed_mps and not obs.safety_stop:
            self._still_since = self._still_since if self._still_since is not None else obs.stamp
            if obs.stamp - self._still_since >= self.cfg.still_s:
                self._next(obs)
        else:
            self._still_since = None
        return ZERO, None

    def _p_plan(self, obs, dt):
        """Ask the pipeline for grasp candidates; choose one (or the geometric default after `grasp_wait_s`)."""
        self._want_grasp = self._target[0]
        seen = self._find_target(obs)
        if seen is None:
            if obs.stamp - self._t_phase > self.cfg.lost_target_s:
                self._finish(FAILED, "lost sight of the object")
            return ZERO, None
        centre = self._to_body(obs.pose, self._target[2])       # the object's centre in the body frame
        pose = choose_grasp(obs.grasp_candidates, centre, self.cfg.palm_to_pinch_m, self.cfg.grasp_gate_m)
        if pose is None and obs.stamp - self._t_phase >= self.cfg.grasp_wait_s:
            pose = Pose(tuple(centre))                          # the geometric default: pinch point at the centre
        if pose is not None:
            x, y, z = pose.position
            self._grasp = Pose((x, y, z - self.cfg.pelvis_height_m), pose.quat)   # body (ground) frame -> pelvis frame
            self._pick_z = float(self._grasp.position[2])
            self._want_grasp = None
            self._next(obs)
        return ZERO, None

    def _flags(self, obs) -> dict:
        return {"estop_clear": obs.safe, "safety_ok": obs.safe and not obs.safety_stop, "pose_ok": obs.pose is not None,
                "base_stopped": obs.base_speed <= self.cfg.still_speed_mps, "object_visible": True}

    def _arm_step(self, obs, started: bool):
        if obs.joints is None:
            self._finish(FAILED, "joint state unavailable")
            return None
        res = self.arm.step(obs.stamp, obs.joints, {**self._flags(obs), "object_visible": started or (self._target is not None and self._find_target(obs) is not None)})
        if res.targets is not None:
            self._arm_targets = res.targets
        return res

    def _p_grasp(self, obs, dt):
        if not self._arm_started:
            if obs.joints is None:
                self._finish(FAILED, "joint state unavailable")
                return ZERO, None
            self.arm.start_pick(obs.stamp, obs.joints, self._grasp, side=self._side)
            self._arm_started = True
        res = self._arm_step(obs, True)
        if res is None:
            return ZERO, None
        if res.status == "succeeded":
            self._arm_started = False
            self._next(obs)
        elif res.status == "failed":
            self._arm_started = False
            self._finish(FAILED, f"pick failed: {res.reason}")
        return ZERO, res.targets

    def _p_place(self, obs, dt):
        if not self._arm_started:
            if obs.joints is None:
                self._finish(FAILED, "joint state unavailable")
                return ZERO, None
            # Only the area the holding arm can reach is a valid place point: the pick spot, shifted sideways for left / right.
            sgn = 1.0 if self.arm.side == "left" else -1.0
            shift = {"front": 0.0, "left": 1.0, "right": -1.0}[self._place_side] * self.cfg.place_side_step_m
            self.arm.start_place(obs.stamp, obs.joints, (self.cfg.reach_x_m + 0.01, sgn * self.cfg.reach_y_m + shift, self._pick_z + 0.005))
            self._arm_started = True
        res = self._arm_step(obs, True)
        if res is None:
            return ZERO, None
        if res.status == "succeeded":
            self._arm_started = False
            self._next(obs)
        elif res.status == "failed":
            self._arm_started = False
            self._finish(FAILED, f"place failed: {res.reason}")
        return ZERO, res.targets

    def _p_rest(self, obs, dt):
        if not self._arm_started:
            self.arm.start_rest(obs.stamp, obs.joints)
            self._arm_started = True
        res = self._arm_step(obs, True)
        if res is None:
            return ZERO, None
        if res.status == "succeeded":
            self._arm_started = False
            self._arm_targets = None          # the arm is home: hand control back to the planner
            self._next(obs)
        elif res.status == "failed":
            self._arm_started = False
            self._finish(FAILED, f"return to rest failed: {res.reason}")
        return ZERO, res.targets
