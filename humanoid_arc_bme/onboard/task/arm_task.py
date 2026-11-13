"""ArmPickPlace: one arm and its Dex3 hand carrying out pick and place through the skill contract.

The existing `PickPlaceMachine` (skills/state_pick_place.py) says WHICH named waypoint the wrist should be at and whether
the hand should be closed; this module supplies the geometry and the sensing around it:

* waypoints (pelvis frame, metres) are derived from one grasp pose: `pick` = the grasp, `above_pick` = backed off along
  the approach axis and lifted, `place`/`above_place`/`ready_place` likewise around the place point; `rest` = where the
  arm was when the task began (the safe place to return to);
* motion is a Cartesian setpoint that moves toward the active waypoint at a bounded speed; each tick the IK turns it
  into 7 joint targets, warm-started from the previous solution. The controller (S1) then clamps and rate-limits them;
* success is judged from MEASURED joints only: wrist position by forward kinematics, an object in the hand by the
  Dex3 fingers stalling on it (`Dex3Hand.grasp_state`), "lifted" by the wrist rising after the grasp. Nothing is
  taken from the commands the task itself issued.

The pelvis frame is taken equal to the body frame used by perception (calibrated camera extrinsics map into it).
Waist joints and the other arm are held at their measured values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from humanoid_arc_bme.onboard.g1_layout import LEFT_ARM_JOINTS, LEFT_HAND_JOINTS, RIGHT_ARM_JOINTS, RIGHT_HAND_JOINTS, UPPER_BODY_JOINTS
from humanoid_arc_bme.onboard.pipeline.interfaces import Pose
from humanoid_arc_bme.onboard.pipeline.manipulation.pick_skill import PickSkill, PlaceSkill, invariants
from humanoid_arc_bme.onboard.pipeline.skill_contract import SkillRunner, WorldFacts
from humanoid_arc_bme.onboard.validation import is_num


@dataclass(frozen=True)
class ArmTaskConfig:
    cart_speed: float = 0.12        # m/s   setpoint speed (S1 wrist_speed_max is 0.2)
    ang_speed: float = 0.6          # rad/s
    reach_tol_m: float = 0.02       # wrist within this of a waypoint (measured by FK) = reached
    settle_speed_mps: float = 0.03  # and slower than this
    standoff_m: float = 0.08        # above_pick/ready_place: back off along the approach axis
    lift_m: float = 0.10            # above_pick: and up. Must clear the 3 cm "lifted" threshold with margin: the planned grasp height can sit ~1 cm above the true centre
    place_lift_m: float = 0.06      # above_place height over the place point
    hand_rate_eps: float = 0.25     # closure change per second below which the hand counts as settled
    hand_settle_s: float = 0.3
    grasp_min_closure: float = 0.25  # fingers that stop BELOW this have not closed on anything: that is not "an object in the hand"
    lift_margin_m: float = 0.02      # subtracted from the measured wrist rise before it is reported as the object's lift (height uncertainty of the plan)
    closing_s: float = 0.5           # the hand needs this long after a close command before a stall counts as evidence
    ik_fail_limit: int = 15         # consecutive unconverged IK solves before the task fails
    ik_pos_ok_m: float = 0.015      # an IK solution this close in position is used even if the orientation is not met (a cup is 8 cm wide)
    ik_ori_weight: float = 0.05     # orientation matters little for a power grasp of a cup; position decides
    pick_timeout_s: float = 60.0
    place_timeout_s: float = 40.0

    def __post_init__(self):
        for n in ("cart_speed", "ang_speed", "reach_tol_m", "ik_pos_ok_m", "ik_ori_weight", "settle_speed_mps", "standoff_m", "lift_m", "place_lift_m",
                  "hand_rate_eps", "hand_settle_s", "grasp_min_closure", "closing_s", "lift_margin_m", "pick_timeout_s", "place_timeout_s"):
            v = getattr(self, n)
            if not is_num(v) or v <= 0:
                raise ValueError(f"{n}: must be a finite number > 0, got {v!r}")
        if self.lift_m <= 0.03:
            raise ValueError("lift_m must exceed the skill contract's 3 cm lift threshold")


@dataclass(frozen=True)
class ArmTargets:
    """Joint targets in g1_layout order: body = 17 (waist 3, left arm 7, right arm 7), then 7 per hand (rad)."""

    body: tuple[float, ...]
    left: tuple[float, ...]
    right: tuple[float, ...]


@dataclass(frozen=True)
class ArmStep:
    targets: ArmTargets | None   # None: nothing to command (idle, never started)
    state: str                   # machine state name, "REST", or "IDLE"
    status: str                  # "idle" | "running" | "succeeded" | "failed"
    reason: str = ""


def _nlerp(q0, q1, s: float) -> np.ndarray:
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if float(q0 @ q1) < 0.0:
        q1 = -q1
    q = (1.0 - s) * q0 + s * q1
    return q / np.linalg.norm(q)


def _angle_between(q0, q1) -> float:
    d = abs(float(np.clip(np.dot(q0, q1), -1.0, 1.0)))
    return 2.0 * math.acos(d)


class ArmPickPlace:
    def __init__(self, kin: Mapping[str, object], hands: Mapping[str, object], cfg: ArmTaskConfig | None = None):
        """`kin` / `hands`: {"left": G1ArmKinematics, "right": ...} and {"left": Dex3Hand, ...}."""
        self.cfg = cfg or ArmTaskConfig()
        self.kin, self.hands = dict(kin), dict(hands)
        self._now = 0.0
        self._reset_motion()
        self.holding = False
        self.side: str | None = None

    # ------------------------------------------------------------------ lifecycle
    def _reset_motion(self) -> None:
        self._runner: SkillRunner | None = None
        self._skill = None
        self._mode = "idle"            # idle | pick | place | rest
        self._wp: dict[str, Pose] = {}
        self._sp: tuple[np.ndarray, np.ndarray] | None = None   # Cartesian setpoint (position, quat)
        self._q_goal: np.ndarray | None = None
        self._closure_cmd = 0.0
        self._ik_fails = 0
        self._tool_prev: np.ndarray | None = None
        self._speed = math.inf
        self._closure_prev = math.nan
        self._still_since: float | None = None
        self._close_since: float | None = None
        self._grasp_z = math.nan
        self._last_t = None
        self._place_pos: np.ndarray | None = None
        self._result = ArmStep(None, "IDLE", "idle")

    def _arm_q(self, side: str, q: Mapping[str, float]) -> np.ndarray:
        names = LEFT_ARM_JOINTS if side == "left" else RIGHT_ARM_JOINTS
        return self._vec(q, names)

    def _hand_q(self, side: str, q: Mapping[str, float]) -> np.ndarray:
        return self._vec(q, LEFT_HAND_JOINTS if side == "left" else RIGHT_HAND_JOINTS)

    @staticmethod
    def _vec(q: Mapping[str, float], names) -> np.ndarray:
        try:
            v = np.array([q[n] for n in names], dtype=float)
        except KeyError as e:
            raise ValueError(f"measured joint missing: {e.args[0]}") from None
        if not np.isfinite(v).all():
            raise ValueError("measured joints are not finite")
        return v

    def _pick_side(self, y: float, side: str | None) -> str:
        if side in ("left", "right"):
            return side
        return "left" if y >= 0.0 else "right"

    def start_pick(self, now: float, q: Mapping[str, float], grasp: Pose, side: str | None = None) -> None:
        """Begin a pick at the wrist pose `grasp` (pelvis frame). Does not move anything by itself."""
        if self.holding:
            raise RuntimeError("the hand already holds an object")
        self._reset_motion()
        self._now = now
        self.side = self._pick_side(grasp.position[1], side)
        a = grasp.rotation_matrix()[:, 0]                       # hand x axis = approach direction (fingers)
        p = np.asarray(grasp.position)
        up = np.array([0.0, 0.0, 1.0])
        c = self.cfg
        self._wp = {"pick": grasp,
                    "above_pick": Pose(tuple(p - c.standoff_m * a + c.lift_m * up), grasp.quat)}
        self._begin(q)
        self._grasp_z = float(p[2])
        self._skill = PickSkill(timeout_s=c.pick_timeout_s)
        self._mode = "pick"
        self._runner = SkillRunner(self._skill, lambda: self._now, invariants=invariants(), max_fact_age_s=1.0)
        self._started = False

    def start_place(self, now: float, q: Mapping[str, float], goal: tuple[float, float, float], side: str | None = None) -> None:
        """Begin placing the held object at `goal` (pelvis frame). Needs a held object (the pick that ran before)."""
        if not self.holding or self.side is None:
            raise RuntimeError("nothing is held")
        s = self.side if side is None else side
        if s != self.side:
            raise RuntimeError("the object is in the other hand")
        keep_rest = self._rest
        self._reset_motion()
        self._rest = keep_rest
        self._now = now
        self.side = s
        tool = self.kin[s].fk(self._arm_q(s, q))
        quat = tool.quat
        a = tool.rotation_matrix()[:, 0]
        g = np.asarray(goal, float)
        c = self.cfg
        up = np.array([0.0, 0.0, 1.0])
        self._wp = {"place": Pose(tuple(g), quat),
                    "above_place": Pose(tuple(g + c.place_lift_m * up), quat),
                    "ready_place": Pose(tuple(g - c.standoff_m * a + 0.04 * up), quat)}
        self._place_pos = g
        self._begin(q, keep_rest=True)
        # the object is already held: continue "closed" rather than starting from an open hand
        self._closure_cmd = 1.0
        self._close_since = now - 10.0
        self._still_since = now - 1.0
        self._closure_prev = self.hands[s].joints_to_closure(self._hand_q(s, q))
        self._skill = PlaceSkill(timeout_s=c.place_timeout_s)
        self._mode = "place"
        self._runner = SkillRunner(self._skill, lambda: self._now, invariants=invariants(), max_fact_age_s=1.0)
        self._started = False

    def start_rest(self, now: float, q: Mapping[str, float]) -> None:
        """Return the arm to where it was before the pick (hand open, nothing held)."""
        if self.side is None or self._rest is None:
            raise RuntimeError("no rest pose recorded")
        side, rest = self.side, self._rest
        self._reset_motion()
        self._rest, self.side, self._now = rest, side, now
        self._wp = {"rest": rest}
        self._begin(q, keep_rest=True)
        self._mode = "rest"

    def _begin(self, q: Mapping[str, float], keep_rest: bool = False) -> None:
        s = self.side
        qa = self._arm_q(s, q)
        tool = self.kin[s].fk(qa)
        if not keep_rest:
            self._rest = tool
        self._sp = (np.asarray(tool.position, float), np.asarray(tool.quat, float))
        self._q_goal = qa.copy()
        self._wp.setdefault("rest", self._rest)
        self._tool_prev = np.asarray(tool.position, float)
        self._last_t = self._now

    _rest: Pose | None = None

    # ------------------------------------------------------------------ one tick
    def step(self, now: float, q: Mapping[str, float], flags: Mapping[str, bool]) -> ArmStep:
        """`q`: measured joints by name (waist, arms, hands). `flags`: estop_clear, safety_ok, pose_ok, base_stopped,
        object_visible (booleans from the rest of the system; missing = False = unsafe). Never raises on a failed skill:
        the outcome is in the returned status."""
        if self._mode == "idle":
            return self._result
        if self._result.status in ("succeeded", "failed") and self._mode != "rest":
            return self._hold(q, self._result)
        dt = now - self._last_t if self._last_t is not None else 0.0
        if not (math.isfinite(now) and dt >= 0.0):
            return self._fail("bad_clock")
        self._now = now
        s = self.side
        qa, qh = self._arm_q(s, q), self._hand_q(s, q)
        tool = self.kin[s].fk(qa)
        tp = np.asarray(tool.position, float)
        if dt > 1e-6:
            self._speed = float(np.linalg.norm(tp - self._tool_prev) / dt)
        self._tool_prev = tp
        hand = self.hands[s]
        closure = hand.joints_to_closure(qh)
        hstate = hand.grasp_state(qh, self._closure_cmd)
        settled = self._hand_settled(now, closure)

        wp_name, closure_cmd = None, self._closure_cmd
        if self._mode == "rest":
            wp_name, closure_cmd = "rest", 0.0
        else:
            facts = self._facts(now, tp, closure, hstate, settled, flags)
            if not self._started:
                res = self._runner.start(facts)
                self._started = res.state == "running"
                if not self._started:
                    return self._fail(res.code)
                if not self._skill.start(facts):            # publishes the machine's first command (SkillRunner has no start hook)
                    return self._fail("precondition:start")
            else:
                res = self._runner.tick(facts, max(dt, 1e-3))
                if res.state == "succeeded":
                    self.holding = self._mode == "pick"
                    self._result = ArmStep(self._targets(q, self._q_goal, qh_cmd=self._hand_cmd(hand)), "DONE", "succeeded")
                    return self._result
                if res.state == "failed":
                    return self._fail(res.code)
            cmd = self._skill.command
            wp_name, closure_cmd = cmd.waypoint, cmd.hand_closure
        self._closure_cmd = float(closure_cmd)

        target = self._wp.get(wp_name)
        if target is None:
            return self._fail(f"no_waypoint:{wp_name}")
        self._move_setpoint(target, dt)
        res_ik = self.kin[s].ik(Pose(tuple(self._sp[0]), tuple(self._sp[1])), self._q_goal, ori_weight=self.cfg.ik_ori_weight)
        if res_ik.converged or res_ik.pos_err_m <= self.cfg.ik_pos_ok_m:
            self._q_goal, self._ik_fails = np.asarray(res_ik.q, float), 0
        else:
            self._ik_fails += 1
            if self._ik_fails > self.cfg.ik_fail_limit:
                return self._fail(f"ik_failed:{wp_name}:{res_ik.pos_err_m:.3f}m")
        self._last_t = now
        if self._mode == "rest" and self._reached(tp, target) and settled and self._closure_cmd == 0.0:
            self._result = ArmStep(self._targets(q, self._q_goal, qh_cmd=self._hand_cmd(hand)), "REST", "succeeded")
            self._mode_done_rest()
            return self._result
        state = "REST" if self._mode == "rest" else self._skill.machine.state.value
        self._result = ArmStep(self._targets(q, self._q_goal, qh_cmd=self._hand_cmd(hand)), state, "running")
        return self._result

    # ------------------------------------------------------------------ helpers
    def _mode_done_rest(self) -> None:
        self.holding = False

    def _hand_cmd(self, hand) -> np.ndarray:
        return hand.closure_to_joints(self._closure_cmd)

    def _hold(self, q, last: ArmStep) -> ArmStep:
        s = self.side
        return ArmStep(self._targets(q, self._q_goal, qh_cmd=self._hand_cmd(self.hands[s])), last.state, last.status, last.reason)

    def _fail(self, reason: str) -> ArmStep:
        # Hold the last targets: opening the hand would drop a held object, moving could hit something.
        self._result = ArmStep(self._result.targets, self._result.state, "failed", reason)
        if self._runner is not None:
            self._runner.abort()
        return self._result

    def _reached(self, tp: np.ndarray, wp: Pose) -> bool:
        return float(np.linalg.norm(tp - np.asarray(wp.position))) <= self.cfg.reach_tol_m and self._speed <= self.cfg.settle_speed_mps

    def _hand_settled(self, now: float, closure: float) -> bool:
        if not math.isfinite(closure):
            return False
        prev = self._closure_prev
        self._closure_prev = closure
        dt = max(now - (self._last_t if self._last_t is not None else now), 1e-3)
        moving = math.isfinite(prev) and abs(closure - prev) / dt > self.cfg.hand_rate_eps
        if moving:
            self._still_since = None
            return False
        if self._still_since is None:
            self._still_since = now
        return now - self._still_since >= self.cfg.hand_settle_s

    def _holds(self, now: float, closure: float, hstate, settled: bool) -> bool:
        """An object is in the hand when the fingers, commanded closed for `closing_s`, have settled somewhere between
        `grasp_min_closure` and the commanded closure: they moved, then something stopped them."""
        if self._closure_cmd < 0.9:
            self._close_since = None
            return False
        if self._close_since is None:
            self._close_since = now
        return bool(hstate.stalled and settled and now - self._close_since >= self.cfg.closing_s
                    and math.isfinite(closure) and closure >= self.cfg.grasp_min_closure)

    def _move_setpoint(self, target: Pose, dt: float) -> None:
        p, q = self._sp
        tp, tq = np.asarray(target.position, float), np.asarray(target.quat, float)
        step = self.cfg.cart_speed * dt
        d = tp - p
        n = float(np.linalg.norm(d))
        p = tp if n <= step else p + d * (step / n)
        ang = _angle_between(q, tq)
        s = 1.0 if ang <= self.cfg.ang_speed * dt else (self.cfg.ang_speed * dt) / ang
        self._sp = (p, _nlerp(q, tq, s))

    def _facts(self, now, tp, closure, hstate, settled, flags) -> WorldFacts:
        f = {k: bool(flags.get(k, False)) for k in ("estop_clear", "safety_ok", "pose_ok", "base_stopped", "object_visible")}
        wps = self._wp
        near = lambda n: n in wps and self._reached(tp, wps[n])   # noqa: E731
        held = self._holds(now, closure, hstate, settled)
        f.update(
            grasp_available=self._mode == "pick",
            gripper_empty=(not hstate.stalled) and (not self.holding if self._mode == "pick" else True),
            object_in_hand=held,
            hand_closed=bool(settled and self._closure_cmd >= 0.9 and (hstate.closed or held)),
            hand_open=bool(settled and self._closure_cmd <= 0.05 and hstate.open),
            ph_pregrasp_reached=near("above_pick"),
            ph_grasp_pose_reached=near("pick"),
            ph_place_pose_reached=near("above_place") or near("place"),
        )
        nums = {"object_speed_mps": self._speed}
        if self._mode == "pick":
            nums["object_height_above_table_m"] = max(0.0, float(tp[2]) - self._grasp_z - self.cfg.lift_margin_m) if held else 0.0
        if self._place_pos is not None:
            d = float(np.linalg.norm(tp - self._place_pos))
            nums["object_to_goal_m"] = d
            nums["wrist_to_object_m"] = d
        if self._mode == "place":
            nums["object_height_above_table_m"] = 0.1
        return WorldFacts({**f, **nums}, now)

    def _targets(self, q: Mapping[str, float], q_arm: np.ndarray, qh_cmd: np.ndarray) -> ArmTargets:
        body = {n: float(q[n]) for n in UPPER_BODY_JOINTS}           # everything not being moved stays where it is
        arm_names = LEFT_ARM_JOINTS if self.side == "left" else RIGHT_ARM_JOINTS
        for n, v in zip(arm_names, q_arm):
            body[n] = float(v)
        left = [float(q[n]) for n in LEFT_HAND_JOINTS]
        right = [float(q[n]) for n in RIGHT_HAND_JOINTS]
        if self.side == "left":
            left = [float(v) for v in qh_cmd]
        else:
            right = [float(v) for v in qh_cmd]
        return ArmTargets(tuple(body[n] for n in UPPER_BODY_JOINTS), tuple(left), tuple(right))
