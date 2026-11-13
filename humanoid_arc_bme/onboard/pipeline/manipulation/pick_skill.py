"""D2: PickSkill / PlaceSkill, the existing PickPlaceMachine on the strict skill contract (stdlib + numpy via interfaces.is_num).

The machine (skills/state_pick_place.py) is unchanged and only used as the motion expert. Success is judged by the
contract's postconditions from sensor facts, never by machine.state (a DONE machine with disagreeing facts keeps
running until timeout). If the machine ends FAILED the step raises -> failure code ``exception`` with the machine's
reason in the log (the contract has no better code).

FACT NAMES (single contract perception/sim -> skill layer). bool unless a unit is given. Missing = unknown = False.
  Safety / pre:   estop_clear, safety_ok, pose_ok, base_stopped, gripper_empty, object_visible, grasp_available
  Sensor truth:   object_height_above_table_m [m], object_in_hand (object inside the hand pocket, <= 8 cm),
                  hand_closed, hand_open, hand_contact (optional; if present it must be True for pick success),
                  object_to_goal_m [m], object_speed_mps [m/s], wrist_to_object_m [m]
  Phase hints for the machine only (never used for success): ph_pregrasp_reached, ph_grasp_pose_reached,
                  ph_place_pose_reached (wrist at the named waypoint).
Thresholds (proposed, same as state_runner): lift 0.03 m, place 0.02 m, rest 0.05 m/s, hand clear 0.20 m.

START: ``skill.command`` is None until ``start(facts)`` has passed the skill's preconditions, so nothing is
actuated by merely constructing a skill. SkillRunner has no start hook, so the first ``step`` calls ``start(facts)``
itself (the runner has already checked the preconditions on its own start; a failing re-check raises -> failure code
``exception`` and the machine is never touched). PlaceSkill sets the machine's CARRY start state inside ``start``.

ACTUATION: after each step ``skill.command`` is the machine's Command (waypoint name, hand closure 0..1) for the
caller to execute. abort() FREEZES the machine (never stepped again) and holds the LAST waypoint and LAST hand
closure: it does not open the hand, so a held object is not dropped, and does not move the arm. The runner also
calls abort() after success, so after a pick the hand stays closed on the object at the lifted pose.
"""

from __future__ import annotations

from typing import Mapping, Optional

from humanoid_arc_bme.onboard.skills.state_pick_place import Command, Observation, PickPlaceMachine, State
from humanoid_arc_bme.onboard.validation import is_num
from humanoid_arc_bme.onboard.pipeline.skill_contract import Condition, WorldFacts

LIFT_M, PLACE_M, REST_MPS, CLEAR_M, HELD_M = 0.03, 0.02, 0.05, 0.20, 0.08
CLOSED_MIN, OPEN_MAX = 0.95, 0.05

# base_stopped is a mid-run invariant for pick AND place: if the base starts moving while the arm works, the skill
# aborts (SkillRunner code ``postcondition_lost``). object_visible is deliberately NOT an invariant: after the grasp
# the hand legitimately occludes the object, so losing sight of it is expected and not a fault.
_INVARIANTS = ("estop_clear", "safety_ok", "pose_ok", "base_stopped")


def _invariants():
    return [Condition.is_true(n) for n in _INVARIANTS]


def _num(f: WorldFacts, name: str):
    v = f.get(name)
    return None if v is None or isinstance(v, bool) else v


def _nonneg(f: WorldFacts, name: str):
    """A distance or speed: negative is a sensor/sign error and is treated as unknown (never as 'at the goal')."""
    v = _num(f, name)
    return v if v is not None and v >= 0.0 else None


def _object_lifted(f: WorldFacts) -> bool:
    h = _num(f, "object_height_above_table_m")
    if h is None or h <= LIFT_M or f.get("object_in_hand") is not True or f.get("hand_closed") is not True:
        return False
    return f.get("hand_contact") is not False if "hand_contact" not in f else f.get("hand_contact") is True


def _object_at_goal(f: WorldFacts) -> bool:
    d, v = _nonneg(f, "object_to_goal_m"), _nonneg(f, "object_speed_mps")
    return d is not None and v is not None and d < PLACE_M and v < REST_MPS


def _hand_empty(f: WorldFacts) -> bool:
    return f.get("object_in_hand") is False and f.get("hand_open") is True


def observation_from_facts(f: WorldFacts) -> Observation:
    """Facts -> machine Observation (the only coupling to the machine)."""
    t = lambda n: f.get(n) is True  # noqa: E731
    d, w = _nonneg(f, "object_to_goal_m"), _nonneg(f, "wrist_to_object_m")
    h, v = _num(f, "object_height_above_table_m"), _nonneg(f, "object_speed_mps")
    held = t("object_in_hand")
    return Observation(
        object_seen=t("object_visible"),
        pregrasp_reached=t("ph_pregrasp_reached"),
        grasp_pose_reached=t("ph_grasp_pose_reached"),
        closure_confirmed=t("hand_closed") and held,
        object_held=held,
        lifted=h is not None and h > LIFT_M,
        place_pose_reached=t("ph_place_pose_reached"),
        placed=d is not None and v is not None and d < PLACE_M and v < REST_MPS,
        hand_open=t("hand_open"),
        hand_clear=w is not None and w >= CLEAR_M,
    )


class _MachineSkill:
    name = ""
    start_state: Optional[State] = None

    def __init__(self, timeout_s: float = 30.0, machine: Optional[PickPlaceMachine] = None):
        self.timeout_s = timeout_s
        self.machine = machine if machine is not None else PickPlaceMachine()
        self.command: Optional[Command] = None   # None until start() passed the preconditions: nothing actuated
        self.started = False
        self.aborted = False
        self.aborts = 0  # times abort() was called (idempotence is checked on actuation, see abort_actuations)
        self.abort_actuations = 0
        self._t = 0.0

    def start(self, facts: WorldFacts) -> bool:
        """Check the preconditions on `facts`; only if all hold, set the start state and publish the first command.
        Returns False (and touches nothing) otherwise. Idempotent once started."""
        if self.started:
            return True
        for c in self.preconditions:
            try:
                ok = c.check(facts) is True
            except Exception:  # noqa: BLE001 - a failing check is a failed precondition
                ok = False
            if not ok:
                return False
        if self.start_state is not None:
            self.machine.state = self.start_state
        self.started = True
        self.command = self.machine.command()
        return True

    def step(self, facts: WorldFacts, dt: float) -> None:
        if self.aborted:
            return
        if not self.start(facts):
            raise RuntimeError("preconditions do not hold at the first step")
        self._t += dt
        self.command = self.machine.step(self._t, observation_from_facts(facts))
        if self.machine.state is State.FAILED:
            raise RuntimeError(f"machine failed: {self.machine.reason}")

    def abort(self) -> None:
        self.aborts += 1
        if self.aborted:
            return
        self.aborted = True
        self.abort_actuations += 1  # self.command is left as is: hold last waypoint and hand closure


class PickSkill(_MachineSkill):
    name = "pick"

    def __init__(self, timeout_s: float = 30.0, machine=None):
        super().__init__(timeout_s, machine)
        self.preconditions = [Condition.is_true(n) for n in
                              ("gripper_empty", "object_visible", "base_stopped", "safety_ok", "grasp_available")]
        self.postconditions = [Condition("object_lifted", _object_lifted)]


class PlaceSkill(_MachineSkill):
    name = "place"
    start_state = State.CARRY  # the object is already held: skip locate..lift

    def __init__(self, timeout_s: float = 20.0, machine=None):
        super().__init__(timeout_s, machine)
        self.preconditions = [Condition.is_true(n) for n in
                              ("object_in_hand", "hand_closed", "base_stopped", "safety_ok")]
        self.postconditions = [Condition("object_at_goal", _object_at_goal), Condition("hand_empty", _hand_empty)]


def invariants():
    """Fresh invariant list for SkillRunner(skill, clock, invariants=invariants())."""
    return _invariants()


def facts_from_sim(obs: Mapping, stamp: Optional[float]) -> WorldFacts:
    """Sim observation dict -> WorldFacts. Keys (all optional; missing = unknown): the safety/pre bools above,
    object_height_above_table_m, object_to_pocket_m [m], hand_closure (0..1), hand_contact, object_to_goal_m,
    object_speed_mps, wrist_to_object_m, ph_* hints. Derived: object_in_hand = pocket < 8 cm, hand_closed = closure
    >= 0.95, hand_open = closure <= 0.05. gripper_empty is passed through only if given (a sensor, not derived)."""
    out = {k: v for k, v in dict(obs).items()
           if k not in ("object_to_pocket_m", "hand_closure")}
    p, c = obs.get("object_to_pocket_m"), obs.get("hand_closure")
    if is_num(p) and p >= 0.0:        # a negative distance is a sign error -> unknown
        out["object_in_hand"] = bool(p <= HELD_M)
    if is_num(c) and 0.0 <= c <= 1.0:   # closure outside [0, 1] -> unknown (neither closed nor open)
        out["hand_closed"], out["hand_open"] = bool(c >= CLOSED_MIN), bool(c <= OPEN_MAX)
    return WorldFacts(out, stamp)
