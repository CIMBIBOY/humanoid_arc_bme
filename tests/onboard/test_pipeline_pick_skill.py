"""D2 PickSkill/PlaceSkill on the real PickPlaceMachine; CPU only."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from humanoid_arc_bme.onboard.pipeline.manipulation.pick_skill import PickSkill, PlaceSkill, facts_from_sim, invariants
from humanoid_arc_bme.onboard.pipeline.skill_contract import Condition, SkillLog, SkillRunner, WorldFacts
from humanoid_arc_bme.onboard.skills.state_pick_place import State

DT = 0.1
SAFE = dict(estop_clear=True, safety_ok=True, pose_ok=True, base_stopped=True)
PRE = dict(SAFE, gripper_empty=True, object_visible=True, grasp_available=True)


class Clk:
    t = 0.0

    def __call__(self):
        return self.t


def mk(skill, **kw):
    c = Clk()
    r = SkillRunner(skill, c, invariants=invariants(), log=SkillLog(), **kw)
    return r, c


def tick(r, c, facts):
    c.t = round(c.t + DT, 10)
    return r.tick(WorldFacts(facts, c.t), DT)


def start(r, c, facts):
    return r.start(WorldFacts(facts, c.t))


# phases of a pick: dict of facts per phase
SEQ = [
    dict(PRE),
    dict(PRE, ph_pregrasp_reached=True),
    dict(PRE, ph_grasp_pose_reached=True),
    dict(SAFE, object_visible=True, hand_closed=True, object_in_hand=True, hand_contact=True),
    dict(SAFE, object_visible=True, hand_closed=True, object_in_hand=True, hand_contact=True,
         object_height_above_table_m=0.05),
]


def test_pick_success_path_with_agreeing_facts():
    s = PickSkill()
    r, c = mk(s)
    assert start(r, c, PRE).state == "running"
    res = None
    for f in SEQ:
        for _ in range(2):
            res = tick(r, c, f)
    assert res.state == "succeeded"
    assert s.machine.state is State.CARRY  # the machine was only mid-way; facts decided
    assert s.aborts == 1 and s.command.hand_closure == 1.0  # hold closed


def test_machine_done_but_facts_disagree_runs_to_timeout():
    s = PickSkill(timeout_s=20.0)
    r, c = mk(s)
    start(r, c, PRE)
    phases = [dict(PRE, ph_pregrasp_reached=True), dict(PRE, ph_grasp_pose_reached=True),
              dict(SAFE, object_visible=True, hand_closed=True, object_in_hand=True),  # no height, no lift
              ]
    for f in phases:
        tick(r, c, f)
    # make the machine believe everything: lift hint via object_in_hand but height stays low
    res = None
    while c.t < 25:
        res = tick(r, c, dict(SAFE, object_visible=True, hand_closed=True, object_in_hand=True,
                              object_height_above_table_m=0.0))
        if res.terminal:
            break
    assert res.state == "failed" and res.code in ("timeout", "exception")
    assert res.state != "succeeded"
    # direct: force the machine to DONE and keep facts wrong
    s2 = PickSkill(timeout_s=5.0)
    s2.machine.state = State.DONE
    r2, c2 = mk(s2)
    start(r2, c2, PRE)
    last = None
    for _ in range(80):
        last = tick(r2, c2, dict(PRE, object_height_above_table_m=0.0))
        if last.terminal:
            break
    assert last.code == "timeout"


def test_lift_needs_contact_if_reported():
    s = PickSkill(timeout_s=3.0)
    r, c = mk(s)
    start(r, c, PRE)
    f = dict(SAFE, hand_closed=True, object_in_hand=True, object_height_above_table_m=0.1, hand_contact=False)
    res = None
    for _ in range(40):
        res = tick(r, c, f)
        if res.terminal:
            break
    assert res.code in ("timeout", "exception")


def test_precondition_failures_named_and_nothing_moves():
    for name in ("gripper_empty", "object_visible", "base_stopped", "safety_ok", "grasp_available", "estop_clear",
                 "pose_ok"):
        s = PickSkill()
        r, c = mk(s)
        res = start(r, c, dict(PRE, **{name: False}))
        assert res.code == f"precondition:{name}"
        assert s.aborts == 0 and s._t == 0.0 and s.machine.state is State.LOCATE
    s = PickSkill()
    r, c = mk(s)
    f = dict(PRE)
    del f["grasp_available"]  # unknown fails closed
    assert start(r, c, f).code == "precondition:grasp_available"


def test_stale_facts_abort_once():
    s = PickSkill()
    r, c = mk(s)
    start(r, c, PRE)
    tick(r, c, PRE)
    c.t += 5.0
    res = r.tick(WorldFacts(PRE, c.t - 4.0), DT)
    assert res.code == "stale_facts" and s.aborts == 1
    r.tick(WorldFacts(PRE, c.t), DT)
    assert s.aborts == 1
    # stale at start: never moved
    s2 = PickSkill()
    r2, c2 = mk(s2)
    assert r2.start(WorldFacts(PRE, None)).code == "stale_facts" and s2.aborts == 0


def test_invariant_lost_mid_motion_aborts_once_and_holds():
    s = PickSkill()
    r, c = mk(s)
    start(r, c, PRE)
    for f in SEQ[:4]:
        tick(r, c, f)
    cmd = s.command
    res = tick(r, c, dict(SEQ[3], safety_ok=False))
    assert res.code == "postcondition_lost" and s.aborts == 1
    assert s.command == cmd and s.aborted
    t = s._t
    tick(r, c, SEQ[3])
    assert s._t == t and s.abort_actuations == 1


def test_abort_idempotent_and_never_started():
    s = PickSkill()
    s.abort(); s.abort()
    assert s.abort_actuations == 1
    s2 = PickSkill()
    r, c = mk(s2)
    assert r.abort().code == "aborted" and s2.aborts == 0
    start(r, c, PRE)
    r3, c3 = mk(PickSkill())
    start(r3, c3, PRE)
    assert r3.abort().code == "aborted" and r3.abort().code == "aborted"
    assert r3.skill.aborts == 1


def test_machine_failure_logged_as_exception_with_reason():
    s = PickSkill(timeout_s=500.0)
    r, c = mk(s)
    start(r, c, PRE)
    res = None
    for _ in range(2000):
        res = tick(r, c, PRE)
        if res.terminal:
            break
    assert res.code == "exception"
    assert any("timeout:LOCATE" in e[2] or "machine failed" in e[2] for e in r.log.entries)


def test_log_has_codes_and_is_json():
    s = PickSkill()
    r, c = mk(s)
    start(r, c, dict(PRE, safety_ok=False))
    obj = json.loads(json.dumps(r.log.to_json_obj()))
    assert any(e[1] == "failed" and e[2] == "precondition:safety_ok" for e in obj["entries"])


def test_place_success_and_preconditions():
    HELD = dict(SAFE, object_in_hand=True, hand_closed=True, object_to_goal_m=0.3, object_speed_mps=0.0)
    s = PlaceSkill()
    r, c = mk(s)
    assert start(r, c, HELD).state == "running"
    assert s.command is None and s.machine.state is not State.CARRY   # S1: nothing before the first step
    tick(r, c, HELD)
    assert s.machine.state is State.CARRY and s.command is not None
    for f in [dict(HELD, ph_place_pose_reached=True), dict(HELD, object_to_goal_m=0.01),
              dict(SAFE, object_in_hand=True, hand_closed=True, object_to_goal_m=0.01, object_speed_mps=0.0)]:
        tick(r, c, f)
    # object at goal but hand still holding -> not done
    assert not r._result
    res = tick(r, c, dict(SAFE, object_in_hand=False, hand_open=True, hand_closed=False, object_to_goal_m=0.01,
                          object_speed_mps=0.0, wrist_to_object_m=0.3))
    assert res.state == "succeeded"
    s2 = PlaceSkill()
    r2, c2 = mk(s2)
    assert start(r2, c2, dict(HELD, object_in_hand=False)).code == "precondition:object_in_hand"


def test_facts_from_sim_adapter():
    f = facts_from_sim(dict(object_to_pocket_m=0.02, hand_closure=1.0, object_height_above_table_m=0.1,
                            safety_ok=True, junk=float("nan")), 1.0)
    assert f.get("object_in_hand") is True and f.get("hand_closed") is True and f.get("hand_open") is False
    assert "junk" not in f and f.stamp == 1.0
    g = facts_from_sim(dict(object_to_pocket_m=float("nan")), None)
    assert "object_in_hand" not in g and g.stamp is None


def test_existing_machine_tests_unchanged_and_green():
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", str(root / "tests/onboard/test_state_pick_place.py")],
                         capture_output=True, text=True, cwd=root)
    assert out.returncode == 0, out.stdout[-500:]


# ---- audit regressions (S1-S3) -------------------------------------------------------------------------------
HELD0 = dict(SAFE, object_in_hand=True, hand_closed=True)


def test_s1_no_actuation_before_start_and_failed_precondition_never_sets_carry():
    for cls in (PickSkill, PlaceSkill):
        s = cls()
        assert s.command is None and s.started is False
        assert s.machine.state is not State.CARRY
    s = PlaceSkill()
    assert s.start(WorldFacts(dict(HELD0, object_in_hand=False), 0.0)) is False
    assert s.command is None and s.machine.state is not State.CARRY
    assert s.start(WorldFacts(HELD0, 0.0)) is True
    assert s.machine.state is State.CARRY and s.command is not None
    cmd = s.command
    assert s.start(WorldFacts({}, 0.0)) is True and s.command is cmd   # idempotent once started


def test_s1_step_with_failing_preconditions_raises_and_does_not_actuate():
    s = PlaceSkill()
    with pytest.raises(RuntimeError):
        s.step(WorldFacts(dict(HELD0, base_stopped=False), 0.0), 0.1)
    assert s.command is None and s.machine.state is not State.CARRY
    s.abort()                       # abort before start holds "nothing"
    assert s.command is None and s.aborted


def test_s1_precondition_check_that_raises_counts_as_failed():
    s = PlaceSkill()
    s.preconditions = [Condition("boom", lambda f: 1 / 0)]
    assert s.start(WorldFacts(HELD0, 0.0)) is False and s.command is None


def _place_result(d, v=0.0):
    s = PlaceSkill()
    r, c = mk(s)
    start(r, c, dict(HELD0))
    tick(r, c, HELD0)                      # first step starts the machine (preconditions re-checked here)
    return tick(r, c, dict(SAFE, object_in_hand=False, hand_open=True, object_to_goal_m=d,
                           object_speed_mps=v))


@pytest.mark.parametrize("d,v", [(-0.5, 0.0), (-1e6, 0.0), (0.01, -3.0), (-0.01, -0.01)])
def test_s2_negative_distance_or_speed_is_not_placed(d, v):
    assert _place_result(d, v).state == "running"


def test_s2_valid_nonnegative_still_placed():
    assert _place_result(0.01, 0.0).state == "succeeded"
    assert _place_result(0.0, 0.0).state == "succeeded"


def test_s2_negative_goal_and_speed_unknown_to_machine_observation():
    from humanoid_arc_bme.onboard.pipeline.manipulation.pick_skill import observation_from_facts
    o = observation_from_facts(WorldFacts({"object_to_goal_m": -1.0, "object_speed_mps": 0.0,
                                           "wrist_to_object_m": -1.0}, 0.0))
    assert o.placed is False and o.hand_clear is False
    o = observation_from_facts(WorldFacts({"object_to_goal_m": 0.01, "object_speed_mps": 0.0}, 0.0))
    assert o.placed is True


@pytest.mark.parametrize("closure", [-3.0, 7.0, 1.0000001, -1e-9, float("nan"), float("inf"), True, "0.5", None])
def test_s2_hand_closure_outside_unit_interval_is_unknown(closure):
    f = facts_from_sim(dict(object_to_pocket_m=0.01, hand_closure=closure), 0.0)
    assert "hand_closed" not in f and "hand_open" not in f and f.get("object_in_hand") is True


@pytest.mark.parametrize("closure,closed,opened", [(0.0, False, True), (1.0, True, False), (0.5, False, False)])
def test_s2_hand_closure_in_range(closure, closed, opened):
    f = facts_from_sim(dict(hand_closure=closure), 0.0)
    assert f.get("hand_closed") is closed and f.get("hand_open") is opened


def test_s2_negative_pocket_distance_is_unknown():
    assert "object_in_hand" not in facts_from_sim(dict(object_to_pocket_m=-0.5), 0.0)


@pytest.mark.parametrize("make,facts", [(PickSkill, PRE), (PlaceSkill, HELD0)])
def test_s3_base_stopped_is_mid_run_invariant(make, facts):
    s = make()
    r, c = mk(s)
    assert start(r, c, facts).state == "running"
    assert tick(r, c, facts).state == "running"
    res = tick(r, c, dict(facts, base_stopped=False))
    assert res.state == "failed" and res.code == "postcondition_lost" and s.aborted
    # a missing base_stopped fact is unknown = not stopped
    s2 = make()
    r2, c2 = mk(s2)
    start(r2, c2, facts)
    f = {k: v for k, v in facts.items() if k != "base_stopped"}
    assert tick(r2, c2, f).code == "postcondition_lost"


def test_s3_object_visible_is_not_an_invariant():
    """A hand legitimately occludes the object after the grasp: losing object_visible must not abort."""
    s = PickSkill()
    r, c = mk(s)
    start(r, c, PRE)
    res = tick(r, c, dict(SAFE, object_visible=False, gripper_empty=False))
    assert res.code != "postcondition_lost"
    assert "object_visible" not in [x.name for x in invariants()]
