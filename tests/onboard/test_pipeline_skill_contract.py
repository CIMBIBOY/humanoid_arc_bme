"""D1 gate: preconditions enforced, success from sensors only, failure codes, sticky terminal, one abort."""

import json
import math

import pytest

from humanoid_arc_bme.onboard.api.contract import SkillStatus
from humanoid_arc_bme.onboard.pipeline.skill_contract import (
    Condition,
    ReachSkill,
    SkillLog,
    SkillRunner,
    WorldFacts,
    to_status,
)

OK = dict(gripper_empty=True, object_visible=True, hand_object_dist=0.5, estop_clear=True)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def W(facts, stamp=0.0):
    return WorldFacts(facts, stamp=stamp)


def make(skill=None, **kw):
    c = Clock()
    s = skill or ReachSkill(timeout_s=2.0)
    kw.setdefault("max_fact_age_s", 1e6)  # freshness is exercised by its own tests
    return s, c, SkillRunner(s, c, **kw)


def test_precondition_failure_names_condition_and_no_step():
    s, c, r = make()
    res = r.start(W({**OK, "object_visible": False}))
    assert (res.state, res.code) == ("failed", "precondition:object_visible")
    assert r.tick(W(OK), 0.1).code == "precondition:object_visible"
    assert s.steps == 0


def test_unknown_fact_fails_closed():
    s, c, r = make()
    f = dict(OK)
    del f["gripper_empty"]
    assert r.start(W(f)).code == "precondition:gripper_empty"
    assert W({"x": float("nan")}).get("x") is None


def test_raising_check_is_not_a_crash():
    def bad(f):
        raise KeyError("boom")

    s = ReachSkill()
    s.preconditions.append(Condition("bad", bad))
    _, _, r = make(s)
    assert r.start(W(OK)).code == "precondition:bad"
    assert any(e[1] == "check_error" for e in r.log.entries)


def test_success_only_from_postconditions():
    s, c, r = make(ReachSkill(timeout_s=1.0))
    assert r.start(W(OK)).state == "running"
    for _ in range(5):
        c.t += 0.1
        assert r.tick(W(OK), 0.1).state == "running"  # facts disagree with the claim
    c.t += 1.0
    assert r.tick(W(OK), 0.1).code == "timeout"
    s2, c2, r2 = make()
    r2.start(W(OK))
    c2.t += 0.1
    res = r2.tick(W({**OK, "hand_object_dist": 0.01}), 0.1)
    assert res.state == "succeeded" and s2.aborts == 1


def test_timeout_with_injected_clock():
    s, c, r = make()
    r.start(W(OK))
    c.t = 1.99
    assert r.tick(W(OK), 0.1).state == "running"
    c.t = 2.0
    res = r.tick(W(OK), 0.1)
    assert res.code == "timeout" and s.aborts == 1


def test_invariant_lost_aborts_once():
    s, c, r = make(invariants=[Condition.is_true("estop_clear")])
    assert r.start(W(OK)).state == "running"
    res = r.tick(W({**OK, "estop_clear": False}), 0.1)
    assert res.code == "postcondition_lost" and s.aborts == 1
    steps = s.steps
    r.tick(W(OK), 0.1)
    assert s.aborts == 1 and s.steps == steps
    # unknown invariant at start blocks the run
    _, _, r2 = make(invariants=[Condition.is_true("estop_clear")])
    assert r2.start(W({"gripper_empty": True, "object_visible": True})).code == "precondition:estop_clear"


def test_step_exception_aborts_and_does_not_propagate():
    s = ReachSkill()

    def boom(facts, dt):
        raise RuntimeError("x")

    s.step = boom
    _, c, r = make(s)
    r.start(W(OK))
    res = r.tick(W(OK), 0.1)
    assert res.code == "exception" and s.aborts == 1


def test_terminal_sticky_and_abort_idempotent():
    s, c, r = make()
    r.start(W(OK))
    first = r.tick(W({**OK, "hand_object_dist": 0.0}), 0.1)
    assert first.state == "succeeded"
    assert r.tick(W({**OK, "gripper_empty": False}), 0.1) is first
    assert r.abort() is first and r.abort() is first
    assert s.aborts == 1
    s2, c2, r2 = make()
    r2.start(W(OK))
    assert r2.abort().code == "aborted"
    r2.abort()
    r2.tick(W(OK), 0.1)
    assert s2.aborts == 1


def test_bad_dt_and_timeout_rejected():
    s, c, r = make()
    r.start(W(OK))
    assert r.tick(W(OK), float("nan")).code == "invalid_dt"
    assert s.aborts == 1
    for bad in (float("inf"), 0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            SkillRunner(ReachSkill(timeout_s=bad), Clock())


def test_abort_on_never_started_runner_does_not_call_skill_abort():
    s, c, r = make()
    assert r.abort().code == "aborted"
    assert s.aborts == 0 and s.steps == 0
    assert r.abort().code == "aborted" and s.aborts == 0


def test_stale_facts_block_start():
    s, c, r = make(max_fact_age_s=0.5)
    c.t = 10.0
    assert r.start(W(OK, stamp=9.0)).code == "stale_facts"  # 1.0 s old
    assert s.steps == 0 and s.aborts == 0
    _, c2, r2 = make(max_fact_age_s=0.5)
    c2.t = 10.0
    assert r2.start(WorldFacts(OK)).code == "stale_facts"  # no stamp
    assert r2.start(W(OK, stamp=10.0)).code == "stale_facts"  # sticky terminal
    _, c3, r3 = make(max_fact_age_s=0.5)
    c3.t = 10.0
    assert r3.start(W(OK, stamp=9.8)).state == "running"


def test_stale_facts_abort_running_skill_once():
    s, c, r = make(max_fact_age_s=0.5)
    c.t = 10.0
    assert r.start(W(OK, stamp=10.0)).state == "running"
    c.t = 10.1
    assert r.tick(W(OK, stamp=10.1), 0.1).state == "running"
    c.t = 11.0  # sensor frozen at 10.1
    res = r.tick(W(OK, stamp=10.1), 0.1)
    assert (res.state, res.code) == ("failed", "stale_facts") and s.aborts == 1
    r.tick(W(OK, stamp=11.0), 0.1)
    assert s.aborts == 1
    s2, c2, r2 = make(max_fact_age_s=0.5)
    c2.t = 5.0
    r2.start(W(OK, stamp=5.0))
    assert r2.tick(WorldFacts(OK), 0.1).code == "stale_facts" and s2.aborts == 1


def test_max_fact_age_validation():
    for bad in (0.0, -1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            SkillRunner(ReachSkill(), Clock(), max_fact_age_s=bad)


def test_tick_before_start_fails_without_step():
    s, c, r = make()
    assert r.tick(W(OK), 0.1).code == "not_started"
    assert s.steps == 0


def test_to_status_roundtrip():
    from humanoid_arc_bme.onboard.pipeline.skill_contract import SkillResult

    for res in (SkillResult("running"), SkillResult("succeeded"), SkillResult("failed", "timeout")):
        st = to_status(res)
        assert isinstance(st, SkillStatus) and st.state == res.state
    assert to_status(SkillResult("failed", "precondition:a")).reason == "precondition:a"


def test_log_bounded_and_json():
    log = SkillLog(maxlen=3)
    for i in range(10):
        log.append(float(i), "e", str(i))
    assert len(log.entries) == 3 and log.dropped == 7
    json.dumps(log.to_json_obj())
    s, c, r = make()
    r.start(W(OK))
    r.abort()
    d = json.loads(json.dumps(r.log.to_json_obj()))
    assert d["entries"][-1][1:] == ["failed", "aborted"]
    assert math.isfinite(d["entries"][0][0])
