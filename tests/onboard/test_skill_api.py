"""Skill API (api.py, plan A4.5). Stdlib plus the state machine; no Isaac Sim."""

import math
from pathlib import Path

import pytest

from humanoid_arc_bme.onboard.api.contract import SkillStatus
from humanoid_arc_bme.onboard.api.safety import load_limits
from humanoid_arc_bme.onboard.skills.api import SkillApi, status_from_machine
from humanoid_arc_bme.onboard.skills.state_pick_place import Observation, PickPlaceMachine, State

ROOT = Path(__file__).resolve().parents[2]
LIMITS = load_limits(ROOT / "config" / "g1_limits.yaml")
MIN, MAX = LIMITS.ws_min, LIMITS.ws_max
MID = tuple((lo + hi) / 2 for lo, hi in zip(MIN, MAX))
EPS = 1e-6
OBJECTS = {"cup", "box"}


def api():
    return SkillApi(LIMITS, OBJECTS)


def test_unknown_object_fails_and_sets_nothing():
    a = api()
    assert a.pick("spoon") == SkillStatus("failed", "unknown_object")
    assert a.pending is None


def test_known_object_pick_is_running_and_stored():
    a = api()
    assert a.pick("cup") == SkillStatus("running")
    assert a.pending is not None
    assert a.pending.action == "pick" and a.pending.target == "cup"


def test_valid_place_is_running_and_stored_as_floats():
    a = api()
    assert a.place(MID) == SkillStatus("running")
    assert a.pending.action == "place"
    assert a.pending.target == pytest.approx(MID)


def test_pending_is_not_set_by_rejections_and_survives_them():
    a = api()
    assert a.pick("cup").state == "running"
    before = a.pending
    assert a.pick("spoon").reason == "unknown_object"
    assert a.place((MAX[0] + EPS, MID[1], MID[2])).reason == "out_of_workspace"
    assert a.pending is before
    fresh = api()
    fresh.place((MAX[0] + EPS, MID[1], MID[2]))
    assert fresh.pending is None


def _with_axis(axis, value):
    p = list(MID)
    p[axis] = value
    return tuple(p)


FACES = [
    ("x_min", 0, MIN[0] - EPS, MIN[0]),
    ("x_max", 0, MAX[0] + EPS, MAX[0]),
    ("y_min", 1, MIN[1] - EPS, MIN[1]),
    ("y_max", 1, MAX[1] + EPS, MAX[1]),
    ("z_min", 2, MIN[2] - EPS, MIN[2]),
    ("z_max", 2, MAX[2] + EPS, MAX[2]),
]


@pytest.mark.parametrize("face,axis,outside,boundary", FACES, ids=[f[0] for f in FACES])
def test_just_outside_each_face_is_rejected(face, axis, outside, boundary):
    a = api()
    assert a.place(_with_axis(axis, outside)) == SkillStatus("failed", "out_of_workspace")
    assert a.pending is None


@pytest.mark.parametrize("face,axis,outside,boundary", FACES, ids=[f[0] for f in FACES])
def test_on_boundary_and_just_inside_each_face_is_accepted(face, axis, outside, boundary):
    inward = boundary + EPS if "min" in face else boundary - EPS
    for value in (boundary, inward):
        a = api()
        assert a.place(_with_axis(axis, value)) == SkillStatus("running"), (face, value)
        assert a.pending is not None


def test_box_corners_are_accepted():
    for x in (MIN[0], MAX[0]):
        for y in (MIN[1], MAX[1]):
            for z in (MIN[2], MAX[2]):
                assert api().place((x, y, z)).state == "running"


@pytest.mark.parametrize(
    "location",
    [
        (float("nan"), MID[1], MID[2]),
        (MID[0], float("inf"), MID[2]),
        (MID[0], MID[1], float("-inf")),
        (MID[0], MID[1]),
        (MID[0], MID[1], MID[2], 0.0),
        (),
        None,
        5,
        ("a", MID[1], MID[2]),
    ],
)
def test_non_finite_or_wrong_shape_is_out_of_workspace(location):
    a = api()
    assert a.place(location) == SkillStatus("failed", "out_of_workspace")
    assert a.pending is None


def test_list_location_is_accepted():
    a = api()
    assert a.place(list(MID)).state == "running"
    assert a.pending.target == pytest.approx(MID)


def test_status_from_machine_done_is_succeeded():
    m = PickPlaceMachine(max_retries=0)
    m.state = State.DONE
    assert status_from_machine(m) == SkillStatus("succeeded")


def test_status_from_machine_running():
    m = PickPlaceMachine(max_retries=0)
    assert status_from_machine(m) == SkillStatus("running")
    m.step(0.0, Observation(object_seen=True))
    assert m.state is State.PREGRASP
    assert status_from_machine(m) == SkillStatus("running")


def test_status_from_machine_failed_carries_reason():
    m = PickPlaceMachine(max_retries=0)
    m.step(0.0, Observation())
    m.step(30.0, Observation())  # LOCATE times out (3 s) with no retries left
    assert m.state is State.FAILED
    assert m.reason == "timeout:LOCATE"
    status = status_from_machine(m)
    assert status == SkillStatus("failed", "timeout:LOCATE")


def test_limits_are_finite_box_used_by_api():
    assert all(math.isfinite(v) for v in MIN + MAX)
    assert api().place(MAX) == SkillStatus("running")
