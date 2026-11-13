"""Waypoint routing and hand ramp (skills/waypoint_follower.py). No Isaac Sim."""

import ast
from pathlib import Path

import pytest

from humanoid_arc_bme.onboard.skills.waypoint_follower import EDGES, HandRamp, WaypointFollower, route

DT = 0.05


def test_waypoint_names_match_the_timer_script():
    src = (Path(__file__).resolve().parents[2] / "humanoid_arc_bme/skills/scripted_pick_place.py").read_text()
    wps = next(ast.literal_eval(n.value) for n in ast.parse(src).body
               if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "WAYPOINTS")
    assert set(EDGES) == set(wps)


@pytest.mark.parametrize("a,b,expect", [
    ("rest", "above_pick", ["ready_pick", "above_pick"]),
    ("above_pick", "above_place", ["above_place"]),
    ("above_place", "rest", ["ready_place", "rest"]),
    ("pick", "rest", ["above_pick", "ready_pick", "rest"]),
    ("place", "rest", ["above_place", "ready_place", "rest"]),
    ("pick", "pick", []),
])
def test_routes(a, b, expect):
    assert route(a, b) == expect


def test_every_pair_is_routable_and_uses_only_edges():
    for a in EDGES:
        for b in EDGES:
            node = a
            for n in route(a, b):
                assert n in EDGES[node]
                node = n
            assert node == b


def test_never_leaves_rest_straight_to_the_object():
    # the hand must rise to ready_pick first (fingers would sweep the object)
    assert route("rest", "pick")[0] == "ready_pick"
    assert route("rest", "above_pick")[0] == "ready_pick"


def drive(f, target, seconds):
    out, t = [], 0.0
    while t < seconds:
        out.append(f.step(target, DT))
        t += DT
    return out


def test_follower_reaches_target_and_travel_time_is_the_edge_sum():
    f = WaypointFollower()
    t = 0.0
    while f.settled != "above_pick":
        f.step("above_pick", DT)
        t += DT
        assert t < 10
    assert t == pytest.approx(3.0, abs=2 * DT)


def test_segment_in_flight_finishes_before_retarget_and_s_is_monotone_in_segment():
    f = WaypointFollower()
    seq = [f.step("pick", DT) for _ in range(10)]          # heading to ready_pick
    seq += [f.step("rest", DT) for _ in range(60)]         # retarget mid-flight
    assert seq[10][1] == "ready_pick" or seq[10][0] == "ready_pick"  # first segment not abandoned
    s_first = [s for a, b, s in seq if (a, b) == ("rest", "ready_pick")]
    assert s_first == sorted(s_first) and s_first[-1] == 1.0
    assert f.settled == "rest"


def test_speed_scales_time():
    slow, fast = WaypointFollower(speed=1.0), WaypointFollower(speed=2.0)
    n = lambda f: next(i for i in range(1000) if (f.step("ready_pick", DT), f.settled)[1] == "ready_pick")
    assert n(fast) == pytest.approx(n(slow) / 2, abs=2)


def test_at_rest_returns_zero_progress():
    assert WaypointFollower().step("rest", DT) == ("rest", "rest", 0.0)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        WaypointFollower(start="nowhere")
    with pytest.raises(ValueError):
        WaypointFollower().step("nowhere", DT)
    with pytest.raises(ValueError):
        WaypointFollower(speed=0)


def test_hand_ramp_rate_and_hold():
    h = HandRamp(rate=1.0)
    for _ in range(10):
        h.step(1.0, 0.05)
    assert h.closure == pytest.approx(0.5, abs=1e-6) and not h.reached(0.1)
    for _ in range(10):
        h.step(1.0, 0.05)
    assert h.closure == 1.0
    assert not h.reached(0.5)
    for _ in range(10):
        h.step(1.0, 0.05)
    assert h.reached(0.5)
    h.step(0.0, 0.05)                      # new target resets the hold
    assert not h.reached(0.1)
    assert h.closure < 1.0


def test_hand_ramp_rejects_out_of_range():
    with pytest.raises(ValueError):
        HandRamp().step(1.5, 0.05)
