"""Obstacle stage inside OperatorRuntime: every mode passes through it, no-sensor policy, telemetry, explicit idle on stop
(GR00T adapter in the loop, fake transport)."""
import math

import pytest

from humanoid_arc_bme.onboard.runtime.groot_controller import MODE_IDLE, FakeTransport, GrootPlannerController
from humanoid_arc_bme.onboard.nav.obstacle_stage import ObstacleStage
from tests.onboard.test_operator_runtime import arm, make, release_once, run


def armed_rt(tmp_path, obstacle, controllers=None):
    rt, clock, fake = make(tmp_path, controllers=controllers)
    if obstacle is not None:
        rt.obstacle = obstacle
        rt.obstacle_info = None
    run(rt, clock, 1.2)
    assert arm(rt)[0]
    release_once(rt, clock)
    return rt, clock, fake


def steps(fake, n0=0):
    return [c[2] for c in fake.calls if c[0] == "step"][n0:]


def test_default_is_no_sensor_with_warning_and_rear_blocked(tmp_path):
    rt, clock, fake = armed_rt(tmp_path, None)
    assert rt.status()["obstacle"]["reason"] == "no_sensor" and "NO OBSTACLE SENSOR" in rt.status()["obstacle"]["warning"]
    n0 = len(steps(fake))
    run(rt, clock, 2.0, hb=(0.2, 0, 0), held=True)
    fwd = steps(fake, n0)
    assert max(c[0] for c in fwd) == pytest.approx(0.2)  # capped to the S1 limit, still moves
    n0 = len(steps(fake))
    run(rt, clock, 1.0, hb=(-0.1, 0, 0), held=True)
    assert all(c[0] >= 0.0 for c in steps(fake, n0))  # (ramp-down from forward is fine) rear without a sensor: fail closed
    assert any(e["event"] == "obstacle_state" for e in rt._events)
    rt.close()


@pytest.mark.parametrize("mode,hb", [("teleop", (0.2, 0, 0)), ("curvature", {"v": 0.2, "kappa": 0.0}), ("ros", None)])
def test_every_mode_passes_through_stage(tmp_path, mode, hb):
    seen = []

    def dist(d):
        seen.append(d)
        return 0.1  # inside the margin: blocked
    rt, clock, fake = armed_rt(tmp_path, ObstacleStage(dist))
    n0 = len(steps(fake))
    if mode == "ros":
        for _ in range(100):
            rt.set_external_cmd(0.2, 0.0, 0.0)
            rt.heartbeat({"mode": "ros", "held": True})
            rt.tick()
            clock.t += 0.02
    elif mode == "curvature":
        run(rt, clock, 2.0, hb=hb, held=True, mode="curvature")
    else:
        run(rt, clock, 2.0, hb=hb, held=True)
    assert seen, "stage was never consulted"
    assert all(c == (0.0, 0.0, 0.0) for c in steps(fake, n0))
    assert rt.status()["obstacle"]["reason"] == "blocked"
    assert any(e["event"] == "obstacle_cap" for e in rt._events)
    rt.close()


def test_clear_path_passes_and_cap_never_exceeds_input(tmp_path):
    rt, clock, fake = armed_rt(tmp_path, ObstacleStage(lambda d: math.inf))
    n0 = len(steps(fake))
    run(rt, clock, 2.0, hb=(0.2, 0, 0), held=True)
    assert max(c[0] for c in steps(fake, n0)) == pytest.approx(0.2)
    rt.close()


def test_stop_by_guard_sends_explicit_idle_to_groot(tmp_path):
    tx = FakeTransport()
    groot = GrootPlannerController(tx, threaded=False, enabled=True)
    dist = {"d": math.inf}
    rt, clock, _ = armed_rt(tmp_path, ObstacleStage(lambda d: dist["d"]), controllers={"groot": groot})
    run(rt, clock, 1.5, hb=(0.2, 0, 0), held=True)
    planners = [d for t, d in tx.decoded() if t == "planner"]
    assert planners and planners[-1]["mode"] != MODE_IDLE
    dist["d"] = 0.1
    n = len(planners)
    run(rt, clock, 0.5, hb=(0.2, 0, 0), held=True)
    planners = [d for t, d in tx.decoded() if t == "planner"]
    assert len(planners) > n, "must keep streaming (an explicit idle), not go silent"
    assert all(p["mode"] == MODE_IDLE for p in planners[n:])
    assert planners[-1]["speed"] == pytest.approx(-1.0) or planners[-1]["movement"] == (0.0, 0.0, 0.0)
    rt.close()


def test_command_log_carries_obstacle_state(tmp_path):
    rt, clock, fake = armed_rt(tmp_path, ObstacleStage(lambda d: None))
    run(rt, clock, 0.5, hb=(0.2, 0, 0), held=True)
    o = rt.status()["obstacle"]
    assert o["reason"] == "no_reading" and o["scale"] == 0.0
    rt.close()
