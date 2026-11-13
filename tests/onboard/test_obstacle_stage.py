"""ObstacleStage: tightening-only property tests, fail-closed cases, no-sensor policy, idempotence with the follower cap."""
import math
import random

import pytest

from humanoid_arc_bme.onboard.nav.obstacle_stage import ObstacleStage


def rnd_cmd(r):
    return (r.uniform(-0.3, 0.3), r.uniform(-0.2, 0.2), r.uniform(-0.5, 0.5))


def ok(inp, out):
    return all(abs(o) <= abs(i) + 1e-12 and o * i >= 0 for o, i in zip(out, inp))


def test_property_never_increases_never_flips():
    r = random.Random(1)
    srcs = [lambda d: r.uniform(0, 4), lambda d: None, lambda d: math.inf, lambda d: float("nan"), lambda d: -1.0,
            lambda d: 0.2, lambda d: "x", lambda d: 1e9]
    for src in srcs + [None]:
        st = ObstacleStage(src, no_sensor_speed=0.2)
        for _ in range(500):
            c = rnd_cmd(r)
            res = st.apply(c)
            assert ok(c, res.cmd), (c, res)
            assert res.scale <= 1.0


def test_garbage_commands_zero():
    st = ObstacleStage(lambda d: 5.0)
    for c in [(float("nan"), 0, 0), (0, float("inf"), 0), (None, 0, 0), "abc", (1, 2), (True, 0, 0)]:
        assert st.apply(c).cmd == (0.0, 0.0, 0.0)


def test_none_nan_negative_and_raising_source_fail_closed():
    def boom(d):
        raise RuntimeError
    for v in (None, float("nan"), -0.5, -math.inf, "1.0", True, boom):
        src = boom if v is boom else (lambda d, v=v: v)
        res = ObstacleStage(src).apply((0.2, 0.0, 0.1))
        assert res.cmd == (0.0, 0.0, 0.0) and res.reason in ("no_reading", "bad_reading")


def test_stale_depth_source_zero(  ):
    from humanoid_arc_bme.onboard.nav.depth_guard import DepthDistance
    t = [0.0]
    dd = DepthDistance(now_fn=lambda: t[0])
    dd._d, dd._t = math.inf, 0.0
    st = ObstacleStage(dd)
    assert st.apply((0.2, 0, 0)).cmd == (0.2, 0, 0)
    t[0] = 1.0  # older than max_age_s 0.3
    res = st.apply((0.2, 0, 0))
    assert res.cmd == (0.0, 0.0, 0.0) and res.reason == "no_reading"


def test_rear_without_rear_sensor_zero_even_when_front_is_fresh():
    from humanoid_arc_bme.onboard.nav.depth_guard import DepthDistance
    dd = DepthDistance()
    dd._d, dd._t = math.inf, 0.0
    st = ObstacleStage(dd)
    assert st.apply((0.2, 0, 0)).cmd == (0.2, 0, 0)
    assert st.apply((-0.1, 0, 0)).cmd == (0.0, 0.0, 0.0)  # a fresh front reading must not unlock the rear


def test_cap_scales_curve_together_and_blocks_inside_margin():
    st = ObstacleStage(lambda d: 0.5 + 0.1 / 2 * 0.3 * 0 + 0.0, a_stop=0.3, d_margin=0.5)
    assert st.apply((0.2, 0.0, 0.2)).cmd == (0.0, 0.0, 0.0) and st.last.reason == "blocked"
    st = ObstacleStage(lambda d: 0.5 + 0.0225 / 0.6 * 1, a_stop=0.3, d_margin=0.5)  # cap = sqrt(2*.3*.0375)=0.15
    res = st.apply((0.2, 0.0, 0.2))
    assert res.cmd[0] == pytest.approx(0.15) and res.cmd[2] == pytest.approx(0.15) and res.reason == "capped"


def test_pure_turn_passes_with_healthy_sensor_blocked_with_dead_sensor():
    assert ObstacleStage(lambda d: 0.0).apply((0, 0, 0.3)).cmd == (0, 0, 0.3)  # turn away from an adjacent obstacle
    assert ObstacleStage(lambda d: None).apply((0, 0, 0.3)).cmd == (0, 0, 0)


def test_no_sensor_policy_is_explicit():
    st = ObstacleStage(None, no_sensor_speed=0.2)
    r = st.apply((0.3, 0, 0))
    assert r.reason == "no_sensor" and r.warning and r.cmd[0] == pytest.approx(0.2)
    r = st.apply((-0.1, 0, 0))
    assert r.cmd == (0, 0, 0) and "rear" in r.warning
    assert ObstacleStage(None, allow_rear_without_sensor=True).apply((-0.1, 0, 0)).cmd == (-0.1, 0, 0)
    assert st.apply((0, 0, 0.3)).cmd == (0, 0, 0.3)


def test_idempotent_with_second_application():
    r = random.Random(3)
    st = ObstacleStage(lambda d: r.choice([0.6, 0.8, 1.5, math.inf]))
    d = {1: 0.52, -1: 0.52}
    st = ObstacleStage(lambda dr: d.get(dr))
    for _ in range(200):
        c = rnd_cmd(r)
        once = st.apply(c).cmd
        twice = st.apply(once).cmd
        assert twice == pytest.approx(once)
