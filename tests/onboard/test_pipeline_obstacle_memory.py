import math

import pytest

from humanoid_arc_bme.onboard.pipeline.perception.obstacle_memory import ObstacleMemory, ObstacleMemoryConfig

NAN = float("nan")


def seen(m, t=0.0, pose=(0.0, 0.0, 0.0), d=1.0):
    m.observe(t, pose, (d, 0.0), d, blind_until=0.0)


def test_empty_is_inf():
    assert ObstacleMemory().blocked_distance(0.0, (0, 0, 0)) == math.inf


def test_leaves_view_still_blocking():
    m = ObstacleMemory()
    seen(m, 0.0, d=1.0)
    m.observe(0.1, (0.3, 0.0, 0.0), None, None, blind_until=0.0)       # blind / out of view: no evidence
    assert m.blocked_distance(0.1, (0.3, 0.0, 0.0)) == pytest.approx(0.7)


def test_drive_toward_and_stop_with_margin():
    m = ObstacleMemory()
    x, t, margin = 0.0, 0.0, 0.4
    seen(m, t, (x, 0, 0), 1.5)
    while True:
        t += 0.1
        m.observe(t, (x, 0, 0), None, None, blind_until=0.8)   # obstacle now in the blind zone
        if m.blocked_distance(t, (x, 0, 0)) <= margin:
            break
        x += 0.05
        assert t < 30
    assert 1.5 - x >= margin - 1e-9 and x > 0.9


def test_drop_visible_empty():
    m = ObstacleMemory()
    seen(m, 0.0, d=1.0)
    m.observe(0.1, (0, 0, 0), None, 3.0, blind_until=0.5)
    assert len(m) == 0


def test_no_drop_inside_blind_zone_or_insufficient_range():
    m = ObstacleMemory()
    seen(m, 0.0, d=1.0)
    m.observe(0.1, (0, 0, 0), None, 3.0, blind_until=1.0)   # obstacle within blind zone: not observable
    m.observe(0.2, (0, 0, 0), None, 1.2, blind_until=0.5)   # free only to 1.2 < 1.0+0.3
    m.observe(0.3, (0, 0, 0), None, NAN, blind_until=0.5)
    assert len(m) == 1


def test_drop_behind_lateral_ttl():
    m = ObstacleMemory()
    seen(m)
    m.observe(0.1, (1.5, 0, 0), None, None, blind_until=0.0)
    assert len(m) == 0
    seen(m)
    m.observe(0.1, (0, 1.0, 0), None, None, blind_until=0.0)                  # |lateral| 1.0 > 0.55
    assert len(m) == 0
    seen(m, 0.0)
    m.observe(31.0, (0, 0, 0), None, None, blind_until=0.0)
    assert len(m) == 0


def test_frozen_on_bad_pose():
    m = ObstacleMemory()
    seen(m)
    for bad in (None, (NAN, 0, 0), (0, 0, math.inf), (0, 0), "x"):
        m.observe(100.0, bad, None, 5.0, blind_until=0.0)
        assert len(m) == 1
        assert m.blocked_distance(100.0, bad) == 0.0
    assert m.blocked_distance(0.1, (0, 0, 0)) == pytest.approx(1.0)


def test_bad_stamp_and_nan_do_not_forget():
    m = ObstacleMemory()
    seen(m, 5.0)
    for t in (NAN, math.inf, 1.0, None):
        m.observe(t, (0, 0, 0), None, 5.0, blind_until=0.0)
        assert len(m) == 1
    assert m.blocked_distance(1.0, (0, 0, 0)) == pytest.approx(1.0)   # backwards stamp: no TTL forgetting
    assert m.blocked_distance(NAN, (0, 0, 0)) == pytest.approx(1.0)
    m.observe(5.1, (0, 0, 0), (NAN, 0.0), NAN, blind_until=0.0)
    m.observe(5.2, (0, 0, 0), "junk", NAN, blind_until=NAN)
    assert len(m) == 1 and m.blocked_distance(5.2, (0, 0, 0)) == pytest.approx(1.0)


def test_nearest_only_and_yaw_projection():
    m = ObstacleMemory()
    m.observe(0, (0, 0, math.pi / 2), (1.0, 0.0), 1.0, blind_until=0.0)       # obstacle at map (0, 1)
    m.observe(0.1, (0, 0, math.pi / 2), (2.0, 0.0), None, blind_until=0.0)    # farther: ignored in single mode
    assert m.blocked_distance(0.1, (0, 0, math.pi / 2)) == pytest.approx(1.0)
    assert m.blocked_distance(0.1, (0, 0.5, math.pi / 2)) == pytest.approx(0.5)


def test_bounded_size_oldest_dropped():
    m = ObstacleMemory(ObstacleMemoryConfig(max_items=2))
    for i, d in enumerate((1.0, 2.0, 3.0)):
        m.observe(float(i), (0, 0, 0), (d, 0.0), None, blind_until=0.0)
    assert len(m) == 2
    assert m.blocked_distance(2.0, (0, 0, 0)) == pytest.approx(2.0)   # the 1.0 (oldest) was evicted


def test_reset():
    m = ObstacleMemory()
    seen(m)
    m.reset()
    assert m.blocked_distance(0, (0, 0, 0)) == math.inf


@pytest.mark.parametrize("kw", [dict(ttl_s=0), dict(ttl_s=NAN), dict(corridor_half_width=-1), dict(max_items=0),
                                dict(max_items=True), dict(lateral_margin=math.inf)])
def test_config_validation(kw):
    with pytest.raises(ValueError):
        ObstacleMemoryConfig(**kw)


def test_blind_until_is_required_keyword():
    """M1: a caller that forgets the blind zone gets a TypeError, never a silent 'visible and empty'."""
    m = ObstacleMemory()
    m.observe(0.0, (0, 0, 0), (0.2, 0.0), None, blind_until=0.0)
    with pytest.raises(TypeError):
        m.observe(0.1, (0, 0, 0), None, 5.0)                 # type: ignore[call-arg]
    with pytest.raises(TypeError):
        m.observe(0.1, (0, 0, 0), None, 5.0, 0.0)            # keyword-only
    assert len(m) == 1 and m.blocked_distance(0.1, (0, 0, 0)) == pytest.approx(0.2)


@pytest.mark.parametrize("blind", [NAN, math.inf, None, "0.5"])
def test_unknown_blind_zone_never_drops(blind):
    m = ObstacleMemory()
    m.observe(0.0, (0, 0, 0), (0.2, 0.0), None, blind_until=0.0)
    m.observe(0.1, (0, 0, 0), None, 5.0, blind_until=blind)
    assert len(m) == 1


def test_shares_is_num_with_interfaces():
    from humanoid_arc_bme.onboard.pipeline import interfaces
    from humanoid_arc_bme.onboard.pipeline.perception import obstacle_memory
    from humanoid_arc_bme.onboard.pipeline.manipulation import pick_skill
    from humanoid_arc_bme.onboard.pipeline import runner
    assert obstacle_memory._num is interfaces.is_num and runner._is_num is interfaces.is_num
    assert pick_skill.is_num is interfaces.is_num
