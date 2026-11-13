import math

import numpy as np
import pytest

from humanoid_arc_bme.onboard.nav.depth_guard import DepthDistance, DepthGuardConfig, tighten_command
from humanoid_arc_bme.onboard.nav.obstacle import SpeedSeparationGuard

CFG = DepthGuardConfig(cam_pitch=0.0, cam_height=1.0, stride=2)


def frame(fill=10.0):
    return np.full((CFG.height, CFG.width), fill, dtype=np.float32)


def wall(depth, dist, u0=200, u1=440, v0=100, v1=380):
    f = frame(depth)
    f[v0:v1, u0:u1] = dist
    return f


def test_open_space_is_inf():
    assert DepthDistance(CFG).forward_distance(frame(10.0)) == math.inf


def test_wall_ahead_distance():
    d = DepthDistance(CFG).forward_distance(wall(10.0, 1.0))
    assert d == pytest.approx(1.0 + CFG.cam_forward, abs=0.03)


def test_wall_outside_corridor_ignored():
    # narrow block at far left of the image (|x| >> half_width at 1 m)
    f = frame(10.0)
    f[100:380, 0:60] = 1.0
    assert DepthDistance(CFG).forward_distance(f) == math.inf


def test_isolated_noise_pixels_ignored_but_blob_counts():
    f = frame(10.0)
    f[240, 320] = 0.5
    f[240, 322] = 0.5  # only 2 hits after stride -> below min_hits
    assert DepthDistance(CFG).forward_distance(f) == math.inf
    assert DepthDistance(CFG).forward_distance(wall(10.0, 0.5, 300, 340, 220, 260)) < 0.7


def test_ground_ignored_with_pitch():
    cfg = DepthGuardConfig(cam_pitch=0.4, cam_height=1.2)
    z = 1.2 / math.sin(0.4)  # not used directly; build a flat-ground depth image
    v = (np.arange(cfg.height, dtype=np.float32) - cfg.cy) / cfg.fy
    ang = cfg.cam_pitch + np.arctan(v)  # ray angle below horizontal
    depth = np.where(ang > 0.05, cfg.cam_height / np.sin(np.maximum(ang, 0.05)) / np.cos(np.arctan(v)), 50.0)
    f = np.repeat(depth[:, None], cfg.width, axis=1).astype(np.float32)
    assert z > 0
    assert DepthDistance(cfg).forward_distance(f) == math.inf


@pytest.mark.parametrize("bad", [np.nan, 0.0, -1.0, -np.inf])
def test_invalid_frame_stops(bad):
    assert DepthDistance(CFG).forward_distance(frame(bad)) == 0.0


def test_garbage_input_stops():
    dd = DepthDistance(CFG)
    assert dd.forward_distance(None) == 0.0
    assert dd.forward_distance(np.zeros((10, 10))) == 0.0
    assert dd.forward_distance("x") == 0.0
    assert dd.forward_distance(np.zeros((CFG.height, CFG.width, 3))) == 0.0


def test_positive_inf_is_far_not_invalid():
    assert DepthDistance(CFG).forward_distance(frame(np.inf)) == math.inf


def test_stale_and_missing_frame_none_then_guard_stops():
    t = {"now": 0.0}
    dd = DepthDistance(CFG, now_fn=lambda: t["now"])
    assert dd(1) is None
    dd.submit(frame(10.0), 0.0)
    assert dd(1) == math.inf
    t["now"] = 1.0
    assert dd(1) is None
    g = SpeedSeparationGuard(dd, require_sensor=True, stale_s=0.3)
    assert g.speed_cap(5.0, 1) == 0.0


def test_rear_unknown_fails_closed():
    dd = DepthDistance(CFG)
    dd.submit(frame(10.0), 0.0)
    assert dd(-1) is None
    assert SpeedSeparationGuard(dd).speed_cap(0.0, -1) == 0.0


def test_integration_with_guard_slows_and_stops():
    dd = DepthDistance(CFG)
    g = SpeedSeparationGuard(dd, a_stop=0.3, d_margin=0.3)
    dd.submit(frame(10.0), 0.0)
    assert g.speed_cap(0.0, 1) == math.inf
    dd.submit(wall(10.0, 1.5), 0.1)
    assert 0 < g.speed_cap(0.1, 1) < 1.0
    dd.submit(wall(10.0, 0.17), 0.2)
    assert g.speed_cap(0.2, 1) == 0.0


@pytest.mark.parametrize("vx,vy,wz", [(0.2, 0.1, 0.3), (-0.1, 0.0, -0.2), (0.0, 0.1, 0.3), (0.15, -0.05, 0.0)])
@pytest.mark.parametrize("cap", [math.inf, 1.0, 0.1, 0.05, 0.0, -1.0, math.nan, None])
def test_tighten_never_increases(vx, vy, wz, cap):
    out = tighten_command(vx, vy, wz, cap)
    for a, b in zip(out, (vx, vy, wz)):
        assert abs(a) <= abs(b) + 1e-12
        assert a == 0 or math.copysign(1, a) == math.copysign(1, b)


def test_tighten_values_and_failsafe():
    assert tighten_command(0.2, 0.1, 0.3, 0.1) == pytest.approx((0.1, 0.05, 0.15))
    assert tighten_command(0.2, 0.1, 0.3, math.nan) == (0.0, 0.0, 0.0)
    assert tighten_command(math.nan, 0.0, 0.0, 1.0) == (0.0, 0.0, 0.0)
    assert tighten_command(0.2, 0.1, 0.3, math.inf) == (0.2, 0.1, 0.3)
