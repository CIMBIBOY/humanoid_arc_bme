"""B1-fix: gravity alignment + fitted ground plane + corridor clearance, on SYNTHETIC frames (SIM, not MEASURED).

Same renderer (tests/core/_depth_render.py), same placeholder noise and seed-sweep style as
test_pipeline_depth_scenarios.py. Thresholds come from the geometry/noise model in `pipeline/ground.py`; they were not
tuned to make a scenario pass. Set DEPTH_SCENARIO_VERBOSE=1 and run with -s to print the tables.
Attitude convention: see `attitude_matrix` (world-from-body, positive pitch = nose down, positive roll = left up).
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import pytest

from _depth_render import CFG, Box, Hole, noisy, render
from humanoid_arc_bme.onboard.pipeline.perception.ground import (
    analyze_corridor, attitude_matrix, fit_ground, gravity_align, gravity_unalign)
from humanoid_arc_bme.onboard.pipeline.interfaces import DepthFrame, Intrinsics
from humanoid_arc_bme.onboard.pipeline.perception.sensing import camera_pose_from_mount, depth_to_points

V = bool(os.environ.get("DEPTH_SCENARIO_VERBOSE"))
SEEDS = 12
K = Intrinsics(CFG.fx, CFG.fy, CFG.cx, CFG.cy, CFG.width, CFG.height)
MOUNT = camera_pose_from_mount((CFG.cam_forward, 0.0, CFG.cam_height), CFG.cam_pitch)


def say(*a):
    if V:
        print(*a)


def body_points(depth):
    return depth_to_points(DepthFrame(0.0, depth, K, MOUNT), max_range=8.0)


def analyze(depth, roll=0.0, pitch=0.0, valid_fraction=1.0, **kw):
    """Nominal-mount back-projection -> gravity align with the GIVEN attitude -> fit -> corridor."""
    pts = gravity_align(body_points(depth), roll, pitch)
    plane = fit_ground(pts)
    return analyze_corridor(pts, plane, valid_fraction=valid_fraction, **kw), plane


def scene(boxes=(), holes=(), roll=0.0, pitch=0.0, seed=0):
    return noisy(render(CFG, boxes, holes, body_rot=attitude_matrix(roll, pitch)), seed)


def box(x, h, w=0.4):
    return Box(x, x + 0.3, -w / 2, w / 2, h)


# ---------------------------------------------------------------- alignment
def test_attitude_round_trip_and_convention():
    rng = np.random.default_rng(0)
    p = rng.normal(size=(50, 3))
    for r, q in ((0.1, -0.2), (-0.05, 0.3), (0.0, 0.0)):
        assert np.allclose(gravity_unalign(gravity_align(p, r, q), r, q), p, atol=1e-12)
    # nose-down pitch: a point straight ahead moves DOWN; left-up roll: a point on the left moves UP
    assert gravity_align([[1, 0, 0]], 0, 0.2)[0, 2] < 0
    assert gravity_align([[0, 1, 0]], 0.2, 0)[0, 2] > 0
    with pytest.raises(ValueError):
        gravity_align([[1, 0, 0]], math.nan, 0)


def test_tilted_flat_floor_becomes_level():
    for r, q in ((0.0, 0.0), (math.radians(3), math.radians(5)), (math.radians(-3), math.radians(-5))):
        pts = body_points(render(CFG, body_rot=attitude_matrix(r, q)))
        raw_span = np.ptp(pts[:, 2])
        z = gravity_align(pts, r, q)[:, 2]
        assert np.abs(z).max() < 1e-3                     # noise-free: z = const = 0 (body origin is on the floor)
        if r or q:
            assert raw_span > 0.5                         # unaligned the same floor spans a lot of heights


def test_fit_recovers_level_floor_and_fails_closed():
    pl = fit_ground(gravity_align(body_points(scene(seed=1)), 0, 0))
    assert pl is not None and pl.tilt_deg < 1.0 and abs(pl.offset) < 0.02
    assert pl.residual_rms < 0.01 and pl.inlier_fraction > 0.8 and pl.confidence > 0.5
    assert fit_ground(np.zeros((0, 3))) is None
    assert fit_ground(np.full((1000, 3), np.nan)) is None
    wall = np.column_stack([np.full(2000, 1.0), np.random.default_rng(0).uniform(-1, 1, 2000),
                            np.random.default_rng(1).uniform(0, 1.5, 2000)])
    assert fit_ground(wall) is None                       # vertical plane: tilt limit
    rng = np.random.default_rng(2)
    tilted = np.column_stack([rng.uniform(0.2, 3, 3000), rng.uniform(-1, 1, 3000), np.zeros(3000)])
    tilted[:, 2] = np.tan(math.radians(25)) * tilted[:, 0]
    assert fit_ground(tilted) is None                     # 25 deg off gravity-up > max_tilt_deg


# ---------------------------------------------------------------- (a) sway, true attitude
def test_sway_empty_floor_with_true_attitude():
    out = {}
    for label, pm, rm in (("1/0.5", 1, 0.5), ("3/2", 3, 2), ("5/3", 5, 3)):
        rng = np.random.default_rng(7)
        n = 60
        pos = neg = fail = 0
        dmin = math.inf
        for i in range(n):
            r, q = rng.uniform(-rm, rm), rng.uniform(-pm, pm)
            res, _ = analyze(scene(roll=math.radians(r), pitch=math.radians(q), seed=i), math.radians(r), math.radians(q))
            pos += math.isfinite(res.distance_positive)
            neg += math.isfinite(res.distance_negative)
            fail += not res.ground_ok
            dmin = min(dmin, res.distance_positive, res.distance_negative)
        out[label] = (pos / n, neg / n, fail / n)
        say(f"SIM ground fit, EMPTY floor, true attitude, pitch+-{pm} roll+-{rm} deg: phantom positive={pos/n:.3f} "
            f"phantom negative/gap={neg/n:.3f} fail-closed={fail/n:.3f} nearest phantom={dmin}")
    for k, (pp, nn, ff) in out.items():
        assert pp <= 0.02 and nn <= 0.02 and ff == 0.0, k


# ---------------------------------------------------------------- (b) static pitch error, correct IMU
def test_static_pitch_error_with_correct_imu():
    for deg in (-8, -5, -3, 0, 3, 5, 8):
        q = math.radians(deg)
        res, _ = analyze(scene(pitch=q, seed=3), 0.0, q)
        say(f"SIM pitch {deg:+d} deg, correct IMU, empty floor: pos={res.distance_positive} neg={res.distance_negative}")
        assert res.ground_ok and math.isinf(res.distance_positive) and math.isinf(res.distance_negative)
        det = [math.isfinite(analyze(scene([box(1.5, 0.15)], pitch=q, seed=s), 0.0, q)[0].distance_positive)
               for s in range(SEEDS)]
        err = [abs(analyze(scene([box(1.5, 0.15)], pitch=q, seed=s), 0.0, q)[0].distance_positive - 1.5)
               for s in range(3)]
        say(f"SIM pitch {deg:+d} deg: 0.15 m box at 1.5 m detect={np.mean(det):.2f} |err|<={max(err):.3f}")
        assert np.mean(det) == 1.0                         # old guard: 0.00 at -5/-8 deg
        assert max(err) < 0.1


# ---------------------------------------------------------------- (c) negative obstacles
def test_holes_are_bounded_conservatively_never_after_the_true_edge():
    """Safety property: for every hole, every seed, the reported distance is finite (drop seen) and NOT later than
    the true edge by more than one grid cell plus noise margin. An edge inside the blind zone reports 0.0 (stop):
    the hole floor is seen farther out, but nothing verifies where the edge is."""
    rows = {}
    for depth in (0.15, 0.4, 1.0):
        for x in (0.6, 1.2):
            hole = Hole(x, x + 1.5, -0.6, 0.6, depth)
            ds = [analyze(scene(holes=[hole], seed=s))[0] for s in range(SEEDS)]
            dist = [r.distance_negative for r in ds]
            rows[(depth, x)] = dist
            say(f"SIM hole depth={depth} edge={x}: reported {np.round(dist, 2).tolist()} (old guard: undetected)")
            assert all(math.isfinite(d) for d in dist), (depth, x)        # always detected (stop or bound)
            assert all(d <= x + 0.15 for d in dist), (depth, x, dist)      # never later than the true edge
    for depth in (0.15, 0.4, 1.0):
        assert all(d == 0.0 for d in rows[(depth, 0.6)])                   # inside the blind zone: stop
        assert all(0.9 <= d <= 1.35 for d in rows[(depth, 1.2)])           # in view: a useful bound, not 0


def test_gap_behind_positive_obstacle_is_explained():
    res, _ = analyze(scene([Box(1.5, 1.8, -0.2, 0.2, 1.0)], seed=0))
    assert math.isfinite(res.distance_positive) and abs(res.distance_positive - 1.5) < 0.06
    assert math.isinf(res.unknown_ground_from)


# ---------------------------------------------------------------- (d) low / thin
def test_low_and_thin_objects_reported():
    rows = {}
    for name, mk in (("box0.10", lambda x: box(x, 0.10)), ("box0.15", lambda x: box(x, 0.15)),
                     ("pole1cm", lambda x: Box(x, x + 0.01, -0.005, 0.005, 1.0)),
                     ("slab3cm_0.33", lambda x: Box(x, x + 0.03, -0.3, 0.3, 0.33))):
        for x in (1.5, 2.5):
            det = float(np.mean([math.isfinite(analyze(scene([mk(x)], seed=s))[0].distance_positive)
                                 for s in range(SEEDS)]))
            rows[(name, x)] = det
            say(f"SIM {name} at x={x}: detect={det:.2f}")
    h = analyze(scene(seed=0))[0].h_min
    say(f"SIM derived h_min on clean floor = {h:.3f} m (old guard z_min 0.12 m)")
    assert rows[("box0.15", 1.5)] == 1.0
    assert rows[("slab3cm_0.33", 1.5)] == 1.0
    assert 0.03 <= h < 0.06
    # no assertion on 0.10 m / pole beyond being reported: honest numbers only (see findings doc)


# ---------------------------------------------------------------- (e) blind zone is a limit, not fixed
def test_near_field_blind_zone_is_reported_not_fixed():
    for x in (0.3, 0.5, 0.7):
        r = [analyze(scene([Box(x, x + 0.05, -0.2, 0.2, 0.3)], seed=s))[0] for s in range(SEEDS)]
        det = np.mean([np.isfinite(a.distance_positive) for a in r])
        say(f"SIM 0.3 m slab at x={x}: detect={det:.2f}, blind_until={r[0].blind_until:.2f}")
        assert det == 0.0                                  # still invisible: geometry, not a bug
        assert all(a.blind_until > x + 0.05 for a in r)    # ...and the result says so
    clean, _ = analyze(scene(seed=0))
    ground_first = CFG.cam_height / math.tan(math.atan(CFG.cy / CFG.fy) + CFG.cam_pitch) + CFG.cam_forward
    assert abs(clean.blind_until - ground_first) < 0.2
    assert clean.blind_until > 0.9


# ---------------------------------------------------------------- (f) fail closed
@pytest.mark.parametrize("fill", [np.nan, 0.0, -np.inf])
def test_covered_lens_fails_closed(fill):
    res, plane = analyze(np.full((CFG.height, CFG.width), fill, np.float32))
    assert plane is None and not res.ground_ok and res.distance_positive == 0.0 and res.distance_negative == 0.0


def test_fail_closed_inputs():
    good = gravity_align(body_points(scene(seed=0)), 0, 0)
    plane = fit_ground(good)
    bad = good.copy()
    bad[5, 1] = np.nan
    for pts, pl, vf in ((bad, plane, 1.0), (good, None, 1.0), (np.zeros((4, 2)), plane, 1.0),
                        (good, plane, 0.2), (good, plane, math.nan)):
        r = analyze_corridor(pts, pl, valid_fraction=vf)
        assert not r.ground_ok and r.distance_positive == 0.0 and r.distance_negative == 0.0
    assert analyze_corridor(good, plane, valid_fraction=0.9).ground_ok


# ---------------------------------------------------------------- (g) wrong attitude
def test_imu_attitude_error_degradation():
    err = math.radians(3)
    rng = np.random.default_rng(5)
    n, phantom, lost, fail = 40, 0, 0, 0
    for i in range(n):
        r, q = rng.uniform(-3, 3), rng.uniform(-5, 5)
        tr, tq = math.radians(r), math.radians(q)
        er, eq = rng.choice([-err, err]), rng.choice([-err, err])
        res, _ = analyze(scene(roll=tr, pitch=tq, seed=i), tr + er, tq + eq)
        phantom += math.isfinite(res.distance_positive) or math.isfinite(res.distance_negative)
        fail += not res.ground_ok
        res2, _ = analyze(scene([box(1.5, 0.15)], roll=tr, pitch=tq, seed=i), tr + er, tq + eq)
        lost += not (math.isfinite(res2.distance_positive) and abs(res2.distance_positive - 1.5) < 0.15)
    say(f"SIM attitude wrong by 3 deg (roll and pitch): phantom={phantom/n:.2f} fail-closed={fail/n:.2f} "
        f"0.15 m box at 1.5 m missed={lost/n:.2f}")
    # The plane is fitted from the points, so a wrong attitude only matters through the tilt limit and the
    # n_z>0 gravity prior; record the degradation instead of promising immunity.
    assert phantom / n <= 0.1 and lost / n <= 0.2


def test_runtime_under_one_second():
    d = scene([box(1.5, 0.15)], seed=0)
    analyze(d)
    t = time.perf_counter()
    analyze(d)
    dt = time.perf_counter() - t
    say(f"SIM 320x240 analysis time {dt*1000:.0f} ms")
    assert dt < 1.0


# ---------------------------------------------------------------- audit fixes (G1-G5)
def _body_aligned(boxes=(), seed=0, **kw):
    return gravity_align(body_points(scene(boxes, seed=seed, **kw)), 0, 0)


@pytest.mark.parametrize("name,boxes", [
    ("block0.5", [Box(0.5, 3.0, -3, 3, 0.5)]),
    ("block1.0", [Box(0.3, 3.0, -3, 3, 1.0)]),
    ("block0.8_far", [Box(0.5, 6.0, -3, 3, 0.8)]),
    ("stairs0.15", [Box(1.0 + 0.3 * k, 1.3 + 0.3 * k, -3, 3, 0.15 * (k + 1)) for k in range(6)]),
    ("wall", [Box(0.5, 0.6, -3, 3, 2.0)]),
])
def test_g1_raised_surface_filling_view_is_not_ground(name, boxes):
    pts = _body_aligned(boxes)
    assert fit_ground(pts) is None, name
    r, plane = analyze(scene(boxes))
    assert plane is None and not r.ground_ok and r.distance_positive == 0.0 and r.distance_negative == 0.0


def test_g1_ramp_10deg_is_accepted_documented_limit_and_14deg_refused():
    def ramp(deg):
        t = math.tan(math.radians(deg))
        return [Box(x, x + 0.05, -3, 3, t * (x + 0.05)) for x in np.arange(0.3, 3.0, 0.05)]
    pl = fit_ground(_body_aligned(ramp(10)))
    assert pl is not None and abs(pl.tilt_deg - 10) < 1.0 and abs(pl.offset) < 0.1
    assert fit_ground(_body_aligned(ramp(14))) is None          # beyond the default 12 deg tilt limit
    assert fit_ground(_body_aligned(ramp(14)), max_tilt_deg=20.0) is not None   # explicit, documented parameter


def test_g1_offset_limit_is_a_parameter_and_floor_offset_is_near_zero():
    pts = _body_aligned()
    assert abs(fit_ground(pts).offset) < 0.02
    lifted = pts + np.array([0.0, 0.0, 0.2])                      # whole floor 0.2 m above the origin
    assert fit_ground(lifted) is None
    assert fit_ground(lifted, max_plane_offset=0.25) is not None
    assert fit_ground(pts - np.array([0.0, 0.0, 0.2])) is None    # and below
    assert fit_ground(pts + np.array([0.0, 0.0, 0.04])) is not None   # within 0.05 m (sway of body height)


def test_g2_unobserved_near_field_fails_closed():
    rng = np.random.default_rng(0)
    far = np.column_stack([rng.uniform(2, 3, 3000), rng.uniform(-1, 1, 3000), rng.normal(0, 0.005, 3000)])
    pl = fit_ground(far)
    assert pl is not None
    r = analyze_corridor(far, pl, valid_fraction=1.0)
    assert not r.ground_ok and r.distance_positive == 0.0 and r.distance_negative == 0.0
    assert analyze_corridor(far, pl, valid_fraction=1.0, max_blind=2.5).ground_ok   # explicit opt-in
    # no ground at all in the corridor (everything above h_max or empty): closed, not "clear"
    wall = np.column_stack([np.full(2000, 1.0), rng.uniform(-0.3, 0.3, 2000), rng.uniform(0.5, 1.4, 2000)])
    r = analyze_corridor(np.vstack([far, wall]), pl, valid_fraction=1.0, max_range=1.2)
    assert r.ground_ok  # max_range below max_blind: nothing farther than max_blind could be vouched for anyway
    r = analyze_corridor(np.vstack([far, wall]), pl, valid_fraction=1.0)
    assert not r.ground_ok
    nominal, _ = analyze(scene(seed=0))
    assert nominal.ground_ok and 1.0 < nominal.blind_until < 1.5     # nominal mount keeps working


_KNOBS_CORRIDOR = ["corridor_half_width", "max_range", "h_max", "drop_min", "cell", "min_gap", "k_sigma", "h_floor",
                   "max_blind", "h_min", "min_hits", "min_valid_fraction"]
_KNOBS_FIT = ["fit_x_max", "fit_half_width", "band_percentile", "band_height", "inlier_tol", "min_points",
              "min_inlier_frac", "max_tilt_deg", "iters", "max_sample", "max_plane_offset"]
_BAD = [math.nan, math.inf, -math.inf, -1.0, 0, True, "1", None]


@pytest.mark.parametrize("name", _KNOBS_CORRIDOR)
@pytest.mark.parametrize("bad", _BAD)
def test_g3_bad_corridor_config_raises(name, bad):
    if name == "h_min" and bad is None:
        pytest.skip("None means derive")
    good = _body_aligned([box(1.5, 0.15)])
    with pytest.raises(ValueError):
        analyze_corridor(good, fit_ground(good), valid_fraction=1.0, **{name: bad})


@pytest.mark.parametrize("name", _KNOBS_FIT)
@pytest.mark.parametrize("bad", _BAD)
def test_g3_bad_fit_config_raises(name, bad):
    with pytest.raises(ValueError):
        fit_ground(_body_aligned(), **{name: bad})


def test_g3_bad_config_raises_even_when_plane_is_none_and_data_errors_do_not():
    good = _body_aligned()
    with pytest.raises(ValueError):
        analyze_corridor(good, None, valid_fraction=1.0, h_max=math.nan)
    assert not analyze_corridor(good, None, valid_fraction=1.0).ground_ok     # data error: closed, no raise


def test_g4_valid_fraction_is_required_and_checked():
    good = _body_aligned()
    plane = fit_ground(good)
    with pytest.raises(TypeError):
        analyze_corridor(good, plane)
    with pytest.raises(ValueError):
        analyze_corridor(good, plane, valid_fraction="1")
    assert not analyze_corridor(good, plane, valid_fraction=1.5).ground_ok     # not a fraction: closed
    assert not analyze_corridor(good, plane, valid_fraction=0.4).ground_ok
    assert analyze_corridor(good, plane, valid_fraction=0.5).ground_ok


def test_g5_attitude_matrix_matches_explicit_euler():
    rng = np.random.default_rng(1)
    for _ in range(50):
        r, q = rng.uniform(-1.2, 1.2, 2)
        Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
        Ry = np.array([[math.cos(q), 0, math.sin(q)], [0, 1, 0], [-math.sin(q), 0, math.cos(q)]])
        R = attitude_matrix(r, q)
        assert np.allclose(R, Ry @ Rx, atol=1e-12) and np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    from humanoid_arc_bme.onboard.pipeline.perception import ground
    assert not hasattr(ground, "Rx") and ground.quat_to_matrix.__module__.endswith("geometry")
