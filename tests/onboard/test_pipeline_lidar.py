"""Lidar fusion: near-field coverage study (SIM, UNMEASURED mount) and the fused runner.

The Livox sensor model (`_lidar_render.py`) is from memory / UNVERIFIED and the mount is a study parameter, so the
numbers here are SIM results, not MEASURED. Set PIPELINE_VERBOSE=1 and run with -s to print the coverage table.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from _depth_render import CFG, Box, noisy, render
from _lidar_render import LidarSpec, scan
from humanoid_arc_bme.onboard.pipeline.perception.ground import GroundPlane
from humanoid_arc_bme.onboard.pipeline.interfaces import DepthFrame, Intrinsics, LocalizerEstimate, yaw_pose
from humanoid_arc_bme.onboard.pipeline.perception.lidar import LidarFrame, lidar_clearance
from humanoid_arc_bme.onboard.pipeline.runner import PipelineRunner, RunnerConfig, TickInput
from humanoid_arc_bme.onboard.pipeline.perception.sensing import camera_pose_from_mount

V = bool(os.environ.get("PIPELINE_VERBOSE"))
DT = 0.1
VX = 0.2
K = Intrinsics(CFG.fx, CFG.fy, CFG.cx, CFG.cy, CFG.width, CFG.height)
MOUNT = camera_pose_from_mount((CFG.cam_forward, 0.0, CFG.cam_height), CFG.cam_pitch)
DISTS = (0.4, 0.5, 0.7, 1.0)
COVER = 70.0       # mount pitch (deg down, mount height 1.2 m) that the study shows covers the near field
LOW = 0.0          # level mount: sees no near-field floor


def slab(d, w=0.3):
    """0.3 m high, 5 cm deep, 0.6 m wide slab whose front face is at x = d."""
    return Box(d, d + 0.05, -w, w, 0.3)


# ------------------------------------------------------------------ coverage study
def coverage_table(heights=(1.2,), pitches=(0, 15, 30, 45, 55, 60, 65, 70), n=20):
    """rows: (height, pitch, mean near ground returns, mean coverage, near_observed rate, {dist: detection rate}, false-hit
    rate on an empty floor). Detection = a hit within +-0.1 m of the true face."""
    rows = []
    for h in heights:
        for p in pitches:
            sp = LidarSpec(mount_height=h, pitch_deg=p)
            g, c, o, fp = [], [], [], 0
            for s in range(n):
                r = lidar_clearance(scan(sp, seed=s), None)
                g.append(r.ground_points_near), c.append(r.near_coverage), o.append(r.near_observed)
                fp += math.isfinite(r.distance)
            det = {}
            for d in DISTS:
                hit = 0
                for s in range(n):
                    r = lidar_clearance(scan(sp, [slab(d)], seed=100 + s), None)
                    hit += abs(r.distance - d) <= 0.1
                det[d] = hit / n
            rows.append((h, p, float(np.mean(g)), float(np.mean(c)), float(np.mean(o)), det, fp / n))
    return rows


@pytest.fixture(scope="module")
def table():
    t = coverage_table(heights=(1.2, 0.9))
    if V:
        for r in t:
            print(f"H={r[0]} pitch={r[1]:>2} ground_near={r[2]:7.0f} cov={r[3]:.2f} obs={r[4]:.2f} "
                  f"det={ {k: v for k, v in r[5].items()} } fp={r[6]:.2f}")
    return {(r[0], r[1]): r for r in t}


def test_level_mount_sees_neither_near_ground_nor_slab(table):
    r = table[(1.2, 0)]
    assert r[2] == 0 and r[3] == 0.0 and r[4] == 0.0
    assert all(v == 0.0 for v in r[5].values()), r[5]       # the level lidar cannot see a 0.3 m slab at 0.4..1.0 m


def test_pitched_mount_covers_the_near_field(table):
    r = table[(1.2, int(COVER))]
    assert r[4] == 1.0 and r[3] >= 0.8
    assert all(v >= 0.95 for v in r[5].values()), r[5]
    assert r[6] == 0.0                                       # no false hit on an empty floor


def test_coverage_grows_with_pitch_and_is_never_overclaimed(table):
    ps = (0, 15, 30, 45, 55, 60, 65, 70)
    gn = [table[(1.2, p)][2] for p in ps]
    assert gn == sorted(gn)
    for (h, p), r in table.items():
        assert r[6] == 0.0, (h, p)                           # never a false obstacle on empty floor
        if r[4] == 1.0:                                      # near field called observed => every slab must be seen
            assert all(v >= 0.95 for v in r[5].values()), (h, p, r[5])


# ------------------------------------------------------------------ lidar.py units
def test_clearance_slab_synthetic_points():
    rng = np.random.default_rng(0)
    gx, gy = rng.uniform(0.3, 3, 2000), rng.uniform(-1, 1, 2000)
    ground = np.stack([gx, gy, rng.normal(0, 0.01, 2000)], 1)
    sl = np.stack([np.full(40, 0.8), rng.uniform(-.3, .3, 40), rng.uniform(0.0, 0.3, 40)], 1)
    r = lidar_clearance(np.vstack([ground, sl]), None)
    assert r.ok and abs(r.distance - 0.8) < 1e-9 and r.n_hits >= 25 and r.nearest_xy[0] == r.distance
    assert r.near_observed
    assert lidar_clearance(ground, None).distance == math.inf


def test_clearance_uses_the_plane():
    p = np.stack([np.linspace(0.5, 2, 50), np.zeros(50), np.full(50, 0.5)], 1)        # a line at z = 0.5
    assert lidar_clearance(p, None).distance < 0.6                                    # vs z = 0: an obstacle
    plane = GroundPlane((0.0, 0.0, 1.0), -0.5, 1.0, 0.0, 1.0, 0.0, 50)                 # floor is at z = 0.5
    r = lidar_clearance(p, plane)
    assert r.distance == math.inf and r.ground_points_near > 0


@pytest.mark.parametrize("pts", [np.full((5, 3), np.nan), np.zeros((5, 2)), np.zeros(5), "x", None,
                                 np.array([[0.5, 0, 0.2], [np.inf, 0, 0]]), np.array([["a", "b", "c"]])])
def test_clearance_bad_input_fails_closed(pts):
    r = lidar_clearance(pts, None)
    assert not r.ok and r.distance == 0.0 and not r.near_observed


def test_clearance_bad_plane_and_config():
    assert not lidar_clearance(np.zeros((3, 3)), "plane").ok
    for kw in ({"corridor_half_width": math.nan}, {"max_range": 0.0}, {"h_min": -1.0}, {"min_hits": 0},
               {"min_hits": True}, {"h_max": 0.05}, {"near_x": (1.0, 0.5)}, {"min_near_coverage": 0.0}):
        with pytest.raises(ValueError):
            lidar_clearance(np.zeros((3, 3)), None, **kw)


def test_empty_cloud_is_ok_but_unobserved():
    r = lidar_clearance(np.zeros((0, 3)), None)
    assert r.ok and r.distance == math.inf and not r.near_observed


def test_lidar_frame_validation():
    f = LidarFrame(1.0, np.array([[1, 0, 0], [np.nan, 0, 0], [0, np.inf, 1.0]]))
    assert f.points.shape == (1, 3) and f.n_invalid == 2
    assert LidarFrame(1.0, []).points.shape == (0, 3)
    for st in (math.nan, math.inf, "a", None, True):
        with pytest.raises(ValueError):
            LidarFrame(st, np.zeros((1, 3)))
    for pts in (np.zeros((3, 2)), np.zeros(3), np.array(["a"])):
        with pytest.raises(ValueError):
            LidarFrame(0.0, pts)


# ------------------------------------------------------------------ runner end to end
def _tick(t, cam, lid, cmd=(VX, 0.0, 0.0), **kw):
    fd = DepthFrame(t, cam, K, MOUNT)
    return TickInput(t, fd, None, LocalizerEstimate(t, yaw_pose(0.0, 0.0, 0.0, 0.0), (1e-4,) * 6), 0.0, 0.0,
                     command=cmd, lidar=lid, **kw)


def run(d, seed, pitch, cfg=None, n=7, lid_stamp_off=0.0, cmd=(VX, 0.0, 0.0), use_lidar=True, boxes=None, lid_mod=None):
    """Static robot at x=0, slab with its face at d (visible to the lidar model and to the camera model; the camera
    still cannot see it: near-field blind zone). Returns the last TickOutput."""
    r = PipelineRunner(cfg)
    sp = LidarSpec(pitch_deg=pitch)
    bx = [slab(d)] if boxes is None else boxes
    out = None
    for i in range(n):
        t = i * DT
        cam = noisy(render(CFG, bx), seed * 100 + i)
        lid = LidarFrame(t + lid_stamp_off, scan(sp, bx, seed=seed * 100 + i)) if use_lidar else None
        if lid_mod is not None:
            lid = lid_mod(lid)
        out = r.tick(_tick(t, cam, lid, cmd))
    return r, out


def test_camera_alone_does_not_see_the_slab():
    for s in range(5):
        _, o = run(0.5, s, COVER, use_lidar=False)
        assert o.command_out[0] > 0.0 and "obstacle_stop" not in o.safety.reasons, (s, o.safety)


def test_fused_runner_stops_slab_at_half_metre_over_20_seeds():
    for s in range(20):
        r, o = run(0.5, s, COVER)
        assert o.command_out == (0.0, 0.0, 0.0), (s, o.safety)
        assert "obstacle_stop" in o.safety.reasons and o.safety.stop, (s, o.safety)
        assert o.lidar is not None and o.lidar.near_observed and abs(o.lidar.distance - 0.5) < 0.1
        assert r.costmap.nearest_obstacle(0.35, 3.0) is not None          # the lidar points reached the costmap


def test_insufficient_coverage_is_never_claimed_clear():
    cam_only = []
    for s in range(20):
        _, o = run(0.5, s, LOW)
        _, c = run(0.5, s, LOW, use_lidar=False)
        # level lidar: no near-field floor => unobserved, and the result is exactly the camera-only behaviour
        assert o.lidar is not None and not o.lidar.near_observed and o.lidar.ground_points_near == 0
        assert any(e.get("event") == "lidar_near_unobserved" for e in o.events)
        assert o.command_out == c.command_out and o.safety.reasons == c.safety.reasons
        cam_only.append(o.command_out[0] > 0.0)
    assert all(cam_only)        # the existing blind-zone behaviour (not a claim of safety): documented, not fixed here


def test_require_lidar_stops_on_unobserved_near_field():
    for s in range(5):
        _, o = run(0.5, s, LOW, RunnerConfig(require_lidar=True))
        assert o.command_out == (0.0, 0.0, 0.0) and "lidar_near_unobserved" in o.safety.reasons
        _, ok = run(1.8, s, COVER, RunnerConfig(require_lidar=True))      # covered, nothing near: goes
        assert "lidar_near_unobserved" not in ok.safety.reasons and ok.command_out[0] > 0.0, ok.safety


@pytest.mark.parametrize("require", [False, True])
@pytest.mark.parametrize("kind", ["stale", "invalid", "nan", "none"])
def test_stale_or_invalid_lidar(kind, require):
    cfg = RunnerConfig(require_lidar=require)
    kw = {}
    if kind == "stale":
        kw["lid_stamp_off"] = -1.0
    elif kind == "invalid":
        kw["lid_mod"] = lambda f: "not a frame"
    elif kind == "nan":
        def corrupt(f):
            f.points[0, 0] = np.nan          # array mutated after validation
            return f
        kw["lid_mod"] = corrupt
    elif kind == "none":
        kw["use_lidar"] = False
    _, o = run(1.8, 0, COVER, cfg, **kw)
    _, cam = run(1.8, 0, COVER, use_lidar=False)
    assert o.lidar is None
    if require:
        assert "lidar_stale" in o.safety.reasons and o.command_out == (0.0, 0.0, 0.0)
    else:
        assert "lidar_stale" not in o.safety.reasons and o.command_out == cam.command_out
        if kind != "none":
            assert any(e.get("event") == "lidar_unavailable" for e in o.events)


def test_stale_lidar_never_hides_a_camera_stop():
    # camera stops for a tall box at 0.6 m; a stale lidar (require_lidar False) must not loosen the camera's stop
    box = [Box(0.6, 0.7, -0.4, 0.4, 1.0)]
    _, a = run(0, 1, COVER, boxes=box, use_lidar=False)
    _, b = run(0, 1, COVER, boxes=box, lid_stamp_off=-5.0)
    assert a.command_out == b.command_out == (0.0, 0.0, 0.0) and a.safety.reasons == b.safety.reasons and b.safety.stop


def test_tighten_only_and_nan_safe_fused_distance():
    boxes = [[slab(0.5)], [slab(0.9)], [Box(1.5, 1.7, -0.3, 0.3, 0.5)], []]
    for bi, bx in enumerate(boxes):
        for s in range(5):
            r, f = run(0, s, COVER, boxes=bx, cmd=(VX, 0.05, 0.1))
            _, c = run(0, s, COVER, boxes=bx, cmd=(VX, 0.05, 0.1), use_lidar=False)
            assert abs(f.command_out[0]) <= abs(c.command_out[0]) + 1e-12
            assert abs(f.command_out[1]) <= abs(c.command_out[1]) + 1e-12
            assert abs(f.command_out[2]) <= 0.1 + 1e-12
            assert r._dist == r._dist and r._dist >= 0.0
            if f.lidar is not None and f.clearance.ground_ok:
                assert r._dist <= f.lidar.distance + 1e-9            # fused distance is the min


def test_turn_rule_uses_the_fused_costmap():
    # slab at 0.6 m: only the lidar sees it. Soft stop, but raw 0.6 m > turn_clearance_m -> turning is allowed.
    _, far = run(0.6, 2, COVER, cmd=(VX, 0.0, 0.2))
    assert far.safety.stop and "turn_only" in far.safety.reasons
    assert far.command_out[0] == 0.0 and 0.0 < far.command_out[2] <= 0.15 + 1e-12
    # slab at 0.4 m: its raw cells are inside the swept circle (0.45 m) -> NO turn.
    _, near = run(0.4, 2, COVER, cmd=(VX, 0.0, 0.2))
    assert near.safety.stop and "turn_only" not in near.safety.reasons and near.command_out == (0.0, 0.0, 0.0)


def test_obstacle_memory_gets_the_lidar_nearest():
    r, o = run(0.5, 3, COVER, n=6)
    assert len(r._mem) == 1                                          # remembered although the camera never saw it
    xb, _ = r._mem._body(r._mem._items[0], (0.0, 0.0, 0.0))
    assert abs(xb - 0.5) < 0.1


def test_reset_clears_lidar_state():
    r, _ = run(0.5, 1, COVER)
    r.reset()
    assert r._lidar_res is None and len(r._mem) == 0


def test_runner_config_validation():
    for kw in ({"require_lidar": 1}, {"max_lidar_age_s": 0.0}, {"max_lidar_age_s": math.nan},
               {"lidar_min_near_ground_points": 0}, {"lidar_min_near_ground_points": True},
               {"lidar_min_near_coverage": 0.0}, {"lidar_min_near_coverage": 1.5}, {"lidar_h_min": -0.1}):
        with pytest.raises(ValueError):
            PipelineRunner(RunnerConfig(**kw))
    c = RunnerConfig()
    assert c.require_lidar is False and c.max_lidar_age_s == 0.3


# ------------------------------------------------------------------ body self-exclusion (audit)
from humanoid_arc_bme.onboard.pipeline.perception.lidar import self_exclusion_keep  # noqa: E402


def test_self_exclusion_radius_boundaries():
    p = np.array([[0.05, 0.0, 0.5], [0.20, 0.0, 0.5], [0.30, 0.0, 0.5], [0.0, -0.24, 1.0], [0.20, 0.0, 0.0],
                  [0.20, 0.0, 1.6], [3.0, 0.0, 0.5]])
    keep = self_exclusion_keep(p, None, 0.25)
    assert keep.tolist() == [False, False, True, False, True, True, True]    # 0.20 m excluded, 0.30 m seen, floor + overhead kept
    assert self_exclusion_keep(p, None, 0.0).all()
    assert self_exclusion_keep(np.zeros((0, 3)), None).shape == (0,)


def _selfhit_runner(pts, n=30, **cfg):
    r = PipelineRunner(RunnerConfig(**cfg) if cfg else None)
    out = None
    for i in range(n):
        t = i * DT
        cam = noisy(render(CFG, []), 100 + i)
        out = r.tick(_tick(t, cam, LidarFrame(t, pts), cmd=(VX, 0.0, 0.0)))
    return r, out


def test_self_hits_never_cause_a_permanent_obstacle_stop():
    own = np.tile([0.05, 0.0, 0.5], (5, 1))
    _, o = _selfhit_runner(own)
    assert "obstacle_stop" not in o.safety.reasons and o.command_out[0] == pytest.approx(VX)
    _, o = _selfhit_runner(own, lidar_self_radius_m=0.0001)           # control: mask effectively off -> the old stop
    assert "obstacle_stop" in o.safety.reasons


def test_obstacle_just_outside_the_exclusion_radius_is_still_seen():
    _, o = _selfhit_runner(np.tile([0.30, 0.0, 0.5], (5, 1)))
    assert "obstacle_stop" in o.safety.reasons and o.command_out[0] == 0.0
    _, o = _selfhit_runner(np.tile([0.20, 0.0, 0.5], (5, 1)))         # inside the body cylinder: excluded
    assert "obstacle_stop" not in o.safety.reasons
