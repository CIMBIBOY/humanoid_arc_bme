import math
import time

import numpy as np
import pytest

from humanoid_arc_bme.onboard.pipeline.manipulation.grasp_analytic import AnalyticGraspProposer, HandSpec
from humanoid_arc_bme.onboard.pipeline.interfaces import GraspProposer, Pose


def _rot(q):
    return Pose((0, 0, 0), q).rotation_matrix()


def cylinder(radius=0.025, height=0.12, n=1800, seed=1, noise=0.0007):
    """Upright cylinder surface, visible half only (camera on -x): x <= 0.1 r."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(math.pi / 2 - 0.1, 3 * math.pi / 2 + 0.1, n)  # cos<=~0.1 -> x<=0.1r
    z = rng.uniform(0.0, height, n)
    p = np.stack([radius * np.cos(th), radius * np.sin(th), z], 1)
    return p + rng.normal(0, noise, p.shape)


def sphere_half(radius=0.1, n=1500, seed=2):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v = v[v[:, 0] <= 0]
    return radius * v + np.array([0, 0, radius])


def test_protocol():
    assert isinstance(AnalyticGraspProposer(), GraspProposer)


def test_cylinder_pinch():
    cands = AnalyticGraspProposer().propose(cylinder())
    assert len(cands) >= 1
    best = cands[0]
    R = _rot(best.pose.quat)
    closing = R[:, 1]
    assert abs(closing[2]) < 0.3  # perpendicular to the (vertical) cylinder axis
    mid = np.array(best.pose.transform_point((HandSpec().palm_standoff, 0, 0)))
    assert np.linalg.norm(mid[:2]) < 0.01  # midpoint within 1 cm of the axis
    assert best.kind == "pinch" and 0.04 < best.aperture < 0.06


def test_too_wide_returns_empty():
    assert AnalyticGraspProposer().propose(sphere_half(0.1)) == []


@pytest.mark.parametrize(
    "bad",
    [
        None,
        np.zeros((0, 3)),
        np.zeros((5, 3)),
        np.full((200, 3), np.nan),
        np.zeros((100, 2)),
        np.zeros(300),
        np.zeros((10, 10, 3)),
        "abc",
    ],
)
def test_bad_input_returns_empty(bad):
    assert AnalyticGraspProposer().propose(bad) == []


def test_nan_rows_dropped():
    p = cylinder()
    p[::7] = np.nan
    assert len(AnalyticGraspProposer().propose(p)) >= 1


def test_bad_obstacles_fail_closed():
    assert AnalyticGraspProposer().propose(cylinder(), np.zeros((4, 2))) == []


def test_obstacle_removes_candidate_but_alternative_remains():
    prop = AnalyticGraspProposer()
    p = cylinder()
    free = prop.propose(p)
    best = free[0]
    r = prop.spec.finger_clearance_radius
    block = np.array([best.pose.transform_point((prop.spec.palm_standoff * 0.5, best.aperture / 2 + r, 0.0))])
    blocked = prop.propose(p, block)
    assert len(blocked) >= 1
    assert block.shape == (1, 3)
    # the exact best candidate (same position and same orientation) must be gone, and the free list had it
    assert _same_pose(free[0], best)
    assert not any(_same_pose(c, best) for c in blocked)
    assert blocked[0].score <= free[0].score + 1e-12


def _same_pose(a, b):
    return np.allclose(a.pose.position, b.pose.position, atol=1e-9) and np.allclose(a.pose.quat, b.pose.quat, atol=1e-9)


def test_min_z_rejects_below_table():
    assert AnalyticGraspProposer(min_z=10.0).propose(cylinder()) == []


def test_deterministic():
    a = AnalyticGraspProposer(seed=3).propose(cylinder())
    b = AnalyticGraspProposer(seed=3).propose(cylinder())
    assert a == b


def test_scores_sorted_and_quats_unit():
    cands = AnalyticGraspProposer().propose(cylinder())
    scores = [c.score for c in cands]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)
    assert len(cands) <= 20
    for c in cands:
        assert abs(sum(q * q for q in c.pose.quat) - 1.0) < 1e-9
        R = _rot(c.pose.quat)
        assert abs(np.linalg.det(R) - 1.0) < 1e-6


def test_max_candidates():
    assert len(AnalyticGraspProposer(max_candidates=3).propose(cylinder())) <= 3


def test_hand_spec_validation():
    with pytest.raises(ValueError):
        HandSpec(min_aperture=0.2, max_aperture=0.1)
    with pytest.raises(ValueError):
        HandSpec(finger_clearance_radius=-1.0)
    with pytest.raises(ValueError):
        HandSpec(approaches=("up",))


def test_runtime_bound():
    p = cylinder(n=2000)
    t = time.perf_counter()
    AnalyticGraspProposer().propose(p, np.random.default_rng(0).uniform(-0.5, 0.5, (3000, 3)) + np.array([0, 0.3, 0]))
    assert time.perf_counter() - t < 10.0


def _best_and_block():
    prop = AnalyticGraspProposer()
    p = cylinder()
    best = prop.propose(p)[0]
    r = prop.spec.finger_clearance_radius
    block = np.array(best.pose.transform_point((prop.spec.palm_standoff * 0.5, best.aperture / 2 + r, 0.0)))
    return prop, p, best, block


def test_dense_obstacle_cloud_is_not_strided_blocker_still_checked():
    prop, p, best, block = _best_and_block()
    rng = np.random.default_rng(5)
    filler = rng.uniform(1.0, 3.0, (20000, 3))  # far away, many distinct voxels
    obs = np.vstack([filler[:1], block[None], filler[1:]])  # blocker at row 1: the old 5000-row stride skipped it
    blocked = prop.propose(p, obs)
    assert not any(_same_pose(c, best) for c in blocked)


def test_too_dense_obstacle_cloud_fails_closed():
    prop = AnalyticGraspProposer()
    # 60000 points on distinct 5 mm voxels far from the object -> above the voxel cap
    g = np.stack(np.meshgrid(np.arange(250), np.arange(240), [0.0]), -1).reshape(-1, 3) * 0.005 + 5.0
    assert len(g) > 50000
    assert prop.propose(cylinder(), g) == []


def test_nonfinite_obstacle_rows_fail_closed():
    prop = AnalyticGraspProposer()
    for bad in (np.nan, np.inf, -np.inf):
        obs = np.array([[5.0, 5.0, 5.0], [bad, 0.0, 0.0]])
        assert prop.propose(cylinder(), obs) == []
    assert len(prop.propose(cylinder(), np.array([[5.0, 5.0, 5.0]]))) >= 1


def test_target_collision_uses_full_cloud_not_the_candidate_subset():
    prop = AnalyticGraspProposer()
    p = cylinder(n=6000)
    best = prop.propose(p)[0]
    r = prop.spec.finger_clearance_radius
    spot = np.array(best.pose.transform_point((prop.spec.palm_standoff * 0.5, best.aperture / 2 + r, 0.0)))
    q = p.copy()
    q[1] = spot  # row 1 is not in the 2000-row stride used for candidate generation, so generation is unchanged
    assert np.linalg.norm(q[1] - p[1]) > 1e-3
    out = prop.propose(q)
    assert not any(_same_pose(c, best) for c in out)


# ---------------------------------------------------------------- reach constraint / contact refinement
from humanoid_arc_bme.onboard.pipeline.manipulation.grasp_analytic import ReachConstraint  # noqa: E402

LEGACY = dict(reach=None, contact_refinement=False)
# (score, position, aperture) of the top candidates of the ORIGINAL code, recorded before the change.
GOLDEN = {"cyl": [[0.67809, [0.0015, -0.00048, 0.16909], 0.04948], [0.63366, [0.06144, 0.00216, 0.10909], 0.04948], [0.62411, [0.05859, -0.00799, 0.00768], 0.05098], [0.61878, [0.0595, -0.00025, 0.09528], 0.05028]], "box": [[0.3664, [0.34935, 0.36451, 0.76597], 0.05743], [0.3664, [0.24126, 0.41665, 0.76597], 0.05743], [0.33984, [0.29576, 0.38892, 0.82626], 0.06831], [0.33275, [0.26901, 0.33521, 0.76626], 0.06831]]}


def box_partial(yaw=0.3, centre=(0.3, 0.4, 0.75), cam=(0.06, 0.0, 1.1), n=800, seed=3, half=(0.025, 0.025, 0.05)):
    """Visible faces only (outward normal towards the camera) of an upright box, as from one depth view."""
    rng = np.random.default_rng(seed)
    half = np.array(half)
    c = np.array(centre)
    R = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    out = []
    for ax in range(3):
        for sg in (-1, 1):
            nrm = np.zeros(3)
            nrm[ax] = sg
            nw = R @ nrm
            if ax == 2 and sg < 0 or nw @ (np.array(cam) - (c + nw * half[ax])) <= 0:
                continue
            q = rng.uniform(-1, 1, (n, 3)) * half
            q[:, ax] = sg * half[ax]
            out.append(c + q @ R.T)
    return np.vstack(out), c, R, half


def _legacy_box():
    rng = np.random.default_rng(3)
    H = np.array([.025, .025, .05])
    th = 0.3
    R = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    out = []
    for ax, s in ((0, -1), (1, -1), (2, 1)):
        q = rng.uniform(-1, 1, (500, 3)) * H
        q[:, ax] = s * H[ax]
        out.append(q)
    return np.vstack(out) @ R.T + np.array([.3, .4, .75])


@pytest.mark.parametrize("name", ["cyl", "box"])
def test_disabled_mode_reproduces_original_behaviour(name):
    pts = cylinder() if name == "cyl" else _legacy_box()
    cs = AnalyticGraspProposer(min_z=None if name == "cyl" else 0.70, **LEGACY).propose(pts)
    for c, (sc, pos, ap) in zip(cs, GOLDEN[name]):
        assert c.score == pytest.approx(sc, abs=1e-5) and np.allclose(c.pose.position, pos, atol=1e-5)
        assert c.aperture == pytest.approx(ap, abs=1e-5)


def test_reach_constraint_validation():
    for bad in (dict(reference_dir=(0, 0, 0)), dict(cone_half_angle_deg=0), dict(cone_half_angle_deg=181),
                dict(closing_weight=1.5), dict(max_elevation_deg=90), dict(preferred_closing=(1, 2)),
                dict(reference_dir=(1, float("nan"), 0))):
        with pytest.raises(ValueError):
            ReachConstraint(**bad)
    with pytest.raises(TypeError):
        AnalyticGraspProposer(reach="x")
    with pytest.raises(ValueError):
        AnalyticGraspProposer(snap_tol=0)


def test_cone_geometry():
    r = ReachConstraint(reference_dir=(1, 0, 0), cone_half_angle_deg=60)
    ang = lambda d: np.array([math.cos(math.radians(d)), math.sin(math.radians(d)), 0.0])
    assert r.allows(ang(0)) and r.allows(ang(59.9)) and r.allows(ang(-59.9))
    assert not r.allows(ang(60.5)) and not r.allows(ang(180)) and not r.allows(ang(-90))
    assert not r.allows([0.5, 0, -0.8660254])  # top-down by default
    assert ReachConstraint(allow_top_down=True, cone_half_angle_deg=60).allows([0.5, 0, -0.8660254])
    rot = ReachConstraint(reference_dir=(0, 1, 0), cone_half_angle_deg=30)
    assert rot.allows(ang(90)) and rot.allows(ang(110)) and not rot.allows(ang(0))


def test_closing_preference_is_soft():
    r = ReachConstraint(preferred_closing=(0, 1, 0), closing_weight=0.2)
    assert r.closing_factor([0, 1, 0]) == pytest.approx(1.0)
    assert r.closing_factor([0, -1, 0]) == pytest.approx(1.0)
    assert r.closing_factor([1, 0, 0]) == pytest.approx(0.8)  # reduced, never zero
    assert ReachConstraint(closing_weight=0.0).closing_factor([1, 0, 0]) == 1.0


def _approach_dirs(cands):
    return [_rot(c.pose.quat)[:, 0] for c in cands]


def test_hard_filter_keeps_only_cone_side_approaches():
    pts, c, R, half = box_partial(yaw=0.3)
    free = AnalyticGraspProposer(min_z=0.70, reach=None).propose(pts)
    assert any(abs(a[2]) > 0.5 for a in _approach_dirs(free)) or any(a[0] < 0 for a in _approach_dirs(free))
    con = AnalyticGraspProposer(min_z=0.70).propose(pts)
    assert con
    for a in _approach_dirs(con):
        assert a[0] >= math.cos(math.radians(60)) - 1e-6 and abs(a[2]) <= math.sin(math.radians(15)) + 1e-6
    # a different forward direction gives approaches in that cone
    side = AnalyticGraspProposer(min_z=0.70, reach=ReachConstraint(reference_dir=(0, 1, 0))).propose(pts)
    for a in _approach_dirs(side):
        assert a[1] >= math.cos(math.radians(60)) - 1e-6
    # allowing top-down brings the top approaches back
    td = AnalyticGraspProposer(min_z=0.70, reach=ReachConstraint(allow_top_down=True, cone_half_angle_deg=180)).propose(pts)
    assert any(a[2] < -0.9 for a in _approach_dirs(td))


def test_constrained_candidates_are_subset_of_unconstrained():
    pts, *_ = box_partial(yaw=-0.4)
    free = AnalyticGraspProposer(min_z=0.70, reach=None, max_candidates=200).propose(pts)
    con = AnalyticGraspProposer(min_z=0.70, max_candidates=200).propose(pts)
    assert len(con) <= len(free)


def test_determinism_with_new_options():
    pts, *_ = box_partial()
    a = AnalyticGraspProposer(min_z=0.70, seed=5).propose(pts)
    b = AnalyticGraspProposer(min_z=0.70, seed=5).propose(pts)
    assert [(x.score, x.pose, x.aperture) for x in a] == [(x.score, x.pose, x.aperture) for x in b]


def _surf_dist(p, c, R, half):
    """Unsigned distance of a point to the surface of the oriented box (inside points count their depth)."""
    q = np.abs((p - c) @ R) - half
    return float(np.linalg.norm(np.maximum(q, 0)) if (q > 0).any() else -q.max())


def _contacts(cand, spec=HandSpec()):
    R = _rot(cand.pose.quat)
    mid = np.array(cand.pose.position) + R[:, 0] * spec.palm_standoff
    return mid - R[:, 1] * cand.aperture / 2, mid + R[:, 1] * cand.aperture / 2


@pytest.mark.parametrize("yaw", [0.0, 0.3, -0.5, 0.7])
def test_partial_box_refined_contacts_on_true_surface(yaw):
    """Two visible faces only: refinement gives horizontal closing, observed contact on the surface, far contact on
    the (unobserved) opposite face within 1 cm, aperture = true width."""
    pts, c, R, half = box_partial(yaw=yaw)
    best = AnalyticGraspProposer(min_z=0.70).propose(pts)[0]
    c0, c1 = _contacts(best)
    for p in (c0, c1):
        assert _surf_dist(p, c, R, half) < 0.01
    assert abs(_rot(best.pose.quat)[:, 1][2]) < 1e-6
    assert best.aperture == pytest.approx(0.05, abs=0.004)


def test_refinement_beats_legacy_on_partial_clouds():
    ok = {"new": 0, "old": 0}
    for k in range(12):
        pts, c, R, half = box_partial(yaw=-0.7 + 0.12 * k, centre=(0.25 + 0.01 * k, 0.4, 0.75), seed=k)
        for tag, kw in (("new", dict(reach=None)), ("old", LEGACY)):
            cs = AnalyticGraspProposer(min_z=0.70, **kw).propose(pts)
            if cs:
                c0, c1 = _contacts(cs[0])
                d = [_surf_dist(p, c, R, half) for p in (c0, c1)]
                ok[tag] += max(d) <= 0.01 and abs(_rot(cs[0].pose.quat)[:, 1][2]) < 0.35
    assert ok["new"] >= 11 and ok["new"] > ok["old"]


def test_refinement_rejects_unobserved_contacts():
    pts, *_ = box_partial()
    assert AnalyticGraspProposer(min_z=0.70, snap_tol=1e-7).propose(pts) == []  # no observed point on the contact
    rng = np.random.default_rng(0)  # a sheet narrower than the minimum aperture in both horizontal directions
    wall = np.column_stack([np.full(600, 0.3), rng.uniform(0.0, 0.005, 600), rng.uniform(0.7, 0.8, 600)])
    assert AnalyticGraspProposer().propose(wall) == []


def test_refinement_penalises_poor_footprint_fit():
    pts, *_ = box_partial()
    arc = cylinder(radius=0.03, height=0.1)  # curved: footprint is a half-circle, not a rectangle
    s_box = AnalyticGraspProposer(min_z=0.70, reach=None).propose(pts)[0].score
    assert s_box > 0.0
    assert all(0.0 <= c.score <= 1.0 for c in AnalyticGraspProposer(reach=None).propose(arc))


def test_reach_rejects_bool_angles_and_bad_reference_and_normalises():
    for bad in (dict(cone_half_angle_deg=True), dict(max_elevation_deg=False), dict(closing_weight=True),
                dict(reference_dir=(float("inf"), 0, 0)), dict(reference_dir=(0.0, 0.0, 0.0)),
                dict(reference_dir=(1e-9, 0, 0)), dict(reference_dir=(True, 0, 0)),
                dict(preferred_closing=(0, 0, 0))):
        with pytest.raises(ValueError):
            ReachConstraint(**bad)
    r = ReachConstraint(reference_dir=(1e200, 1e200, 0))          # huge but finite: normalised, no overflow
    assert np.linalg.norm(r.reference_dir) == pytest.approx(1.0)
    assert r.allows(np.array([1.0, 1.0, 0.0]))
    assert np.linalg.norm(ReachConstraint(reference_dir=(3, 0, 4)).reference_dir) == pytest.approx(1.0)


def test_huge_coordinates_do_not_warn_or_raise():
    import warnings
    rng = np.random.default_rng(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = AnalyticGraspProposer().propose(rng.uniform(0, 1, (100, 3)) * 1e200)
    assert out == []


def test_collinear_cloud_has_no_footprint_pairs():
    line = np.column_stack([np.linspace(0.4, 0.5, 60), np.zeros(60), np.full(60, 0.85)])
    p = AnalyticGraspProposer(reach=None, n_pairs=0, contact_refinement=True)
    assert p._footprint_pairs(line) == []         # was 1 pair scored as a perfect box fit
    assert p.propose(line) == []
    rng = np.random.default_rng(0)
    box = np.column_stack([rng.uniform(0.5, 0.6, 300), rng.uniform(-0.05, 0.05, 300), rng.uniform(0.8, 0.9, 300)])
    assert len(p._footprint_pairs(box)) >= 1      # a real box is unaffected


# ---- workspace constraint (max_reach_from)
def test_max_reach_from_validation_and_default_off():
    assert ReachConstraint().max_reach_from is None and ReachConstraint().in_workspace([9, 9, 9])
    for bad in [((0, 0), 1.0), ((0, 0, 0), 0.0), ((0, 0, float("nan")), 1.0), ((0, 0, 0), True), "x", 5]:
        with pytest.raises(ValueError):
            ReachConstraint(max_reach_from=bad)
    r = ReachConstraint(max_reach_from=((0, 0, 0), 0.5))
    assert r.in_workspace([0.3, 0.3, 0.0]) and not r.in_workspace([0.4, 0.4, 0.0])


def test_max_reach_from_is_hard_filter_and_default_is_identical():
    pts, c, R, half = box_partial()
    base = AnalyticGraspProposer(min_z=0.70).propose(pts)
    explicit = AnalyticGraspProposer(min_z=0.70, reach=ReachConstraint(max_reach_from=None)).propose(pts)
    assert [(x.score, x.pose.position, x.aperture) for x in base] == [(x.score, x.pose.position, x.aperture) for x in explicit]
    far = AnalyticGraspProposer(min_z=0.70, reach=ReachConstraint(max_reach_from=((5, 5, 5), 0.5))).propose(pts)
    assert far == []
    near = AnalyticGraspProposer(min_z=0.70, reach=ReachConstraint(max_reach_from=(tuple(c), 0.3))).propose(pts)
    assert len(near) == len(base)
