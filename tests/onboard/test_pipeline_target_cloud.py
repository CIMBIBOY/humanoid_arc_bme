import time

import numpy as np
import pytest

from humanoid_arc_bme.onboard.pipeline.manipulation.target_cloud import clean_target_cloud, estimate_table_z

TZ = 0.75


def box_pts(c, half=(0.025, 0.025, 0.05), n=1500, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.uniform(-1, 1, (n, 3)) * half
    ax = rng.integers(0, 3, n)
    sg = rng.choice([-1, 1], n)
    q[np.arange(n), ax] = sg * np.asarray(half)[ax]
    return np.asarray(c) + q


def table(n=4000, seed=1, noise=0.0):
    rng = np.random.default_rng(seed)
    t = np.c_[rng.uniform(0.0, 0.6, n), rng.uniform(0.2, 0.6, n), np.full(n, TZ)]
    return t + rng.normal(0, noise, t.shape) if noise else t


C = (0.3, 0.4, TZ + 0.05)


def scene(noise=0.0):
    return np.vstack([box_pts(C), table(noise=noise)])


def test_table_removed_known_z():
    out, d = clean_target_cloud(scene(), table_z=TZ, prior_centre=C)
    assert d["n_table"] >= 4000 and d["n_out"] == len(out) > 1000
    assert out[:, 2].min() > TZ + 0.003 and np.abs(out - C).max() < 0.06


def test_table_z_estimated():
    z = estimate_table_z(scene())
    assert z == pytest.approx(TZ, abs=0.003)
    out, d = clean_target_cloud(scene(), prior_centre=C)
    assert d["table_z"] == pytest.approx(TZ, abs=0.003) and out[:, 2].min() > TZ


def test_noise_widens_tolerance():
    out, d = clean_target_cloud(scene(noise=0.002), table_z=TZ, prior_centre=C)
    assert d["tol"] > 0.004
    assert out[:, 2].min() > TZ + 0.004 and len(out) > 1000


def test_hand_blob_excluded():
    hand = np.random.default_rng(5).normal(0, 0.012, (400, 3)) + np.array([0.3, 0.30, TZ + 0.05])
    pts = np.vstack([scene(), hand])
    out, d = clean_target_cloud(pts, table_z=TZ, exclusions=[((0.3, 0.30, TZ + 0.05), 0.05)], prior_centre=C)
    assert d["n_excluded"] > 300
    assert out[:, 1].min() > 0.37  # the hand blob (y ~ 0.30) is gone, the box (y 0.375..0.425) remains


def test_touching_blob_without_exclusion_would_remain_but_component_picks_prior():
    other = box_pts((0.45, 0.4, TZ + 0.05), seed=3)
    out, d = clean_target_cloud(np.vstack([scene(), other]), table_z=TZ, prior_centre=C)
    assert d["n_components"] == 2 and np.abs(out - C).max() < 0.06
    out2, _ = clean_target_cloud(np.vstack([scene(), other]), table_z=TZ, prior_centre=(0.45, 0.4, TZ + 0.05))
    assert np.abs(out2 - (0.45, 0.4, TZ + 0.05)).max() < 0.06


def test_no_prior_picks_largest():
    small = box_pts((0.45, 0.4, TZ + 0.05), half=(0.01, 0.01, 0.02), n=200, seed=4)
    out, _ = clean_target_cloud(np.vstack([scene(), small]), table_z=TZ)
    assert np.abs(out - C).max() < 0.06


def test_too_small_and_prior_far_rejected():
    tiny = box_pts(C, half=(0.003, 0.003, 0.003), n=10)
    out, d = clean_target_cloud(tiny, table_z=TZ, prior_centre=C)
    assert out.shape == (0, 3) and d["reason"] == "too_small"
    out, d = clean_target_cloud(scene(), table_z=TZ, prior_centre=(0.0, 0.0, 1.5))
    assert out.shape == (0, 3) and d["reason"] == "prior_too_far"


@pytest.mark.parametrize("bad", [None, [], np.zeros((0, 3)), np.zeros((5, 2)), np.full((50, 3), np.nan), "x", np.zeros(3)])
def test_bad_inputs_fail_closed(bad):
    out, d = clean_target_cloud(bad, table_z=TZ)
    assert out.shape == (0, 3) and d["n_out"] == 0


def test_bad_arguments_fail_closed():
    assert clean_target_cloud(scene(), table_z=TZ, prior_centre=(np.nan, 0, 0))[0].shape == (0, 3)
    assert clean_target_cloud(scene(), table_z=TZ, voxel=0.0)[0].shape == (0, 3)
    out, _ = clean_target_cloud(scene(), table_z=TZ, prior_centre=C, exclusions=[("bad", 1), ((0, 0, 0), -1), ((np.nan, 0, 0), 1)])
    assert len(out) > 1000


def test_nan_rows_dropped():
    pts = scene()
    pts[::7] = np.nan
    out, d = clean_target_cloud(pts, table_z=TZ, prior_centre=C)
    assert len(out) > 500 and np.isfinite(out).all()


def test_everything_excluded():
    out, d = clean_target_cloud(scene(), table_z=TZ, exclusions=[(C, 5.0)])
    assert out.shape == (0, 3) and d["reason"] == "nothing_left"


def test_deterministic():
    a, _ = clean_target_cloud(scene(0.001), prior_centre=C)
    b, _ = clean_target_cloud(scene(0.001), prior_centre=C)
    assert np.array_equal(a, b)


def test_performance_50k():
    rng = np.random.default_rng(0)
    big = np.vstack([scene(), table(46000, seed=9), rng.uniform([0, 0.2, TZ], [0.6, 0.6, TZ + 0.2], (3000, 3))])
    t = time.time()
    out, d = clean_target_cloud(big, prior_centre=C, exclusions=[((0.1, 0.3, TZ + 0.1), 0.05)])
    assert time.time() - t < 1.0 and d["n_in"] > 50000 and len(out) > 500
