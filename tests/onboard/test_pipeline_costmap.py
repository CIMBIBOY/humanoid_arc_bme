import time

import numpy as np
import pytest

from humanoid_arc_bme.onboard.pipeline.perception.costmap import CostmapConfig, LocalCostmap


def wall(x=1.01, y0=-1.0, y1=1.0, z=0.5, n=200, per=4):
    ys = np.repeat(np.linspace(y0, y1, n), per)
    return np.stack([np.full_like(ys, x), ys, np.full_like(ys, z)], axis=1)


def cfg(**kw):
    base = dict(size_x=6.0, size_y=6.0, resolution=0.05, inflation_radius=0.0)
    base.update(kw)
    return CostmapConfig(**base)


def test_wall_cells():
    m = LocalCostmap(cfg())
    m.update(wall(), 0.0)
    occ = m.occupied()
    ii, jj = np.nonzero(occ)
    assert np.allclose(m._cx[np.unique(ii)], 1.025)
    assert np.all(np.abs(m._cy[jj]) <= 1.03)
    assert occ.sum() >= 38


def test_band_filter():
    m = LocalCostmap(cfg())
    low, high = wall(z=0.0), wall(z=2.0)
    m.update(np.vstack([low, high]), 0.0)
    assert not m.occupied().any()


def test_single_noise_point_rejected():
    m = LocalCostmap(cfg(min_points_per_cell=3))
    m.update(np.array([[1.0, 0.0, 0.5]]), 0.0)
    assert not m.occupied().any()
    m.update(np.array([[1.01, 0.0, 0.5]] * 3), 0.1)
    assert m.occupied().sum() == 1


def test_inflation_grows_by_radius():
    m = LocalCostmap(cfg(inflation_radius=0.30))
    m.update(np.array([[1.01, 0.0, 0.5]] * 5), 0.0)
    n = m.inflated().sum()
    area = n * 0.05 ** 2
    assert area == pytest.approx(np.pi * 0.30 ** 2, rel=0.15)
    assert m.occupied().sum() == 1


def test_nearest_obstacle():
    m = LocalCostmap(cfg(inflation_radius=0.30))
    m.update(wall(), 0.0)
    d = m.nearest_obstacle(0.3, 3.0)
    assert d == pytest.approx(1.0 - 0.30, abs=0.06)
    far = LocalCostmap(cfg(inflation_radius=0.30))
    far.update(wall(y0=1.5, y1=2.5), 0.0)
    assert far.nearest_obstacle(0.3, 3.0) is None
    assert m.nearest_obstacle(0.3, 0.5) is None
    assert m.sector_min_distance(1, 0.3, 3.0) == pytest.approx(d)
    assert m.sector_min_distance(-1) == float("inf")


def test_decay_clears():
    m = LocalCostmap(cfg(decay_s=1.0))
    m.update(wall(), 0.0)
    assert m.occupied().any()
    m.update(np.zeros((0, 3)), 0.9)
    assert m.occupied().any()
    m.update(np.zeros((0, 3)), 1.2)
    assert not m.occupied().any()
    assert m.nearest_obstacle(0.3, 3.0) is None


def test_health():
    m = LocalCostmap(cfg(max_age_s=0.5))
    assert not m.healthy(0.0)
    m.update(np.zeros((0, 3)), 10.0)
    assert m.healthy(10.3)
    assert not m.healthy(10.6)
    assert not m.healthy(float("nan"))


def test_nonfinite_and_empty():
    m = LocalCostmap(cfg())
    p = wall()
    bad = np.array([[np.nan, 0, 0.5], [1, np.inf, 0.5], [1, 0, -np.inf]])
    m.update(np.vstack([p, bad]), 0.0)
    assert m.occupied().any()
    m.update(np.zeros((0, 3)), 0.1)
    m.update([], 0.2)
    with pytest.raises(ValueError):
        m.update(np.zeros((4, 2)), 0.3)
    with pytest.raises(ValueError):
        m.update(p, 0.0)  # time going backwards


@pytest.mark.parametrize("kw", [
    dict(size_x=0), dict(size_y=-1), dict(resolution=0), dict(resolution=float("nan")),
    dict(z_min=1.0, z_max=0.5), dict(min_points_per_cell=0), dict(inflation_radius=-0.1),
    dict(decay_s=0), dict(max_age_s=float("inf")), dict(size_x=0.01, resolution=0.05),
])
def test_invalid_config(kw):
    with pytest.raises(ValueError):
        CostmapConfig(**kw)


def test_performance_50k():
    rng = np.random.default_rng(0)
    pts = np.stack([rng.uniform(-3, 3, 50000), rng.uniform(-3, 3, 50000), rng.uniform(0, 2, 50000)], axis=1)
    m = LocalCostmap(cfg())
    m.update(pts, 0.0)  # warm-up
    t = time.perf_counter()
    m.update(pts, 0.1)
    assert time.perf_counter() - t < 0.5


def test_never_updated_map_fails_closed():
    m = LocalCostmap(cfg(inflation_radius=0.30))
    assert m.occupied().all() and m.inflated().all()
    assert m.nearest_obstacle(0.3, 3.0) == 0.0
    assert m.sector_min_distance(1) == 0.0 and m.sector_min_distance(-1) == 0.0
    m.update(np.zeros((0, 3)), 0.0)  # first heartbeat: now a real (empty) map
    assert not m.occupied().any()
    assert m.nearest_obstacle(0.3, 3.0) is None
    assert m.sector_min_distance(1) == float("inf")


def test_backwards_stamp_and_bad_stamp_rejected_and_state_kept():
    m = LocalCostmap(cfg())
    m.update(wall(), 1.0)
    before = m.occupied(1.0).copy()
    with pytest.raises(ValueError, match="backwards"):
        m.update(np.zeros((0, 3)), 0.5)
    for bad in (float("nan"), float("inf"), "1", None):
        with pytest.raises(ValueError):
            m.update(np.zeros((0, 3)), bad)
    assert (m.occupied(1.0) == before).all()


@pytest.mark.parametrize("kw", [
    dict(z_min=float("nan")), dict(z_max=float("inf")), dict(inflation_radius=float("nan")),
    dict(z_min="a"), dict(min_points_per_cell=True), dict(min_points_per_cell=2.5),
    dict(size_y=0.01, resolution=0.05), dict(size_x=True), dict(decay_s=float("nan")),
])
def test_invalid_config_more(kw):
    with pytest.raises(ValueError):
        CostmapConfig(**kw)


def test_nearest_obstacle_argument_validation():
    m = LocalCostmap(cfg())
    m.update(np.zeros((0, 3)), 0.0)
    for hw, mr in ((-0.1, 3.0), (0.3, 0.0), (float("nan"), 3.0), (0.3, float("inf"))):
        with pytest.raises(ValueError):
            m.nearest_obstacle(hw, mr)


def test_huge_coordinates_drop_silently_without_warning():
    import warnings
    m = LocalCostmap(cfg())
    pts = np.vstack([np.tile([1e200, 1e200, 0.5], (5, 1)), np.tile([-1e308, 1e308, 0.5], (5, 1)),
                     np.tile([1e30, 0.0, 0.5], (5, 1)), np.tile([1.0, 0.0, 0.5], (4, 1))])
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # RuntimeWarning (cast / overflow) would raise
        m.update(pts, 0.0)
    occ = m.occupied(0.0)
    assert occ.sum() == 1                        # only the real obstacle at x = 1.0 survives
    assert m.nearest_obstacle(0.3, 3.0) == pytest.approx(1.025, abs=0.03)
