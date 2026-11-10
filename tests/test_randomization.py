"""Randomization ranges (tasks/pick_place/randomization.py). Pure Python, no Isaac Sim.

The module is loaded by path so the test needs neither the project root on
sys.path nor the package __init__ (which imports gymnasium).
"""

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "randomization", _ROOT / "humanoid_arc_bme/tasks/pick_place/randomization.py")
rnd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rnd)

PARAMS = rnd.PARAMS


def test_param_count_and_unique_names():
    assert len(PARAMS) == 24
    assert len({p.name for p in PARAMS}) == 24


def test_all_params_unmeasured():
    assert all(p.measured is False for p in PARAMS)


def test_param_validation():
    with pytest.raises(ValueError):
        rnd.Param("x", 1.0, 1.0, False, "reason")
    with pytest.raises(ValueError):
        rnd.Param("x", 0.0, 1.0, False, "   ")


def test_sample_is_deterministic_per_seed_and_differs_across_seeds():
    assert rnd.sample(7, 50, "in") == rnd.sample(7, 50, "in")
    assert rnd.sample(7, 50, "heldout") == rnd.sample(7, 50, "heldout")
    assert rnd.sample(7, 50, "edge") == rnd.sample(7, 50, "edge")
    assert rnd.sample(7, 50, "in") != rnd.sample(8, 50, "in")


def test_in_mode_always_inside():
    rows = rnd.sample(1, 200, "in")
    assert len(rows) == 200
    assert all(rnd.inside(r) for r in rows)


def test_edge_mode_hits_only_corners_and_reaches_both_ends():
    rows = rnd.sample(2, 200, "edge")
    for p in PARAMS:
        values = {r[p.name] for r in rows}
        assert values <= {p.low, p.high}, p.name
        assert p.low in values and p.high in values, p.name


def test_heldout_mode_outside_and_within_band():
    rows = rnd.sample(3, 200, "heldout")
    for r in rows:
        assert not rnd.inside(r)
    for p in PARAMS:
        band = 0.25 * (p.high - p.low)
        for r in rows:
            v = r[p.name]
            above = p.high < v <= p.high + band
            below = p.low - band <= v < p.low
            assert above or below, (p.name, v)


def test_invalid_mode_and_n_raise():
    with pytest.raises(ValueError):
        rnd.sample(0, 5, "bogus")
    with pytest.raises(ValueError):
        rnd.sample(0, 0, "in")


def test_inside_raises_keyerror_on_unknown_or_missing_key():
    good = rnd.sample(4, 1, "in")[0]
    with pytest.raises(KeyError):
        rnd.inside({**good, "not_a_param": 0.0})
    missing = dict(good)
    del missing["object_mass"]
    with pytest.raises(KeyError):
        rnd.inside(missing)


def test_json_round_trip(tmp_path):
    path = tmp_path / "params.json"
    rnd.to_json(str(path))
    loaded = rnd.from_json(str(path))
    assert isinstance(loaded, tuple)
    assert loaded == PARAMS


def test_describe_marks_every_param_as_guess():
    lines = rnd.describe().splitlines()
    assert len(lines) == 24
    for line, p in zip(lines, PARAMS):
        assert line.startswith(p.name)
        assert "GUESS" in line
        assert p.reason in line


def test_heldout_respects_physical_floor():
    floors = {p.name: p.floor for p in PARAMS if p.floor is not None}
    assert {"object_mass", "depth_dropout", "camera_delay_s", "action_delay_steps"} <= set(floors)
    for row in rnd.sample(3, 500, "heldout"):
        for name, floor in floors.items():
            assert row[name] >= floor
