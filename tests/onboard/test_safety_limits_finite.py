"""R-2: NaN/inf/negative/non-numeric limits are rejected at load; a NaN never turns a clamp off."""
import copy
import math

import pytest

from humanoid_arc_bme.onboard.api.contract import BaseVelocity
from humanoid_arc_bme.onboard.api.safety import clamp_base, limits_from_dict

GOOD = {"validated_on_robot": False, "pos_margin_rad": 0.05, "joint_vel_max_default": 1.0, "wrist_speed_max": 0.2,
        "workspace": {"frame": "pelvis", "min": [-1, -1, -1], "max": [1, 1, 1]},
        "base_vel_max": {"vx": 0.2, "vy": 0.1, "wz": 0.3}, "joints": {"j": {"lower": -1.0, "upper": 1.0}},
        "joint_vel_max": {"j": 2.0}}


def _mut(path, value):
    d = copy.deepcopy(GOOD)
    cur = d
    for k in path[:-1]:
        cur = cur[k]
    cur[path[-1]] = value
    return d


def test_good_limits_load():
    assert limits_from_dict(GOOD).base_vel_max == (0.2, 0.1, 0.3)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -0.1, 0.0, "0.2", None, True])
@pytest.mark.parametrize("path", [("base_vel_max", "vx"), ("base_vel_max", "wz"), ("wrist_speed_max",), ("joint_vel_max_default",),
                                  ("joint_vel_max", "j"), ("pos_margin_rad",), ("joints", "j", "lower"), ("joints", "j", "upper")])
def test_bad_scalar_limits_raise(path, bad):
    if bad == 0.0 and path in (("pos_margin_rad",), ("joints", "j", "lower"), ("joints", "j", "upper")):
        pytest.skip("zero is valid there")
    if bad == -0.1 and path in (("joints", "j", "lower"), ("joints", "j", "upper")):
        pytest.skip("negative joint bound is valid")
    with pytest.raises(ValueError):
        limits_from_dict(_mut(path, bad))


@pytest.mark.parametrize("which", ["min", "max"])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, "x", None])
def test_bad_workspace_raises(which, bad):
    d = copy.deepcopy(GOOD)
    d["workspace"][which][1] = bad
    with pytest.raises(ValueError):
        limits_from_dict(d)


def test_workspace_needs_three_axes_and_missing_keys_raise_value_error():
    d = copy.deepcopy(GOOD)
    d["workspace"]["min"] = [-1, -1]
    with pytest.raises(ValueError):
        limits_from_dict(d)
    d = copy.deepcopy(GOOD)
    del d["base_vel_max"]
    with pytest.raises(ValueError):
        limits_from_dict(d)


def test_nan_cannot_disable_base_clamp():
    with pytest.raises(ValueError):
        limits_from_dict(_mut(("base_vel_max", "vx"), math.nan))
    out, ev = clamp_base(BaseVelocity(5.0, 0, 0), limits_from_dict(GOOD))
    assert out.vx == 0.2 and ev
