"""Dex3 hand closure map and grasp evidence (onboard/manipulation/dex3.py). Synthetic, no Isaac Sim."""

import math

import numpy as np
import pytest

from humanoid_arc_bme.onboard.api.safety import load_limits
from humanoid_arc_bme.onboard.manipulation.dex3 import CLOSED_POSE, OPEN_POSE, Dex3Hand
from humanoid_arc_bme.onboard.paths import DEFAULT_LIMITS

LIMITS = load_limits(DEFAULT_LIMITS)
SIDES = ("left", "right")
GRID = np.linspace(0.0, 1.0, 41)


@pytest.fixture(scope="module")
def hands():
    return {side: Dex3Hand(side) for side in SIDES}


def test_constants_are_the_yaml_limits_in_the_flexing_direction(hands):
    for side in SIDES:
        for name, closed, opened in zip(hands[side].joint_names, CLOSED_POSE[side], OPEN_POSE[side]):
            lower, upper = LIMITS.joint_pos[name]
            assert closed == pytest.approx(upper if closed > 0 else lower, abs=1e-5), name
            assert opened == 0.0 and lower <= 0.0 <= upper, name


def test_closure_endpoints_are_the_open_and_closed_poses(hands):
    for side, hand in hands.items():
        assert np.allclose(hand.closure_to_joints(0.0), hand.open_pose)
        assert np.allclose(hand.closure_to_joints(1.0), hand.closed_pose)
        assert hand.joints_to_closure(hand.closed_pose) == pytest.approx(1.0)
        assert hand.joints_to_closure(hand.open_pose) == pytest.approx(0.0)


def test_closure_map_is_monotone_in_the_closing_direction(hands):
    for side, hand in hands.items():
        path = np.array([hand.closure_to_joints(r) for r in GRID])
        step = np.diff(path, axis=0)
        closing = np.sign(hand.closed_pose - hand.open_pose)
        assert np.all(step * closing >= -1e-12), side


@pytest.mark.parametrize("side", SIDES)
def test_joint_targets_are_always_inside_the_limits(hands, side):
    hand = hands[side]
    for r in list(np.linspace(-1.0, 2.0, 31)) + [math.nan, math.inf, -math.inf]:
        q = hand.closure_to_joints(r)
        assert all(LIMITS.joint_range(n)[0] <= v <= LIMITS.joint_range(n)[1] for n, v in zip(hand.joint_names, q))


@pytest.mark.parametrize("side", SIDES)
def test_inverse_round_trip_on_the_segment(hands, side):
    hand = hands[side]
    for r in GRID:
        assert abs(hand.joints_to_closure(hand.closure_to_joints(r)) - r) < 1e-6


def test_closure_is_clipped_and_nan_means_open(hands):
    hand = hands["left"]
    assert np.allclose(hand.closure_to_joints(-0.5), hand.open_pose)
    assert np.allclose(hand.closure_to_joints(1.5), hand.closed_pose)
    assert np.allclose(hand.closure_to_joints(math.nan), hand.open_pose)


def test_joints_to_closure_handles_bad_input(hands):
    hand = hands["right"]
    assert math.isnan(hand.joints_to_closure([math.nan] + [0.0] * 6))
    assert math.isnan(hand.joints_to_closure([0.0] * 6))
    assert math.isnan(hand.joints_to_closure([0.0] * 8))
    assert math.isnan(hand.joints_to_closure(None))
    assert math.isnan(hand.joints_to_closure("abc"))
    beyond = hand.closed_pose * 3.0
    assert hand.joints_to_closure(beyond) == pytest.approx(1.0)


def test_grasp_state_flags_open_closed_and_stalled(hands):
    hand = hands["left"]
    opened = hand.grasp_state(hand.open_pose, 0.0)
    assert (opened.closed, opened.open, opened.stalled) == (False, True, False)
    closed = hand.grasp_state(hand.closed_pose, 1.0)
    assert (closed.closed, closed.open, closed.stalled) == (True, False, False)
    blocked = hand.grasp_state(hand.closure_to_joints(0.5), 1.0)
    assert blocked.closure == pytest.approx(0.5)
    assert blocked.stalled is True
    assert (blocked.closed, blocked.open) == (False, False)
    released = hand.grasp_state(hand.closure_to_joints(0.5), 0.0)
    assert released.stalled is False


def test_grasp_state_is_nan_safe(hands):
    hand = hands["right"]
    for q in ([math.nan] * 7, [0.0] * 3, None):
        state = hand.grasp_state(q, 1.0)
        assert math.isnan(state.closure)
        assert (state.closed, state.open, state.stalled) == (False, False, False)
    state = hand.grasp_state(hand.closure_to_joints(0.5), math.nan)
    assert state.stalled is False
    assert state.closure == pytest.approx(0.5)
