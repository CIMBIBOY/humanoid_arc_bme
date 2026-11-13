"""G1 arm kinematics (onboard/manipulation/kinematics.py): FK, Jacobian, damped least-squares IK. Synthetic, no Isaac Sim."""

import math
import time

import numpy as np
import pytest

from humanoid_arc_bme.onboard.api.safety import load_limits
from humanoid_arc_bme.onboard.g1_layout import WAIST_JOINTS
from humanoid_arc_bme.onboard.manipulation.kinematics import G1ArmKinematics
from humanoid_arc_bme.onboard.paths import DEFAULT_LIMITS
from humanoid_arc_bme.onboard.pipeline.interfaces import Pose

LIMITS = load_limits(DEFAULT_LIMITS)
# Mirror of the right arm's joint values across the sagittal plane: roll, yaw and the wrist roll/yaw flip sign.
MIRROR_SIGNS = np.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0])


def in_joint_range(kin, q) -> bool:
    return all(LIMITS.joint_range(n)[0] <= v <= LIMITS.joint_range(n)[1] for n, v in zip(kin.joint_names, q))


def random_config(kin, rng) -> np.ndarray:
    return rng.uniform(kin.lower, kin.upper)


def numeric_jacobian(kin, q, h=1e-6) -> np.ndarray:
    """Central differences of `fk`: linear rows from the positions, angular rows from R(q+h) R(q-h)^T = I + 2h[w]x."""
    jac = np.zeros((6, len(q)))
    for i in range(len(q)):
        e = np.zeros(len(q))
        e[i] = h
        plus, minus = kin.fk(q + e), kin.fk(q - e)
        jac[:3, i] = (np.array(plus.position) - np.array(minus.position)) / (2 * h)
        d = plus.rotation_matrix() @ minus.rotation_matrix().T
        jac[3:, i] = np.array([d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]]) / (4 * h)
    return jac


@pytest.fixture(scope="module")
def arms():
    return {side: G1ArmKinematics(side) for side in ("left", "right")}


def test_zero_pose_is_a_sane_tool_position(arms):
    for side, kin in arms.items():
        pose = kin.fk(np.zeros(7))
        x, y, z = pose.position
        assert 0.1 < x < 0.4, side
        assert (y > 0.05) if side == "left" else (y < -0.05), side
        assert all(lo <= v <= hi for v, lo, hi in zip(pose.position, LIMITS.ws_min, LIMITS.ws_max)), side
        assert math.isclose(pose.quat[0] ** 2 + sum(c * c for c in pose.quat[1:]), 1.0, abs_tol=1e-12)


def test_left_and_right_chains_mirror_each_other(arms):
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = random_config(arms["left"], rng)
        left = arms["left"].fk(q).position
        right = arms["right"].fk(q * MIRROR_SIGNS).position
        assert left[0] == pytest.approx(right[0], abs=1e-4)
        assert left[1] == pytest.approx(-right[1], abs=1e-4)
        assert left[2] == pytest.approx(right[2], abs=1e-4)


@pytest.mark.parametrize("side", ["left", "right"])
def test_analytic_jacobian_matches_central_differences(arms, side):
    kin = arms[side]
    rng = np.random.default_rng(2)
    for _ in range(20):
        q = random_config(kin, rng)
        assert np.max(np.abs(kin.jacobian(q) - numeric_jacobian(kin, q))) < 1e-5


@pytest.mark.parametrize("side", ["left", "right"])
def test_ik_round_trip_from_perturbed_warm_start(arms, side):
    kin = arms[side]
    rng = np.random.default_rng(3)
    converged, pos_errors = 0, []
    for _ in range(200):
        q_true = random_config(kin, rng)
        q_init = np.clip(q_true + rng.uniform(-0.3, 0.3, len(q_true)), kin.lower, kin.upper)
        result = kin.ik(kin.fk(q_true), q_init)
        assert in_joint_range(kin, result.q)
        if result.converged:
            converged += 1
            pos_errors.append(result.pos_err_m)
            assert result.pos_err_m < 2e-3
    assert converged >= 0.95 * 200
    assert max(pos_errors) < 2e-3


def test_unreachable_target_fails_without_raising(arms):
    kin = arms["left"]
    q_init = np.full(7, 0.5)
    result = kin.ik(Pose((2.0, 0.0, 0.0)), q_init)
    assert result.converged is False
    assert np.all(np.isfinite(result.q))
    assert in_joint_range(kin, result.q)
    assert result.pos_err_m > 0.5


def test_non_finite_target_returns_clipped_warm_start(arms):
    kin = arms["right"]
    with pytest.raises(ValueError):
        Pose((math.nan, 0.0, 0.0))
    bad = Pose((0.2, -0.2, 0.1))
    object.__setattr__(bad, "position", (math.nan, 0.0, 0.0))  # a pose that bypassed validation
    q_init = np.array([5.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = kin.ik(bad, q_init)
    assert result.converged is False
    assert np.array_equal(result.q, np.clip(q_init, kin.lower, kin.upper))


def test_position_only_mode_converges_to_the_position(arms):
    kin = arms["left"]
    rng = np.random.default_rng(4)
    converged = 0
    for _ in range(50):
        q_true = random_config(kin, rng)
        q_init = np.clip(q_true + rng.uniform(-0.3, 0.3, len(q_true)), kin.lower, kin.upper)
        result = kin.ik(kin.fk(q_true), q_init, ori_weight=0.0)
        assert in_joint_range(kin, result.q)
        if result.converged:
            converged += 1
            assert result.pos_err_m < 1e-3
    assert converged >= 0.9 * 50


def test_use_waist_adds_three_joints_first():
    kin = G1ArmKinematics("right", use_waist=True)
    assert kin.joint_names[:3] == WAIST_JOINTS
    assert len(kin.joint_names) == 10
    assert kin.fk(np.zeros(10)).position[0] > 0.1
    rng = np.random.default_rng(5)
    converged = 0
    for _ in range(50):
        q_true = random_config(kin, rng)
        q_init = np.clip(q_true + rng.uniform(-0.3, 0.3, 10), kin.lower, kin.upper)
        result = kin.ik(kin.fk(q_true), q_init)
        assert in_joint_range(kin, result.q)
        converged += result.converged
    assert converged >= 0.8 * 50


def test_waist_fixed_at_a_nonzero_value_moves_the_tool():
    base = G1ArmKinematics("left")
    turned = G1ArmKinematics("left", waist=(0.3, 0.0, 0.0))
    q = np.zeros(7)
    assert not np.allclose(base.fk(q).position, turned.fk(q).position)
    with pytest.raises(ValueError):
        G1ArmKinematics("left", waist=(0.9, 0.0, 0.0))


def test_ik_is_deterministic_and_fast(arms):
    kin = arms["left"]
    rng = np.random.default_rng(6)
    cases = []
    for _ in range(200):
        q_true = random_config(kin, rng)
        q_init = np.clip(q_true + rng.uniform(-0.3, 0.3, 7), kin.lower, kin.upper)
        cases.append((kin.fk(q_true), q_init))
    first = kin.ik(*cases[0])
    second = kin.ik(*cases[0])
    assert np.array_equal(first.q, second.q) and first.iters == second.iters
    start = time.perf_counter()
    for target, q_init in cases:
        kin.ik(target, q_init)
    mean_ms = 1e3 * (time.perf_counter() - start) / len(cases)
    print(f"\nik mean solve time: {mean_ms:.3f} ms over {len(cases)} warm-started solves")
    assert mean_ms < 15.0


def test_reachable_reports_convergence(arms):
    kin = arms["left"]
    q_true = np.array([0.2, 0.3, -0.1, 0.8, 0.0, 0.1, 0.0])
    assert kin.reachable(kin.fk(q_true), q_true + 0.05) is True
    assert kin.reachable(Pose((2.0, 0.0, 0.0)), q_true) is False


def test_bad_arguments_raise(arms):
    with pytest.raises(ValueError):
        G1ArmKinematics("middle")
    with pytest.raises(ValueError):
        arms["left"].fk(np.zeros(6))
    with pytest.raises(ValueError):
        arms["left"].ik(Pose((0.2, 0.2, 0.0)), np.array([math.nan] * 7))
