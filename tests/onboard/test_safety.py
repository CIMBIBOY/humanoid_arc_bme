"""S1 deterministic safety layer (api/safety.py). No Isaac Sim."""

import math
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from humanoid_arc_bme.onboard.api.contract import ArmCommand, BaseVelocity, Imu, RobotState
from humanoid_arc_bme.onboard.api.safety import (
    SafetyLayer,
    Watchdog,
    clamp_arm,
    clamp_base,
    clamp_joint_targets,
    hold_commands,
    load_limits,
)

ROOT = Path(__file__).resolve().parents[2]
LIMITS = load_limits(ROOT / "config" / "g1_limits.yaml")
URDF = ROOT / "humanoid_arc_bme" / "robots" / "g1" / "g1_29dof_with_hand_only_kinematics.urdf"
Q = (1.0, 0.0, 0.0, 0.0)
rng = random.Random(0)


def rnd(lo, hi):
    return rng.uniform(lo, hi)


def state(pose=(0.2, 0.0, 0.1) + Q):
    return RobotState(
        stamp=1.0,
        joint_names=("a",),
        joint_pos=(0.0,),
        joint_vel=(0.0,),
        base_vel=(0.0, 0.0, 0.0),
        imu=Imu(Q, (0, 0, 0), (0, 0, 9.81)),
        wrist_poses={"left": pose, "right": pose},
    )


def test_limits_file_is_placeholder_and_refuses_hardware():
    assert LIMITS.validated_on_robot is False
    with pytest.raises(RuntimeError):
        LIMITS.require_validated()


def test_limits_cover_every_urdf_joint():
    root = ET.parse(URDF).getroot()
    urdf = {j.get("name"): j.find("limit") for j in root.iter("joint") if j.get("type") in ("revolute", "continuous")}
    assert set(urdf) == set(LIMITS.joint_pos)
    for name, lim in urdf.items():
        assert LIMITS.joint_pos[name] == pytest.approx((float(lim.get("lower")), float(lim.get("upper"))), abs=1e-4)


def test_base_never_leaves_limits_and_in_limit_is_unchanged():
    m = LIMITS.base_vel_max
    for _ in range(1000):
        cmd = BaseVelocity(rnd(-1, 1), rnd(-1, 1), rnd(-2, 2))
        out, events = clamp_base(cmd, LIMITS)
        assert abs(out.vx) <= m[0] and abs(out.vy) <= m[1] and abs(out.wz) <= m[2]
        assert bool(events) == (out != cmd)
        assert clamp_base(out, LIMITS) == (out, [])  # idempotent
    inside = BaseVelocity(0.1, -0.05, 0.2)
    assert clamp_base(inside, LIMITS) == (inside, [])


def test_arm_inside_box_and_speed_limited():
    dt = 0.02
    max_step = LIMITS.wrist_speed_max * dt
    for _ in range(1000):
        ref = tuple(rnd(lo, hi) for lo, hi in zip(LIMITS.ws_min, LIMITS.ws_max))
        cmd = ArmCommand("left", tuple(rnd(-2, 2) for _ in range(3)) + Q, "pelvis")
        out, events = clamp_arm(cmd, ref, dt, LIMITS)
        p = out.wrist_pose[:3]
        assert all(lo - 1e-12 <= x <= hi + 1e-12 for x, lo, hi in zip(p, LIMITS.ws_min, LIMITS.ws_max))
        assert math.dist(p, ref) <= max_step + 1e-12
        assert out.wrist_pose[3:] == cmd.wrist_pose[3:]  # orientation untouched
        assert bool(events) == (out != cmd)
        out2, ev2 = clamp_arm(out, ref, dt, LIMITS)  # idempotent
        assert (out2, ev2) == (out, [])


def test_arm_in_limit_command_is_unchanged_and_reports_nothing():
    ref = (0.2, 0.0, 0.1)
    cmd = ArmCommand("right", (0.2005, 0.0, 0.1) + Q, "pelvis")
    assert clamp_arm(cmd, ref, 0.02, LIMITS) == (cmd, [])


def test_arm_reference_outside_box_still_obeys_speed_limit():
    ref = (LIMITS.ws_max[0] + 0.2, 0.0, 0.1)
    cmd = ArmCommand("left", (0.2, 0.0, 0.1) + Q, "pelvis")
    out, events = clamp_arm(cmd, ref, 0.02, LIMITS)
    assert math.dist(out.wrist_pose[:3], ref) <= LIMITS.wrist_speed_max * 0.02 + 1e-12
    assert {e.rule for e in events} == {"wrist_speed"}


def test_joint_targets_within_range_and_speed():
    names = sorted(LIMITS.joint_pos)
    dt = 0.02
    # per-joint speed limits (legs, W4) override the default; every other joint keeps joint_vel_max_default
    assert LIMITS.joint_vel_limit("left_elbow_joint") == LIMITS.joint_vel_max
    assert LIMITS.joint_vel_limit("left_knee_joint") > LIMITS.joint_vel_max
    for _ in range(300):
        cur, tgt = [], []
        for n in names:
            lo, hi = LIMITS.joint_range(n)
            cur.append(rnd(lo, hi))
            tgt.append(rnd(lo - 1, hi + 1))
        out, events = clamp_joint_targets(names, tgt, cur, dt, LIMITS)
        for n, o, c in zip(names, out, cur):
            lo, hi = LIMITS.joint_range(n)
            assert lo - 1e-12 <= o <= hi + 1e-12
            assert abs(o - c) <= LIMITS.joint_vel_limit(n) * dt + 1e-12
        out2, ev2 = clamp_joint_targets(names, out, cur, dt, LIMITS)
        assert out2 == out and ev2 == []


def test_joint_outside_range_moves_toward_it_at_speed_limit():
    n = "left_elbow_joint"
    lo, hi = LIMITS.joint_range(n)
    cur = hi + 0.5
    out, events = clamp_joint_targets([n], [0.0], [cur], 0.02, LIMITS)
    assert out[0] == pytest.approx(cur - LIMITS.joint_vel_max * 0.02)
    assert any(e.rule == "joint_speed" for e in events)


def test_unknown_joint_and_bad_input_fail_closed():
    with pytest.raises(ValueError):
        clamp_joint_targets(["not_a_joint"], [0.0], [0.0], 0.02, LIMITS)
    with pytest.raises(ValueError):
        clamp_joint_targets(["left_elbow_joint"], [float("nan")], [0.0], 0.02, LIMITS)
    with pytest.raises(ValueError):
        clamp_joint_targets(["left_elbow_joint"], [0.0], [0.0], 0.0, LIMITS)
    with pytest.raises(ValueError):
        Watchdog(0)


def test_watchdog_stale_returns_hold_and_fresh_passes():
    layer = SafetyLayer(LIMITS, Watchdog(0.2))
    st = state()
    arm = ArmCommand("left", (0.2, 0.0, 0.1) + Q, "pelvis")
    # never fed -> hold
    base, arms, events = layer.filter(0.0, 0.02, st, BaseVelocity(0.1, 0, 0), [arm])
    assert base == BaseVelocity(0.0, 0.0, 0.0) and set(arms) == {"left", "right"}
    assert [e.rule for e in events] == ["watchdog"]
    assert all(a.wrist_pose == st.wrist_poses[a.side] for a in arms.values())
    # fed -> passes
    layer.command_received(1.0)
    base, arms, events = layer.filter(1.1, 0.02, st, BaseVelocity(0.1, 0, 0), [arm])
    assert base == BaseVelocity(0.1, 0, 0) and events == []
    # stale again (0.2 s limit)
    base, _, events = layer.filter(1.31, 0.02, st, BaseVelocity(0.1, 0, 0), [arm])
    assert base == BaseVelocity(0.0, 0.0, 0.0) and events[0].rule == "watchdog"
    # exactly at the timeout is still fresh
    layer.command_received(2.0)
    assert layer.filter(2.19, 0.02, st, BaseVelocity(0.1, 0, 0), [])[2] == []


def test_layer_tracks_last_applied_target_so_a_far_goal_is_approached_at_the_speed_limit():
    layer = SafetyLayer(LIMITS, Watchdog(1.0))
    st = state((0.2, 0.0, 0.1) + Q)
    goal = ArmCommand("left", (0.5, 0.0, 0.1) + Q, "pelvis")
    pos = st.wrist_poses["left"][0]
    for i in range(5):
        layer.command_received(float(i) * 0.02)
        _, arms, _ = layer.filter(float(i) * 0.02, 0.02, st, None, [goal])
        new = arms["left"].wrist_pose[0]
        assert new - pos == pytest.approx(LIMITS.wrist_speed_max * 0.02)
        pos = new


def test_layer_rejects_non_pelvis_frame():
    layer = SafetyLayer(LIMITS, Watchdog(1.0))
    layer.command_received(0.0)
    with pytest.raises(ValueError):
        layer.filter(0.0, 0.02, state(), None, [ArmCommand("left", (0.2, 0, 0.1) + Q, "map")])


def test_hold_commands():
    base, arms = hold_commands(state())
    assert base == BaseVelocity(0.0, 0.0, 0.0)
    assert arms["left"].frame == "pelvis"


def test_imports_without_heavy_dependencies():
    code = (
        "import sys, humanoid_arc_bme.onboard.api.safety;"
        "bad = {'isaaclab', 'torch', 'rclpy', 'numpy', 'omni', 'yaml'} & set(sys.modules);"
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
