import math

import pytest

from humanoid_arc_bme.onboard.task.motion import ServoLimits, Slew, go_to_point, turn_to, wrap

LIM = ServoLimits()


def integrate(pose, cmd, dt):
    x, y, yaw = pose
    vx, vy, wz = cmd
    return (x + (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt, y + (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt, wrap(yaw + wz * dt))


def run(pose, step, n=4000, dt=0.05):
    slew = Slew(LIM)
    for i in range(n):
        cmd, done = step(pose)
        if done:
            return pose, i * dt, True
        pose = integrate(pose, slew(cmd, dt), dt)
    return pose, n * dt, False


@pytest.mark.parametrize("goal", [(2.0, 0.0), (1.0, 1.0), (0.5, -0.8), (-0.6, 0.3), (0.2, 0.0)])
def test_goal_reached_without_overshoot_and_never_reversing(goal):
    pose0, tmax, ok = (0.0, 0.0, 0.0), 0.0, False
    pose, t, ok = run(pose0, lambda p: go_to_point(p, goal, LIM))
    assert ok, (goal, pose)
    assert math.hypot(goal[0] - pose[0], goal[1] - pose[1]) <= LIM.pos_tol + 0.02


def test_straight_move_holds_heading():
    pose, _, ok = run((0.0, 0.0, 0.0), lambda p: go_to_point(p, (2.0, 0.0), LIM, hold_yaw=0.0))
    assert ok and abs(pose[1]) < 0.03 and abs(wrap(pose[2])) < 0.05


def test_commands_respect_limits_and_vy_is_zero():
    pose, slew = (0.0, 0.0, 1.0), Slew(LIM)
    for _ in range(600):
        cmd, done = go_to_point(pose, (3.0, -2.0), LIM)
        out = slew(cmd, 0.05)
        assert 0.0 <= out[0] <= LIM.vx_max + 1e-9 and out[1] == 0.0 and abs(out[2]) <= LIM.wz_max + 1e-9
        pose = integrate(pose, out, 0.05)


def test_turns_in_place_first_when_the_goal_is_off_axis():
    cmd, done = go_to_point((0.0, 0.0, 0.0), (0.0, 2.0), LIM)
    assert cmd[0] == 0.0 and cmd[2] > 0 and not done


def test_goal_behind_never_reverses():
    cmd, _ = go_to_point((0.0, 0.0, 0.0), (-2.0, 0.0), LIM)
    assert cmd[0] == 0.0 and abs(cmd[2]) > 0


@pytest.mark.parametrize("goal", [math.pi / 2, -math.pi / 2, math.pi, 0.2])
def test_turn_reaches_goal(goal):
    pose, _, ok = run((0.0, 0.0, 0.0), lambda p: turn_to(p, goal, LIM))
    assert ok and abs(wrap(goal - pose[2])) <= LIM.yaw_tol + 0.01


def test_slew_limits_acceleration():
    s = Slew(LIM)
    out = s((0.2, 0.0, 0.3), 0.1)
    assert out[0] == pytest.approx(LIM.slew_v * 0.1) and out[2] == pytest.approx(LIM.slew_w * 0.1)
    assert s((0.2, 0.0, 0.0), math.nan) == (0.0, 0.0, 0.0)


def test_bad_limits_rejected():
    with pytest.raises(ValueError):
        ServoLimits(vx_max=math.nan)
    with pytest.raises(ValueError):
        ServoLimits(decel=0.0)
