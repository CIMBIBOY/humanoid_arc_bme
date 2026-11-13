"""humanoid_arc_bme.onboard.nav: paths, limits, controllers, follower, odometry, obstacle hook, runtime hook, ROS 2 helpers.
CPU only, stdlib only (no Isaac, no ROS)."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from humanoid_arc_bme.onboard.api.contract import BaseVelocity, Imu, RobotState, WholeBodyCommand
from humanoid_arc_bme.onboard.nav import path as P
from humanoid_arc_bme.onboard.nav.controllers import CONTROLLERS, make_controller
from humanoid_arc_bme.onboard.nav.follower import FollowerConfig, PathFollower
from humanoid_arc_bme.onboard.nav.limits import CommandShaper, NavLimits, speed_profile, stop_speed
from humanoid_arc_bme.onboard.nav.obstacle import SpeedSeparationGuard, sector_min_distance
from humanoid_arc_bme.onboard.nav.odometry import DeadReckoning, ExternalPoseOdometry, FusedOdometry, TruthOdometry, yaw_from_quat
from humanoid_arc_bme.onboard.nav.runtime_hook import NavPolicy, NavSession, plan_from_spec, vendor_velocity_args
from humanoid_arc_bme.onboard.nav.sim import CONDITIONS, run_episode

# ---------------------------------------------------------------- paths


def test_line_and_arc_geometry():
    ln = P.line(2.0)
    assert ln.length == pytest.approx(2.0)
    a = P.arc(0.5, math.pi / 2)
    assert a.length == pytest.approx(0.5 * math.pi / 2, rel=1e-3)
    assert all(k == pytest.approx(2.0, rel=0.02) for k in a.kappa[2:-2])
    assert a.end[0] == pytest.approx(0.5, abs=1e-6) and a.end[1] == pytest.approx(0.5, abs=1e-6)
    r = P.arc(0.5, -math.pi / 2)
    assert r.kappa[5] < 0 and r.end[1] == pytest.approx(-0.5, abs=1e-6)


def test_clothoid_curvature_is_linear():
    c = P.clothoid(2.0, 0.0, 1.0, ds=0.01)
    mid = len(c) // 2
    assert c.kappa[mid] == pytest.approx(0.5, abs=0.02)
    assert c.heading[-1] == pytest.approx(1.0, abs=0.02)  # integral of kappa = 0.5 * 2 * 1


def test_spline_passes_through_waypoints_and_corner_rounding():
    wps = [(0, 0), (1, 0), (2, 1), (3, 1)]
    sp = P.catmull_rom(wps)
    for x, y in wps:
        assert min(math.hypot(x - a, y - b) for a, b in zip(sp.x, sp.y)) < 1e-6
    sharp = P.from_waypoints([(0, 0), (2, 0), (2, 2)])
    round_ = P.from_waypoints([(0, 0), (2, 0), (2, 2)], corner_radius=0.4)
    assert max(map(abs, sharp.kappa)) > 10
    assert max(map(abs, round_.kappa)) == pytest.approx(1 / 0.4, rel=0.15)


def test_errors_sign_and_backward_body_yaw():
    ln = P.line(2.0)
    ey, eyaw, s, _ = ln.errors(P.Pose2D(1.0, 0.1, 0.2))
    assert ey == pytest.approx(0.1) and eyaw == pytest.approx(0.2) and s == pytest.approx(1.0)
    back = P.line(2.0, P.Pose2D(0, 0, math.pi), direction=-1)
    assert back.body_yaw(0) == pytest.approx(0.0, abs=1e-9)  # walking backward toward -x faces +x
    _, eyaw, _, _ = back.errors(P.Pose2D(-1.0, 0.0, 0.0))
    assert eyaw == pytest.approx(0.0, abs=1e-9)


def test_closed_loop_projection_does_not_jump_to_start():
    c = P.arc(1.0, 2 * math.pi - 1e-3)
    i, _, s = c.nearest(c.x[5], c.y[5], start=0, window=40)
    assert s < 0.5
    i, _, s = c.nearest(c.x[-3], c.y[-3], start=len(c) - 10, window=40)
    assert s > c.length - 0.5


def test_split_at_corners():
    sharp = P.from_waypoints([(0, 0), (2, 0), (2, 2)])
    parts = P.split_at_corners(sharp)
    assert len(parts) == 2 and parts[0].end == pytest.approx((2.0, 0.0))
    assert len(P.split_at_corners(P.from_waypoints([(0, 0), (2, 0), (2, 2)], corner_radius=0.4))) == 1
    assert len(P.split_at_corners(P.arc(1.0, math.pi))) == 1


def test_path_validation():
    with pytest.raises(ValueError):
        P.Path([0.0], [0.0])
    with pytest.raises(ValueError):
        P.Path([0.0, 1.0], [0.0, 0.0], direction=2)
    with pytest.raises(ValueError):
        P.Path([0.0, 0.0], [0.0, 0.0])


# ---------------------------------------------------------------- limits


def test_speed_profile_respects_limits():
    L = NavLimits()
    p = P.concat(P.line(1.0), P.arc(0.5, math.pi / 2, P.end_pose(P.line(1.0))), )
    v = speed_profile(p, L)
    assert v[-1] == 0.0
    for i, (vi, k) in enumerate(zip(v, p.kappa)):
        assert vi <= L.v_fwd_max + 1e-9
        if abs(k) > 1e-6:
            assert vi * abs(k) <= L.wz_max + 1e-6
            assert vi * vi * abs(k) <= L.a_lat_max + 1e-6
    for i in range(1, len(v)):  # decel limit
        ds = p.s[i] - p.s[i - 1]
        assert v[i - 1] ** 2 <= v[i] ** 2 + 2 * L.d_max * ds + 1e-9
    assert max(speed_profile(P.line(2.0, P.Pose2D(0, 0, math.pi), direction=-1), L)) <= L.v_back_max + 1e-9


def test_stop_speed_with_latency():
    L = NavLimits()
    for d in (0.05, 0.3, 1.0):
        for T in (0.0, 0.3):
            v = stop_speed(d, L, T)
            assert v * T + v * v / (2 * L.d_max) == pytest.approx(d, rel=1e-9)


def test_shaper_ramps_and_keeps_curvature():
    L = NavLimits()
    sh = CommandShaper(L)
    vx, wz = sh.saturate(0.4, 0.4)  # curvature 1 kept
    assert vx == pytest.approx(0.2) and wz == pytest.approx(0.2)
    vx, wz = sh.saturate(0.1, 0.6)
    assert wz == pytest.approx(L.wz_max) and wz / vx == pytest.approx(6.0)
    sh.reset()
    out = [sh(0.2, 0.0, 0.05)[0] for _ in range(40)]
    assert out[0] == pytest.approx(L.a_max * 0.05)
    assert all(b - a <= L.a_max * 0.05 + 1e-12 for a, b in zip([0.0] + out, out))
    assert out[-1] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        NavLimits(v_fwd_max=-1)


def test_from_safety_limits_are_not_looser():
    lim = SimpleNamespace(base_vel_max=(0.2, 0.1, 0.3))
    n = NavLimits.from_safety(lim)
    assert n.v_fwd_max == 0.2 and n.wz_max == 0.3 and n.v_back_max <= 0.2


# ---------------------------------------------------------------- controllers on the harness


@pytest.mark.parametrize("ctrl", sorted(CONTROLLERS))
@pytest.mark.parametrize("path_name", ["straight", "backward", "s_curve", "circle", "figure_eight", "sharp_corner",
                                       "out_and_back"])
def test_every_controller_completes_every_path(ctrl, path_name):
    r = run_episode(P.standard_paths()[path_name], ctrl, CONDITIONS["nominal"])
    assert r["success"], r
    assert r["cte_max"] < 0.15
    assert r["final_err"] < NavLimits().goal_pos_tol


def test_backward_path_commands_negative_vx():
    plan = P.standard_paths()["backward"]
    for name in CONTROLLERS:
        r = run_episode(plan, name, CONDITIONS["nominal"], trace=True)
        vxs = [row[4] for row in r["trace"] if abs(row[4]) > 1e-3]
        assert vxs and all(v < 0 for v in vxs), name
        assert max(-v for v in vxs) <= NavLimits().v_back_max + 1e-9


def test_laggy_plant_with_deadband_still_completes_the_corner():
    for name in CONTROLLERS:
        r = run_episode(P.standard_paths()["sharp_corner"], name, CONDITIONS["laggy"])
        assert r["success"], (name, r["reason"])


def test_unknown_controller():
    with pytest.raises(ValueError):
        make_controller("teb")
    assert make_controller("rpp").name == "regulated_pure_pursuit"


# ---------------------------------------------------------------- follower


class _Kinematic:
    """Perfect unicycle: pose follows the command exactly (for state-machine tests)."""

    def __init__(self):
        self.p = P.Pose2D(0, 0, 0)

    def step(self, cmd, dt):
        th = self.p.yaw + cmd.wz * dt
        self.p = P.Pose2D(self.p.x + cmd.vx * math.cos(th) * dt, self.p.y + cmd.vx * math.sin(th) * dt, P.wrap(th))


def _run(fol, robot, seconds, dt=0.05, t0=0.0):
    t = t0
    states = []
    while t < t0 + seconds:
        cmd = fol.step(t)
        robot.step(cmd, dt)
        states.append((t, fol.state, cmd))
        t += dt
        if fol.done:
            break
    return states


def test_settle_before_success_and_goal_yaw():
    robot = _Kinematic()
    L = NavLimits()
    fol = PathFollower(TruthOdometry(lambda: robot.p), "rpp", L)
    fol.follow(P.Path([0, 0.5, 1.0], [0, 0, 0], goal_yaw=math.pi / 2), 0.0)
    states = _run(fol, robot, 60)
    names = [s for _, s, _ in states]
    assert "SETTLE" in names and "FINAL_ALIGN" in names and names[-1] == "SUCCEEDED"
    t_settle = next(t for t, s, _ in states if s == "SETTLE")
    t_done = next(t for t, s, _ in states if s == "SUCCEEDED")
    assert t_done - t_settle >= L.settle_s
    assert abs(P.wrap(robot.p.yaw - math.pi / 2)) <= L.goal_yaw_tol
    assert fol.status().state == "succeeded"


def test_align_turns_in_place_first():
    robot = _Kinematic()
    robot.p = P.Pose2D(0, 0, math.pi / 2)  # facing +y, path along +x
    fol = PathFollower(TruthOdometry(lambda: robot.p), "pp", NavLimits())
    fol.follow(P.line(1.0), 0.0)
    states = _run(fol, robot, 40)
    align = [c for _, s, c in states if s == "ALIGN"]
    assert align and all(abs(c.vx) < 1e-9 for c in align)
    assert states[-1][1] == "SUCCEEDED"


def test_no_pose_fails_and_commands_zero():
    odo = ExternalPoseOdometry(stale_s=0.3)
    fol = PathFollower(odo, "rpp", NavLimits())
    fol.follow(P.line(1.0), 0.0)
    cmd = fol.step(0.1)
    assert fol.state == "FAILED" and fol.reason == "no_pose" and cmd.vx == 0.0 and cmd.wz == 0.0
    assert fol.status().reason == "no_pose"


def test_operator_stop_ramps_to_zero():
    robot = _Kinematic()
    fol = PathFollower(TruthOdometry(lambda: robot.p), "rpp", NavLimits())
    fol.follow(P.line(3.0), 0.0)
    _run(fol, robot, 5.0)
    v_before = fol.shaper.vx
    assert v_before > 0.1
    fol.stop()
    cmd = fol.step(5.05)
    assert 0 < cmd.vx < v_before  # ramped, not a step
    _run(fol, robot, 5.0, t0=5.1)
    assert fol.done and fol.status().reason == "stopped"


def test_obstacle_guard_stops_then_fails_blocked():
    robot = _Kinematic()
    dist = {"d": 5.0}
    guard = SpeedSeparationGuard(lambda direction: dist["d"], a_stop=0.3, d_margin=0.3)
    fol = PathFollower(TruthOdometry(lambda: robot.p), "rpp", NavLimits(), guard, FollowerConfig(blocked_timeout_s=2.0))
    fol.follow(P.line(5.0), 0.0)
    _run(fol, robot, 4.0)
    dist["d"] = 0.25  # inside the margin
    _run(fol, robot, 10.0, t0=4.0)
    assert fol.state == "FAILED" and fol.reason == "blocked"
    assert abs(fol.shaper.vx) < 1e-9


def test_speed_separation_rule_and_fail_closed():
    g = SpeedSeparationGuard(lambda d: None, a_stop=0.3, d_margin=0.3, require_sensor=True)
    assert g.speed_cap(0.0, 1) == 0.0  # no reading ever: stop
    g2 = SpeedSeparationGuard(lambda d: 1.0, a_stop=0.3, d_margin=0.3)
    assert g2.speed_cap(0.0, 1) == pytest.approx(math.sqrt(2 * 0.3 * 0.7))
    assert g2.cap_for(0.3) == 0.0
    assert sector_min_distance([(1.0, 0.0), (0.5, 0.6), (-0.4, 0.0), (0.05, 0.0)], 1) == 1.0
    assert sector_min_distance([(1.0, 0.0), (-0.4, 0.0)], -1) == pytest.approx(0.4)
    assert sector_min_distance([], 1) == math.inf


@pytest.mark.parametrize("kw", [dict(a_stop=math.nan), dict(a_stop=math.inf), dict(a_stop=0.0), dict(a_stop=-1.0),
                                dict(d_margin=math.nan), dict(d_margin=math.inf), dict(d_margin=-0.1),
                                dict(v_nominal=math.nan), dict(v_nominal=0.0), dict(v_nominal=-1.0),
                                dict(stale_s=math.nan), dict(stale_s=math.inf), dict(stale_s=0.0), dict(stale_s=-1.0),
                                dict(a_stop="x"), dict(a_stop=True)])
def test_guard_rejects_bad_constructor_knobs(kw):
    with pytest.raises(ValueError):
        SpeedSeparationGuard(lambda d: 1.0, **kw)


def test_guard_accepts_inf_v_nominal_and_numpy_floats():
    SpeedSeparationGuard(lambda d: 1.0, a_stop=np.float32(0.3), d_margin=0, v_nominal=math.inf)


def test_guard_direction_without_sensor_stops_immediately_even_after_forward_reading():
    dist = {1: 5.0, -1: None}
    g = SpeedSeparationGuard(lambda d: dist[d], a_stop=0.3, d_margin=0.3, stale_s=0.3)
    assert g.speed_cap(0.0, 1) > 0
    assert g.speed_cap(0.0, -1) == 0.0          # rear never seen: 0 at once, not after stale_s
    assert g.speed_cap(0.01, -1) == 0.0
    assert g.speed_cap(0.02, 1) > 0             # forward unaffected


def test_guard_none_after_valid_holds_last_cap_then_zero():
    dist = {"d": 1.0}
    g = SpeedSeparationGuard(lambda d: dist["d"], a_stop=0.3, d_margin=0.3, stale_s=0.3)
    c = g.speed_cap(0.0, 1)
    dist["d"] = None
    assert g.speed_cap(0.2, 1) == c             # never raised to v_nominal (inf)
    assert g.speed_cap(0.31, 1) == 0.0          # stale window over: fail closed
    dist["d"] = 1.0
    assert g.speed_cap(0.4, 1) == c             # recovers on a valid reading


def test_guard_require_sensor_false_passes_none_but_not_invalid_numbers():
    g = SpeedSeparationGuard(lambda d: None, require_sensor=False, v_nominal=0.5)
    assert g.speed_cap(0.0, 1) == 0.5
    for bad in (math.nan, -0.1, -math.inf):
        assert SpeedSeparationGuard(lambda d, b=bad: b, require_sensor=False).speed_cap(0.0, 1) == 0.0


@pytest.mark.parametrize("bad", [math.nan, -0.1, -math.inf, -1e-9, "x", True])
def test_guard_nan_or_negative_distance_is_zero_cap(bad):
    g = SpeedSeparationGuard(lambda d: bad, a_stop=0.3, d_margin=0.3)
    assert g.speed_cap(0.0, 1) == 0.0
    ok = {"d": 1.0}
    g = SpeedSeparationGuard(lambda d: ok["d"], a_stop=0.3, d_margin=0.3)
    assert g.speed_cap(0.0, 1) > 0
    ok["d"] = bad
    assert g.speed_cap(0.01, 1) == 0.0          # invalid evidence stops at once, no stale window


def test_guard_inf_distance_is_clear_and_refreshes_validity():
    d = {"d": math.inf}
    g = SpeedSeparationGuard(lambda x: d["d"], v_nominal=0.4, stale_s=0.3)
    assert g.speed_cap(0.0, 1) == 0.4
    d["d"] = None
    assert g.speed_cap(0.2, 1) == 0.4 and g.speed_cap(0.5, 1) == 0.0


def test_guard_alone_stops_reverse_without_rear_sensor():
    """The runner also blocks reverse (`reverse_unsensed`); the guard must be safe on its own."""
    g = SpeedSeparationGuard(lambda d: 3.0 if d > 0 else None, a_stop=0.3, d_margin=0.3)
    for k in range(10):
        assert g.speed_cap(k * 0.05, 1) > 0
        assert g.speed_cap(k * 0.05, -1) == 0.0


def test_whole_body_command_output():
    cmd = PathFollower.to_whole_body(1.0, BaseVelocity(0.1, 0.0, 0.05), 0.72)
    assert isinstance(cmd, WholeBodyCommand) and cmd.velocity_height.vx == 0.1 and cmd.velocity_height.height == 0.72


# ---------------------------------------------------------------- odometry


def test_dead_reckoning_and_imu_yaw():
    dr = DeadReckoning("command", max_dt=2.0)
    dr.on_command(0.0, 0.2, 0.0, 0.0)
    assert dr.pose(1.0).x == pytest.approx(0.2)
    dr.set_yaw(1.0)  # first IMU reading defines the offset: yaw stays 0
    assert dr.pose(1.0).yaw == pytest.approx(0.0)
    dr.set_yaw(1.5)
    assert dr.pose(1.0).yaw == pytest.approx(0.5)
    dr2 = DeadReckoning("command", max_dt=0.2)
    dr2.on_command(0.0, 0.2, 0.0, 0.0)
    assert dr2.pose(10.0).x == pytest.approx(0.04)  # a long gap is not integrated


def test_external_and_fused_odometry():
    ex = ExternalPoseOdometry(stale_s=0.5, extrapolate_s=0.2)
    assert ex.pose(0.0) is None
    ex.set_pose(0.0, P.Pose2D(1, 2, 0))
    ex.on_command(0.0, 0.1, 0, 0)
    assert ex.pose(0.1).x == pytest.approx(1.01)
    assert ex.pose(0.6) is None
    fu = FusedOdometry(DeadReckoning())
    fu.on_command(0.0, 0.2, 0, 0)
    fu.fix(1.0, P.Pose2D(5.0, 0.0, 0.3))
    p = fu.pose(1.0)
    assert (p.x, p.yaw) == (pytest.approx(5.0), pytest.approx(0.3))
    assert yaw_from_quat((math.cos(0.25), 0, 0, math.sin(0.25))) == pytest.approx(0.5)


# ---------------------------------------------------------------- runtime hook


def _state(stamp, yaw=0.0):
    q = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    return RobotState(stamp, ("j",), (0.0,), (0.0,), (0.0, 0.0, 0.0), Imu(q, (0, 0, 0), (0, 0, 9.81)), {})


def test_nav_session_json_round_trip_and_adapters():
    nav = NavSession("rpp")
    summary = nav.load({"waypoints": [[0, 0], [1, 0], [1, 1]], "smoothing": "corner", "corner_radius": 0.3})
    assert summary["segments"][0]["length_m"] > 1.5
    nav.start(0.0)
    cmd = nav.tick(0.05, _state(0.05))
    assert isinstance(cmd, BaseVelocity) and cmd.vy == 0.0
    st = nav.status()
    json.dumps(st)  # JSON friendly
    assert st["skill"] == "running" and st["controller"] == "regulated_pure_pursuit"
    src = nav.command_source(0.72)
    assert src() == (cmd.vx, cmd.vy, cmd.wz, 0.72)
    base, arms = NavPolicy(nav, lambda: 0.1).act(_state(0.1), {})
    assert isinstance(base, BaseVelocity) and arms == []
    assert vendor_velocity_args(BaseVelocity(0.1, 0, 0.2), 0.05) == (0.1, 0, 0.2, pytest.approx(0.15))
    nav.stop()
    assert nav.status()["skill"] == "failed" and nav.status()["reason"] == "stopped"


def test_nav_session_relative_frame_and_external_pose():
    nav = NavSession("pp")
    nav.reset_odometry(1.0, 1.0, math.pi / 2)
    nav.load({"waypoints": [[0, 0], [1, 0]], "frame": "start", "smoothing": "none"})
    nav.start(0.0)
    seg = nav.follower.segment
    assert seg.end == (pytest.approx(1.0), pytest.approx(2.0))  # 1 m ahead of a robot facing +y
    nav.external_pose(0.1, 1.0, 1.5, math.pi / 2)
    assert nav.odom.pose(0.1).y == pytest.approx(1.5) and nav.status()["external_fixes"] == 1


def test_plan_spec_validation():
    with pytest.raises(ValueError):
        plan_from_spec({"waypoints": [[0, 0]]})
    with pytest.raises(ValueError):
        plan_from_spec({"waypoints": [[0, 0], [1, 0]], "smoothing": "bezier"})
    plan = plan_from_spec({"segments": [{"waypoints": [[0, 0], [1, 0]]},
                                        {"waypoints": [[1, 0], [0, 0]], "direction": -1}]})
    assert [p.direction for p in plan.segments] == [1, -1]


# ---------------------------------------------------------------- ROS 2 helpers (no rclpy needed)


def _pose_stamped(x, y, yaw):
    return SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y, z=0.0),
                                                orientation=SimpleNamespace(w=math.cos(yaw / 2), x=0.0, y=0.0,
                                                                            z=math.sin(yaw / 2))))


def test_ros2_helpers_without_rclpy():
    from humanoid_arc_bme.onboard.nav import ros2_node as R

    msg = SimpleNamespace(poses=[_pose_stamped(0.05 * i, 0.0, 0.0) for i in range(21)])
    plan = R.path_from_msg(msg, direction=0)
    assert plan.segments[0].direction == 1 and plan.segments[0].length == pytest.approx(1.0)
    back = SimpleNamespace(poses=[_pose_stamped(-0.05 * i, 0.0, 0.0) for i in range(21)])
    assert R.path_from_msg(back, direction=0).segments[0].direction == -1
    odom = SimpleNamespace(pose=_pose_stamped(1.0, 2.0, 0.5))
    p = R.pose_from_odom(odom)
    assert (p.x, p.y, p.yaw) == (1.0, 2.0, pytest.approx(0.5))
    assert R.twist_fields(BaseVelocity(0.1, 0.0, 0.2)) == {"linear": (0.1, 0.0, 0.0), "angular": (0.0, 0.0, 0.2)}


def test_ros2_main_needs_rclpy():
    pytest.importorskip("rclpy")


def test_open_loop_driver_ramps_deadman_and_script():
    from humanoid_arc_bme.onboard.nav.runtime_hook import OpenLoopDriver

    L = NavLimits()
    d = OpenLoopDriver(L, deadman_s=0.5)
    d.set(0.0, 0.5, 1.0)  # above the limits: saturated keeping curvature
    out = [d.tick(0.05 * k) for k in range(1, 9)]  # 0.05 .. 0.4 s, inside the dead-man window
    assert all(abs(c.vx) <= L.v_fwd_max + 1e-9 and abs(c.wz) <= L.wz_max + 1e-9 for c in out)
    assert out[0].vx == 0.0  # first tick has dt = 0
    assert out[1].vx == pytest.approx(L.a_max * 0.05)
    c = d.tick(2.0)  # dead man expired at 0.5 s: ramping down
    assert c.vx < out[-1].vx
    total = d.play(3.0, [(0.1, 0.0, 1.0), (-0.05, 0.0, 1.0)])
    assert total == 2.0
    assert d._target(3.5) == (0.1, 0.0) and d._target(4.5) == (-0.05, 0.0) and d._target(5.5) == (0.0, 0.0)
    with pytest.raises(ValueError):
        d.play(0.0, [(0.1, 0.0, 0.0)])


# ---- guard hardening (audit): finite hold, monotonic clock, safe huge ints ----------------------------------------
def _hold_guard(**kw):
    box = {"v": 0.5}
    return SpeedSeparationGuard(lambda d: box["v"], **kw), box


def test_guard_hold_is_never_infinite():
    g, box = _hold_guard(hold_fallback=0.1)
    box["v"] = math.inf
    assert g.speed_cap(0.0, 1) == math.inf
    box["v"] = None
    cap = g.speed_cap(0.1, 1)
    assert math.isfinite(cap) and cap == pytest.approx(0.1)
    assert g.speed_cap(0.5, 1) == 0.0              # and expires after stale_s


@pytest.mark.parametrize("t_next", [-1e9, -math.inf, 999.0, math.nan, math.inf, 0.0 - 1e-6])
def test_guard_hold_needs_forward_monotonic_clock(t_next):
    g, box = _hold_guard()
    g.speed_cap(1000.0 if t_next == 999.0 else 0.0, 1)
    box["v"] = None
    assert g.speed_cap(t_next, 1) == 0.0           # backwards / non-finite clock never holds a cap


def test_guard_hold_forward_clock_within_stale_still_holds():
    g, box = _hold_guard()
    first = g.speed_cap(0.0, 1)
    box["v"] = None
    assert 0.0 < g.speed_cap(0.1, 1) <= first


def test_guard_huge_int_distance_does_not_raise():
    g, box = _hold_guard()
    box["v"] = 10 ** 400
    assert g.speed_cap(0.0, 1) == math.inf         # huge positive = nothing in range
    box["v"] = -(10 ** 400)
    assert g.speed_cap(0.1, 1) == 0.0              # huge negative = invalid


def test_guard_hold_fallback_validated():
    for bad in (math.nan, -1.0, math.inf, True):
        with pytest.raises(ValueError):
            SpeedSeparationGuard(lambda d: None, hold_fallback=bad)
