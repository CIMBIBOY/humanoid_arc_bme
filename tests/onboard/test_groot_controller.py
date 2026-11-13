"""GrootPlannerController with a fake transport: wire format, S1-clamped command mapping, stop = IDLE (never stop-control),
step-deadline watchdog, disabled by default. No zmq, no robot."""

import math

import pytest

from humanoid_arc_bme.onboard.runtime.groot_controller import (HEADER_SIZE, MODE_IDLE, MODE_SLOW_WALK, FakeTransport, GrootPlannerController,
                                         build_command_message, build_planner_message, decode_message)


def make(**kw):
    tx = FakeTransport()
    kw.setdefault("enabled", True)
    return tx, GrootPlannerController(tx, threaded=False, **kw)


def planners(tx):
    return [d for t, d in tx.decoded() if t == "planner"]


def test_wire_format_layout():
    m = build_planner_message(1, (1, 0, 0), (1, 0, 0), 0.2, -1)
    assert m.startswith(b"planner")
    assert len(m) == 7 + HEADER_SIZE + 4 + 12 + 12 + 4 + 4
    t, d = decode_message(m)
    assert t == "planner" and d["mode"] == 1 and d["movement"] == (1.0, 0.0, 0.0) and d["speed"] == pytest.approx(0.2)
    c = build_command_message(False, True, True, 0.5)
    assert c.startswith(b"command") and len(c) == 7 + HEADER_SIZE + 3 + 4
    assert decode_message(c)[1] == {"start": 0, "stop": 1, "planner": 1, "delta_heading": pytest.approx(0.5)}


def test_disabled_by_default_sends_nothing():
    tx = FakeTransport()
    c = GrootPlannerController(tx, threaded=False)
    assert c.ready()[0] is False
    c.on_arm(0.0, None)
    c.step(0.01, 0.01, None, (0.1, 0, 0))
    c.on_disarm(0.02, "x")
    assert tx.sent == []


def test_arm_selects_planner_but_does_not_start_control():
    tx, c = make()
    c.on_arm(0.0, None)
    assert tx.decoded() == [("command", {"start": 0, "stop": 0, "planner": 1})]


def test_forward_command_maps_to_slow_walk_with_speed():
    tx, c = make()
    c.on_arm(0.0, None)
    c.step(0.05, 0.05, None, (0.2, 0.0, 0.0))
    d = planners(tx)[-1]
    assert d["mode"] == MODE_SLOW_WALK and d["speed"] == pytest.approx(0.2)
    assert d["movement"] == pytest.approx((1.0, 0.0, 0.0)) and d["facing"] == pytest.approx((1.0, 0.0, 0.0))


def test_zero_command_is_idle_and_turn_in_place_changes_facing():
    tx, c = make()
    c.on_arm(0.0, None)
    for i in range(1, 11):
        c.step(0.05 * i, 0.05, None, (0.0, 0.0, 0.3))
    d = planners(tx)[-1]
    assert d["mode"] == MODE_IDLE and d["movement"] == (0.0, 0.0, 0.0)
    assert math.atan2(d["facing"][1], d["facing"][0]) == pytest.approx(0.3 * 0.5, abs=1e-4)  # left turn = +yaw


def test_movement_is_rotated_into_heading_frame():
    tx, c = make()
    c.on_arm(0.0, None)
    c.yaw = math.pi / 2
    c.step(0.05, 0.0, None, (0.1, 0.0, 0.0))  # forward in the body = +y in the reference frame
    assert planners(tx)[-1]["movement"] == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)


def test_disarm_stop_emergency_send_idle_never_stop_control():
    tx, c = make()
    c.on_arm(0.0, None)
    c.step(0.05, 0.05, None, (0.2, 0, 0))
    n = len(tx.sent)
    c.on_disarm(0.1, "deadman")
    c.stop(0.2)
    c.emergency_stop(0.3)
    last = tx.decoded()[n:]
    assert len(last) == 3 and all(t == "planner" and d["mode"] == MODE_IDLE for t, d in last)
    assert not any(t == "command" and d["stop"] for t, d in tx.decoded())
    assert "not supported" in c.damp(0.4)


def test_step_deadline_sends_idle_and_latches_fault():
    tx, c = make(step_deadline_s=0.15)
    c.on_arm(0.0, None)
    c.step(0.05, 0.05, None, (0.2, 0, 0))
    c.pump(0.4)  # the control loop went silent for 0.35 s
    assert planners(tx)[-1]["mode"] == MODE_IDLE and "silent" in c.fault()
    n = len(tx.sent)
    c.pump(0.5)
    assert len(tx.sent) == n  # stopped streaming: the deploy's own 1 s timeout is the second layer


def test_stream_rate_and_nonfinite_rejected():
    tx, c = make(send_hz=20.0)
    c.on_arm(0.0, None)
    for i in range(1, 21):
        c.step(0.01 * i, 0.01, None, (0.1, 0, 0))
    assert 3 <= len(planners(tx)) <= 5  # ~20 Hz over 0.2 s
    with pytest.raises(ValueError):
        c.step(0.3, 0.01, None, (float("nan"), 0, 0))


def test_constructor_guards():
    with pytest.raises(ValueError):
        GrootPlannerController(FakeTransport(), send_hz=1.0, threaded=False)
    with pytest.raises(ValueError):
        GrootPlannerController(FakeTransport(), step_deadline_s=2.0, threaded=False)


# ---- arms and Dex3 hands through the planner topic's optional fields -------------------------------------------------
from pathlib import Path  # noqa: E402

from humanoid_arc_bme.onboard.api.contract import Imu, RobotState  # noqa: E402
from humanoid_arc_bme.onboard.api.safety import load_limits  # noqa: E402
from humanoid_arc_bme.onboard.g1_layout import LEFT_HAND_JOINTS, RIGHT_HAND_JOINTS, UPPER_BODY_JOINTS  # noqa: E402
from humanoid_arc_bme.onboard.paths import DEFAULT_LIMITS  # noqa: E402

ALL = tuple(UPPER_BODY_JOINTS) + tuple(LEFT_HAND_JOINTS) + tuple(RIGHT_HAND_JOINTS)
LIM = load_limits(DEFAULT_LIMITS)


def rstate(q=0.0, names=ALL):
    n = len(names)
    return RobotState(0.0, tuple(names), (q,) * n, (0.0,) * n, (0.0, 0.0, 0.0), Imu((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), {})


def test_optional_fields_round_trip_and_are_validated():
    m = build_planner_message(0, (0, 0, 0), (1, 0, 0), upper_body_position=[0.1] * 17, left_hand_joints=[0.0] * 7)
    t, d = decode_message(m)
    assert d["upper_body_position"] == pytest.approx((0.1,) * 17) and len(d["left_hand_joints"]) == 7 and "right_hand_joints" not in d
    for bad in ({"upper_body_position": [0.0] * 16}, {"left_hand_joints": [math.nan] * 7}, {"nonsense": [0.0]}):
        with pytest.raises(ValueError):
            build_planner_message(0, (0, 0, 0), (1, 0, 0), **bad)


def test_arm_commands_are_refused_without_s1_limits():
    tx, c = make()
    with pytest.raises(RuntimeError):
        c.set_upper_body([0.0] * 17, [0.0] * 7, [0.0] * 7)


@pytest.mark.parametrize("body,lh,rh", [([0.0] * 16, [0.0] * 7, [0.0] * 7), ([math.nan] + [0.0] * 16, [0.0] * 7, [0.0] * 7),
                                        ([0.0] * 17, [0.0] * 6, [0.0] * 7), ([0.0] * 17, [0.0] * 7, [math.inf] + [0.0] * 6)])
def test_arm_command_shape_and_finiteness(body, lh, rh):
    tx, c = make(limits=LIM)
    with pytest.raises(ValueError):
        c.set_upper_body(body, lh, rh)


def test_arm_targets_are_rate_limited_and_streamed_with_every_planner_message():
    tx, c = make(limits=LIM)
    c.on_arm(0.0, None)
    c.set_upper_body([0.5] * 17, [0.0] * 7, [0.0] * 7)
    st = rstate(0.0)
    c.step(0.05, 0.05, st, (0.0, 0.0, 0.0))
    c.step(0.10, 0.05, st, (0.0, 0.0, 0.0))
    msgs = [d for d in planners(tx) if "upper_body_position" in d]
    assert msgs, "the arm fields must ride on the planner messages"
    first = msgs[0]["upper_body_position"]
    step_max = LIM.joint_vel_limit(UPPER_BODY_JOINTS[3]) * 0.05
    assert max(first) <= step_max + 1e-6 and max(first) > 0.0        # moved, but no faster than S1 allows
    later = msgs[-1]["upper_body_position"]
    assert max(later) >= max(first)                                  # keeps walking toward the goal


def test_arm_targets_stay_inside_the_joint_range():
    tx, c = make(limits=LIM)
    c.on_arm(0.0, None)
    c.set_upper_body([50.0] * 17, [50.0] * 7, [50.0] * 7)
    st = rstate(0.0)
    for i in range(1, 400):
        c.step(i * 0.05, 0.05, st, (0.0, 0.0, 0.0))
    last = [d for d in planners(tx) if "upper_body_position" in d][-1]
    for name, v in zip(UPPER_BODY_JOINTS, last["upper_body_position"]):
        lo, hi = LIM.joint_range(name)
        assert lo - 1e-6 <= v <= hi + 1e-6, (name, v)


def test_missing_joint_in_state_refuses_to_move_the_arms():
    tx, c = make(limits=LIM)
    c.on_arm(0.0, None)
    c.set_upper_body([0.0] * 17, [0.0] * 7, [0.0] * 7)
    with pytest.raises(ValueError, match="every body joint"):
        c.step(0.05, 0.05, rstate(0.0, names=UPPER_BODY_JOINTS[:5]), (0.0, 0.0, 0.0))


def test_release_and_disarm_stop_sending_arm_fields():
    tx, c = make(limits=LIM)
    c.on_arm(0.0, None)
    c.set_upper_body([0.2] * 17, [0.0] * 7, [0.0] * 7)
    c.step(0.05, 0.05, rstate(0.0), (0.0, 0.0, 0.0))
    c.release_upper_body()
    n = len(tx.sent)
    c.step(0.10, 0.05, rstate(0.0), (0.0, 0.0, 0.0))
    assert "upper_body_position" not in planners(tx)[-1] and len(tx.sent) > n
    c.set_upper_body([0.2] * 17, [0.0] * 7, [0.0] * 7)
    c.step(0.15, 0.05, rstate(0.0), (0.0, 0.0, 0.0))
    c.on_disarm(0.2, "operator")
    assert "upper_body_position" not in planners(tx)[-1]
    assert c.status()["upper_body"] is False
