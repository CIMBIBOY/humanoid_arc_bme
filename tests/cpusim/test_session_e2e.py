"""The whole onboard stack in a closed loop against the CPU world (see humanoid_arc_bme/cpusim): text command -> perception ->
task executor -> pipeline runner (down-camera guard, close-range mode) -> GR00T controller wire format -> arm IK -> kinematic robot.
A KINEMATIC check of the logic and wiring; grasp physics belongs to Isaac Lab."""

import math

import pytest

from humanoid_arc_bme.cpusim.session import make_session
from humanoid_arc_bme.cpusim.world import Box, Cylinder, World


def table_world(red_xy=(1.04, 0.04), blue_xy=(1.07, -0.15)):
    return World(boxes=[Box(0.95, 1.6, -0.5, 0.5, 0.72)],
                 cylinders=[Cylinder(*red_xy, 0.04, 0.12, (200, 30, 30), "cup", base_z=0.72),
                            Cylinder(*blue_xy, 0.04, 0.12, (30, 60, 200), "cup", base_z=0.72)])


def run_command(s, text, seconds=120):
    st = s.say(text)
    if st.state == "running":
        s.run(seconds, until=lambda s: not s.stack.executor.busy())
    return s.stack.executor.status if st.state == "running" else st


@pytest.fixture
def floor_session():
    s = make_session(World())
    s.run(1.0)
    return s


def test_go_forward_reaches_the_goal(floor_session):
    s = floor_session
    st = run_command(s, "go forward 0.5 meter")
    assert st.state == "succeeded", st
    assert abs(s.robot.x - 0.5) < 0.08 and abs(s.robot.y) < 0.05


def test_turn_left_ninety_degrees(floor_session):
    s = floor_session
    st = run_command(s, "turn left 90 degrees")
    assert st.state == "succeeded", st
    assert abs(math.remainder(s.robot.yaw - math.pi / 2, 2 * math.pi)) < 0.12


def test_lateral_move_turns_toward_the_side_and_never_reverses(floor_session):
    s = floor_session
    st = run_command(s, "move left 0.4 meter")
    assert st.state == "succeeded", st
    assert s.robot.y > 0.3 and all(row[4][0] >= -1e-9 for row in s.log)      # row[4] = (vx, vy, wz) after the runner


def test_backward_is_refused_with_a_reason(floor_session):
    st = floor_session.say("go back 1 meter")
    assert st.state == "failed" and "rear sensor" in st.message


def test_stop_cancels_a_running_motion_and_zeroes_the_base(floor_session):
    s = floor_session
    assert s.say("go forward 2 meters").state == "running"
    s.run(3.0)
    assert s.robot.x > 0.05
    s.say("stop")
    out = s.tick()
    assert not s.stack.executor.busy() and out.base == (0.0, 0.0, 0.0)
    s.run(2.0)
    assert s.robot.speed() < 0.02


def test_a_second_command_while_busy_is_refused(floor_session):
    s = floor_session
    s.say("go forward 1 meter")
    st = s.say("turn left")
    assert st.state == "failed" and "busy" in st.message


def test_status_describes_the_scene():
    s = make_session(table_world())
    s.run(1.0)
    s.say("what do you see")
    s.tick()
    assert "red cup" in s.out.info and "blue cup" in s.out.info


@pytest.mark.parametrize("text,needle", [("pick up the green cup", "do not see"), ("pick up the cup", "say which colour"),
                                         ("pick up the bottle", "do not see")])
def test_ambiguous_or_missing_targets_are_refused_before_moving(text, needle):
    s = make_session(table_world())
    s.run(1.0)
    st = s.say(text)
    assert st.state == "failed" and needle in st.message, st
    s.run(1.0)
    assert s.robot.speed() < 0.01


def test_pick_and_place_cycle():
    w = table_world()
    s = make_session(w)
    s.run(1.0)
    red = w.cylinders[0]
    st = run_command(s, "pick up the red cup")
    assert st.state == "succeeded" and "red cup" in st.message, (st, s.stack.executor.last_error)
    assert s.robot.held is red and red.base_z > 0.72 + 0.03                    # lifted off the table
    blue_before = (w.cylinders[1].x, w.cylinders[1].y)
    st = run_command(s, "put it down on the left")
    assert st.state == "succeeded", (st, s.stack.executor.last_error)
    assert s.robot.held is None and abs(red.base_z - 0.72) < 0.02             # released onto the table
    assert red.y > 0.08 and (w.cylinders[1].x, w.cylinders[1].y) == blue_before
    assert s.stack.executor.arm.holding is False
    # a second pick of the other cup works from where the robot stands (or is refused with a reason, never a crash)
    st = s.say("pick up the blue cup")
    assert st.state in ("running", "failed")


def test_pick_is_refused_for_an_object_out_of_reach_height():
    w = World(boxes=[Box(0.95, 1.6, -0.5, 0.5, 0.95)],
              cylinders=[Cylinder(1.04, 0.04, 0.04, 0.12, (200, 30, 30), "cup", base_z=0.95)])
    s = make_session(w)
    s.robot.x = 0.5                                          # close enough for the high camera to see over the table edge
    s.run(1.0)
    st = s.say("pick up the red cup")
    assert st.state == "failed" and "out of reach" in st.message


def test_a_lost_pose_fails_the_command_and_stops_the_base(floor_session):
    s = floor_session
    s.say("go forward 2 meters")
    s.run(1.0)
    s.stack._xyyaw = lambda f: None                        # the localizer disappears
    s.run(4.0)
    assert s.stack.executor.status.state == "failed" and "pose" in s.stack.executor.status.message
    assert s.robot.speed() < 0.03


def test_arm_tasks_are_refused_when_hand_feedback_is_missing():
    s = make_session(table_world())
    s.run(1.0)
    orig = s._frame
    s._frame = lambda: __import__("dataclasses").replace(orig(), joints=None)
    st = s.say("pick up the red cup")
    assert st.state == "failed" and "joint feedback" in st.message
