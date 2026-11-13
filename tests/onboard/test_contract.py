"""The robot-facing contract (api/contract.py): construction, round trips, rejection of bad shapes. No Isaac Sim."""

import json
import subprocess
import sys

import pytest

from humanoid_arc_bme.onboard.api.contract import (
    ArmCommand,
    BaseVelocity,
    HandCommand,
    Imu,
    RobotState,
    SkillStatus,
)

IDENT = (1.0, 0.0, 0.0, 0.0)
POSE = (0.2, 0.3, 0.9) + IDENT


def make_state(**over):
    kw = dict(
        stamp=12.5,
        joint_names=("a", "b"),
        joint_pos=(0.1, 0.2),
        joint_vel=(0.0, 0.0),
        base_vel=(0.0, 0.0, 0.0),
        imu=Imu(quat=IDENT, gyro=(0, 0, 0), accel=(0, 0, 9.81)),
        wrist_poses={"left": POSE, "right": POSE},
    )
    kw.update(over)
    return RobotState(**kw)


GOOD = [
    ArmCommand("left", POSE, "pelvis"),
    HandCommand("right", 0.5),
    BaseVelocity(0.2, 0.0, -0.1),
    SkillStatus("running"),
    SkillStatus("failed", "unknown_object"),
    Imu(IDENT, (0, 0, 0), (0, 0, 9.81)),
    make_state(),
]


@pytest.mark.parametrize("msg", GOOD, ids=lambda m: type(m).__name__)
def test_round_trip_dict_and_json(msg):
    d = msg.to_dict()
    assert type(msg).from_dict(d) == msg
    # plain JSON types only
    assert type(msg).from_dict(json.loads(json.dumps(d))) == msg


BAD = [
    ("arm side", lambda: ArmCommand("middle", POSE, "pelvis")),
    ("arm pose length", lambda: ArmCommand("left", POSE[:6], "pelvis")),
    ("arm quaternion not unit", lambda: ArmCommand("left", (0, 0, 0, 2, 0, 0, 0), "pelvis")),
    ("arm nan", lambda: ArmCommand("left", (float("nan"),) + POSE[1:], "pelvis")),
    ("arm empty frame", lambda: ArmCommand("left", POSE, "")),
    ("hand above 1", lambda: HandCommand("left", 1.01)),
    ("hand below 0", lambda: HandCommand("left", -0.01)),
    ("hand not a number", lambda: HandCommand("left", "closed")),
    ("base inf", lambda: BaseVelocity(float("inf"), 0, 0)),
    ("skill state", lambda: SkillStatus("done")),
    ("failed without reason", lambda: SkillStatus("failed")),
    ("imu accel length", lambda: Imu(IDENT, (0, 0, 0), (0, 9.81))),
    ("state joint length", lambda: make_state(joint_vel=(0.0,))),
    ("state duplicate names", lambda: make_state(joint_names=("a", "a"))),
    ("state base_vel length", lambda: make_state(base_vel=(0.0, 0.0))),
    ("state wrist side", lambda: make_state(wrist_poses={"top": POSE})),
    ("state wrist pose length", lambda: make_state(wrist_poses={"left": POSE[:5]})),
]


@pytest.mark.parametrize("name,build", BAD, ids=[b[0] for b in BAD])
def test_rejects_bad_input(name, build):
    with pytest.raises(ValueError):
        build()


def test_from_dict_rejects_missing_and_unknown_keys():
    d = BaseVelocity(0.1, 0.0, 0.0).to_dict()
    with pytest.raises(ValueError):
        BaseVelocity.from_dict({k: v for k, v in d.items() if k != "wz"})
    with pytest.raises(ValueError):
        BaseVelocity.from_dict({**d, "extra": 1})
    with pytest.raises(ValueError):
        BaseVelocity.from_dict([0.1, 0.0, 0.0])


def test_messages_are_immutable():
    cmd = BaseVelocity(0.1, 0.0, 0.0)
    with pytest.raises(Exception):
        cmd.vx = 1.0


def test_imports_without_heavy_dependencies():
    """The contract must import on the robot and in sim alike: no isaaclab, torch, rclpy, numpy."""
    code = (
        "import sys, humanoid_arc_bme.onboard.api.contract;"
        "bad = {'isaaclab', 'torch', 'rclpy', 'numpy', 'omni'} & set(sys.modules);"
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# ---------------------------------------------------------------- schema v2 (2026-10-23)

from humanoid_arc_bme.onboard.api.contract import (  # noqa: E402
    SCHEMA_VERSION,
    ActionChunk,
    BaseVelocityHeight,
    CameraFrame,
    JointTargets,
    TaskSpec,
    WholeBodyCommand,
)

V2_GOOD = [
    HandCommand("left", joint_targets=(0.0,) * 7),
    make_state(base_vel_source="sim_truth", foot_contacts=(True, False)),
    CameraFrame("head", 1.0, 2, 2, "rgb8", bytes(range(12)), K=(1, 0, 1, 0, 1, 1, 0, 0, 1), T_pelvis_camera=POSE),
    CameraFrame("depth", 1.0, 2, 1, "depth16", b"\x01\x02\x03\x04"),
    TaskSpec("put the red box on the left plate", "pick_red_left", goal_name="plate_left"),
    TaskSpec("go there", "nav", goal_pose=POSE, frame="world"),
    JointTargets(("a", "b"), (0.1, -0.2)),
    BaseVelocityHeight(0.3, 0.0, 0.1, 0.72),
    WholeBodyCommand(2.0, velocity_height=BaseVelocityHeight(0.3, 0.0, 0.1, 0.72)),
    WholeBodyCommand(2.0, wrists=(ArmCommand("left", POSE, "pelvis"), ArmCommand("right", POSE, "pelvis"))),
    WholeBodyCommand(2.0, joints=JointTargets(("a",), (0.0,))),
    ActionChunk(3.0, 0.02, "abs_pelvis", ("x", "y"), ((0.0, 1.0), (0.1, 1.1), (0.2, 1.2)), execute_horizon=2),
]


@pytest.mark.parametrize("msg", V2_GOOD, ids=lambda m: type(m).__name__)
def test_v2_round_trip_dict_and_json(msg):
    d = msg.to_dict()
    assert type(msg).from_dict(json.loads(json.dumps(d))) == msg


V2_BAD = [
    ("hand both", lambda: HandCommand("left", 0.5, (0.0,) * 7)),
    ("hand neither", lambda: HandCommand("left")),
    ("hand 6 joints", lambda: HandCommand("left", joint_targets=(0.0,) * 6)),
    ("state source", lambda: make_state(base_vel_source="guess")),
    ("state contacts", lambda: make_state(foot_contacts=(1, 0))),
    ("camera size", lambda: CameraFrame("h", 0.0, 4, 4, "rgb8", b"\x00" * 10)),
    ("camera encoding", lambda: CameraFrame("h", 0.0, 1, 1, "rgba", b"\x00" * 4)),
    ("camera K", lambda: CameraFrame("h", 0.0, 1, 1, "mono8", b"\x00", K=(1, 0, 0))),
    ("task no goal", lambda: TaskSpec("do it", "t")),
    ("task empty text", lambda: TaskSpec("", "t", goal_name="g")),
    ("joints length", lambda: JointTargets(("a", "b"), (0.0,))),
    ("joints dup", lambda: JointTargets(("a", "a"), (0.0, 0.0))),
    ("height", lambda: BaseVelocityHeight(0, 0, 0, 2.0)),
    ("wbc two", lambda: WholeBodyCommand(0.0, velocity_height=BaseVelocityHeight(0, 0, 0, 0.7), joints=JointTargets(("a",), (0.0,)))),
    ("wbc none", lambda: WholeBodyCommand(0.0)),
    ("wbc same side twice", lambda: WholeBodyCommand(0.0, wrists=(ArmCommand("left", POSE, "pelvis"),) * 2)),
    ("chunk row length", lambda: ActionChunk(0.0, 0.02, "abs_pelvis", ("x", "y"), ((0.0,),), 1)),
    ("chunk horizon", lambda: ActionChunk(0.0, 0.02, "abs_pelvis", ("x",), ((0.0,),), 2)),
    ("chunk repr", lambda: ActionChunk(0.0, 0.02, "torque", ("x",), ((0.0,),), 1)),
    ("chunk dt", lambda: ActionChunk(0.0, 0.0, "abs_pelvis", ("x",), ((0.0,),), 1)),
]


@pytest.mark.parametrize("name,build", V2_BAD, ids=[b[0] for b in V2_BAD])
def test_v2_rejects_bad_input(name, build):
    with pytest.raises(ValueError):
        build()


def test_v1_dicts_still_load():
    """A v1 RobotState / HandCommand dict (no v2 keys) loads and means the same as before."""
    d = make_state().to_dict()
    for k in ("base_vel_source", "foot_contacts"):
        d.pop(k)
    st = RobotState.from_dict(d)
    assert st == make_state() and st.base_vel_source == "none"
    assert HandCommand.from_dict({"side": "left", "closure": 0.3}) == HandCommand("left", 0.3)
    assert SCHEMA_VERSION == 2


def test_chunk_k_and_camera_base64():
    c = ActionChunk(0.0, 0.02, "joint_pos", ("a",), tuple((float(i),) for i in range(50)), 25)
    assert c.K == 50
    f = CameraFrame("h", 0.0, 1, 1, "rgb8", b"\x01\x02\x03")
    assert isinstance(f.to_dict()["data"], str) and CameraFrame.from_dict(f.to_dict()).data == b"\x01\x02\x03"
