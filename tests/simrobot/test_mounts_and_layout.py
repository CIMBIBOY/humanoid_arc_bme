"""Pure-Python tests of the simulated robot's data: motor table, sensor mounts, frame conversions, deploy stand-in input logic."""

import math

import numpy as np
import pytest

from humanoid_arc_bme.isaac import g1_model, lowlevel
from humanoid_arc_bme.onboard import g1_layout as L
from humanoid_arc_bme.onboard.pipeline.interfaces import DepthFrame
from humanoid_arc_bme.onboard.pipeline.perception.sensing import depth_to_points
from humanoid_arc_bme.onboard.sensors import mounts as MT


def test_motor_table_is_the_unitree_order_and_matches_the_planner_fields():
    assert L.MOTOR_JOINTS[0] == "left_hip_pitch_joint" and L.MOTOR_JOINTS[5] == "left_ankle_roll_joint" and L.MOTOR_JOINTS[6] == "right_hip_pitch_joint"
    assert L.MOTOR_JOINTS[12:15] == ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
    assert L.MOTOR_JOINTS[15] == "left_shoulder_pitch_joint" and L.MOTOR_JOINTS[21] == "left_wrist_yaw_joint" and L.MOTOR_JOINTS[28] == "right_wrist_yaw_joint"
    assert len(set(L.MOTOR_JOINTS)) == 29 and L.MOTOR_JOINTS[L.UPPER_BODY_MOTOR_OFFSET:] == L.UPPER_BODY_JOINTS
    assert [n.split("_hand_")[1] for n in L.LEFT_HAND_JOINTS] == ["thumb_0_joint", "thumb_1_joint", "thumb_2_joint", "middle_0_joint", "middle_1_joint",
                                                                 "index_0_joint", "index_1_joint"]


def test_motor_model_covers_every_joint():
    kp, kd = g1_model.gains()
    assert len(kp) == len(kd) == 29 and min(kp) > 0 and kp[3] == 200.0 and kp[4] == 20.0
    assert len(g1_model.gains(g1_model.HAND_JOINTS)[0]) == 14 and len(g1_model.armature(L.MOTOR_JOINTS)) == 29
    q0 = g1_model.default_pose()
    assert q0[L.MOTOR_JOINTS.index("left_knee_joint")] == 0.30 and q0[L.MOTOR_JOINTS.index("right_hip_pitch_joint")] == -0.10 and sum(map(abs, q0[12:])) == 0


def test_mounts_file_loads_and_feeds_the_stack_config():
    m = MT.load_mounts()
    assert m.frame("d435")["tag"] == "ASSET" and m.frame("head_rgbd")["tag"] == "PLACEHOLDER"
    o = m.autonomy_overrides("sim")
    assert o["camera"]["mount_torso_xyz"] == [0.0537, 0.0175, 0.4739] and o["camera"]["pitch_down_deg"] == 47.6
    assert abs(o["lidar"]["mount_xyz"][2] - (0.4602 + 0.72)) < 1e-9
    K = m.intrinsics("d435")
    assert abs(2 * math.degrees(math.atan((K.width / 2) / K.fx)) - 87.0) < 1e-9 and K.cx == (K.width - 1) / 2


def test_optical_conversion_and_camera_mount_are_consistent():
    """A point 2 m straight ahead of a level camera: USD camera frame (0, 0, -2) -> optical (0, 0, 2) -> body x = +2."""
    assert np.allclose(MT.usd_depth_to_optical([[0.0, 0.0, -2.0], [0.3, 0.1, -1.0]]), [[0.0, 0.0, 2.0], [0.3, -0.1, 1.0]])
    assert np.allclose(MT.OPTICAL_FROM_USD @ MT.OPTICAL_FROM_USD, np.eye(3))
    # the camera prim placed in a link frame looks along the link's +x with +z up and +y to its left
    R = MT.USD_CAMERA_IN_LINK
    assert np.allclose(R @ [0, 0, -1], [1, 0, 0]) and np.allclose(R @ [0, 1, 0], [0, 0, 1]) and np.allclose(R @ [1, 0, 0], [0, -1, 0])
    assert np.isclose(np.linalg.det(R), 1.0)
    # the stack's extrinsics: a down-pitched camera sees the floor where geometry says it is
    m = MT.load_mounts()
    pose, K = m.camera_pose("d435", "sim"), m.intrinsics("d435")
    h = pose.position[2]
    a = math.radians(47.6)
    depth = np.full((K.height, K.width), h / math.sin(a), dtype=np.float32)        # every pixel at the centre ray's range (only the centre is checked)
    pts = depth_to_points(DepthFrame(0.0, depth, K, pose, camera="down"), max_range=10.0)
    c = pts.reshape(K.height, K.width, 3)[K.height // 2, K.width // 2]
    assert abs(c[2]) < 0.02 and abs(c[0] - (pose.position[0] + h / math.tan(a))) < 0.02 and abs(c[1] - pose.position[1]) < 0.02


def test_lidar_pose_pitches_the_beam_down():
    p = MT.load_mounts().lidar_pose("sim")
    R = np.array(MT.link_rotation_for_pitch(2.3))
    assert R[2, 0] < 0 and abs(math.degrees(math.asin(-R[2, 0])) - 2.3) < 1e-9 and p.position[2] > 1.1


def test_loopback_link_hands_over_the_newest_messages():
    link = lowlevel.LoopbackLink()
    assert link.read_cmd() is None and link.read_state() == (None, None)
    c = lowlevel.LowCmd()
    link.write_cmd(c)
    assert link.read_cmd() is c
