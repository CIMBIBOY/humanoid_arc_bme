import math
from types import SimpleNamespace

import numpy as np

from humanoid_arc_bme.onboard.api.contract import Imu, RobotState
from humanoid_arc_bme.onboard.g1_layout import LEFT_HAND_JOINTS, RIGHT_HAND_JOINTS
from humanoid_arc_bme.onboard.pipeline.interfaces import DepthFrame, Intrinsics, LocalizerEstimate, Pose, yaw_pose
from humanoid_arc_bme.onboard.sensors.hub import SensorHub, roll_pitch

K = Intrinsics(100.0, 100.0, 9.5, 7.5, 20, 16)
DEPTH = DepthFrame(1.0, np.ones((16, 20), np.float32), K, Pose((0, 0, 1.0)))


def state(quat=(1.0, 0.0, 0.0, 0.0)):
    names = ("waist_yaw_joint", "left_elbow_joint")
    return RobotState(1.0, names, (0.1, 0.2), (0.0, 0.0), (0, 0, 0), Imu(quat, (0, 0, 0), (0, 0, 9.8)), {})


class Cam:
    def __init__(self, depth=DEPTH):
        self.d = depth

    def latest(self):
        return (np.zeros((16, 20, 3), np.uint8), self.d)


class Loc:
    def __init__(self):
        self.x = 0.0

    def estimate(self, stamp):
        return LocalizerEstimate(stamp, yaw_pose(self.x, 0.0, 0.0, 0.0), (1e-4,) * 6)


class Hands:
    def latest(self):
        return ((0.0,) * 7, (0.0,) * 7)


def test_roll_pitch_of_known_attitudes():
    assert roll_pitch((1, 0, 0, 0)) == (0.0, 0.0)
    s = math.sin(math.radians(10) / 2)
    r, p = roll_pitch((math.cos(math.radians(10) / 2), s, 0, 0))
    assert abs(r - math.radians(10)) < 1e-9 and abs(p) < 1e-9


def test_frame_needs_camera_and_state():
    hub = SensorHub(Cam(depth=None))
    assert hub.frame(1.0, state()) is None
    assert SensorHub(Cam()).frame(1.0, None) is None


def test_joints_merge_body_and_hands_or_are_none_without_a_hand_feed():
    f = SensorHub(Cam(), localizer=Loc(), hands=Hands()).frame(1.0, state())
    assert f.joints["left_elbow_joint"] == 0.2 and all(n in f.joints for n in LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS)
    assert SensorHub(Cam(), localizer=Loc()).frame(1.0, state()).joints is None


def test_speed_is_measured_from_successive_poses():
    loc = Loc()
    hub = SensorHub(Cam(), localizer=loc)
    hub.frame(1.0, state())
    for t in (1.1, 1.2, 1.3, 1.4, 1.5, 1.6):
        loc.x = (t - 1.0) * 0.2
        f = hub.frame(t, state())
    assert abs(f.base_speed - 0.2) < 0.03
