"""A deterministic stand-in robot for tests and for exercising the viewer without hardware."""

from __future__ import annotations

import math

from humanoid_arc_bme.onboard.api.contract import Imu, RobotState

from ..types import StateMeta
from .base import RobotIO

G1_JOINTS = tuple(
    f"{side}_{j}_joint"
    for side, js in (
        ("left", ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")),
        ("right", ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")),
    )
    for j in js
) + ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint") + tuple(
    f"{side}_{j}_joint"
    for side in ("left", "right")
    for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
)


class FakeRobotIO(RobotIO):
    simulated = True
    def __init__(self, clock, joint_names=G1_JOINTS):
        self.clock, self.names = clock, tuple(joint_names)
        self.sent: list = []

    def read_state(self):
        t = self.clock()
        q = tuple(0.1 * math.sin(t + i) for i in range(len(self.names)))
        dq = tuple(0.1 * math.cos(t + i) for i in range(len(self.names)))
        imu = Imu(quat=(1.0, 0.0, 0.0, 0.0), gyro=(0.0, 0.0, 0.0), accel=(0.0, 0.0, 9.81))
        pose = (0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        st = RobotState(stamp=t, joint_names=self.names, joint_pos=q, joint_vel=dq, base_vel=(0.0, 0.0, 0.0), imu=imu,
                        wrist_poses={"left": pose, "right": pose})
        return st, StateMeta()

    def send_joint_targets(self, names, targets, kp=None, kd=None):
        self.sent.append((tuple(names), tuple(targets)))
