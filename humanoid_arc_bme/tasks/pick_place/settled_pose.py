"""Settled standing pose of the G1 (agile_legs base) in the pick-place demo env.

Measured, not hand-tuned: the robot spawned in G1_29DOF_CFG's default pose,
commanded to stand (Agile, [0,0,0,0.72]) with the arms at rest, after 700
steps (14s), by which point it no longer moves. Spawning in it skips the
first 2-4s of every episode, in which the robot drops 4cm, leans 6.5deg
forward and steps 6.7cm back.
"""

ROOT_HEIGHT = 0.7219
ROOT_QUAT = (0.7081, -0.0388, 0.0410, 0.7038)  # w, x, y, z: the 90deg yaw of G1_29DOF_CFG plus the settled lean
JOINT_POS = {
    "left_hip_pitch_joint": -0.6420,
    "right_hip_pitch_joint": -0.6453,
    "waist_yaw_joint": 0.0026,
    "left_hip_roll_joint": -0.0064,
    "right_hip_roll_joint": 0.0075,
    "waist_roll_joint": -0.0059,
    "left_hip_yaw_joint": -0.0137,
    "right_hip_yaw_joint": 0.0136,
    "waist_pitch_joint": 0.1076,
    "left_knee_joint": 0.9312,
    "right_knee_joint": 0.9352,
    "left_shoulder_pitch_joint": -0.2680,
    "right_shoulder_pitch_joint": -0.2598,
    "left_ankle_pitch_joint": -0.4022,
    "right_ankle_pitch_joint": -0.4027,
    "left_shoulder_roll_joint": 0.0522,
    "right_shoulder_roll_joint": -0.0396,
    "left_ankle_roll_joint": -0.0067,
    "right_ankle_roll_joint": 0.0005,
    "left_shoulder_yaw_joint": 0.0465,
    "right_shoulder_yaw_joint": -0.0354,
    "left_elbow_joint": -0.2035,
    "right_elbow_joint": -0.2279,
    "left_wrist_roll_joint": -0.0445,
    "right_wrist_roll_joint": 0.0377,
    "left_wrist_pitch_joint": 0.2459,
    "right_wrist_pitch_joint": 0.2582,
    "left_wrist_yaw_joint": 0.0430,
    "right_wrist_yaw_joint": -0.0446,
    "left_hand_index_0_joint": -0.0000,
    "left_hand_middle_0_joint": -0.0000,
    "left_hand_thumb_0_joint": 0.0000,
    "right_hand_index_0_joint": 0.0000,
    "right_hand_middle_0_joint": -0.0000,
    "right_hand_thumb_0_joint": 0.0000,
    "left_hand_index_1_joint": -0.0000,
    "left_hand_middle_1_joint": -0.0000,
    "left_hand_thumb_1_joint": 0.0000,
    "right_hand_index_1_joint": 0.0000,
    "right_hand_middle_1_joint": -0.0000,
    "right_hand_thumb_1_joint": 0.0000,
    "left_hand_thumb_2_joint": 0.0000,
    "right_hand_thumb_2_joint": 0.0000,
}
