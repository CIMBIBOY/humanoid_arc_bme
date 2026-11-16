"""Joint layout of the Unitree G1 (29 DoF) with Dex3 hands: the motor order of rt/lowcmd / rt/lowstate and the planner topic's fields.

Motor order (OPENED 2026-11-14, unitree_sdk2_python `example/g1/low_level/g1_low_level_example.py`, class G1JointIndex): motors 0-5
left leg (hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll), 6-11 right leg, 12-14 waist (yaw, roll, pitch), 15-21 left
arm, 22-28 right arm. `MOTOR_JOINTS[i]` is the joint of motor i. Unitree's own Isaac simulation maps arm motors 15-28 and the Dex3
motors 0-6 (thumb 0, 1, 2, middle 0, 1, index 0, 1) to the same joint names (OPENED, unitree_sim_isaaclab
`action_provider/action_provider_wh_dds.py`), and every name exists in Unitree's G1 USD (checked in the simulator by scripts/isaac/g1_loco_gate.py).

Source: gear_sonic_deploy `zmq_manager.hpp` (OPENED 2026-11-13) decodes the optional planner-message fields
`upper_body_position` / `upper_body_velocity` (17 values), `left_hand_joints` and `right_hand_joints` (7 values each).
The deploy does not name the joints; the order below is the G1 29-DoF motor order after the 12 leg motors (waist, left
arm, right arm) and the Dex3 order of the repository's URDF. UNVERIFIED on hardware: confirm it once against the deploy
(run the deploy's MuJoCo sim, command one joint, watch which one moves) before the first arm motion.
"""

from __future__ import annotations

_ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
_HAND = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")

_LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")

LEFT_LEG_JOINTS = tuple(f"left_{n}_joint" for n in _LEG)
RIGHT_LEG_JOINTS = tuple(f"right_{n}_joint" for n in _LEG)
WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
LEFT_ARM_JOINTS = tuple(f"left_{n}_joint" for n in _ARM)
RIGHT_ARM_JOINTS = tuple(f"right_{n}_joint" for n in _ARM)
UPPER_BODY_JOINTS = WAIST_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS          # 17
LEFT_HAND_JOINTS = tuple(f"left_hand_{n}_joint" for n in _HAND)                # 7
RIGHT_HAND_JOINTS = tuple(f"right_hand_{n}_joint" for n in _HAND)              # 7

MOTOR_JOINTS = LEFT_LEG_JOINTS + RIGHT_LEG_JOINTS + UPPER_BODY_JOINTS             # 29: index = unitree_hg motor index
UPPER_BODY_MOTOR_OFFSET = 12                                                    # upper_body_position[i] drives motor 12 + i

assert len(UPPER_BODY_JOINTS) == 17 and len(LEFT_HAND_JOINTS) == len(RIGHT_HAND_JOINTS) == 7 and len(MOTOR_JOINTS) == 29
assert MOTOR_JOINTS[UPPER_BODY_MOTOR_OFFSET:] == UPPER_BODY_JOINTS
