"""Joint layout of the Unitree G1 (29 DoF) with Dex3 hands, as the GR00T deploy's planner topic expects it.

Source: gear_sonic_deploy `zmq_manager.hpp` (OPENED 2026-11-13) decodes the optional planner-message fields
`upper_body_position` / `upper_body_velocity` (17 values), `left_hand_joints` and `right_hand_joints` (7 values each).
The deploy does not name the joints; the order below is the G1 29-DoF motor order after the 12 leg motors (waist, left
arm, right arm) and the Dex3 order of the repository's URDF. UNVERIFIED on hardware: confirm it once against the deploy
(run the deploy's MuJoCo sim, command one joint, watch which one moves) before the first arm motion.
"""

from __future__ import annotations

_ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
_HAND = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")

WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
LEFT_ARM_JOINTS = tuple(f"left_{n}_joint" for n in _ARM)
RIGHT_ARM_JOINTS = tuple(f"right_{n}_joint" for n in _ARM)
UPPER_BODY_JOINTS = WAIST_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS          # 17
LEFT_HAND_JOINTS = tuple(f"left_hand_{n}_joint" for n in _HAND)                # 7
RIGHT_HAND_JOINTS = tuple(f"right_hand_{n}_joint" for n in _HAND)              # 7

assert len(UPPER_BODY_JOINTS) == 17 and len(LEFT_HAND_JOINTS) == len(RIGHT_HAND_JOINTS) == 7
