"""The simulated G1's motor model: what the simulator must know about the robot that no command carries. Plain data, no Isaac.

A real motor obeys `tau = tau_ff + kp (q_des - q) + kd (dq_des - dq)` with the gains that arrive in every rt/lowcmd; the simulator
applies the same law per motor (as a PhysX joint drive, which integrates it implicitly and stays stable at arm gains an explicit
200 Hz loop cannot hold). Effort limits come from Unitree's USD (`drive:angular:physics:maxForce`: 88 / 139 / 50 N m legs, 25 / 5 N m
arms). Armature is the reflected rotor inertia the controller was trained with.

STAND-IN values (the SONIC deploy brings its own gains in rt/lowcmd; these are what the stand-in policy sends):
gains, armature and the default pose are those of the Isaac Lab configuration NVIDIA's AGILE lower-body policy was trained on
(isaaclab_assets `G1_29DOF_CFG`, OPENED in the local Isaac Lab checkout). Arm and waist gains of 3000 / 5000 N m/rad are simulation
values, far stiffer than anyone runs on the hardware.
"""

from __future__ import annotations

import re

from humanoid_arc_bme.onboard.g1_layout import LEFT_HAND_JOINTS, MOTOR_JOINTS, RIGHT_HAND_JOINTS

HAND_JOINTS = LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS
ROOT_HEIGHT = 0.80                      # pelvis above the floor at spawn: feet a few centimetres up, the robot lands on its stance
MAX_DEPENETRATION_VELOCITY = 1.0        # m/s (G1_29DOF_CFG rigid_props)
SOLVER_ITERATIONS = (8, 4)              # position, velocity (G1_29DOF_CFG articulation_props)
LOWCMD_TIMEOUT_S = 0.10                 # GUESS: with no rt/lowcmd for this long the simulated motors go passive (damping only). The
#                                         real G1's behaviour when its controller exits is UNKNOWN (docs/plan/groot-behind-s1.md 1.1)
PASSIVE_KD = 2.0                        # GUESS: N m s/rad of the passive state

# (regex on the joint name, kp N m/rad, kd N m s/rad, armature kg m^2), first match wins
_TABLE = (
    (r".*_hip_(yaw|roll|pitch)_joint", 100.0, 2.5, 0.03),
    (r".*_knee_joint", 200.0, 5.0, 0.03),
    (r".*_ankle_pitch_joint", 20.0, 0.2, 0.03),
    (r".*_ankle_roll_joint", 20.0, 0.1, 0.03),
    (r"waist_.*_joint", 5000.0, 5.0, 0.001),
    (r".*_(shoulder|elbow|wrist)_.*joint", 3000.0, 10.0, 0.001),
    (r".*_hand_.*_joint", 20.0, 2.0, 0.001),
)
_DEFAULT_POSE = ((r".*_hip_pitch_joint", -0.10), (r".*_knee_joint", 0.30), (r".*_ankle_pitch_joint", -0.20))


def _lookup(name: str) -> tuple[float, float, float]:
    for pat, kp, kd, arm in _TABLE:
        if re.fullmatch(pat, name):
            return kp, kd, arm
    raise KeyError(name)


def gains(names=MOTOR_JOINTS) -> tuple[list[float], list[float]]:
    """(kp, kd) per joint in the order of `names` (default: motor order)."""
    return [_lookup(n)[0] for n in names], [_lookup(n)[1] for n in names]


def armature(names) -> list[float]:
    return [_lookup(n)[2] for n in names]


def default_pose(names=MOTOR_JOINTS) -> list[float]:
    """Standing joint positions (rad) in the order of `names`."""
    return [next((v for pat, v in _DEFAULT_POSE if re.fullmatch(pat, n)), 0.0) for n in names]
