"""Proportional walk-and-turn controller for the Agile lower-body policy's
`[vx, vy, wz, hip_height]` command. Vectorized over envs; walking and turning
run simultaneously (curved approach), not stop-then-spin-then-walk.
"""

import torch

from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

MAX_LINEAR_SPEED = 0.5  # m/s, within the range verified in eval_velocity.py
MAX_YAW_RATE = 0.8  # rad/s
POSITION_GAIN = 1.5
YAW_GAIN = 2.0
# Tightening these 4x/3x (0.03 m / 0.05 rad) left the tour's placement error
# unchanged, so arrival precision is not what limits it (the cause was the
# arm's reach, see docs/audit.md). Kept at the looser values.
POSITION_TOLERANCE = 0.12  # m
YAW_TOLERANCE = 0.15  # rad
STAND_HIP_HEIGHT = 0.72
# The Agile policy stands still for small commands (learned, not a coded
# deadband): wz=0.3 rad/s / vx=0.11 m/s left a G1 standing 0.153 rad off its
# heading, just outside YAW_TOLERANCE, forever. The proportional law decays
# into that zone near the target, so outside tolerance it's floored here.
MIN_YAW_RATE = 0.5  # rad/s
MIN_LINEAR_SPEED = 0.2  # m/s


def navigate(
    pelvis_pos_xy: torch.Tensor,
    pelvis_quat: torch.Tensor,
    target_xy: torch.Tensor,
    target_heading: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a lower-body command steering the pelvis toward a pose.

    Args:
        pelvis_pos_xy: (N, 2) current pelvis position, env-local frame.
        pelvis_quat: (N, 4) current pelvis orientation.
        target_xy: (N, 2) target position, env-local frame.
        target_heading: (N,) target yaw, radians.

    Returns:
        command: (N, 4) `[vx, vy, wz, hip_height]` for the lower-body action term.
        arrived: (N,) bool, within position and heading tolerance.
    """
    delta_world = target_xy - pelvis_pos_xy
    distance = delta_world.norm(dim=-1)
    yaw = euler_xyz_from_quat(pelvis_quat)[2]

    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    forward_error = cos_yaw * delta_world[:, 0] + sin_yaw * delta_world[:, 1]
    lateral_error = -sin_yaw * delta_world[:, 0] + cos_yaw * delta_world[:, 1]

    speed = torch.clamp(POSITION_GAIN * distance, max=MAX_LINEAR_SPEED)
    speed = torch.where(distance > POSITION_TOLERANCE, speed.clamp(min=MIN_LINEAR_SPEED), speed)
    scale = speed / distance.clamp(min=1e-3)
    vx = forward_error * scale
    vy = lateral_error * scale

    yaw_error = wrap_to_pi(target_heading - yaw)
    wz = torch.clamp(YAW_GAIN * yaw_error, min=-MAX_YAW_RATE, max=MAX_YAW_RATE)
    floor = torch.sign(yaw_error) * MIN_YAW_RATE
    wz = torch.where((yaw_error.abs() > YAW_TOLERANCE) & (wz.abs() < MIN_YAW_RATE), floor, wz)

    arrived = (distance < POSITION_TOLERANCE) & (yaw_error.abs() < YAW_TOLERANCE)
    hip_height = torch.full_like(vx, STAND_HIP_HEIGHT)
    return torch.stack([vx, vy, wz, hip_height], dim=-1), arrived
