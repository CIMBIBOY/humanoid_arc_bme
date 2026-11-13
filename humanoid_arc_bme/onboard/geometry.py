"""Small rotation helpers shared by the onboard modules. Quaternions are (w, x, y, z), the contract's order."""

import math

import numpy as np


def quat_to_matrix(q) -> np.ndarray:
    """(w, x, y, z) unit quaternion -> 3x3 rotation matrix (float64)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def yaw_from_quat(q) -> float:
    """Yaw of a (w, x, y, z) quaternion."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
