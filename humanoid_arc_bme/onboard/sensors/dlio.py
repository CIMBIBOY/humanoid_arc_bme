"""DLIO lidar-inertial odometry as the body localiser: nav_msgs/Odometry -> `LocalizerEstimate`.

Stdlib + numpy at import time (rclpy only when no node is injected). The pose is `pose.pose` (body pose in the odometry
map frame). ROS quaternions are (x, y, z, w) and are converted to the package's (w, x, y, z), normalised. The
covariance is the row-major 6x6 pose covariance in the order (x, y, z, roll, pitch, yaw); its diagonal becomes the
estimate's variance (m^2, rad^2). Fail closed: malformed or non-finite input gives `healthy=False`, a unit quaternion
(identity pose if the pose itself is unusable) and UNKNOWN_VARIANCE, never an exception.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import numpy as np

from humanoid_arc_bme.onboard.pipeline.interfaces import LocalizerEstimate, Pose
from humanoid_arc_bme.onboard.sensors.livox import ros_subscribe
from humanoid_arc_bme.onboard.validation import finite_vec, is_num

ODOMETRY = "nav_msgs/msg/Odometry"
UNKNOWN_VARIANCE = 1e6                       # m^2 / rad^2 reported when the source gave nothing usable
_POSITION_AXES = 3                           # x, y, z are checked against max_sigma_m; roll, pitch, yaw are not


def _pose_from(msg: Any) -> Pose:
    """Body pose from an Odometry-like message. Raises ValueError if non-finite or the quaternion has zero norm."""
    p = msg.pose.pose
    xyz = finite_vec("position", (p.position.x, p.position.y, p.position.z), 3)
    q = np.array(finite_vec("orientation", (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w), 4))
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        raise ValueError("orientation: zero quaternion")
    x, y, z, w = q / norm
    return Pose(xyz, (w, x, y, z))


def _covariance_from(msg: Any) -> np.ndarray:
    """The 36 covariance entries as (6, 6) float64. Raises ValueError on a wrong size or any non-finite entry."""
    cov = np.asarray(msg.pose.covariance, dtype=np.float64)
    if cov.size != 36:
        raise ValueError("covariance: expected 36 values")
    cov = cov.reshape(6, 6)
    if not np.isfinite(cov).all():
        raise ValueError("covariance: non-finite entry")
    return cov


def odometry_to_estimate(msg: Any, stamp: float, source: str = "dlio", max_sigma_m: float = 0.5) -> LocalizerEstimate:
    """Estimate for `stamp` from an Odometry-like message. `healthy` is False if the pose or covariance is unusable,
    any covariance diagonal entry is negative, or sqrt(variance) of x, y or z exceeds `max_sigma_m` (metres).
    `stamp` (the caller's clock) must be finite; the message's own header stamp is not used."""
    if not (is_num(max_sigma_m) and max_sigma_m > 0):
        raise ValueError("max_sigma_m: must be a finite number > 0")
    try:
        pose = _pose_from(msg)
    except (AttributeError, TypeError, ValueError):
        return LocalizerEstimate(stamp, Pose((0.0, 0.0, 0.0)), (UNKNOWN_VARIANCE,) * 6, False, source)
    try:
        cov = _covariance_from(msg)
    except (AttributeError, TypeError, ValueError):
        return LocalizerEstimate(stamp, pose, (UNKNOWN_VARIANCE,) * 6, False, source)
    var = cov.diagonal().copy()
    if (var < 0).any():
        return LocalizerEstimate(stamp, pose, (UNKNOWN_VARIANCE,) * 6, False, source)
    healthy = bool((np.sqrt(var[:_POSITION_AXES]) <= max_sigma_m).all())
    return LocalizerEstimate(stamp, pose, tuple(float(v) for v in var), healthy, source)


class DlioSource:
    """Newest DLIO estimate, stamped on arrival. `estimate(stamp)` returns it, or None if none has arrived or it is
    more than `max_age_s` older than `stamp`. An unhealthy estimate is returned as is (the consumer stops on it).
    The subscription callback runs on the ROS thread; a lock guards the handoff."""

    def __init__(self, topic: str = "/dlio/odom_node/odom", node: Any = None,
                 clock: Callable[[], float] = time.monotonic, max_age_s: float = 0.5):
        if not (is_num(max_age_s) and max_age_s > 0):
            raise ValueError("max_age_s: must be a finite number > 0")
        self._clock = clock
        self._max_age = float(max_age_s)
        self._lock = threading.Lock()
        self._latest: LocalizerEstimate | None = None
        self._node = ros_subscribe(node, ODOMETRY, topic, self._on_odometry, "dlio_source")

    def _on_odometry(self, msg: Any) -> None:
        est = odometry_to_estimate(msg, self._clock())
        with self._lock:
            self._latest = est

    def estimate(self, stamp: float) -> LocalizerEstimate | None:
        with self._lock:
            est = self._latest
        if est is None or stamp - est.stamp > self._max_age:
            return None
        return est
