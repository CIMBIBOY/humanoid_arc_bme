"""Livox MID-360 lidar as a pipeline source: PointCloud2 -> xyz, lidar-to-body extrinsics, latest-scan holder.

Stdlib + numpy at import time; rclpy and rosidl_runtime_py are imported only when no node is injected. Frames: points
arrive in the lidar frame and leave in the body frame (x forward, y left, z up); metres, seconds. The result is a
`LidarFrame`, which expects body-frame points (gravity alignment is applied downstream, as for the camera cloud).
"""

from __future__ import annotations

import math
import operator
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from humanoid_arc_bme.onboard.pipeline.interfaces import Pose
from humanoid_arc_bme.onboard.pipeline.perception.lidar import LidarFrame

POINTCLOUD2 = "sensor_msgs/msg/PointCloud2"
FIELD_FLOAT32 = 7       # sensor_msgs/PointField datatype codes
FIELD_FLOAT64 = 8
_FLOAT_FIELDS = {FIELD_FLOAT32: "f4", FIELD_FLOAT64: "f8"}
_QUEUE_DEPTH = 5        # queue depth handed to an injected node (a real node gets the sensor-data QoS profile)


@dataclass(frozen=True)
class PointField:
    """One entry of a PointCloud2 `fields` list (sensor_msgs/PointField)."""

    name: str
    offset: int
    datatype: int
    count: int = 1


@dataclass(frozen=True)
class PointCloud2:
    """Layout-compatible stand-in for sensor_msgs/PointCloud2 (tests and replay); `data` is the raw byte buffer."""

    height: int
    width: int
    fields: list
    point_step: int
    row_step: int
    data: bytes
    is_bigendian: bool = False


def _xyz_dtype(fields: dict, step: int, endian: str) -> np.dtype:
    formats, offsets = [], []
    for name in ("x", "y", "z"):
        f = fields.get(name)
        if f is None:
            raise ValueError(f"pointcloud2: no '{name}' field")
        if f.count != 1 or f.datatype not in _FLOAT_FIELDS or f.offset < 0:
            raise ValueError(f"pointcloud2: '{name}' must be one FLOAT32 or FLOAT64 value at a non-negative offset")
        fmt = endian + _FLOAT_FIELDS[f.datatype]
        if f.offset + np.dtype(fmt).itemsize > step:
            raise ValueError(f"pointcloud2: '{name}' does not fit in point_step")
        formats.append(fmt)
        offsets.append(int(f.offset))
    return np.dtype({"names": ["x", "y", "z"], "formats": formats, "offsets": offsets, "itemsize": step})


def pointcloud2_to_xyz(msg: Any) -> np.ndarray:
    """(N, 3) float32 xyz of a sensor_msgs/PointCloud2-like object, in the sensor frame. Other fields (intensity, tag,
    timestamp, padding) are ignored; rows with a non-finite coordinate are dropped. Raises ValueError on a malformed
    message (missing or mistyped header, fields that are not float x/y/z, a buffer shorter than height * row_step)."""
    try:
        height, width = operator.index(msg.height), operator.index(msg.width)
        step, row = operator.index(msg.point_step), operator.index(msg.row_step)
        endian = ">" if msg.is_bigendian else "<"
        fields = {f.name: f for f in msg.fields}
        dtype = _xyz_dtype(fields, step, endian) if height * width > 0 else None
    except (AttributeError, TypeError) as e:
        raise ValueError(f"pointcloud2: malformed message ({e})") from e
    if height < 0 or width < 0 or step <= 0 or row < width * step:
        raise ValueError("pointcloud2: bad height/width/point_step/row_step")
    if dtype is None:
        return np.zeros((0, 3), dtype=np.float32)
    try:
        raw = np.frombuffer(msg.data, dtype=np.uint8, count=height * row)
    except (TypeError, ValueError) as e:
        raise ValueError("pointcloud2: data shorter than height * row_step") from e
    rows = raw.reshape(height, row)[:, : width * step]          # drop the per-row padding
    pts = np.ascontiguousarray(rows).view(dtype).reshape(-1)
    with np.errstate(over="ignore", invalid="ignore"):          # float64 beyond float32 range becomes inf: dropped below
        xyz = np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float32)
    return xyz[np.isfinite(xyz).all(axis=1)]


def lidar_to_body(points_lidar: np.ndarray, lidar_to_body: Pose) -> np.ndarray:
    """(N, 3) lidar-frame points -> (N, 3) float32 body-frame points: p_body = R p_lidar + t (`lidar_to_body` extrinsics)."""
    p = np.asarray(points_lidar, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("points_lidar: expected an (N, 3) array")
    R = lidar_to_body.rotation_matrix()
    return (p @ R.T + np.asarray(lidar_to_body.position)).astype(np.float32)


def ros_subscribe(node: Any, type_name: str, topic: str, callback: Callable[[Any], None], node_name: str) -> Any:
    """Subscribe `callback` to `topic` and return the node that owns the subscription (keep the reference alive).
    node=None creates an rclpy node named `node_name` and resolves `type_name` to its message class; the caller must
    spin that node (it is not spun here). An injected node receives the type name and an integer queue depth."""
    if node is None:
        import rclpy                                            # lazy: ROS is only needed on the robot
        from rclpy.qos import qos_profile_sensor_data
        from rosidl_runtime_py.utilities import get_message

        if not rclpy.ok():
            rclpy.init()
        node = rclpy.create_node(node_name)
        node.create_subscription(get_message(type_name), topic, callback, qos_profile_sensor_data)
    else:
        node.create_subscription(type_name, topic, callback, _QUEUE_DEPTH)
    return node


class LivoxSource:
    """Newest Livox scan, converted to the body frame and randomly subsampled to at most `max_points` on arrival.
    The subscription callback runs on the ROS thread and `latest()` on the control loop; a lock guards the handoff.
    A malformed message is counted in `dropped` and the previous scan is kept (it ages out via `age`)."""

    def __init__(self, lidar_to_body: Pose, topic: str = "/livox/lidar", node: Any = None,
                 clock: Callable[[], float] = time.monotonic, max_points: int = 20000, seed: int = 0):
        if type(max_points) is not int or max_points < 1:
            raise ValueError("max_points: must be an int >= 1")
        self._pose = lidar_to_body
        self._clock = clock
        self._max_points = max_points
        self._rng = np.random.default_rng(seed)     # used only from the subscription callback (serial per node)
        self._lock = threading.Lock()
        self._frame: LidarFrame | None = None
        self.dropped = 0
        self._node = ros_subscribe(node, POINTCLOUD2, topic, self._on_cloud, "livox_source")

    def _on_cloud(self, msg: Any) -> None:
        try:
            xyz = pointcloud2_to_xyz(msg)
        except ValueError:
            self.dropped += 1
            return
        if len(xyz) > self._max_points:
            keep = np.sort(self._rng.choice(len(xyz), self._max_points, replace=False))
            xyz = xyz[keep]
        frame = LidarFrame(self._clock(), lidar_to_body(xyz, self._pose))
        with self._lock:
            self._frame = frame

    def latest(self) -> LidarFrame | None:
        """Newest body-frame scan, or None before the first message."""
        with self._lock:
            return self._frame

    def age(self, now: float) -> float:
        """Seconds between `now` and the newest scan's arrival; inf before the first message."""
        frame = self.latest()
        return math.inf if frame is None else now - frame.stamp
