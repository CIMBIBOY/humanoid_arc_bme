"""Real-sensor adapters with fakes: RealSense frames, Livox PointCloud2 parsing and lidar extrinsics, DLIO odometry."""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from humanoid_arc_bme.onboard.pipeline.interfaces import Intrinsics, Pose, yaw_pose
from humanoid_arc_bme.onboard.pipeline.perception.lidar import LidarFrame
from humanoid_arc_bme.onboard.sensors.dlio import UNKNOWN_VARIANCE, DlioSource, odometry_to_estimate
from humanoid_arc_bme.onboard.sensors.livox import (
    POINTCLOUD2, LivoxSource, PointCloud2, PointField, lidar_to_body, pointcloud2_to_xyz,
)
from humanoid_arc_bme.onboard.sensors.realsense import RealSenseSource, depth_to_metres

ROOT = Path(__file__).resolve().parents[2]
K = Intrinsics(fx=2.0, fy=2.0, cx=1.5, cy=1.0, width=4, height=3)
CAM = Pose((0.1, 0.0, 0.5), (1.0, 0.0, 0.0, 0.0))
RGB = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
RAW = np.array([[1000, 0, 2000, 20000], [500, 11000, 65535, 1500], [1, 2, 3, 4]], dtype=np.uint16)


# ------------------------------------------------------------------ RealSense
class FakePipeline:
    """Serves queued framesets; an Exception instance in the queue is raised (a driver timeout or a bad frame)."""

    def __init__(self, frames, depth_scale=0.001):
        self.frames = list(frames)
        self.depth_scale = depth_scale
        self.intrinsics = K
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def wait_for_frames(self):
        item = self.frames.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_depth_to_metres_scale_zero_and_range():
    d = depth_to_metres(RAW, 0.001)
    assert d.dtype == np.float32 and d.shape == (3, 4)
    assert d[0, 0] == pytest.approx(1.0) and d[0, 2] == pytest.approx(2.0)
    assert d[1, 0] == pytest.approx(0.5) and d[1, 3] == pytest.approx(1.5)
    assert d[2, 0] == pytest.approx(0.001) and d[2, 3] == pytest.approx(0.004)
    nan_at = np.isnan(d)
    assert nan_at[0, 1] and nan_at[0, 3] and nan_at[1, 1] and nan_at[1, 2]   # 0, 20 m, 11 m, 65.5 m
    assert nan_at.sum() == 4
    assert depth_to_metres(RAW, 0.002)[0, 0] == pytest.approx(2.0)          # the scale is the device's, not a constant


@pytest.mark.parametrize("scale", [0.0, -0.001, math.nan, math.inf, True, None])
def test_depth_to_metres_bad_scale(scale):
    with pytest.raises(ValueError):
        depth_to_metres(RAW, scale)


@pytest.mark.parametrize("raw", [RAW.astype(np.float32), RAW[0], [[1.0, 2.0]]])
def test_depth_to_metres_bad_array(raw):
    with pytest.raises(ValueError):
        depth_to_metres(raw, 0.001)


def test_latest_returns_rgb_depth_intrinsics_and_stamp():
    pipe = FakePipeline([(RGB, RAW)])
    src = RealSenseSource(CAM, pipeline=pipe, clock=lambda: 42.0)
    assert pipe.started == 1
    rgb, frame = src.latest()
    assert np.array_equal(rgb, RGB) and rgb.dtype == np.uint8
    assert frame.stamp == 42.0 and frame.camera == "down" and frame.cam_to_body == CAM
    assert frame.intrinsics == K                       # colour intrinsics, used as given (OpenCV centre convention)
    assert frame.depth.dtype == np.float32 and frame.depth.shape == (3, 4)
    assert np.isnan(frame.depth[0, 1]) and frame.depth[0, 0] == pytest.approx(1.0)
    assert src.dropped == 0


def test_timeouts_and_malformed_framesets_are_dropped_not_raised():
    bad = [RuntimeError("Frame didn't arrive within 5000"),
           (RGB[:2], RAW),                                  # rgb does not match intrinsics
           (RGB, RAW[:2]),                                  # depth does not match intrinsics
           (RGB, RAW.astype(np.float32))]                   # depth is not z16
    pipe = FakePipeline(bad + [(RGB, RAW)])
    src = RealSenseSource(CAM, pipeline=pipe, clock=lambda: 1.0)
    for _ in bad:
        assert src.latest() == (None, None)
    assert src.dropped == len(bad)
    rgb, frame = src.latest()
    assert rgb is not None and frame is not None and src.dropped == len(bad)


def test_close_stops_once_and_latest_is_empty_afterwards():
    pipe = FakePipeline([(RGB, RAW)])
    src = RealSenseSource(CAM, pipeline=pipe)
    src.close()
    src.close()
    assert pipe.stopped == 1
    assert src.latest() == (None, None)


def test_bad_depth_scale_from_device_fails_at_construction():
    with pytest.raises(ValueError):
        RealSenseSource(CAM, pipeline=FakePipeline([], depth_scale=0.0))


# ------------------------------------------------------------------ PointCloud2
POINT_DTYPE = np.dtype({"names": ["x", "y", "z", "intensity", "tag", "line", "timestamp"],
                        "formats": ["<f4", "<f4", "<f4", "<f4", "u1", "u1", "<f8"],
                        "offsets": [0, 4, 8, 12, 16, 17, 24], "itemsize": 32})
FIELDS = [PointField("x", 0, 7), PointField("y", 4, 7), PointField("z", 8, 7), PointField("intensity", 12, 7),
          PointField("tag", 16, 2), PointField("line", 17, 2), PointField("timestamp", 24, 8)]


def cloud_from_xyz(xyz, height=1, row_pad=0) -> PointCloud2:
    """Livox-like cloud: 32-byte points with intensity, tag, line and timestamp, optional bytes of padding per row."""
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    width = len(xyz) // height
    rec = np.zeros(len(xyz), dtype=POINT_DTYPE)
    rec["x"], rec["y"], rec["z"] = xyz.T
    rec["intensity"], rec["timestamp"] = 7.0, 1.5
    data = b"".join(rec[i * width:(i + 1) * width].tobytes() + b"\xff" * row_pad for i in range(height))
    return PointCloud2(height, width, FIELDS, 32, width * 32 + row_pad, data)


def test_pointcloud2_parses_xyz_and_ignores_extra_fields():
    xyz = np.array([[1.0, 2.0, 3.0], [-0.5, 0.25, 4.0]])
    out = pointcloud2_to_xyz(cloud_from_xyz(xyz))
    assert out.dtype == np.float32 and out.shape == (2, 3)
    assert np.allclose(out, xyz)


def test_pointcloud2_row_padding_and_nan_rows_removed():
    xyz = np.array([[1, 0, 0], [0, 1, 0], [np.nan, 0, 0], [0, 0, 1], [0, np.inf, 0], [2, 2, 2]], dtype=np.float32)
    out = pointcloud2_to_xyz(cloud_from_xyz(xyz, height=2, row_pad=7))   # 3 points per row, 7 pad bytes per row
    assert np.allclose(out, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 2, 2]])


def test_pointcloud2_float64_fields():
    rec = np.zeros(2, dtype={"names": ["x", "y", "z"], "formats": ["<f8"] * 3, "offsets": [0, 8, 16], "itemsize": 24})
    rec["x"], rec["y"], rec["z"] = [0.1, 1e3], [0.2, -2.0], [0.3, 4.0]
    fields = [PointField("x", 0, 8), PointField("y", 8, 8), PointField("z", 16, 8)]
    msg = PointCloud2(1, 2, fields, 24, 48, rec.tobytes())
    out = pointcloud2_to_xyz(msg)
    assert out.dtype == np.float32 and np.allclose(out, [[0.1, 0.2, 0.3], [1e3, -2.0, 4.0]], rtol=1e-6)


def test_pointcloud2_big_endian():
    rec = np.zeros(1, dtype={"names": ["x", "y", "z"], "formats": [">f4"] * 3, "offsets": [0, 4, 8], "itemsize": 12})
    rec["x"], rec["y"], rec["z"] = 1.0, 2.0, 3.0
    msg = PointCloud2(1, 1, FIELDS[:3], 12, 12, rec.tobytes(), is_bigendian=True)
    assert np.allclose(pointcloud2_to_xyz(msg), [[1, 2, 3]])


def test_pointcloud2_empty_cloud():
    assert pointcloud2_to_xyz(PointCloud2(0, 0, FIELDS, 32, 0, b"")).shape == (0, 3)


MALFORMED = {
    "missing z": lambda c: replace(c, fields=[f for f in c.fields if f.name != "z"]),
    "int8 x": lambda c: replace(c, fields=[replace(c.fields[0], datatype=1)] + c.fields[1:]),
    "count 2": lambda c: replace(c, fields=[replace(c.fields[0], count=2)] + c.fields[1:]),
    "field past point_step": lambda c: replace(c, fields=[replace(c.fields[0], offset=30)] + c.fields[1:]),
    "row_step too small": lambda c: replace(c, row_step=10),
    "buffer too short": lambda c: replace(c, data=c.data[:-1]),
    "negative height": lambda c: replace(c, height=-1),
    "non-integer width": lambda c: replace(c, width=2.5),
    "data not bytes": lambda c: replace(c, data=[1, 2, 3]),
    "no fields": lambda c: replace(c, fields=None),
}


@pytest.mark.parametrize("name", sorted(MALFORMED))
def test_pointcloud2_malformed_raises_value_error(name):
    with pytest.raises(ValueError):
        pointcloud2_to_xyz(MALFORMED[name](cloud_from_xyz([[1, 2, 3], [4, 5, 6]])))


def test_pointcloud2_not_a_message_raises_value_error():
    with pytest.raises(ValueError):
        pointcloud2_to_xyz(object())


# ------------------------------------------------------------------ extrinsics
def test_lidar_to_body_matches_analytic_rotation():
    theta = 0.3
    ext = yaw_pose(1.0, 2.0, 3.0, theta)
    p = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, -0.4, 0.9]])
    c, s = math.cos(theta), math.sin(theta)
    expected = np.stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1], p[:, 2]], axis=1) + [1.0, 2.0, 3.0]
    out = lidar_to_body(p, ext)
    assert out.dtype == np.float32 and np.allclose(out, expected, atol=1e-6)


def test_lidar_to_body_quarter_turn_and_empty_and_bad_shape():
    out = lidar_to_body(np.array([[1.0, 0, 0]]), yaw_pose(1, 2, 3, math.pi / 2))
    assert np.allclose(out, [[1, 3, 3]], atol=1e-6)
    assert lidar_to_body(np.zeros((0, 3)), Pose((0, 0, 0))).shape == (0, 3)
    with pytest.raises(ValueError):
        lidar_to_body(np.zeros((3, 2)), Pose((0, 0, 0)))


# ------------------------------------------------------------------ LivoxSource
class FakeNode:
    def __init__(self):
        self.subs = []

    def create_subscription(self, msg_type, topic, callback, qos):
        self.subs.append((msg_type, topic, callback, qos))


def test_livox_subscribes_and_converts_on_arrival():
    node = FakeNode()
    t = [10.0]
    src = LivoxSource(yaw_pose(0, 0, 0.5, math.pi / 2), node=node, clock=lambda: t[0])
    msg_type, topic, callback, _ = node.subs[0]
    assert msg_type == POINTCLOUD2 and topic == "/livox/lidar"
    assert src.latest() is None and src.age(11.0) == math.inf
    callback(cloud_from_xyz([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    frame = src.latest()
    assert isinstance(frame, LidarFrame) and frame.stamp == 10.0
    assert np.allclose(frame.points, [[0, 1, 0.5], [-1, 0, 0.5]], atol=1e-6)   # yaw 90 deg, lifted 0.5 m
    t[0] = 12.5
    assert src.age(12.5) == pytest.approx(2.5)


def test_livox_subsamples_to_cap_reproducibly_and_keeps_small_clouds_whole():
    xyz = np.stack([np.arange(100.0), np.zeros(100), np.zeros(100)], axis=1)
    node_a, node_b = FakeNode(), FakeNode()
    a = LivoxSource(Pose((0, 0, 0)), node=node_a, max_points=10, seed=3)
    b = LivoxSource(Pose((0, 0, 0)), node=node_b, max_points=10, seed=3)
    node_a.subs[0][2](cloud_from_xyz(xyz))
    node_b.subs[0][2](cloud_from_xyz(xyz))
    pa, pb = a.latest().points, b.latest().points
    assert pa.shape == (10, 3) and np.array_equal(pa, pb)
    assert len(set(pa[:, 0].tolist())) == 10 and set(pa[:, 0].tolist()) <= set(range(100))
    node_a.subs[0][2](cloud_from_xyz(xyz[:5]))
    assert a.latest().points.shape == (5, 3)


def test_livox_malformed_message_is_counted_and_previous_scan_kept():
    node = FakeNode()
    src = LivoxSource(Pose((0, 0, 0)), node=node, clock=lambda: 1.0)
    cb = node.subs[0][2]
    cb(object())
    assert src.latest() is None and src.dropped == 1
    cb(cloud_from_xyz([[1.0, 2.0, 3.0]]))
    cb(object())
    assert src.dropped == 2 and src.latest().points.shape == (1, 3)


@pytest.mark.parametrize("bad", [0, -1, 2.0, True])
def test_livox_max_points_validation(bad):
    with pytest.raises(ValueError):
        LivoxSource(Pose((0, 0, 0)), node=FakeNode(), max_points=bad)


# ------------------------------------------------------------------ DLIO
ROS_YAW90 = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))


def odom(pos=(0.0, 0.0, 0.0), q=(0.0, 0.0, 0.0, 1.0), cov=None):
    """nav_msgs/Odometry-like object; `q` is ROS order (x, y, z, w); `cov` is 36 values (default: zeros)."""
    c = [0.0] * 36 if cov is None else list(cov)
    return SimpleNamespace(pose=SimpleNamespace(
        pose=SimpleNamespace(position=SimpleNamespace(x=pos[0], y=pos[1], z=pos[2]),
                             orientation=SimpleNamespace(x=q[0], y=q[1], z=q[2], w=q[3])),
        covariance=c))


def cov_with_diag(diag, junk=9.0):
    """36 values with `diag` on the diagonal positions 0, 7, ... and a large junk value everywhere else."""
    c = [junk] * 36
    for i, v in enumerate(diag):
        c[7 * i] = v
    return c


SMALL = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)


def test_odometry_quaternion_is_converted_from_ros_order():
    est = odometry_to_estimate(odom(pos=(1.0, 2.0, 0.5), q=ROS_YAW90, cov=cov_with_diag(SMALL)), stamp=3.0)
    assert est.stamp == 3.0 and est.source == "dlio" and est.healthy
    assert est.pose.position == (1.0, 2.0, 0.5)
    assert np.allclose(est.pose.quat, (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)), atol=1e-9)
    assert est.pose.yaw() == pytest.approx(math.pi / 2)


def test_odometry_quaternion_is_normalised():
    est = odometry_to_estimate(odom(q=(0.0, 0.0, 0.0, 2.0), cov=cov_with_diag(SMALL)), stamp=0.0)
    assert est.pose.quat == (1.0, 0.0, 0.0, 0.0) and est.healthy


def test_covariance_diagonal_is_extracted():
    est = odometry_to_estimate(odom(cov=cov_with_diag(SMALL)), stamp=0.0)
    assert est.variance == pytest.approx(SMALL)


def test_health_depends_on_position_sigma_only():
    ok = odometry_to_estimate(odom(cov=cov_with_diag((0.16, 0.16, 0.16, 9.0, 9.0, 9.0))), stamp=0.0)
    assert ok.healthy                                              # angular sigma is not gated
    assert not odometry_to_estimate(odom(cov=cov_with_diag((0.36, 0.0, 0.0, 0.0, 0.0, 0.0))), 0.0).healthy
    assert not odometry_to_estimate(odom(cov=cov_with_diag((0.0, 0.0, 0.36, 0.0, 0.0, 0.0))), 0.0).healthy
    assert odometry_to_estimate(odom(cov=cov_with_diag((0.36, 0.0, 0.0, 0.0, 0.0, 0.0))), 0.0, max_sigma_m=0.7).healthy


@pytest.mark.parametrize("make", [
    lambda: odom(cov=cov_with_diag((0.01, 0.01, 0.01, -1.0, 0.01, 0.01))),   # negative variance
    lambda: odom(cov=cov_with_diag((math.nan, 0.01, 0.01, 0.01, 0.01, 0.01))),
    lambda: odom(cov=cov_with_diag((0.01, math.inf, 0.01, 0.01, 0.01, 0.01))),
    lambda: odom(cov=[0.01] * 35),                                          # wrong size
    lambda: odom(pos=(math.nan, 0.0, 0.0), cov=cov_with_diag(SMALL)),
    lambda: odom(q=(0.0, 0.0, 0.0, 0.0), cov=cov_with_diag(SMALL)),         # zero quaternion
    lambda: odom(q=(math.inf, 0.0, 0.0, 1.0), cov=cov_with_diag(SMALL)),
    lambda: SimpleNamespace(),                                              # not an odometry message
    lambda: object(),
    lambda: odom(cov=[math.nan] * 36),
])
def test_unusable_odometry_is_unhealthy_and_never_raises(make):
    est = odometry_to_estimate(make(), stamp=5.0)
    assert est.stamp == 5.0 and not est.healthy
    assert all(v >= 0 and math.isfinite(v) for v in est.variance)


def test_unhealthy_pose_fallbacks():
    bad_pos = odometry_to_estimate(odom(pos=(math.nan, 0.0, 0.0), cov=cov_with_diag(SMALL)), 0.0)
    assert bad_pos.pose.quat == (1.0, 0.0, 0.0, 0.0) and bad_pos.variance == (UNKNOWN_VARIANCE,) * 6
    neg = odometry_to_estimate(odom(q=ROS_YAW90, cov=cov_with_diag((0.01, 0.01, 0.01, -1.0, 0.01, 0.01))), 0.0)
    assert neg.pose.yaw() == pytest.approx(math.pi / 2) and neg.variance == (UNKNOWN_VARIANCE,) * 6


def test_odometry_max_sigma_validation():
    with pytest.raises(ValueError):
        odometry_to_estimate(odom(), 0.0, max_sigma_m=0.0)


def test_dlio_source_newest_wins_and_staleness():
    node = FakeNode()
    t = [5.0]
    src = DlioSource(node=node, clock=lambda: t[0], max_age_s=0.5)
    msg_type, topic, callback, _ = node.subs[0]
    assert msg_type == "nav_msgs/msg/Odometry" and topic == "/dlio/odom_node/odom"
    assert src.estimate(5.0) is None
    callback(odom(pos=(1.0, 0.0, 0.0), cov=cov_with_diag(SMALL)))
    assert src.estimate(5.2).pose.position == (1.0, 0.0, 0.0)
    assert src.estimate(5.5) is not None                           # exactly max_age_s old: still fresh
    assert src.estimate(5.6) is None                               # stale: the consumer stops
    t[0] = 6.0
    callback(odom(pos=(2.0, 0.0, 0.0), cov=cov_with_diag(SMALL)))
    assert src.estimate(6.1).pose.position == (2.0, 0.0, 0.0)


def test_dlio_source_passes_unhealthy_estimates_through_and_survives_garbage():
    node = FakeNode()
    src = DlioSource(node=node, clock=lambda: 1.0)
    node.subs[0][2](object())
    est = src.estimate(1.0)
    assert est is not None and not est.healthy


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan])
def test_dlio_source_max_age_validation(bad):
    with pytest.raises(ValueError):
        DlioSource(node=FakeNode(), max_age_s=bad)


# ------------------------------------------------------------------ imports
def test_importing_adapters_loads_neither_realsense_nor_ros():
    code = ("import sys\n"
            "import humanoid_arc_bme.onboard.sensors.realsense, humanoid_arc_bme.onboard.sensors.livox, "
            "humanoid_arc_bme.onboard.sensors.dlio\n"
            "print([m for m in ('pyrealsense2', 'rclpy') if m in sys.modules])\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
