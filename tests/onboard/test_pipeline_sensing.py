import math
import time

import numpy as np
import pytest

from humanoid_arc_bme.onboard.pipeline.interfaces import DepthFrame, Intrinsics, Pose
from humanoid_arc_bme.onboard.pipeline.perception.sensing import (
    DepthNoise, apply_depth_noise, camera_pose_from_mount, depth_to_points, dump_extrinsics, load_extrinsics,
    matrix_to_quat, quat_to_matrix,
)

K = Intrinsics(fx=300.0, fy=300.0, cx=159.5, cy=119.5, width=320, height=240)


def frame(depth, pose=Pose((0, 0, 0)), k=K):
    return DepthFrame(0.0, np.asarray(depth, dtype=np.float32), k, pose)


def cam_to_body_identityish(xyz=(0, 0, 0)):
    return camera_pose_from_mount(xyz, 0.0)


def test_wall_at_2m():
    off = (0.1, 0.05, 1.2)
    pts = depth_to_points(frame(np.full((240, 320), 2.0), camera_pose_from_mount(off, 0.0)))
    assert pts.shape == (240 * 320, 3) and pts.dtype == np.float32
    assert np.allclose(pts[:, 0], 2.0 + off[0], atol=1e-5)


def test_principal_ray_is_optical_axis():
    k = Intrinsics(100.0, 100.0, 50.0, 40.0, 101, 81)
    d = np.full((81, 101), np.nan)
    d[40, 50] = 3.0  # pixel (u=50, v=40) is the principal point
    pts = depth_to_points(frame(d, camera_pose_from_mount((0, 0, 0), 0.0), k))
    assert np.allclose(pts, [[3.0, 0.0, 0.0]], atol=1e-6)


def test_invalid_dropped():
    d = np.full((240, 320), 1.0)
    d[0, 0] = np.nan; d[0, 1] = -np.inf; d[0, 2] = np.inf; d[0, 3] = 0.0; d[0, 4] = -1.0
    d[0, 5] = 0.05; d[0, 6] = 9.0
    pts = depth_to_points(frame(d), max_range=5.0, min_range=0.1)
    assert len(pts) == 240 * 320 - 7
    assert depth_to_points(frame(np.full((240, 320), np.nan))).shape == (0, 3)


def test_stride_and_validation():
    f = frame(np.full((240, 320), 1.0))
    assert len(depth_to_points(f, stride=2)) == 120 * 160
    for bad in (0, -1, 1.5):
        with pytest.raises(ValueError):
            depth_to_points(f, stride=bad)


def test_pitched_camera_independent_rotation():
    a = math.radians(30)
    pose = camera_pose_from_mount((0.2, 0.0, 0.5), a)
    # independent: optical->body-level, then pitch down about body y
    R0 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    assert np.allclose(pose.rotation_matrix(), Ry @ R0, atol=1e-9)
    d = np.full((240, 320), 1.5)
    pts = depth_to_points(frame(d, pose)).astype(np.float64)
    u, v = np.meshgrid(np.arange(320), np.arange(240))
    cam = np.stack([(u - K.cx) * 1.5 / K.fx, (v - K.cy) * 1.5 / K.fy, np.full(u.shape, 1.5)], -1).reshape(-1, 3)
    exp = cam @ (Ry @ R0).T + np.array([0.2, 0, 0.5])
    assert np.allclose(pts, exp, atol=1e-4)


def test_mount_axes():
    xyz = (0.3, -0.1, 0.9)
    p0 = camera_pose_from_mount(xyz, 0.0)
    assert np.allclose(p0.transform_point((0, 0, 1)), (1.3, -0.1, 0.9), atol=1e-9)
    assert np.allclose(p0.transform_point((1, 0, 0)), (0.3, -1.1, 0.9), atol=1e-9)  # optical right = body -y
    assert np.allclose(p0.transform_point((0, 1, 0)), (0.3, -0.1, -0.1), atol=1e-9)  # optical down = body -z
    p90 = camera_pose_from_mount(xyz, math.pi / 2)
    assert np.allclose(p90.transform_point((0, 0, 1)), (0.3, -0.1, -0.1), atol=1e-9)
    py = camera_pose_from_mount((0, 0, 0), 0.0, yaw_rad=math.pi / 2)
    assert np.allclose(py.transform_point((0, 0, 1)), (0, 1, 0), atol=1e-9)  # yaw left


def _stats(z, n=200, seed=0):
    d = np.full((n, n), z, dtype=np.float32)
    return apply_depth_noise(d, DepthNoise(p_drop=0.0, p_edge=0.0), np.random.default_rng(seed)) - d


@pytest.mark.parametrize("z", [0.5, 4.0])
def test_noise_statistics(z):
    cfg = DepthNoise()
    err = _stats(z).astype(np.float64).ravel()
    sigma = cfg.sigma0 + cfg.sigma_k * z * z
    assert abs(err.mean()) < 0.05 * sigma
    assert err.std() == pytest.approx(sigma, rel=0.1)


def test_dropout_and_edges():
    cfg = DepthNoise(p_drop=0.1, p_edge=0.6)
    d = np.full((200, 200), 1.0, dtype=np.float32)
    d[:, 100:] = 3.0  # step: columns 99 and 100 are edge pixels
    out = apply_depth_noise(d, cfg, np.random.default_rng(1))
    nan = np.isnan(out)
    edge_frac = nan[:, 99:101].mean()
    flat_frac = np.concatenate([nan[:, :98].ravel(), nan[:, 102:].ravel()]).mean()
    assert flat_frac == pytest.approx(0.1, abs=0.02)
    assert edge_frac == pytest.approx(0.6, abs=0.08)
    assert edge_frac > flat_frac


def test_noise_invalid_determinism_no_mutation():
    d = np.full((50, 50), 2.0, dtype=np.float32)
    d[0, 0] = np.nan; d[0, 1] = np.inf; d[0, 2] = -np.inf; d[0, 3] = 0.0
    ref = d.copy()
    cfg = DepthNoise(quant_step=0.01)
    a = apply_depth_noise(d, cfg, np.random.default_rng(5))
    b = apply_depth_noise(d, cfg, np.random.default_rng(5))
    assert np.array_equal(a, b, equal_nan=True)
    assert np.array_equal(d, ref, equal_nan=True)
    assert a is not d and a.dtype == d.dtype
    assert np.isnan(a[0, 0]) and a[0, 1] == np.inf and a[0, 2] == -np.inf and a[0, 3] == 0.0
    ok = a[np.isfinite(a)]
    assert np.allclose(ok / 0.01, np.round(ok / 0.01), atol=1e-3)
    with pytest.raises(ValueError):
        DepthNoise(p_drop=1.5)


GOOD = {"cameras": {"neck_forward": {"xyz": [0.1, 0.0, 0.5], "quat_wxyz": [0.5, -0.5, 0.5, -0.5]},
                    "wrist_down": {"xyz": [0.0, 0.1, 0.2], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}}}


def test_extrinsics_roundtrip():
    poses = load_extrinsics(GOOD)
    assert set(poses) == {"neck_forward", "wrist_down"} and isinstance(poses["wrist_down"], Pose)
    assert load_extrinsics(dump_extrinsics(poses)) == poses
    mount = {"a": camera_pose_from_mount((0.1, 0.2, 0.3), 0.37, 0.2)}
    assert load_extrinsics(dump_extrinsics(mount)) == mount


def _mut(f):
    import copy
    d = copy.deepcopy(GOOD)
    f(d)
    return d


@pytest.mark.parametrize("bad,field", [
    (_mut(lambda d: d.update(extra=1)), "extra"),
    ({}, "cameras"),
    (_mut(lambda d: d["cameras"]["neck_forward"].update(foo=1)), "foo"),
    (_mut(lambda d: d["cameras"]["neck_forward"].pop("xyz")), "neck_forward.xyz"),
    (_mut(lambda d: d["cameras"]["wrist_down"].pop("quat_wxyz")), "wrist_down.quat_wxyz"),
    (_mut(lambda d: d["cameras"]["neck_forward"].update(xyz=[0, 0])), "neck_forward.xyz"),
    (_mut(lambda d: d["cameras"]["neck_forward"].update(quat_wxyz=[1, 0, 0])), "neck_forward.quat_wxyz"),
    (_mut(lambda d: d["cameras"]["neck_forward"].update(quat_wxyz=[2, 0, 0, 0])), "neck_forward.quat_wxyz"),
    (_mut(lambda d: d["cameras"]["neck_forward"].update(xyz=[0, "a", 0])), "neck_forward.xyz"),
    (_mut(lambda d: d["cameras"]["neck_forward"].update(xyz=[0, float("nan"), 0])), "neck_forward.xyz"),
    (_mut(lambda d: d["cameras"].update(x=5)), "cameras.x"),
])
def test_extrinsics_rejections(bad, field):
    with pytest.raises(ValueError, match=field):
        load_extrinsics(bad)


def test_speed_640x480():
    k = Intrinsics(380.0, 380.0, 319.5, 239.5, 640, 480)
    d = np.random.default_rng(0).uniform(0.3, 6.0, (480, 640)).astype(np.float32)
    f = frame(d, camera_pose_from_mount((0, 0, 1), 0.4), k)
    t = time.perf_counter()
    pts = depth_to_points(f)
    assert time.perf_counter() - t < 1.0
    assert pts.shape[1] == 3


def test_single_rotation_implementation():
    from humanoid_arc_bme.onboard.pipeline import interfaces
    from humanoid_arc_bme.onboard.pipeline.perception import sensing

    assert sensing.quat_to_matrix is interfaces.quat_to_matrix  # re-export, not a copy
    q = camera_pose_from_mount((0, 0, 0), 0.3, 0.2).quat
    assert np.allclose(Pose((0, 0, 0), q).rotation_matrix(), quat_to_matrix(q))
    assert np.allclose(quat_to_matrix(matrix_to_quat(quat_to_matrix(q))), quat_to_matrix(q), atol=1e-9)
