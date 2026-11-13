"""Scene assembly: analytic back-projection through a pitched camera, ordering, point thresholds, NaN handling, colour."""

import math
import subprocess
import sys

import numpy as np
import pytest

from humanoid_arc_bme.onboard.perception.colors import erode_mask
from humanoid_arc_bme.onboard.perception.scene import SceneObject, build_scene, scene_summary
from humanoid_arc_bme.onboard.pipeline.interfaces import DepthFrame, Detection, Intrinsics, Pose

K = Intrinsics(fx=200.0, fy=200.0, cx=159.5, cy=119.5, width=320, height=240)
PITCH = 0.3
MOUNT = (0.1, 0.05, 1.2)
CAM_TO_BODY = Pose(MOUNT, (math.cos(PITCH / 2), 0.0, math.sin(PITCH / 2), 0.0))


def box_mask(u0, u1, v0, v1, h=240, w=320):
    m = np.zeros((h, w), dtype=bool)
    m[v0:v1, u0:u1] = True
    return m


def frame(depth, cam_to_body=CAM_TO_BODY):
    return DepthFrame(0.0, np.asarray(depth, dtype=np.float32), K, cam_to_body)


def det(label, mask, conf=0.9, box=None):
    return Detection(label, conf, mask, box)


def test_plane_box_matches_analytic_position():
    # Fronto-parallel plane at z = 2 m. The box is symmetric about the principal point, so the median camera point is
    # exactly (0, 0, 2). Pitch about body y by PITCH: p_body = (2 sin p, 0, 2 cos p) + mount.
    depth = np.full((240, 320), 2.0)
    m = box_mask(140, 180, 100, 140)
    (obj,) = build_scene([det("cup", m)], None, frame(depth))
    expected = (2.0 * math.sin(PITCH) + MOUNT[0], MOUNT[1], 2.0 * math.cos(PITCH) + MOUNT[2])
    assert np.allclose(obj.position_body, expected, atol=1e-6)
    assert obj.range_m == pytest.approx(math.hypot(expected[0], expected[1]), abs=1e-9)
    assert obj.n_points == 38 * 38  # 40 x 40 box, eroded by one pixel on each side
    assert obj.label == "cup" and obj.color == "unknown" and obj.box is None


def test_identity_mount_position_is_camera_point():
    depth = np.full((240, 320), 1.5)
    m = box_mask(150, 170, 110, 130)
    (obj,) = build_scene([det("bottle", m)], None, frame(depth, Pose((0.0, 0.0, 0.0))))
    assert np.allclose(obj.position_body, (0.0, 0.0, 1.5), atol=1e-6)  # identity mount: body == camera coordinates


def test_ordering_by_range_nearest_first():
    depth = np.full((240, 320), 2.0)
    depth[:, 200:] = 1.0
    near = box_mask(210, 230, 100, 140)
    far = box_mask(60, 80, 100, 140)
    scene = build_scene(
        [det("far", far), det("near", near)], None, frame(depth, Pose((0.0, 0.0, 0.0))), max_range_m=3.0
    )
    assert [o.label for o in scene] == ["near", "far"]
    assert scene[0].range_m < scene[1].range_m


def test_min_points_rejection():
    depth = np.full((240, 320), 2.0)
    small = box_mask(100, 104, 100, 104)  # 4x4 -> 2x2 after erosion
    assert build_scene([det("tiny", small)], None, frame(depth), min_points=30) == []
    assert len(build_scene([det("tiny", small)], None, frame(depth), min_points=4)) == 1


def test_nan_and_nonfinite_depth_excluded_from_points():
    depth = np.full((240, 320), 2.0)
    m = box_mask(140, 180, 100, 140)
    depth[100:140, 140:150] = np.nan  # part of the box has no depth
    depth[100:105, 170:180] = np.inf
    f = frame(depth, Pose((0.0, 0.0, 0.0)))
    (obj,) = build_scene([det("cup", m)], None, f)
    # Expected points: eroded box, finite depth only. Constant depth 2 m, so x and y are the median pixel back-projected.
    valid = erode_mask(m) & np.isfinite(depth)
    ys, xs = np.nonzero(valid)
    expected_xy = (np.median((xs - K.cx) * 2.0 / K.fx), np.median((ys - K.cy) * 2.0 / K.fy))
    assert obj.n_points == valid.sum()
    assert np.isfinite(obj.position_body).all()
    assert obj.position_body[:2] == pytest.approx(expected_xy, abs=1e-9)
    assert obj.position_body[2] == pytest.approx(2.0)


def test_fully_invalid_depth_yields_nothing():
    depth = np.full((240, 320), np.nan)
    assert build_scene([det("cup", box_mask(140, 180, 100, 140))], None, frame(depth)) == []


def test_range_filter_drops_far_objects():
    # Pitch of 90 degrees: camera forward (z) becomes body +x, so a 4 m object is 4.1 m horizontally from the body.
    c = math.sqrt(0.5)
    forward = Pose((0.1, 0.05, 1.2), (c, 0.0, c, 0.0))
    depth = np.full((240, 320), 4.0)
    m = box_mask(140, 180, 100, 140)
    assert build_scene([det("far", m)], None, frame(depth, forward), max_range_m=3.0) == []
    (obj,) = build_scene([det("far", m)], None, frame(depth, forward), max_range_m=5.0)
    assert obj.range_m == pytest.approx(math.hypot(4.1, 0.05), abs=1e-6)


def test_min_conf_and_missing_mask_skipped():
    depth = np.full((240, 320), 2.0)
    m = box_mask(140, 180, 100, 140)
    f = frame(depth)
    assert build_scene([det("cup", m, conf=0.4)], None, f, min_conf=0.5) == []
    assert build_scene([Detection("cup", 0.9, None)], None, f) == []
    assert build_scene([det("cup", np.zeros((10, 10), dtype=bool))], None, f) == []


def test_colour_from_rgb_mask():
    depth = np.full((240, 320), 2.0)
    m = box_mask(140, 180, 100, 140)
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    rgb[:] = (30, 60, 220)
    (obj,) = build_scene([det("cup", m)], rgb, frame(depth))
    assert obj.color == "blue"


def test_colour_with_rgb_at_other_resolution():
    depth = np.full((240, 320), 2.0)
    m = box_mask(140, 180, 100, 140)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[:] = (220, 20, 20)
    (obj,) = build_scene([det("cup", m)], rgb, frame(depth))
    assert obj.color == "red"


def test_build_scene_rejects_bad_parameters():
    depth = np.full((240, 320), 2.0)
    f = frame(depth)
    with pytest.raises(ValueError):
        build_scene([], None, f, min_points=0)
    with pytest.raises(ValueError):
        build_scene([], None, f, max_range_m=float("nan"))


def test_scene_summary_formats():
    def obj(label, color, r):
        return SceneObject(label, color, 0.9, (r, 0.0, 0.0), 100, None, r)

    assert scene_summary([]) == "nothing"
    assert scene_summary([obj("cup", "red", 0.8), obj("bottle", "blue", 1.2)]) == "red cup 0.8 m, blue bottle 1.2 m"
    assert scene_summary([obj("cup", "red", 0.8), obj("cup", "red", 1.0)]) == "2x red cup 0.8 m"
    assert scene_summary([obj("cup", "unknown", 0.5)]) == "cup 0.5 m"


def test_no_heavy_imports():
    code = (
        "import sys, humanoid_arc_bme.onboard.perception.scene as s; "
        "bad = [m for m in ('torch', 'isaaclab', 'rclpy', 'onnxruntime', 'cv2') if m in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
