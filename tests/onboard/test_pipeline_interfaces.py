"""W1 gate: pipeline interfaces validate inputs, stubs satisfy the Protocols, no Isaac/torch/ROS import."""

import math
import subprocess
import sys

import numpy as np
import pytest

from humanoid_arc_bme.onboard.pipeline.interfaces import (
    DepthFrame,
    DepthSource,
    Detection,
    GraspCandidate,
    Intrinsics,
    Localizer,
    LocalizerEstimate,
    ObjectAnchor,
    ObjectMemory,
    Pose,
    SemanticsModel,
    yaw_pose,
)
from humanoid_arc_bme.onboard.pipeline.stubs import (
    GroundTruthSemantics,
    LatestObjectMemory,
    NoisedLocalizer,
    NullSemantics,
    TruthLocalizer,
    _with_yaw,
)

K = Intrinsics(100.0, 100.0, 4.0, 3.0, 8, 6)


def frame(depth=None, cam_to_body=None, stamp=1.0):
    d = np.full((6, 8), 2.0, dtype=np.float32) if depth is None else depth
    return DepthFrame(stamp, d, K, cam_to_body or Pose((0.0, 0.0, 0.0)))


def test_no_heavy_imports():
    code = "import sys, humanoid_arc_bme.onboard.pipeline.stubs; bad=[m for m in ('torch','isaaclab','rclpy') if m in sys.modules]; sys.exit(1 if bad else 0)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


@pytest.mark.parametrize("q", [(0, 0, 0, 0), (1, 1, 0, 0), (math.nan, 0, 0, 0), (1, 0, 0)])
def test_pose_rejects_bad_quat(q):
    with pytest.raises(ValueError):
        Pose((0, 0, 0), q)


def test_pose_rejects_nonfinite_position():
    with pytest.raises(ValueError):
        Pose((0, math.inf, 0))


def test_yaw_roundtrip_and_transform():
    p = yaw_pose(1.0, 2.0, 0.0, math.pi / 2)
    assert p.yaw() == pytest.approx(math.pi / 2)
    x, y, z = p.transform_point((1.0, 0.0, 0.5))  # +x local -> +y map after a 90 deg yaw
    assert (x, y, z) == pytest.approx((1.0, 3.0, 0.5))


def test_intrinsics_and_frame_validation():
    with pytest.raises(ValueError):
        Intrinsics(0, 100, 1, 1, 8, 6)
    with pytest.raises(ValueError):
        Intrinsics(100, 100, 1, 1, 0, 6)
    for cx, cy in ((math.nan, 1), (1, math.nan), (math.inf, 1), (1, -math.inf)):
        with pytest.raises(ValueError):
            Intrinsics(100, 100, cx, cy, 8, 6)
    with pytest.raises(ValueError):
        frame(np.zeros((5, 8), dtype=np.float32))  # shape mismatch
    with pytest.raises(ValueError):
        frame(np.zeros((6, 8), dtype=np.int32))  # not float
    with pytest.raises(ValueError):
        DepthFrame(math.nan, np.zeros((6, 8), dtype=np.float32), K, Pose((0, 0, 0)))


def test_value_object_validation():
    with pytest.raises(ValueError):
        LocalizerEstimate(0.0, Pose((0, 0, 0)), (0, 0, 0, 0, 0, -1.0))
    for bad in ("yes", 1, 0, None):
        with pytest.raises(ValueError):
            LocalizerEstimate(0.0, Pose((0, 0, 0)), (0,) * 6, healthy=bad)
    with pytest.raises(ValueError):
        Detection("", 0.5)
    with pytest.raises(ValueError):
        Detection("cup", 1.5)
    with pytest.raises(ValueError):
        Detection("cup", 0.5, mask=np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError):
        ObjectAnchor("cup", (0, 0, math.nan), 0.5, 0.0)
    with pytest.raises(ValueError):
        GraspCandidate(Pose((0, 0, 0)), -0.01, 0.5)
    with pytest.raises(ValueError):
        GraspCandidate(Pose((0, 0, 0)), 0.05, 0.5, kind="hook")


def test_stubs_satisfy_protocols():
    truth = TruthLocalizer(lambda t: Pose((t, 0, 0)))
    assert isinstance(truth, Localizer)
    assert isinstance(NoisedLocalizer(truth), Localizer)
    assert isinstance(NullSemantics(), SemanticsModel)
    assert isinstance(GroundTruthSemantics(lambda r, d: (np.zeros((6, 8), int), {})), SemanticsModel)
    assert isinstance(LatestObjectMemory(), ObjectMemory)

    class Src:
        def latest(self):
            return None

    assert isinstance(Src(), DepthSource)


def test_truth_localizer_none_passthrough():
    assert TruthLocalizer(lambda t: None).estimate(0.0) is None


def test_noised_localizer_is_deterministic_and_bounded():
    def run(seed):
        loc = NoisedLocalizer(TruthLocalizer(lambda t: yaw_pose(0, 0, 0, 0)), drift_xy=0.01, jitter_xy=0.005, seed=seed)
        return [loc.estimate(0.1 * i) for i in range(200)]

    a, b, c = run(1), run(1), run(2)
    assert [e.pose.position for e in a] == [e.pose.position for e in b]
    assert [e.pose.position for e in a] != [e.pose.position for e in c]
    assert all(e.healthy for e in a)
    # variance grows with elapsed time (drift term) and matches the configured model
    assert a[-1].variance[0] == pytest.approx(0.005**2 + 0.01**2 * 19.9, rel=1e-6)
    assert a[-1].variance[0] > a[1].variance[0]


def test_noised_localizer_zero_noise_is_identity():
    loc = NoisedLocalizer(TruthLocalizer(lambda t: yaw_pose(1.0, 2.0, 0.0, 0.7)))
    e = loc.estimate(5.0)
    assert e.pose.position == (1.0, 2.0, 0.0)
    assert e.pose.yaw() == pytest.approx(0.7)


def test_noised_localizer_dropout_rate_and_validation():
    loc = NoisedLocalizer(TruthLocalizer(lambda t: Pose((0, 0, 0))), dropout=0.3, seed=3)
    n = sum(loc.estimate(0.1 * i) is None for i in range(2000))
    assert 0.25 < n / 2000 < 0.35
    for bad in (dict(dropout=1.5), dict(drift_xy=-1.0), dict(jitter_yaw=math.nan)):
        with pytest.raises(ValueError):
            NoisedLocalizer(TruthLocalizer(lambda t: Pose((0, 0, 0))), **bad)


def test_noised_localizer_non_monotonic_stamp_does_not_drift_backwards():
    loc = NoisedLocalizer(TruthLocalizer(lambda t: Pose((0, 0, 0))), drift_xy=1.0, seed=0)
    loc.estimate(10.0)
    e = loc.estimate(5.0)  # stamp went back: dt clamped to 0, no extra drift, no crash
    assert e is not None


def test_ground_truth_semantics_masks_and_boxes():
    ids = np.zeros((6, 8), int)
    ids[1:3, 2:5] = 7
    ids[4, 0] = 9  # id with no name is ignored
    det = GroundTruthSemantics(lambda r, d: (ids, {7: "cup"})).detect(None, frame())
    assert len(det) == 1 and det[0].label == "cup" and det[0].box == (2, 1, 5, 3)
    with pytest.raises(ValueError):
        GroundTruthSemantics(lambda r, d: (np.zeros((2, 2), int), {})).detect(None, frame())


def test_object_memory_lifts_to_map_frame():
    ids = np.zeros((6, 8), int)
    ids[3, 4] = 1  # pixel at the principal point: lies on the optical axis
    depth = np.full((6, 8), 2.0, dtype=np.float32)
    # Extrinsics here are a pure translation (no rotation), so the optical z axis stays the body z axis.
    f = frame(depth, cam_to_body=Pose((0.1, 0.0, 1.0)))
    det = GroundTruthSemantics(lambda r, d: (ids, {1: "cup"})).detect(None, f)
    mem = LatestObjectMemory()
    mem.update(det, f, LocalizerEstimate(1.0, yaw_pose(5.0, 0.0, 0.0, 0.0), (0,) * 6))
    (a,) = mem.query("cup")
    assert a.position == pytest.approx((5.1, 0.0, 3.0))  # (0, 0, 2) cam + (0.1, 0, 1) extrinsics + (5, 0, 0) body
    assert mem.query("nothing") == []


def test_object_memory_ignores_unhealthy_pose_and_invalid_depth():
    ids = np.zeros((6, 8), int)
    ids[3, 4] = 1
    bad = np.full((6, 8), np.nan, dtype=np.float32)
    f_bad, f_ok = frame(bad), frame()
    det = GroundTruthSemantics(lambda r, d: (ids, {1: "cup"})).detect(None, f_ok)
    mem = LatestObjectMemory()
    mem.update(det, f_ok, LocalizerEstimate(1.0, Pose((0, 0, 0)), (0,) * 6, healthy=False))
    mem.update(det, f_bad, LocalizerEstimate(1.0, Pose((0, 0, 0)), (0,) * 6))
    assert mem.query("cup") == []


def test_object_memory_rejects_time_mismatched_pose():
    ids = np.zeros((6, 8), int)
    ids[3, 4] = 1
    f = frame(stamp=1.0)
    det = GroundTruthSemantics(lambda r, d: (ids, {1: "cup"})).detect(None, f)
    est = lambda t: LocalizerEstimate(t, Pose((0, 0, 0)), (0,) * 6)
    mem = LatestObjectMemory()
    mem.update(det, f, est(1.5))  # 0.5 s apart > default 0.2
    mem.update(det, f, est(0.7))
    assert mem.query("cup") == []
    mem.update(det, f, est(1.1))
    assert len(mem.query("cup")) == 1
    loose = LatestObjectMemory(max_pose_age_s=1.0)
    loose.update(det, f, est(1.5))
    assert len(loose.query("cup")) == 1
    with pytest.raises(ValueError):
        LatestObjectMemory(max_pose_age_s=-1.0)


def test_with_yaw_non_planar_preserves_roll_pitch():
    q = (0.9, 0.2, 0.3, 0.1)
    n = math.sqrt(sum(c * c for c in q))
    base = Pose((1.0, 2.0, 3.0), tuple(c / n for c in q))
    new = _with_yaw(base, 4.0, 5.0, 6.0, base.yaw() + 0.7)
    assert new.position == (4.0, 5.0, 6.0)
    assert new.yaw() == pytest.approx(base.yaw() + 0.7)
    # undoing the yaw change recovers the original rotation: roll and pitch were preserved
    back = _with_yaw(new, 0.0, 0.0, 0.0, base.yaw())
    assert np.allclose(back.rotation_matrix(), base.rotation_matrix(), atol=1e-9)
    # the body z axis tilt (roll/pitch) is unchanged by a yaw change
    assert new.rotation_matrix()[2, 2] == pytest.approx(base.rotation_matrix()[2, 2])
