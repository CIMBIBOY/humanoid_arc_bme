"""Reference implementations of the pipeline interfaces for simulation and tests.

- `TruthLocalizer`, `NoisedLocalizer`: sim ground truth, and the same with drift/jitter/dropout (A3). Real DLIO
  replaces them behind the `Localizer` Protocol; the noise wrapper is how we test how much error the pipeline tolerates.
- `NullSemantics`, `GroundTruthSemantics`, `LatestObjectMemory`: DAVE PLACEHOLDERS. Deliberately minimal; the real
  semantics model and object memory replace them. Do not grow these.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import numpy as np

from humanoid_arc_bme.onboard.pipeline.perception.sensing import matrix_to_quat
from humanoid_arc_bme.onboard.pipeline.interfaces import DepthFrame, Detection, LocalizerEstimate, ObjectAnchor, Pose, yaw_pose


class TruthLocalizer:
    """Reports the simulator's true body pose with zero variance. `truth(stamp)` returns a Pose or None."""

    def __init__(self, truth: Callable[[float], Pose | None], source: str = "sim_truth"):
        self._truth = truth
        self._source = source

    def estimate(self, stamp: float) -> LocalizerEstimate | None:
        pose = self._truth(stamp)
        if pose is None:
            return None
        return LocalizerEstimate(stamp, pose, (0.0,) * 6, True, self._source)


class NoisedLocalizer:
    """Wraps a Localizer with the error a LIO estimator actually has: slow drift (random walk in x, y, yaw), per-call
    white jitter, and dropouts. Seeded and deterministic. `drift_xy` is m / sqrt(s), `drift_yaw` rad / sqrt(s),
    `jitter_xy` m, `jitter_yaw` rad (1-sigma), `dropout` probability per call of returning None.
    The reported variance is the model's own: jitter^2 plus drift^2 * elapsed time since the first call."""

    def __init__(self, base, drift_xy=0.0, drift_yaw=0.0, jitter_xy=0.0, jitter_yaw=0.0, dropout=0.0, seed=0):
        for n, v in (("drift_xy", drift_xy), ("drift_yaw", drift_yaw), ("jitter_xy", jitter_xy), ("jitter_yaw", jitter_yaw)):
            if not (math.isfinite(v) and v >= 0):
                raise ValueError(f"{n}: must be a finite number >= 0")
        if not (math.isfinite(dropout) and 0.0 <= dropout <= 1.0):
            raise ValueError("dropout: must be in [0, 1]")
        self._base = base
        self._p = (drift_xy, drift_yaw, jitter_xy, jitter_yaw, dropout)
        self._rng = random.Random(seed)
        self._off = [0.0, 0.0, 0.0]  # drift offset x, y, yaw
        self._t0: float | None = None
        self._last: float | None = None

    def estimate(self, stamp: float) -> LocalizerEstimate | None:
        drift_xy, drift_yaw, jit_xy, jit_yaw, dropout = self._p
        est = self._base.estimate(stamp)
        if est is None:
            return None
        if self._t0 is None:
            self._t0 = self._last = stamp
        dt = max(0.0, stamp - self._last)
        self._last = max(self._last, stamp)
        if dt > 0.0:
            s = math.sqrt(dt)
            self._off[0] += self._rng.gauss(0.0, drift_xy * s)
            self._off[1] += self._rng.gauss(0.0, drift_xy * s)
            self._off[2] += self._rng.gauss(0.0, drift_yaw * s)
        if dropout > 0.0 and self._rng.random() < dropout:
            return None
        x, y, z = est.pose.position
        yaw = est.pose.yaw()
        x += self._off[0] + self._rng.gauss(0.0, jit_xy)
        y += self._off[1] + self._rng.gauss(0.0, jit_xy)
        yaw += self._off[2] + self._rng.gauss(0.0, jit_yaw)
        elapsed = self._last - self._t0
        vxy = jit_xy**2 + drift_xy**2 * elapsed
        vyaw = jit_yaw**2 + drift_yaw**2 * elapsed
        base_var = est.variance
        var = (base_var[0] + vxy, base_var[1] + vxy, base_var[2], base_var[3], base_var[4], base_var[5] + vyaw)
        # Roll and pitch are kept from the base pose; only yaw is perturbed (planar-drift model).
        pose = _with_yaw(est.pose, x, y, z, yaw)
        return LocalizerEstimate(stamp, pose, var, est.healthy, f"noised({est.source})")


def _with_yaw(pose: Pose, x: float, y: float, z: float, yaw: float) -> Pose:
    """Replace x, y, z and the yaw of `pose`, keeping its roll and pitch (R = Rz(yaw) Ry(pitch) Rx(roll))."""
    w, qx, qy, qz = pose.quat
    if abs(qx) < 1e-9 and abs(qy) < 1e-9:
        return yaw_pose(x, y, z, yaw)
    R = pose.rotation_matrix()
    # Remove the old yaw (rotate about world z), then apply the new one: Rz(new - old) @ R.
    d = yaw - pose.yaw()
    c, s = math.cos(d), math.sin(d)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Pose((x, y, z), matrix_to_quat(Rz @ R))


# ---- DAVE placeholders ------------------------------------------------------------------------------------------


class NullSemantics:
    """Placeholder SemanticsModel: sees nothing."""

    def detect(self, rgb, depth: DepthFrame) -> list[Detection]:
        return []


class GroundTruthSemantics:
    """Placeholder SemanticsModel for simulation: turns the simulator's instance-segmentation map into detections.
    `seg(rgb, depth)` returns (ids, names): an (H, W) int array (0 / negative = background) and {id: label}."""

    def __init__(self, seg: Callable[[object, DepthFrame], tuple[np.ndarray, dict[int, str]]]):
        self._seg = seg

    def detect(self, rgb, depth: DepthFrame) -> list[Detection]:
        ids, names = self._seg(rgb, depth)
        ids = np.asarray(ids)
        if ids.shape != depth.depth.shape:
            raise ValueError("segmentation shape must match the depth frame")
        out = []
        for i in np.unique(ids):
            if i <= 0 or int(i) not in names:
                continue
            mask = ids == i
            ys, xs = np.nonzero(mask)
            out.append(Detection(names[int(i)], 1.0, mask, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)))
        return out


class LatestObjectMemory:
    """Placeholder ObjectMemory: newest anchor per label. Position = median valid depth over the mask, lifted
    camera -> body -> map. No tracking, no merging, no aging. An update whose body pose stamp differs from the depth stamp by more than
    `max_pose_age_s` is skipped (the pose would not match the view)."""

    def __init__(self, max_pose_age_s: float = 0.2):
        if not (math.isfinite(max_pose_age_s) and max_pose_age_s >= 0):
            raise ValueError("max_pose_age_s: must be a finite number >= 0")
        self._max_pose_age = float(max_pose_age_s)
        self._anchors: dict[str, list[ObjectAnchor]] = {}

    def update(self, detections: list[Detection], depth: DepthFrame, body_pose: LocalizerEstimate) -> None:
        if not body_pose.healthy or abs(body_pose.stamp - depth.stamp) > self._max_pose_age:
            return  # unhealthy or time-mismatched pose: lifting with it would misplace the object
        k = depth.intrinsics
        for det in detections:
            if det.mask is None:
                continue
            ys, xs = np.nonzero(det.mask)
            z = depth.depth[ys, xs]
            ok = np.isfinite(z) & (z > 0)
            if not ok.any():
                continue
            zc = float(np.median(z[ok]))
            u, v = float(np.median(xs[ok])), float(np.median(ys[ok]))
            p_cam = ((u - k.cx) * zc / k.fx, (v - k.cy) * zc / k.fy, zc)
            p_map = body_pose.pose.transform_point(depth.cam_to_body.transform_point(p_cam))
            self._anchors[det.label] = [ObjectAnchor(det.label, p_map, det.confidence, depth.stamp)]

    def query(self, label: str) -> list[ObjectAnchor]:
        return list(self._anchors.get(label, []))
