"""Pipeline interfaces: the seams between sensing, localization, semantics, grasping and the skill layer.

Stdlib + numpy only. No Isaac, torch or ROS imports here: simulator and robot adapters implement these Protocols
elsewhere. Conventions (same as api/contract.py): metres, radians, seconds; quaternions (w, x, y, z), unit length;
a `Pose` always names its frame in the owner's field, never implicitly.

Ownership: `SemanticsModel` and `ObjectMemory` belong to DAVE. Only the interface and a minimal placeholder exist
here (see stubs.py); do not grow them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from humanoid_arc_bme.onboard.geometry import quat_to_matrix  # noqa: F401 - re-exported: part of this module's API
from humanoid_arc_bme.onboard.validation import finite_vec, is_num  # noqa: F401 - re-exported: part of this module's API

QUAT_NORM_TOL = 1e-3
GRASP_KINDS = ("pinch", "power")


@dataclass(frozen=True)
class Pose:
    """Position (x, y, z) and unit quaternion (w, x, y, z). The frame is named by whoever holds the pose."""

    position: tuple[float, float, float]
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self):
        object.__setattr__(self, "position", finite_vec("position", self.position, 3))
        q = finite_vec("quat", self.quat, 4)
        if abs(math.sqrt(sum(c * c for c in q)) - 1.0) > QUAT_NORM_TOL:
            raise ValueError("quat: must be unit length")
        object.__setattr__(self, "quat", q)

    def yaw(self) -> float:
        w, x, y, z = self.quat
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def rotation_matrix(self) -> np.ndarray:
        """3x3 rotation matrix (float64) of the unit quaternion; the single implementation in the package."""
        return quat_to_matrix(self.quat)

    def transform_point(self, p) -> tuple[float, float, float]:
        """Map a point from this pose's local frame into the frame this pose is expressed in: R p + t."""
        px, py, pz = finite_vec("point", p, 3)
        w, x, y, z = self.quat
        # v' = v + 2w(u x v) + 2 u x (u x v), u = (x, y, z)
        cx, cy, cz = y * pz - z * py, z * px - x * pz, x * py - y * px
        dx, dy, dz = y * cz - z * cy, z * cx - x * cz, x * cy - y * cx
        t = self.position
        return (px + 2 * (w * cx + dx) + t[0], py + 2 * (w * cy + dy) + t[1], pz + 2 * (w * cz + dz) + t[2])


def yaw_pose(x: float, y: float, z: float, yaw: float) -> Pose:
    return Pose((x, y, z), (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)))


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics in pixels, OpenCV convention: the CENTRE of pixel (u, v) is at the integer coordinates
    (u, v), so a centred sensor has cx = (W-1)/2, cy = (H-1)/2. Sources using the pixel-edge convention (IsaacLab:
    cx = W/2) must subtract 0.5 before constructing this."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self):
        for n in ("fx", "fy"):
            v = getattr(self, n)
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"{n}: must be a positive finite number")
        for n in ("cx", "cy"):
            if not math.isfinite(getattr(self, n)):
                raise ValueError(f"{n}: must be finite")
        if not (isinstance(self.width, int) and isinstance(self.height, int) and self.width > 0 and self.height > 0):
            raise ValueError("width/height: must be positive ints")


@dataclass(frozen=True)
class DepthFrame:
    """One metric depth image. `depth` is (H, W) float32 metres in the camera optical frame (x right, y down,
    z forward); invalid pixels are NaN, <= 0 or -inf, far pixels +inf. `cam_to_body` maps camera optical points to
    the body frame (calibrated extrinsics, see config)."""

    stamp: float
    depth: np.ndarray
    intrinsics: Intrinsics
    cam_to_body: Pose
    camera: str = "depth"

    def __post_init__(self):
        if not math.isfinite(self.stamp):
            raise ValueError("stamp: must be finite")
        d = np.asarray(self.depth)
        if d.ndim != 2 or not np.issubdtype(d.dtype, np.floating):
            raise ValueError("depth: expected a 2-D float array")
        if d.shape != (self.intrinsics.height, self.intrinsics.width):
            raise ValueError(f"depth: shape {d.shape} does not match intrinsics {(self.intrinsics.height, self.intrinsics.width)}")
        object.__setattr__(self, "depth", d)


@dataclass(frozen=True)
class LocalizerEstimate:
    """Pose of the body in the map frame. `variance` is the diagonal of the covariance (x, y, z, roll, pitch, yaw),
    in m^2 and rad^2. `healthy` False means the source itself reports it cannot be trusted; consumers stop."""

    stamp: float
    pose: Pose
    variance: tuple[float, float, float, float, float, float]
    healthy: bool = True
    source: str = "unknown"

    def __post_init__(self):
        if not math.isfinite(self.stamp):
            raise ValueError("stamp: must be finite")
        if not isinstance(self.healthy, bool):
            raise ValueError("healthy: must be a bool")
        v = finite_vec("variance", self.variance, 6)
        if any(x < 0 for x in v):
            raise ValueError("variance: entries must be >= 0")
        object.__setattr__(self, "variance", v)


@dataclass(frozen=True)
class Detection:
    """One recognised object in one frame (owner: DAVE). `mask` is (H, W) bool in the image the detector saw."""

    label: str
    confidence: float
    mask: np.ndarray | None = field(default=None, compare=False)
    box: tuple[int, int, int, int] | None = None  # x0, y0, x1, y1 pixels

    def __post_init__(self):
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label: expected a non-empty string")
        if not (math.isfinite(self.confidence) and 0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence: must be in [0, 1]")
        if self.mask is not None and (np.asarray(self.mask).ndim != 2 or np.asarray(self.mask).dtype != bool):
            raise ValueError("mask: expected a 2-D bool array")


@dataclass(frozen=True)
class ObjectAnchor:
    """A remembered object position in the map frame (owner: DAVE)."""

    label: str
    position: tuple[float, float, float]
    confidence: float
    stamp: float

    def __post_init__(self):
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label: expected a non-empty string")
        object.__setattr__(self, "position", finite_vec("position", self.position, 3))
        if not (math.isfinite(self.confidence) and 0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence: must be in [0, 1]")
        if not math.isfinite(self.stamp):
            raise ValueError("stamp: must be finite")


@dataclass(frozen=True)
class GraspCandidate:
    """A grasp in the frame of the input point cloud: hand base pose, finger aperture (m), score in [0, 1]."""

    pose: Pose
    aperture: float
    score: float
    kind: str = "pinch"

    def __post_init__(self):
        if not (math.isfinite(self.aperture) and self.aperture >= 0):
            raise ValueError("aperture: must be a finite number >= 0")
        if not (math.isfinite(self.score) and 0.0 <= self.score <= 1.0):
            raise ValueError("score: must be in [0, 1]")
        if self.kind not in GRASP_KINDS:
            raise ValueError(f"kind: expected one of {GRASP_KINDS}")


@runtime_checkable
class DepthSource(Protocol):
    def latest(self) -> DepthFrame | None:
        """Newest frame, or None if there is none. Never raises on a missing sensor."""


@runtime_checkable
class Localizer(Protocol):
    def estimate(self, stamp: float) -> LocalizerEstimate | None:
        """Pose at `stamp`, or None if no estimate exists. A consumer treats None or `healthy=False` as 'stop'."""


@runtime_checkable
class SemanticsModel(Protocol):
    """DAVE's model. Placeholder in stubs.py."""

    def detect(self, rgb: np.ndarray | None, depth: DepthFrame) -> list[Detection]: ...


@runtime_checkable
class ObjectMemory(Protocol):
    """DAVE's memory. Placeholder in stubs.py."""

    def update(self, detections: list[Detection], depth: DepthFrame, body_pose: LocalizerEstimate) -> None: ...

    def query(self, label: str) -> list[ObjectAnchor]: ...


@runtime_checkable
class GraspProposer(Protocol):
    def propose(self, points: np.ndarray, obstacles: np.ndarray | None = None) -> list[GraspCandidate]:
        """`points` (N, 3) target cloud, `obstacles` (M, 3) scene cloud for collision checks, both in one frame.
        Returns candidates best first. An empty or invalid cloud returns []. Never raises on bad geometry."""
