"""Scene assembly: segmented detections -> body-frame objects with a colour name, position and horizontal range.

Stdlib + numpy only. Frames: depth pixels are back-projected in the camera optical frame (x right, y down, z forward),
then mapped to the body frame (x forward, y left, z up) with `depth.cam_to_body`. Metres throughout. Range is the
horizontal distance sqrt(x^2 + y^2) in the body frame.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from humanoid_arc_bme.onboard.perception.colors import erode_mask, name_color
from humanoid_arc_bme.onboard.pipeline.interfaces import DepthFrame, Detection


@dataclass(frozen=True)
class SceneObject:
    """One object in the body frame. `position_body` is the median of its valid back-projected points (metres);
    `n_points` is how many depth pixels contributed; `box` is the detection box (pixels) or None; `range_m` is
    sqrt(x^2 + y^2) of `position_body`."""

    label: str
    color: str
    confidence: float
    position_body: tuple[float, float, float]
    n_points: int
    box: tuple[int, int, int, int] | None
    range_m: float


def _colour(rgb, mask: np.ndarray) -> str:
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3:
        return "unknown"
    hw = rgb.shape[:2]
    if mask.shape != hw:
        mask = _resize_nearest(mask, hw)
    return name_color(rgb, mask)[0]


def _resize_nearest(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resample of a bool (H, W) mask to `hw`. Used when RGB and depth resolutions differ."""
    h1, w1 = mask.shape
    h2, w2 = hw
    rows = np.minimum(np.floor((np.arange(h2) + 0.5) * (h1 / h2)).astype(np.intp), h1 - 1)
    cols = np.minimum(np.floor((np.arange(w2) + 0.5) * (w1 / w2)).astype(np.intp), w1 - 1)
    return mask[np.ix_(rows, cols)]


def build_scene(
    detections: Sequence[Detection],
    rgb: np.ndarray | None,
    depth: DepthFrame,
    *,
    min_points: int = 30,
    max_range_m: float = 3.0,
    min_conf: float = 0.0,
) -> list[SceneObject]:
    """Turn detections with masks into body-frame objects, nearest first.

    Per detection: the mask is eroded by one pixel, then restricted to finite depth pixels in (0, 2 * max_range_m].
    Fewer than `min_points` such pixels drops the detection. The position is the coordinate-wise median of the
    back-projected points in the body frame. Objects beyond `max_range_m` (horizontal) are dropped, as are detections
    below `min_conf`. Detections without a mask, or whose mask does not match the depth frame, are skipped.
    `rgb` is used only for the colour name; without it the colour is "unknown".
    """
    if not (isinstance(min_points, int) and not isinstance(min_points, bool) and min_points >= 1):
        raise ValueError("min_points: must be an int >= 1")
    if not (math.isfinite(max_range_m) and max_range_m > 0):
        raise ValueError("max_range_m: must be a positive finite number")
    d = depth.depth
    h, w = d.shape
    k = depth.intrinsics
    R = depth.cam_to_body.rotation_matrix()
    t = np.asarray(depth.cam_to_body.position, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(d) & (d > 0) & (d <= 2.0 * max_range_m)

    out: list[SceneObject] = []
    for det in detections:
        if det.mask is None or det.confidence < min_conf or det.mask.shape != (h, w):
            continue
        sel = erode_mask(det.mask) & valid
        ys, xs = np.nonzero(sel)
        if ys.size < min_points:
            continue
        z = d[ys, xs].astype(np.float64)
        cam = np.empty((ys.size, 3), dtype=np.float64)
        cam[:, 0] = (xs - k.cx) * z / k.fx
        cam[:, 1] = (ys - k.cy) * z / k.fy
        cam[:, 2] = z
        pos = np.median(cam @ R.T + t, axis=0)
        if not np.isfinite(pos).all():
            continue
        rng = math.hypot(pos[0], pos[1])
        if rng > max_range_m:
            continue
        out.append(
            SceneObject(
                label=det.label,
                color=_colour(rgb, det.mask),
                confidence=float(det.confidence),
                position_body=(float(pos[0]), float(pos[1]), float(pos[2])),
                n_points=int(ys.size),
                box=det.box,
                range_m=float(rng),
            )
        )
    out.sort(key=lambda o: o.range_m)
    return out


def scene_summary(scene: Sequence[SceneObject]) -> str:
    """Short text for logs and the operator, e.g. "red cup 0.8 m, 2x blue bottle 1.2 m", or "nothing".
    Duplicates are grouped under the nearest distance, in the order the scene lists them."""
    if not scene:
        return "nothing"
    groups: dict[str, list[SceneObject]] = {}
    for o in scene:
        name = o.label if o.color == "unknown" else f"{o.color} {o.label}"
        groups.setdefault(name, []).append(o)
    parts = []
    for name, objs in groups.items():
        nearest = min(o.range_m for o in objs)
        prefix = f"{len(objs)}x " if len(objs) > 1 else ""
        parts.append(f"{prefix}{name} {nearest:.1f} m")
    return ", ".join(parts)
