"""Lidar (Livox) near-field obstacle clearance, to cover the forward depth camera's near-field blind zone.

Stdlib + numpy only. Frames: body, x forward / y left / z up; metres, seconds. A `LidarFrame` holds points ALREADY in
the body frame; the caller gravity-aligns them (`ground.gravity_align`, same transform as the camera cloud) before
`lidar_clearance`, which then measures heights above the camera's fitted `GroundPlane` (or above z = 0 of the
gravity-aligned frame when no plane exists; that assumes the body origin is on the floor plane, as the sim does).

What this module can and cannot say
  * It only ADDS detection: a positive obstacle (a return `h_min..h_max` above the ground) in the corridor gives a
    distance. It never vouches for safety by itself.
  * `ground_points_near` / `near_coverage` say whether the lidar actually OBSERVES the near field (returns on the
    floor in x in `near_x`). A lidar that sees no near-field floor (e.g. a level mount, whose lowest ray misses the
    floor for the first metres) cannot vouch that the near field is clear; the caller must then treat it as unobserved.
  * It does NOT replace negative-obstacle (drop / step-down / hole) detection: a hole is the ABSENCE of returns and
    a sparse, non-repetitive lidar pattern cannot tell a hole from dropout. Drops stay with the camera guard.
  * `h_min` default is 0.10 m (not the camera's 0.03 m): lidar range noise of ~2 cm (UNVERIFIED datasheet value) on
    thousands of floor returns would otherwise produce false hits with `min_hits` = 3.
Fail closed: non-finite / ill-shaped points or bad config give `ok=False, distance=0.0` (config raises ValueError).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from humanoid_arc_bme.onboard.pipeline.perception.ground import GroundPlane
from humanoid_arc_bme.onboard.validation import is_num


@dataclass(frozen=True)
class LidarFrame:
    """One lidar scan, `points` (N, 3) float64 in the BODY frame. Rows with a non-finite coordinate are dropped
    (dropout markers); `n_invalid` counts them. A non-finite stamp or a wrong shape raises ValueError."""

    stamp: float
    points: np.ndarray
    n_invalid: int = 0

    def __post_init__(self):
        if not is_num(self.stamp):
            raise ValueError("stamp: must be finite")
        p = np.asarray(self.points)
        if p.size == 0:
            p = p.reshape(0, 3)
        if p.ndim != 2 or p.shape[1] != 3 or not (np.issubdtype(p.dtype, np.floating) or np.issubdtype(p.dtype, np.integer)):
            raise ValueError("points: expected a numeric (N, 3) array")
        p = p.astype(np.float64, copy=False)
        ok = np.isfinite(p).all(axis=1)
        object.__setattr__(self, "n_invalid", int(self.n_invalid) + int((~ok).sum()))
        object.__setattr__(self, "points", p[ok])


@dataclass(frozen=True)
class LidarClearance:
    distance: float                  # m ahead to the nearest obstacle in the corridor; inf = none; 0.0 = fail closed
    ok: bool                         # False: bad input, distance is 0.0
    nearest_xy: tuple[float, float] | None   # body-frame (x, y) of the nearest hit (the min_hits-th nearest by x), or None
    n_hits: int
    ground_points_near: int          # floor returns in the corridor with x in near_x
    near_coverage: float             # fraction of 0.1 m x-bins in near_x that hold >= min_hits floor returns (0..1)
    near_observed: bool = False      # ground_points_near >= min_near_points AND near_coverage >= min_near_coverage


def _closed() -> LidarClearance:
    return LidarClearance(0.0, False, None, 0, 0, 0.0, False)


def self_exclusion_keep(points, plane_or_none=None, radius: float = 0.25, z_lo: float = 0.05,
                        z_hi: float = 1.5) -> np.ndarray:
    """Boolean keep-mask (N,) for gravity-aligned body-frame `points`: False for a return inside the BODY exclusion
    cylinder, i.e. hypot(x, y) < `radius` and height above the ground in [z_lo, z_hi] (ground = `plane_or_none` or
    z = 0). radius 0.25 m, z band 0.05..1.5 m are PLACEHOLDERS (torso half-width ~0.2 m plus a little; measure on the
    real mount). The band starts just above the floor so floor returns beside the feet are not removed.
    A return that close, at a height the body occupies, is the robot itself or already a collision, so excluding
    it cannot hide a usable obstacle; it must NEVER remove one farther than `radius` (the test pins 0.30 m seen,
    0.20 m excluded). radius <= 0 disables the mask. Shared by the costmap update and `lidar_clearance` input so both
    see the same points. Non-finite rows are kept (the callers drop/handle them)."""
    p = np.asarray(points, dtype=np.float64)
    if p.size == 0 or radius <= 0.0:
        return np.ones(len(p), dtype=bool)
    h = plane_or_none.height(p) if plane_or_none is not None else p[:, 2]
    with np.errstate(invalid="ignore"):
        inside = (np.hypot(p[:, 0], p[:, 1]) < radius) & (h >= z_lo) & (h <= z_hi)
    return ~inside


def corridor_mask(p: np.ndarray, half_width: float, min_range: float) -> np.ndarray:
    """Forward corridor: |y| <= half_width, x > 0 and range > min_range (the one definition `lidar_clearance` uses)."""
    return (np.abs(p[:, 1]) <= half_width) & (p[:, 0] > 0.0) & (np.hypot(p[:, 0], p[:, 1]) > min_range)


def _pos(name, v):
    if not is_num(v) or v <= 0.0:
        raise ValueError(f"{name}: must be a finite number > 0, got {v!r}")


def lidar_clearance(points, plane_or_none, corridor_half_width: float = 0.35, max_range: float = 3.0,
                    h_min: float = 0.10, h_max: float = 1.5, min_hits: int = 3, min_range: float = 0.15,
                    *, near_x: tuple[float, float] = (0.3, 1.2), ground_tol: float | None = None,
                    cell: float = 0.1, min_near_points: int = 10, min_near_coverage: float = 0.8) -> LidarClearance:
    """Clearance in |y| <= corridor_half_width, min_range < x <= max_range for returns h_min..h_max above the ground
    (`plane_or_none.height(p)` or p[:, 2]). `points` must be gravity-aligned. The k-th (= min_hits) nearest x is
    reported. Floor returns are |h| < ground_tol (default h_min). See the module docstring for what it cannot do."""
    for n, v in (("corridor_half_width", corridor_half_width), ("max_range", max_range), ("h_min", h_min),
                 ("h_max", h_max), ("min_range", min_range), ("cell", cell)):
        _pos(n, v)
    if type(min_near_points) is not int or min_near_points < 1 or not is_num(min_near_coverage) \
            or not 0.0 < min_near_coverage <= 1.0:
        raise ValueError("min_near_points: int >= 1; min_near_coverage: number in (0, 1]")
    if type(min_hits) is not int or min_hits < 1:
        raise ValueError(f"min_hits: must be an int >= 1, got {min_hits!r}")
    if h_max <= h_min:
        raise ValueError("h_max: must be > h_min")
    x0, x1 = near_x
    if not (is_num(x0) and is_num(x1) and 0.0 <= x0 < x1):
        raise ValueError("near_x: expected (lo, hi) with 0 <= lo < hi")
    gt = h_min if ground_tol is None else ground_tol
    _pos("ground_tol", gt)
    if plane_or_none is not None and not isinstance(plane_or_none, GroundPlane):
        return _closed()
    try:
        p = np.asarray(points)
        if p.ndim != 2 or p.shape[1] != 3 or not np.issubdtype(p.dtype, np.number) or not np.isfinite(p).all():
            return _closed()
        p = p.astype(np.float64, copy=False)
        h = plane_or_none.height(p) if plane_or_none is not None else p[:, 2]
        if not np.isfinite(h).all():
            return _closed()
    except (TypeError, ValueError):
        return _closed()
    inside = corridor_mask(p, corridor_half_width, min_range)
    hit = inside & (p[:, 0] <= max_range) & (h >= h_min) & (h <= h_max)
    xs = p[hit, 0]
    if xs.size >= min_hits:
        order = np.argsort(xs)
        k = order[min_hits - 1]
        dist = float(xs[k])
        nxy = (float(xs[k]), float(p[hit][k, 1]))
    else:
        dist, nxy = math.inf, None
    g = inside & (np.abs(h) < gt) & (p[:, 0] > x0) & (p[:, 0] <= x1)
    gx = p[g, 0]
    nb = max(1, int(round((x1 - x0) / cell)))
    bins = np.minimum(((gx - x0) / cell).astype(int), nb - 1)
    cov = float((np.bincount(bins, minlength=nb) >= min_hits).sum()) / nb
    obs = bool(gx.size >= min_near_points and cov >= min_near_coverage)
    return LidarClearance(dist, True, nxy, int(xs.size), int(gx.size), cov, obs)
