"""Ground-plane geometry for the depth guard: gravity alignment, robust plane fit, corridor clearance.

Stdlib + numpy only. Frame: body, x forward, y left, z up (metres, radians). Everything fails closed: if the ground
cannot be established the result says so and reports distance 0.0 (stop), never "clear".

Why this exists (docs/plan/pipeline-depth-guard-findings.md): the rule guard used a fixed camera pose and an absolute
height band, so sway made phantom obstacles, a pitched-up camera hid low boxes, and holes were invisible. Here heights
are measured against a plane FITTED to the frame, so any rigid tilt (sway, mount error) is absorbed by the fit, and
negative obstacles are visible as points below the plane or as ground that should be there but is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from humanoid_arc_bme.onboard.geometry import quat_to_matrix


# ------------------------------------------------------------------ gravity alignment

def attitude_matrix(roll: float, pitch: float) -> np.ndarray:
    """World-from-body rotation R = Ry(pitch) Rx(roll) (yaw is irrelevant for heights and omitted).

    Convention: right-handed, body x forward / y left / z up. Positive pitch = nose DOWN (Ry(a) takes +x toward -z,
    same as `camera_pose_from_mount`); positive roll = left side UP (Rx(a) takes +y toward +z). A world point is
    p_world = R p_body, so z of the result is the height along gravity.
    """
    if not (math.isfinite(roll) and math.isfinite(pitch)):
        raise ValueError("roll/pitch: must be finite")
    # Ry(pitch) Rx(roll) as the unit quaternion qy * qx, turned into a matrix by the shared helper (tested equal to the
    # explicit matrix product in tests/core/test_pipeline_ground.py::test_attitude_matrix_matches_explicit_euler).
    cr, sr, cp, sp = math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    return quat_to_matrix((cp * cr, cp * sr, sp * cr, -sp * sr))


def gravity_align(points_body: np.ndarray, roll: float, pitch: float) -> np.ndarray:
    """Rotate (N, 3) body-frame points into the gravity-aligned frame (z = up). Origin unchanged."""
    p = np.asarray(points_body, dtype=np.float64).reshape(-1, 3)
    return p @ attitude_matrix(roll, pitch).T


def gravity_unalign(points_aligned: np.ndarray, roll: float, pitch: float) -> np.ndarray:
    """Exact inverse of `gravity_align`."""
    p = np.asarray(points_aligned, dtype=np.float64).reshape(-1, 3)
    return p @ attitude_matrix(roll, pitch)


# ------------------------------------------------------------------ config validation

def _pos(name: str, v) -> float:
    """Programmer error (ValueError) unless v is a finite real number > 0 (bools and strings rejected)."""
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) or not (math.isfinite(v) and v > 0):
        raise ValueError(f"{name}: must be a finite number > 0, got {v!r}")
    return float(v)


def _count(name: str, v, lo: int = 1) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < lo:
        raise ValueError(f"{name}: must be an int >= {lo}, got {v!r}")
    return int(v)


def _frac(name: str, v) -> float:
    v = _pos(name, v)
    if v > 1.0:
        raise ValueError(f"{name}: must be in (0, 1], got {v!r}")
    return v


# ------------------------------------------------------------------ plane fit

@dataclass(frozen=True)
class GroundPlane:
    """Plane n . p + d = 0 with unit normal n (n_z > 0, so `height(p) = n . p + d` is signed height above ground)."""

    normal: tuple[float, float, float]
    offset: float
    inlier_fraction: float   # inliers / candidate points in the fit region
    residual_rms: float      # metres, over inliers
    confidence: float        # 0..1
    tilt_deg: float          # angle between normal and gravity-up
    n_inliers: int

    def height(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64) @ np.asarray(self.normal) + self.offset


def _plane_lsq(p: np.ndarray) -> tuple[np.ndarray, float]:
    c = p.mean(axis=0)
    n = np.linalg.svd(p - c, full_matrices=False)[2][-1]
    if n[2] < 0:
        n = -n
    return n, float(-n @ c)


def fit_ground(points_aligned: np.ndarray, *, fit_x_max: float = 3.0, fit_half_width: float = 1.0,
               band_percentile: float = 40.0, band_height: float = 0.4, inlier_tol: float = 0.03,
               min_points: int = 300, min_inlier_frac: float = 0.3, max_tilt_deg: float = 12.0,
               iters: int = 120, max_sample: int = 4000, seed: int = 0,
               max_plane_offset: float = 0.05) -> GroundPlane | None:
    """Seeded, vectorised RANSAC plane fit on gravity-aligned points. None = fail closed.

    Region: 0 < x <= fit_x_max, |y| <= fit_half_width, and the LOWER height band z <= percentile(z, band_percentile)
    + band_height (the ground is the lowest big surface; box tops/walls above are excluded, holes below are kept).
    `inlier_tol` 0.03 m ~ 3 sigma of ground height noise at 3 m for the placeholder stereo model (depth sigma 0.016 m
    times sin(grazing angle) ~0.37 = 0.006 m, plus margin for plane non-flatness). Candidate planes steeper than
    `max_tilt_deg` from level are rejected, so a wall or a big box face can never be mistaken for the floor.
    Returns None if: non-finite input, < min_points in the region, inlier fraction < min_inlier_frac, or no
    candidate within the tilt limit or the offset limit.

    Tilt: default `max_tilt_deg` 12 = 5 deg sway + 3 deg IMU error (tests) + margin; a 10 deg ramp is an ACCEPTED,
    documented limit (it is taken as ground; steeper slopes fail closed). Pass an explicit value to change it.

    Height prior (`max_plane_offset`): `GroundPlane.height(p) = n.p + d`, so `d` is the plane's height at the body
    origin, and the body origin is at ground level (the robot stands on the ground it looks at). A genuine ground
    plane therefore has d ~ 0: gravity alignment is a rotation about the origin, so a tilt or IMU error does not move
    it, and a level floor stays at d = 0 whatever the pitch. d is allowed to deviate by
    `max_plane_offset + tan(tilt) * x_near`: 0.05 m for sway of the body height / floor non-flatness, plus the
    height at the origin of a ramp of that tilt that starts at the nearest fit point x_near (5th percentile of x of
    the fit region; for a ramp starting at x_near, d = -tan(tilt) * x_near exactly). A block top, table, stair tread
    or step-down filling the view is parallel to the floor but at d = -/+ its height (0.15 m stair > 0.05 m) and is
    refused. Candidates violating this are skipped in RANSAC, and the refined plane is re-checked (fail closed).
    """
    fit_x_max, fit_half_width = _pos("fit_x_max", fit_x_max), _pos("fit_half_width", fit_half_width)
    band_percentile, band_height = _pos("band_percentile", band_percentile), _pos("band_height", band_height)
    if band_percentile > 100.0:
        raise ValueError("band_percentile: must be in (0, 100]")
    inlier_tol, max_plane_offset = _pos("inlier_tol", inlier_tol), _pos("max_plane_offset", max_plane_offset)
    min_inlier_frac, max_tilt_deg = _frac("min_inlier_frac", min_inlier_frac), _pos("max_tilt_deg", max_tilt_deg)
    if max_tilt_deg >= 90.0:
        raise ValueError("max_tilt_deg: must be < 90")
    min_points, iters, max_sample = _count("min_points", min_points, 3), _count("iters", iters), _count("max_sample", max_sample, 3)
    p = np.asarray(points_aligned)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        return None
    p = p.astype(np.float64, copy=False)
    sel = (p[:, 0] > 0.0) & (p[:, 0] <= fit_x_max) & (np.abs(p[:, 1]) <= fit_half_width)
    reg = p[sel]
    if reg.shape[0] < min_points:
        return None
    reg = reg[reg[:, 2] <= np.percentile(reg[:, 2], band_percentile) + band_height]
    if reg.shape[0] < min_points:
        return None
    x_near = float(np.percentile(reg[:, 0], 5.0))
    rng = np.random.default_rng(seed)
    sub = reg if reg.shape[0] <= max_sample else reg[rng.choice(reg.shape[0], max_sample, replace=False)]
    idx = rng.integers(0, sub.shape[0], size=(iters, 3))
    a, b, c = sub[idx[:, 0]], sub[idx[:, 1]], sub[idx[:, 2]]
    n = np.cross(b - a, c - a)
    nrm = np.linalg.norm(n, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / nrm[:, None]
    n[n[:, 2] < 0] *= -1.0
    ok = (nrm > 1e-9) & (n[:, 2] >= math.cos(math.radians(max_tilt_deg)))
    n, a = n[ok], a[ok]
    d = -np.einsum("ij,ij->i", n, a)
    allow = max_plane_offset + np.sqrt(np.clip(1.0 - n[:, 2] ** 2, 0.0, 1.0)) / n[:, 2] * x_near
    keep = np.abs(d) <= allow
    if not keep.any():
        return None
    n, d = n[keep], d[keep]
    counts = (np.abs(sub @ n.T + d) < inlier_tol).sum(axis=0)
    k = int(np.argmax(counts))
    nv, dv = n[k], float(d[k])
    for _ in range(3):                                   # refine on the full region
        inl = np.abs(reg @ nv + dv) < inlier_tol
        if inl.sum() < 3:
            return None
        nv, dv = _plane_lsq(reg[inl])
    res = reg @ nv + dv
    inl = np.abs(res) < inlier_tol
    n_in = int(inl.sum())
    frac = n_in / reg.shape[0]
    tilt = math.degrees(math.acos(min(1.0, float(nv[2]))))
    if frac < min_inlier_frac or tilt > max_tilt_deg:
        return None
    if not abs(dv) <= max_plane_offset + math.tan(math.radians(tilt)) * x_near:   # NaN fails too
        return None
    rms = float(np.sqrt(np.mean(res[inl] ** 2)))
    conf = min(1.0, frac / 0.6) * (1.0 - min(1.0, rms / inlier_tol))
    return GroundPlane(tuple(float(v) for v in nv), dv, float(frac), rms, float(conf), tilt, n_in)


# ------------------------------------------------------------------ corridor analysis

@dataclass(frozen=True)
class ClearanceResult:
    """Distances are metres ahead along body +x (inf = nothing found in range; 0.0 = stop).

    distance_positive: nearest object rising h_min..h_max above the plane. distance_negative: CONSERVATIVE bound on where a
    drop starts: the end of the contiguous verified ground before a detected drop (points more than drop_min below
    the plane, or unexplained missing ground), 0.0 if no ground was verified before it (the edge may be in the blind
    zone). It is never later than the true edge by more than about one cell.  unknown_ground_from: start of
    the first unexplained missing-ground run (inf if none) - reported separately so callers can treat it differently.
    blind_until: range from the body to the first visible ground in the corridor; nothing nearer was observed (the
    mount's geometric near-field blind zone; this module does NOT fix it). ground_ok: the plane was established and
    the input was valid; False always comes with distances 0.0.
    """

    distance_positive: float
    distance_negative: float
    unknown_ground_from: float
    blind_until: float
    ground_ok: bool
    confidence: float
    h_min: float = math.nan


def _closed() -> ClearanceResult:
    return ClearanceResult(0.0, 0.0, 0.0, math.inf, False, 0.0)


def _kth(x: np.ndarray, k: int) -> float:
    return math.inf if x.size < k else float(np.partition(x, k - 1)[k - 1])


def analyze_corridor(points_aligned: np.ndarray, plane: GroundPlane | None, corridor_half_width: float = 0.35,
                     max_range: float = 3.0, h_min: float | None = None, h_max: float = 1.5, drop_min: float = 0.08,
                     *, min_hits: int = 3, cell: float = 0.1, min_gap: float = 0.3, k_sigma: float = 3.0,
                     h_floor: float = 0.03, valid_fraction: float, min_valid_fraction: float = 0.5,
                     max_blind: float = 1.5) -> ClearanceResult:
    """Clearance in the corridor |y| <= corridor_half_width, 0 < x <= max_range, heights measured above `plane`.

    h_min (positive-obstacle threshold) is at least k_sigma * plane.residual_rms (3 sigma of ground noise, so noise
    alone does not trip it) and at least h_floor (0.03 m, unmodelled floor roughness); None derives it. drop_min is
    raised to at least h_min. A hit needs `min_hits` points (the k-th nearest is reported), like the old guard.
    Missing ground: cells (size `cell`) from the first visible ground cell outward with no return of ANY height form
    runs; a run >= min_gap long is a potential drop unless a positive obstacle is nearer (it is occluded by that obstacle, which
    already stops the robot; ground beside it can stay visible, so 'directly behind' is not a usable test).
    `valid_fraction` (REQUIRED, fraction of valid pixels in the frame, 0..1) must be >= `min_valid_fraction`; the
    caller states it explicitly so a coverage check can never be skipped by omission.
    `max_blind` (default 1.5 m): if the first visible ground in the corridor is farther than this (or no ground is
    seen at all), the near field is unobserved and the result is fail closed. Nominal mount: the first ground is at
    ~1.13 m (tests), so 1.5 m leaves 0.37 m for noise/sway/pitch while staying below the stopping distance at
    walking speed (0.5 m/s plus latency, margin ~1 m, plus the 0.3 m body reach), so an unseen near field cannot hide
    an obstacle the robot cannot stop for.
    Fail closed (ground_ok False, distances 0.0) on no plane, non-finite/ill-shaped input, too few valid pixels or an
    unobserved near field. Config knobs (all of the above) that are not finite and positive raise ValueError
    (programmer error); data problems never raise.
    """
    cfg = dict(corridor_half_width=corridor_half_width, max_range=max_range, h_max=h_max, drop_min=drop_min,
               cell=cell, min_gap=min_gap, k_sigma=k_sigma, h_floor=h_floor, max_blind=max_blind)
    for k, v in cfg.items():
        _pos(k, v)
    if h_min is not None:
        _pos("h_min", h_min)
    _count("min_hits", min_hits)
    _frac("min_valid_fraction", min_valid_fraction)
    if isinstance(valid_fraction, bool) or not isinstance(valid_fraction, (int, float, np.integer, np.floating)):
        raise ValueError(f"valid_fraction: must be a number, got {valid_fraction!r}")
    if plane is None:
        return _closed()
    p = np.asarray(points_aligned)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        return _closed()
    if not (0.0 <= valid_fraction <= 1.0 and valid_fraction >= min_valid_fraction):   # NaN fails too
        return _closed()
    p = p.astype(np.float64, copy=False)
    hmin = max(k_sigma * plane.residual_rms, h_floor, h_min if h_min is not None else 0.0)
    drop = max(drop_min, hmin)
    sel = (np.abs(p[:, 1]) <= corridor_half_width) & (p[:, 0] > 0.0) & (p[:, 0] <= max_range)
    q = p[sel]
    x, h = q[:, 0], plane.height(q)
    dpos = _kth(x[(h >= hmin) & (h <= h_max)], min_hits)
    dneg_a = _kth(x[h <= -drop], min_hits)
    gx = x[np.abs(h) < hmin]
    first_ground = _kth(gx, min_hits)
    if not (first_ground <= max_blind):                  # unobserved near field (or no ground at all): fail closed
        return _closed() if max_range > max_blind else ClearanceResult(dpos, 0.0, 0.0, max_range, True, plane.confidence, hmin)
    nc = int(math.ceil(max_range / cell))
    ci = np.minimum((x / cell).astype(int), nc - 1)
    occ = np.bincount(ci, minlength=nc) > 0
    pos = np.bincount(ci[h >= hmin], minlength=nc) > 0
    gap_from, i, need = math.inf, int(first_ground / cell), max(1, int(round(min_gap / cell)))
    while i < nc:
        if occ[i]:
            i += 1
            continue
        j = i
        while j < nc and not occ[j]:
            j += 1
        if j - i >= need and not pos[:i].any():
            gap_from = i * cell
            break
        i = j
    dneg = min(dneg_a, gap_from)
    if math.isfinite(dneg):
        # Where the drop starts is only bounded by the ground that was actually seen, never by where it was detected:
        # a hole floor is often seen farther out than the edge. No verified ground before the detection (the edge
        # may be inside the blind zone) => stop (0.0); otherwise the end of the contiguous verified ground run.
        if first_ground > dneg:
            dneg = 0.0
        else:
            on_ground = np.bincount(ci[np.abs(h) < hmin], minlength=nc) > 0
            k = int(first_ground / cell)
            while k < nc and on_ground[k]:
                k += 1
            dneg = min(dneg, k * cell)
    return ClearanceResult(dpos, dneg, gap_from, first_ground, True, plane.confidence, hmin)
