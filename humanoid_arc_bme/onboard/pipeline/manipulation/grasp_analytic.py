"""Analytic, training-free, licence-free pinch-grasp proposer for a three-finger hand (Dex3-1 class).

Implements `interfaces.GraspProposer`. numpy + stdlib only. Units: metres; quaternions (w, x, y, z); z is up.

Assumptions (all UNMEASURED placeholders until the real Dex3-1 is characterised, see `HandSpec`):
  * The hand is modelled as two opposing "fingers" (thumb vs. finger pair) closing along one axis, plus a palm.
    Only a pinch (kind='pinch') is proposed. Power / wrap grasps are OUT OF SCOPE.
  * The input is ONE object's PARTIAL cloud from a single depth view; the far side is unobserved.
  * Surface normals are unsigned (local PCA); sign is not needed for the antipodal test (|n . c| is used).

Hand frame of a returned pose (right-handed, columns of R): x = approach (palm towards the object),
y = closing axis (thumb to finger), z = x cross y. `pose.position` is the hand BASE (palm) point; the pinch
midpoint is base + x * palm_standoff. `aperture` is the separation (m) of the two contact points.

Candidate sources
  1. Antipodal pairs: two surface points whose normals both lie within `friction_cone_deg` of the line joining
     them, separation in [min_aperture, max_aperture]. Pairs are drawn with a seeded Generator.
  2. Visible-side fallback: for each principal axis of the cloud, contacts at the 1% / 99% extent of the visible
     cloud. The far side is unobserved so the true width may differ: scored lower (see below).
Each pair is turned into up to three hand poses: 'top' (approach -z) and 'side' (horizontal, both signs),
limited by `HandSpec.approaches`. Closing axes closer than ~20 deg to vertical are skipped for both.

Collision: fingers are spheres (radius finger_clearance_radius) placed outward of each contact along the closing
axis and swept back along the approach axis to the palm, plus a row of palm spheres. A candidate is rejected if
an obstacle point lies within r of any sphere, a target point (other than within 1.5 r of the two contacts) lies
within 0.8 r, or (with `min_z`) a sphere's bottom is below the table plane.

Score in [0, 1]:  S = f_app * f_src * (0.40 q_ant + 0.15 q_ap + 0.20 q_cen + 0.25 q_clr)
  q_ant = clip((mean|n.c| - cos(cone)) / (1 - cos(cone)), 0, 1)    (fallback: 0.5)
  q_ap  = 1 - |d - mid_range| / half_range                          (prefer mid-range aperture)
  q_cen = 1 / (1 + (|midpoint - centroid| / s)^2), s = half bbox diagonal
  q_clr = clip((min distance sphere-to-obstacle/target - r) / (2 r), 0, 1)   (1 if nothing nearby)
  f_app = 1.0 top-down, 0.9 side;  f_src = 1.0 antipodal, 0.6 visible-side fallback.
Collision clouds (obstacles, and the full target cloud) are voxel-downsampled with cell r/2 (r = finger_clearance_radius)
so no region is dropped (quantisation error <= r*sqrt(3)/4); above 50000 voxels, or with any non-finite obstacle row,
proposing returns [] (fail closed). Only candidate generation uses a strided target subset.
Output is NMS-filtered (similar midpoint/axes dropped), sorted descending, at most `max_candidates`.
Reach constraint (`ReachConstraint`, default ON as a HARD filter; `reach=None` disables it): the approach direction must
lie in a cone (default half-angle 60 deg) around `reference_dir` (default +x, the robot forward when the cloud is in the
body frame; pass the forward direction when the cloud is in another frame) and, unless `allow_top_down`, be nearly
horizontal. Optional `max_reach_from=((x,y,z), radius)` (off by default) drops proposals whose pinch midpoint lies outside that
sphere (the arm workspace around the shoulder/torso). The closing axis is only a soft preference: score *= 1 - w (1 - |y . preferred|).
Contact refinement (`contact_refinement=True`, default ON; False restores the original visible-side fallback exactly):
a single view of a convex box shows two adjacent side faces, so ANTIPODAL pairs (needing two opposing visible faces)
almost never exist and the old fallback (contacts on a line through the centroid of the visible points, 3-D PCA axes)
put contacts off the surface and tilted. The footprint fallback instead fits the minimum-area rectangle to the cloud's
horizontal projection (a box's two visible faces give BOTH footprint sides exactly), takes the closing axes along the
rectangle sides (horizontal by construction), puts the pinch at the mid-height of the cloud, SNAPS the observed
contact to the nearest observed point (rejecting it if none is within `snap_tol` = 1 cm) and places the far contact one
rectangle width behind it. The far side is unobserved, so this is a convex-footprint (box-like) prior; it is
penalised by the footprint fit residual (a half-cylinder arc fits a rectangle badly), never trusted blindly.
Not validated: real Dex3 geometry, kinematic reachability, contact forces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from humanoid_arc_bme.onboard.pipeline.interfaces import GraspCandidate, Pose
from humanoid_arc_bme.onboard.pipeline.perception.sensing import matrix_to_quat

_UP = np.array([0.0, 0.0, 1.0])
_MAX_POINTS = 2000
_MAX_CLOUD_VOXELS = 50000  # hard cap on the collision clouds AFTER voxel downsampling; above it we fail closed
_TILT_LIMIT = 0.94  # |c . z| above this (about 20 deg from vertical): closing axis too vertical
_PRESELECT = 120
_EXT_PCT = (0.25, 99.75)  # extent percentiles of the footprint fit: a grazing visible face holds < 1 % of the points
_SNAP_SHIFT = 0.03  # m: the snapped (observed) contact may sit this far from the centred (mid-width, mid-height) point


@dataclass(frozen=True)
class HandSpec:
    """Placeholder Dex3-1 numbers. UNMEASURED: replace from the datasheet / a measured hand before real use."""

    max_aperture: float = 0.10  # m, UNMEASURED
    min_aperture: float = 0.015  # m, UNMEASURED
    finger_clearance_radius: float = 0.01  # m, UNMEASURED
    palm_standoff: float = 0.06  # m, palm to pinch midpoint along the approach axis, UNMEASURED
    approaches: tuple[str, ...] = ("top", "side")

    def __post_init__(self):
        for n in ("max_aperture", "min_aperture", "finger_clearance_radius", "palm_standoff"):
            v = getattr(self, n)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise ValueError(f"{n}: must be a positive finite number")
        if self.min_aperture >= self.max_aperture:
            raise ValueError("min_aperture must be < max_aperture")
        if not self.approaches or any(a not in ("top", "side") for a in self.approaches):
            raise ValueError("approaches: non-empty subset of ('top', 'side')")


def _real(x) -> bool:
    """A real number that is not a bool (True/False are not angles or weights)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


@dataclass(frozen=True)
class ReachConstraint:
    """Which approaches the arm can execute, in the proposer's frame (= the frame of the input cloud).

    A candidate approach direction `a` (palm towards object) is kept iff angle(a, reference_dir) <= cone_half_angle_deg
    and, unless `allow_top_down`, |a_z| <= sin(max_elevation_deg) (near-horizontal approaches only). The closing axis
    is NOT forced: `closing_weight` in [0, 1] scales the score by 1 - w (1 - |y . preferred_closing|); the scripted
    grasp closes along the robot's left axis, hence the default (0, 1, 0)."""

    reference_dir: tuple[float, float, float] = (1.0, 0.0, 0.0)
    cone_half_angle_deg: float = 60.0
    allow_top_down: bool = False
    max_elevation_deg: float = 15.0
    preferred_closing: tuple[float, float, float] = (0.0, 1.0, 0.0)
    closing_weight: float = 0.15
    max_reach_from: tuple | None = None  # ((x, y, z), radius): HARD filter on the pinch midpoint; None = off

    def __post_init__(self):
        if self.max_reach_from is not None:
            try:
                c, r = self.max_reach_from
                c = tuple(float(x) for x in c)
                ok = len(c) == 3 and all(math.isfinite(x) for x in c) and _real(r) and math.isfinite(r) and r > 0
            except (TypeError, ValueError):
                ok = False
            if not ok:
                raise ValueError("max_reach_from: ((x, y, z) finite, radius > 0) or None")
            object.__setattr__(self, "max_reach_from", (c, float(r)))
        for n in ("reference_dir", "preferred_closing"):
            v = getattr(self, n)
            if len(v) != 3 or not all(_real(x) and math.isfinite(x) for x in v):
                raise ValueError(f"{n}: need 3 finite numbers")
            nrm = math.hypot(*[float(x) for x in v])   # hypot does not overflow at 1e200
            if not math.isfinite(nrm) or nrm < 1e-6:
                raise ValueError(f"{n}: must be a finite, non-zero direction")
            object.__setattr__(self, n, tuple(float(x) / nrm for x in v))   # stored normalised
        if not (_real(self.cone_half_angle_deg) and 0.0 < self.cone_half_angle_deg <= 180.0):
            raise ValueError("cone_half_angle_deg: in (0, 180]")
        if not (_real(self.max_elevation_deg) and 0.0 <= self.max_elevation_deg < 90.0):
            raise ValueError("max_elevation_deg: in [0, 90)")
        if not (_real(self.closing_weight) and 0.0 <= self.closing_weight <= 1.0):
            raise ValueError("closing_weight: in [0, 1]")

    def allows(self, a: np.ndarray) -> bool:
        ref = np.asarray(self.reference_dir, dtype=float)
        ref = ref / np.linalg.norm(ref)
        a = np.asarray(a, dtype=float)
        a = a / np.linalg.norm(a)
        if float(a @ ref) < math.cos(math.radians(self.cone_half_angle_deg)) - 1e-9:
            return False
        if not self.allow_top_down and abs(float(a[2])) > math.sin(math.radians(self.max_elevation_deg)) + 1e-9:
            return False
        return True

    def in_workspace(self, mid) -> bool:
        """True if `max_reach_from` is off or the pinch midpoint lies within `radius` of the anchor (shoulder/torso
        position in the cloud frame; the radius is the arm's usable reach to a pinch point, measured per robot)."""
        if self.max_reach_from is None:
            return True
        c, r = self.max_reach_from
        return float(np.linalg.norm(np.asarray(mid, dtype=float) - np.asarray(c))) <= r

    def closing_factor(self, y: np.ndarray) -> float:
        pref = np.asarray(self.preferred_closing, dtype=float)
        pref = pref / np.linalg.norm(pref)
        return 1.0 - self.closing_weight * (1.0 - min(1.0, abs(float(np.asarray(y) @ pref))))


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return None if n < 1e-9 else v / n


def _clean(arr, drop_nonfinite: bool = True) -> np.ndarray | None:
    """(N,3) float array, or None if the input is not an (N,3) numeric array. Non-finite rows are dropped, or (with
    drop_nonfinite=False) make the whole input invalid (None)."""
    try:
        a = np.asarray(arr, dtype=float)
    except (TypeError, ValueError):
        return None
    if a.ndim != 2 or a.shape[1] != 3:
        return None
    finite = np.isfinite(a).all(axis=1)
    if not drop_nonfinite:
        if not finite.all():
            return None
        return a[(np.abs(a) <= _MAX_COORD).all(axis=1)]
    a = a[finite]
    return a[(np.abs(a) <= _MAX_COORD).all(axis=1)]   # absurd coordinates are not objects; no overflow downstream


_MAX_COORD = 1e3              # m: points farther than this from the origin are dropped (and cannot overflow squares)
_MIN_FOOTPRINT_SIDE = 0.002   # m: a rectangle side below this is noise, not a box (PLACEHOLDER)
_MIN_FOOTPRINT_AREA = 4e-6    # m^2
_KEY_LIM = 2.0 ** 52   # voxel index clip (well inside int64 and exactly representable)


def _voxel_downsample(a: np.ndarray, cell: float) -> np.ndarray:
    """Keep the first point of every `cell`-sized voxel (deterministic, order preserving). Every occupied region
    keeps at least one point, within cell * sqrt(3) / 2 of any dropped one."""
    if len(a) == 0:
        return a
    # clip before the int cast: a 1e200 coordinate must not overflow (it keeps a unique-enough key at the clip)
    keys = np.clip(np.floor(a / cell), -_KEY_LIM, _KEY_LIM).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return a[np.sort(first)]


def _min_dist(spheres: np.ndarray, cloud: np.ndarray, margin: float) -> float:
    """Smallest sphere-centre to point distance, exact when it is < margin, else any value >= margin (inf if no
    point is in the spheres' bounding box grown by margin). Chunked so memory stays bounded."""
    if len(cloud) == 0:
        return math.inf
    lo, hi = spheres.min(0) - margin, spheres.max(0) + margin
    near = cloud[((cloud >= lo) & (cloud <= hi)).all(axis=1)]
    best = math.inf
    for i in range(0, len(near), 2000):
        chunk = near[i:i + 2000]
        best = min(best, float(np.linalg.norm(spheres[:, None, :] - chunk[None, :, :], axis=2).min()))
    return best


def _stride(a: np.ndarray, cap: int) -> np.ndarray:
    if len(a) <= cap:
        return a
    return a[np.linspace(0, len(a) - 1, cap).astype(int)]


def estimate_normals(pts: np.ndarray, k: int = 12) -> np.ndarray:
    """Unsigned unit normals by local PCA over the k nearest neighbours (brute force). (n,3) -> (n,3)."""
    n = len(pts)
    k = min(k, n)
    sq = (pts * pts).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * pts @ pts.T
    idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
    nb = pts[idx]  # (n,k,3)
    nb = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    _, vecs = np.linalg.eigh(cov)
    return vecs[:, :, 0]


class AnalyticGraspProposer:
    """See module docstring. Deterministic for a fixed `seed`."""

    def __init__(
        self,
        spec: HandSpec | None = None,
        seed: int = 0,
        max_candidates: int = 20,
        min_z: float | None = None,
        min_points: int = 30,
        friction_cone_deg: float = 15.0,
        n_pairs: int = 6000,
        reach: ReachConstraint | None = ReachConstraint(),
        contact_refinement: bool = True,
        snap_tol: float = 0.01,
    ):
        """`reach=None` disables the executability filter; `reach=None, contact_refinement=False` is the original
        (pre-2026-11-10) behaviour exactly."""
        if reach is not None and not isinstance(reach, ReachConstraint):
            raise TypeError("reach: ReachConstraint or None")
        self.reach = reach
        self.contact_refinement = bool(contact_refinement)
        self.snap_tol = float(snap_tol)
        if not (math.isfinite(self.snap_tol) and self.snap_tol > 0):
            raise ValueError("snap_tol: positive finite")
        self.spec = spec or HandSpec()
        self.seed = int(seed)
        self.max_candidates = max(1, int(max_candidates))
        self.min_z = None if min_z is None else float(min_z)
        self.min_points = max(5, int(min_points))
        self.cos_cone = math.cos(math.radians(friction_cone_deg))
        self.n_pairs = int(n_pairs)

    # ------------------------------------------------------------------ public
    def propose(self, points: np.ndarray, obstacles: np.ndarray | None = None) -> list[GraspCandidate]:
        try:
            return self._propose(points, obstacles)
        except np.linalg.LinAlgError:  # degenerate geometry: fail closed
            return []

    # ----------------------------------------------------------------- internals
    def _propose(self, points, obstacles) -> list[GraspCandidate]:
        pts = _clean(points)
        if pts is None or len(pts) < self.min_points:
            return []
        obs = np.zeros((0, 3))
        if obstacles is not None:
            obs = _clean(obstacles, drop_nonfinite=False)
            if obs is None:
                return []  # malformed or non-finite obstacle cloud: fail closed
        cell = self.spec.finger_clearance_radius / 2.0
        # Collision clouds are voxel-downsampled, never strided: every obstacle region still contributes a point.
        obs = _voxel_downsample(obs, cell)
        tgt_all = _voxel_downsample(pts, cell)
        if len(obs) > _MAX_CLOUD_VOXELS or len(tgt_all) > _MAX_CLOUD_VOXELS:
            return []  # too dense to check exhaustively: fail closed
        pts = _stride(pts, _MAX_POINTS)  # candidate generation only
        centroid = pts.mean(0)
        half_diag = 0.5 * float(np.linalg.norm(pts.max(0) - pts.min(0)))
        scale = max(half_diag, 1e-6)

        if self.contact_refinement:
            raw = self._antipodal_pairs(pts) + self._footprint_pairs(pts)
        else:
            raw = self._antipodal_pairs(pts) + self._fallback_pairs(pts, centroid)
        raw.sort(key=lambda t: -t[4])
        raw = raw[:_PRESELECT]

        cands = []
        for p0, p1, d, q_ant, pre, src_f in raw:
            mid = 0.5 * (p0 + p1)
            c = (p1 - p0) / d
            for a in self._approaches(c):
                scored = self._check_and_score(tgt_all, obs, mid, c, a, d, q_ant, centroid, scale, src_f)
                if scored is not None:
                    cands.append(scored)
        cands.sort(key=lambda t: -t[0])
        return self._nms(cands)

    def _antipodal_pairs(self, pts: np.ndarray) -> list:
        sp = self.spec
        n = len(pts)
        normals = estimate_normals(pts)
        rng = np.random.default_rng(self.seed)
        i = rng.integers(0, n, self.n_pairs)
        j = rng.integers(0, n, self.n_pairs)
        diff = pts[j] - pts[i]
        d = np.linalg.norm(diff, axis=1)
        ok = (d >= sp.min_aperture) & (d <= sp.max_aperture)
        i, j, d, diff = i[ok], j[ok], d[ok], diff[ok]
        if len(d) == 0:
            return []
        c = diff / d[:, None]
        ci = np.abs((normals[i] * c).sum(1))
        cj = np.abs((normals[j] * c).sum(1))
        ok = (ci >= self.cos_cone) & (cj >= self.cos_cone)
        i, j, d, ci, cj = i[ok], j[ok], d[ok], ci[ok], cj[ok]
        out = []
        for a, b, dd, x, y in zip(i, j, d, ci, cj):
            q_ant = float(np.clip(((x + y) / 2 - self.cos_cone) / (1 - self.cos_cone), 0, 1))
            out.append((pts[a], pts[b], float(dd), q_ant, 0.4 * q_ant + 0.15 * self._q_ap(float(dd)), 1.0))
        return out

    def _fallback_pairs(self, pts: np.ndarray, centroid: np.ndarray) -> list:
        sp = self.spec
        rel = pts - centroid
        _, vecs = np.linalg.eigh(rel.T @ rel)
        out = []
        for e in vecs.T:
            t = rel @ e
            lo, hi = np.percentile(t, [1.0, 99.0])
            w = float(hi - lo)
            if not (sp.min_aperture <= w <= sp.max_aperture):
                continue
            # contacts on the cloud's visible extent along e, through the centroid in the other axes
            p0, p1 = centroid + e * lo, centroid + e * hi
            out.append((p0, p1, w, 0.5, 0.4 * 0.5 * 0.6 + 0.15 * self._q_ap(w) * 0.6, 0.6))
        return out

    def _footprint_pairs(self, pts: np.ndarray) -> list:
        """Box-prior pairs from the minimum-area rectangle of the horizontal projection (see module docstring)."""
        sp = self.spec
        xy = pts[:, :2]
        c2 = xy.mean(0)
        rel = xy - c2
        best = None
        for deg in range(0, 90):  # a rectangle repeats every 90 deg
            th = math.radians(deg)
            e0 = np.array([math.cos(th), math.sin(th)])
            e1 = np.array([-math.sin(th), math.cos(th)])
            t0, t1 = rel @ e0, rel @ e1
            lo0, hi0 = np.percentile(t0, _EXT_PCT)
            lo1, hi1 = np.percentile(t1, _EXT_PCT)
            area = (hi0 - lo0) * (hi1 - lo1)
            if best is None or area < best[0] - 1e-12:
                best = (area, e0, e1, lo0, hi0, lo1, hi1, t0, t1)
        area, e0, e1, lo0, hi0, lo1, hi1, t0, t1 = best
        # degenerate rectangle (collinear / near-point cloud): a zero-area box fits every point "perfectly", so it is
        # rejected before the residual is computed
        if min(hi0 - lo0, hi1 - lo1) < _MIN_FOOTPRINT_SIDE or area < _MIN_FOOTPRINT_AREA:
            return []
        # footprint fit residual: distance of every projected point to the rectangle boundary
        inside = np.minimum.reduce([t0 - lo0, hi0 - t0, t1 - lo1, hi1 - t1])
        resid = float(np.sqrt(np.mean(np.minimum(np.abs(inside), 0.05) ** 2)))
        q_fit = float(np.clip(1.0 - resid / 0.01, 0.0, 1.0))
        zlo, zhi = np.percentile(pts[:, 2], _EXT_PCT)
        zmid = 0.5 * (zlo + zhi)
        out = []
        for (e, lo, hi, e_o, lo_o, hi_o) in ((e0, lo0, hi0, e1, lo1, hi1), (e1, lo1, hi1, e0, lo0, hi0)):
            w = float(hi - lo)
            if not (sp.min_aperture <= w <= sp.max_aperture):
                continue
            s_mid = 0.5 * (lo_o + hi_o)
            t_e, s_e = rel @ e, rel @ e_o
            ends = []
            for t_end in (lo, hi):
                # observed points on this end plane (within snap_tol along e), nearest to the centred contact point
                slab = np.abs(t_e - t_end) <= self.snap_tol
                if not slab.any():
                    continue
                idx = np.flatnonzero(slab)
                dd = np.hypot(s_e[idx] - s_mid, pts[idx, 2] - zmid)
                j = int(np.argmin(dd))
                if dd[j] <= _SNAP_SHIFT:
                    ends.append((float(dd[j]), pts[idx[j]], t_end))
            if not ends:
                continue  # neither end is on an observed surface: reject (contact refinement)
            ends.sort(key=lambda t: t[0])
            dn, near, t_near = ends[0]
            sign = 1.0 if t_near == lo else -1.0  # far contact lies on the other side of the near one
            ev = np.array([e[0], e[1], 0.0])
            p0, p1 = near, near + sign * ev * w
            src_f = 0.6 * (0.5 + 0.5 * q_fit)
            q_ant = 0.5
            out.append((p0, p1, w, q_ant, 0.4 * q_ant * 0.6 + 0.15 * self._q_ap(w) * 0.6, src_f))
        return out

    def _q_ap(self, d: float) -> float:
        lo, hi = self.spec.min_aperture, self.spec.max_aperture
        half = 0.5 * (hi - lo)
        return float(np.clip(1.0 - abs(d - 0.5 * (lo + hi)) / half, 0.0, 1.0))

    def _approaches(self, c: np.ndarray) -> list[np.ndarray]:
        """Approach directions (unit, perpendicular to c) allowed by the spec."""
        if abs(float(c @ _UP)) > _TILT_LIMIT:
            return []
        return [a for a in self._approaches_unfiltered(c) if self.reach is None or self.reach.allows(a)]

    def _approaches_unfiltered(self, c: np.ndarray) -> list[np.ndarray]:
        out = []
        if "top" in self.spec.approaches:
            a = -_UP - (-_UP @ c) * c
            a = _unit(a)
            if a is not None:
                out.append(a)
        if "side" in self.spec.approaches:
            a = _unit(np.cross(_UP, c))
            if a is not None:
                out += [a, -a]
        return out

    def _spheres(self, mid, y, a, d) -> np.ndarray:
        """World-frame sphere centres of the two fingers (palm to contact) and the palm row."""
        sp = self.spec
        r = sp.finger_clearance_radius
        n_f = int(math.ceil(sp.palm_standoff / (0.8 * r))) + 1
        half = d / 2 + r
        n_p = int(math.ceil(2 * half / (0.8 * r))) + 1
        base = mid - a * sp.palm_standoff
        t = np.linspace(0.0, sp.palm_standoff, n_f)
        pl = np.linspace(-half, half, n_p)
        fingers = [base + a * ti + y * s for ti in t for s in (half, -half)]
        palm = [base + y * s for s in pl]
        return np.asarray(fingers + palm)

    def _check_and_score(self, pts, obs, mid, c, a, d, q_ant, centroid, scale, src_f):
        sp = self.spec
        r = sp.finger_clearance_radius
        y = c
        # canonical sign of the closing axis (deterministic)
        if y[int(np.argmax(np.abs(y)))] < 0:
            y = -y
        if self.reach is not None and not self.reach.in_workspace(mid):
            return None
        z = np.cross(a, y)
        R = np.stack([a, y, z], axis=1)
        spheres = self._spheres(mid, y, a, d)
        if self.min_z is not None and float(spheres[:, 2].min()) - r < self.min_z - 1e-9:
            return None
        c0, c1 = mid - y * d / 2, mid + y * d / 2
        near_contact = (np.linalg.norm(pts - c0, axis=1) < 1.5 * r) | (np.linalg.norm(pts - c1, axis=1) < 1.5 * r)
        tgt = pts[~near_contact]
        clr = np.inf
        if len(tgt):
            dt = _min_dist(spheres, tgt, 3 * r)
            if dt < 0.8 * r:
                return None
            clr = min(clr, dt)
        if len(obs):
            do = _min_dist(spheres, obs, 3 * r)
            if do < r:
                return None
            clr = min(clr, do)
        q_clr = 1.0 if not math.isfinite(clr) else float(np.clip((clr - r) / (2 * r), 0, 1))
        q_cen = 1.0 / (1.0 + (float(np.linalg.norm(mid - centroid)) / scale) ** 2)
        f_app = 1.0 if abs(float(a @ _UP)) > 0.9 else 0.9
        score = f_app * src_f * (0.40 * q_ant + 0.15 * self._q_ap(d) + 0.20 * q_cen + 0.25 * q_clr)
        if self.reach is not None:
            score *= self.reach.closing_factor(y)
        score = float(np.clip(score, 0.0, 1.0))
        base = mid - a * sp.palm_standoff
        return score, mid, y, a, GraspCandidate(Pose(tuple(base), matrix_to_quat(R)), float(d), score, "pinch")

    def _nms(self, cands) -> list[GraspCandidate]:
        kept = []
        for s, mid, y, a, cand in cands:
            dup = any(
                np.linalg.norm(mid - m2) < 0.01 and abs(float(y @ y2)) > 0.95 and float(a @ a2) > 0.9
                for _, m2, y2, a2, _ in kept
            )
            if not dup:
                kept.append((s, mid, y, a, cand))
                if len(kept) >= self.max_candidates:
                    break
        return [k[4] for k in kept]

