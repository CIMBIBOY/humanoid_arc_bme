"""Target-cloud cleaning: table removal, robot-body exclusion, connected component around a prior (numpy only).

The segmentation mask of the target (DAVE, later) is not perfect: the cloud inside it can hold table points under the
object, points of the robot's own hand/arm and a neighbouring object. `clean_target_cloud` removes these with geometry:

(a) Table: points within `tol` of the table plane z = table_z are dropped. If `table_z` is None the plane is estimated
    robustly from `context` (or the points themselves): the LOWEST z-histogram peak holding >= `table_frac` of the points
    (bin 5 mm), refined to the median of the points within 1 cm of it. `tol` = max(`tol_min`, 3 * sigma), sigma = 1.4826
    MAD of the z residuals of the points within 2 cm of the plane (the depth noise). Only points ABOVE the plane stay.
(b) Exclusions: points within `radius` of any `(centre, radius)` sphere are dropped. In simulation the centres are the
    articulation's hand/arm link positions; on the real robot they come from forward kinematics of the joint state (the
    link radii must cover the link geometry plus the depth noise).
(c) Connected component: occupied voxels (edge `voxel`, 26-neighbourhood) are labelled; with `prior_centre` the component
    whose points come closest to it is kept (ties: the larger), else the largest. A component whose closest point is
    farther than `max_prior_dist` from `prior_centre` is rejected, as is a result with fewer than `min_points` points.
Fail closed: empty, non-(N,3), all-non-finite input, invalid arguments' degenerate results => an empty (0,3) array.
Units: metres; frames: whatever frame all inputs share (the world frame in the eval).
"""

from __future__ import annotations

import numpy as np

_EMPTY = np.zeros((0, 3))


def _as_points(a) -> np.ndarray | None:
    try:
        p = np.asarray(a, dtype=float)
    except (TypeError, ValueError):
        return None
    if p.ndim != 2 or p.shape[1] != 3:
        return None
    return p[np.isfinite(p).all(axis=1)]


def estimate_table_z(points, bin_size: float = 0.005, table_frac: float = 0.10) -> float | None:
    """Height of the lowest dominant horizontal plane in `points`, or None (see module docstring)."""
    p = _as_points(points)
    if p is None or len(p) < 20:
        return None
    z = p[:, 2]
    lo = float(z.min())
    nb = int(np.ceil((float(z.max()) - lo) / bin_size)) + 1
    if nb > 100000:
        return None
    h = np.bincount(((z - lo) / bin_size).astype(int), minlength=nb)
    h = h + np.roll(h, 1) * (np.arange(nb) > 0) + np.roll(h, -1) * (np.arange(nb) < nb - 1)  # +-1 bin smoothing
    ok = np.flatnonzero(h >= table_frac * len(z))
    if ok.size == 0:
        return None
    b = int(ok[0])
    # climb to the local maximum of that first significant peak
    while b + 1 < nb and h[b + 1] > h[b]:
        b += 1
    z0 = lo + (b + 0.5) * bin_size
    near = z[np.abs(z - z0) < 0.01]
    return float(np.median(near)) if near.size else None


def _noise_sigma(z: np.ndarray, plane: float, window: float = 0.02) -> float:
    r = z[np.abs(z - plane) < window] - plane
    if r.size < 10:
        return 0.0
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def _components(vox: np.ndarray) -> np.ndarray:
    """Component label per row of the unique integer voxel array (26-neighbourhood), via union-find."""
    n = len(vox)
    mn = vox.min(0) - 1
    d = (vox - mn).astype(np.int64)
    dims = d.max(0) + 2
    key = (d[:, 0] * dims[1] + d[:, 1]) * dims[2] + d[:, 2]
    order = np.argsort(key)
    skey = key[order]
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) > (0, 0, 0)]
    for a, b, c in offs:
        nk = key + (a * dims[1] + b) * dims[2] + c
        pos = np.searchsorted(skey, nk)
        pos[pos >= n] = n - 1
        hit = skey[pos] == nk
        src = np.flatnonzero(hit)
        dst = order[pos[hit]]
        for i, j in zip(src.tolist(), dst.tolist()):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    return np.array([find(i) for i in range(n)])


def clean_target_cloud(points, table_z: float | None = None, exclusions=(), prior_centre=None, *, context=None,
                       tol_min: float = 0.004, noise_k: float = 3.0, voxel: float = 0.01, min_points: int = 30,
                       max_prior_dist: float = 0.10, table_frac: float = 0.10):
    """Return `(cloud (M,3), diagnostics)`; `cloud` is empty (0,3) on any failure. See the module docstring."""
    diag = {"n_in": 0, "n_table": 0, "n_excluded": 0, "n_components": 0, "n_out": 0, "table_z": None, "tol": None,
            "reason": None}
    p = _as_points(points)
    if p is None or len(p) == 0:
        diag["reason"] = "empty_or_invalid"
        return _EMPTY.copy(), diag
    diag["n_in"] = int(len(p))
    if not (voxel > 0 and np.isfinite(voxel)):
        diag["reason"] = "bad_voxel"
        return _EMPTY.copy(), diag
    pc = None
    if prior_centre is not None:
        pc = np.asarray(prior_centre, dtype=float)
        if pc.shape != (3,) or not np.isfinite(pc).all():
            diag["reason"] = "bad_prior_centre"
            return _EMPTY.copy(), diag

    # (a) table plane
    if table_z is None:
        table_z = estimate_table_z(context if context is not None else p, table_frac=table_frac)
    elif not np.isfinite(table_z):
        table_z = None
    if table_z is not None:
        ref = _as_points(context) if context is not None else p
        sigma = _noise_sigma(ref[:, 2], float(table_z)) if ref is not None and len(ref) else 0.0
        tol = max(tol_min, noise_k * sigma)
        keep = p[:, 2] > table_z + tol
        diag["n_table"] = int((~keep).sum())
        diag["table_z"], diag["tol"] = float(table_z), float(tol)
        p = p[keep]

    # (b) exclusion spheres
    for ex in exclusions:
        try:
            c, r = ex
            c = np.asarray(c, dtype=float)
            r = float(r)
        except (TypeError, ValueError):
            continue
        if c.shape != (3,) or not np.isfinite(c).all() or not np.isfinite(r) or r <= 0 or len(p) == 0:
            continue
        keep = np.linalg.norm(p - c, axis=1) > r
        diag["n_excluded"] += int((~keep).sum())
        p = p[keep]

    if len(p) == 0:
        diag["reason"] = "nothing_left"
        return _EMPTY.copy(), diag

    # (c) connected component
    vox = np.floor(p / voxel).astype(np.int64)
    uvox, inv = np.unique(vox, axis=0, return_inverse=True)
    inv = inv.ravel()
    lab = _components(uvox)[inv]
    ul, lab_i = np.unique(lab, return_inverse=True)
    lab_i = lab_i.ravel()
    diag["n_components"] = int(len(ul))
    counts = np.bincount(lab_i)
    if pc is None:
        best = int(np.argmax(counts))
    else:
        dist = np.linalg.norm(p - pc, axis=1)
        dmin = np.full(len(ul), np.inf)
        np.minimum.at(dmin, lab_i, dist)
        best = min(range(len(ul)), key=lambda i: (round(float(dmin[i]), 6), -counts[i]))
        if dmin[best] > max_prior_dist:
            diag["reason"] = "prior_too_far"
            return _EMPTY.copy(), diag
    out = p[lab_i == best]
    if len(out) < min_points:
        diag["reason"] = "too_small"
        return _EMPTY.copy(), diag
    diag["n_out"] = int(len(out))
    return out, diag
