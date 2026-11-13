"""SIM Livox-Mid-360-like lidar: ray-casts the same Box/Hole/ground scenes as `_depth_render`. Not a test module.

All sensor parameters are FROM MEMORY / UNVERIFIED (not checked against a datasheet or the real unit): 360 deg
azimuth, vertical FOV -7..+52 deg about the sensor axis, ~200k points/s (20k per 10 Hz frame), range noise ~2 cm
(1 sigma), min range 0.1 m. The real non-repetitive rosette pattern is approximated by independent uniform random
azimuth/elevation per frame (so per-frame coverage is a Poisson-like scatter, not the real pattern). The mount
(height, forward offset, PITCH down) is UNMEASURED: it is a study parameter.

Frames: body x fwd / y left / z up with the origin ON the floor (the depth sim's convention). Sensor pitch is
positive = sensor axis tilted NOSE DOWN (same sign as `attitude_matrix`). Returned points are in the body frame.
The box/hole intersection is a vectorised copy of `_depth_render.render` for arbitrary rays (that function builds rays
from a pinhole camera and cannot take them); the `Box`/`Hole` classes are imported from it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from _depth_render import Box, Hole  # noqa: F401  (re-exported)


@dataclass(frozen=True)
class LidarSpec:
    mount_height: float = 1.2        # m above the floor (UNMEASURED; matches the depth sim's camera height)
    mount_forward: float = 0.1       # m ahead of the body origin
    pitch_deg: float = 0.0           # mount pitch, + = axis tilted down
    fov_down_deg: float = -7.0       # lowest ray relative to the sensor axis (UNVERIFIED, from memory)
    fov_up_deg: float = 52.0         # highest ray (UNVERIFIED, from memory)
    n_points: int = 20000            # per frame (200k pts/s at 10 Hz, UNVERIFIED)
    min_range: float = 0.1
    max_range: float = 20.0
    range_sigma: float = 0.02        # m, Gaussian along the ray
    dropout: float = 0.05            # probability a return is lost


def cast(ox, oy, oz, d, boxes=(), holes=()):
    """Range t along each unit ray d (3, N) from (ox, oy, oz); inf = no hit. Floor z=0, holes, axis-aligned boxes."""
    dx, dy, dz = d
    t = np.full(dx.shape, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = np.where(dz < 0, -oz / dz, np.inf)
        t = np.minimum(t, tg)
        for h in holes:
            gx, gy = ox + tg * dx, oy + tg * dy
            inside = (gx >= h.x0) & (gx <= h.x1) & (gy >= h.y0) & (gy <= h.y1) & np.isfinite(tg)
            tb = (-h.depth - oz) / dz
            bx, by = ox + tb * dx, oy + tb * dy
            floor_ok = inside & (bx <= h.x1) & (by >= h.y0) & (by <= h.y1)
            tw = (h.x1 - ox) / dx
            t = np.where(inside, np.where(floor_ok, tb, tw), t)
        for b in boxes:
            lo = np.array([b.x0 - ox, b.y0 - oy, 0.0 - oz])
            hi = np.array([b.x1 - ox, b.y1 - oy, b.z1 - oz])
            t0, t1 = np.zeros(dx.shape), np.full(dx.shape, np.inf)
            ok = np.ones(dx.shape, bool)
            for i in range(3):
                inv = 1.0 / d[i]
                a, c = lo[i] * inv, hi[i] * inv
                ta, tb2 = np.minimum(a, c), np.maximum(a, c)
                par = np.abs(d[i]) < 1e-12
                ok &= np.where(par, (lo[i] <= 0) & (hi[i] >= 0), True)
                t0 = np.where(par, t0, np.maximum(t0, ta))
                t1 = np.where(par, t1, np.minimum(t1, tb2))
            hit = ok & (t0 <= t1) & (t0 > 0)
            t = np.where(hit, np.minimum(t, t0), t)
    return t


def scan(spec: LidarSpec, boxes=(), holes=(), seed: int = 0) -> np.ndarray:
    """One frame, (M, 3) body-frame points, seeded. Random az/el per frame, Gaussian range noise, dropout."""
    rng = np.random.default_rng(seed)
    n = spec.n_points
    az = rng.uniform(-math.pi, math.pi, n)
    # uniform in solid angle (sin of elevation) over the vertical FOV, like a scanner that fills the FOV evenly
    lo, hi = math.sin(math.radians(spec.fov_down_deg)), math.sin(math.radians(spec.fov_up_deg))
    el = np.arcsin(rng.uniform(lo, hi, n))
    ce = np.cos(el)
    s = np.stack([ce * np.cos(az), ce * np.sin(az), np.sin(el)])        # sensor frame (x fwd, y left, z up)
    p = math.radians(spec.pitch_deg)
    cp, sp = math.cos(p), math.sin(p)
    d = np.stack([s[0] * cp + s[2] * sp, s[1], -s[0] * sp + s[2] * cp])  # Ry(pitch): +x tilts toward -z
    ox, oz = spec.mount_forward, spec.mount_height
    t = cast(ox, 0.0, oz, d, boxes, holes)
    t = t + rng.normal(0.0, spec.range_sigma, n)
    keep = np.isfinite(t) & (t >= spec.min_range) & (t <= spec.max_range) & (rng.random(n) >= spec.dropout)
    pts = np.stack([ox + t * d[0], t * d[1], oz + t * d[2]], axis=1)[keep]
    return pts
