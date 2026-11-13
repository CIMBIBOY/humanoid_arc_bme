"""Shared SIM depth renderer for the B1 depth tests (ground plane, axis-aligned boxes, rectangular holes).

Not a test module. Camera model = the depth guard's placeholder mount (1.2 m high, 0.3 rad down, 0.1 m forward),
rendered at 320x240 with halved intrinsics. Ray directions are in (forward, right, up); `body_rot` (optional,
world-from-body, x fwd / y left / z up) tilts the whole rig about the body origin, which stays on the ground plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from humanoid_arc_bme.onboard.nav.depth_guard import DepthGuardConfig
from humanoid_arc_bme.onboard.pipeline.perception.sensing import DepthNoise, apply_depth_noise

CFG = DepthGuardConfig(fx=190.0, fy=190.0, cx=160.0, cy=120.0, width=320, height=240, stride=1)
NOISE = DepthNoise()


@dataclass(frozen=True)
class Box:
    x0: float
    x1: float
    y0: float
    y1: float
    z1: float  # top height above ground (bottom is the floor)


@dataclass(frozen=True)
class Hole:
    x0: float
    x1: float
    y0: float
    y1: float
    depth: float


_GRID = {}


def _rays(cfg):
    k = (cfg.width, cfg.height, cfg.fx, cfg.fy, cfg.cx, cfg.cy)
    if k not in _GRID:
        u, v = np.meshgrid(np.arange(cfg.width, dtype=np.float64), np.arange(cfg.height, dtype=np.float64))
        _GRID[k] = ((u - cfg.cx) / cfg.fx, (v - cfg.cy) / cfg.fy)
    return _GRID[k]


def render(cfg, boxes=(), holes=(), pitch_j=0.0, roll_j=0.0, z_off=0.0, body_rot=None):
    """Metric depth (optical z, as the guard expects). pitch_j/roll_j (rad) perturb the TRUE camera pose only
    (about the camera, roll about the optical axis). `body_rot` instead rotates rays and camera position about the
    body origin (a real torso attitude)."""
    xn, yn = _rays(cfg)
    cr, sr = math.cos(roll_j), math.sin(roll_j)
    xr, yr = xn * cr - yn * sr, xn * sr + yn * cr          # roll about the optical axis
    p = cfg.cam_pitch + pitch_j
    sp, cp = math.sin(p), math.cos(p)
    dx, dy, dz = cp - yr * sp, xr, -sp - yr * cp          # ray direction in body (fwd, right, up); lateral sign irrelevant
    ox, oy, oz = cfg.cam_forward, 0.0, cfg.cam_height + z_off
    if body_rot is not None:
        R = np.asarray(body_rot, dtype=np.float64)
        d = np.einsum("ij,jhw->ihw", R, np.stack([dx, -dy, dz]))      # (fwd, left, up) -> world
        dx, dy, dz = d[0], -d[1], d[2]
        o = R @ np.array([ox, 0.0, oz])
        ox, oy, oz = o[0], -o[1], o[2]
    t = np.full(xn.shape, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = np.where(dz < 0, -oz / dz, np.inf)
        t = np.minimum(t, tg)
        for h in holes:
            gx, gy = ox + tg * dx, oy + tg * dy
            inside = (gx >= h.x0) & (gx <= h.x1) & (gy >= h.y0) & (gy <= h.y1) & np.isfinite(tg)
            tb = (-h.depth - oz) / dz                       # hole floor
            bx, by = ox + tb * dx, oy + tb * dy
            floor_ok = inside & (bx <= h.x1) & (by >= h.y0) & (by <= h.y1)
            tw = (h.x1 - ox) / dx                           # far wall of the hole
            t = np.where(inside, np.where(floor_ok, tb, tw), t)
        for b in boxes:
            lo = np.array([b.x0 - ox, b.y0 - oy, 0.0 - oz])
            hi = np.array([b.x1 - ox, b.y1 - oy, b.z1 - oz])
            d = np.stack([dx, dy, dz])
            t0, t1 = np.zeros(xn.shape), np.full(xn.shape, np.inf)
            ok = np.ones(xn.shape, bool)
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
    out = np.where(np.isfinite(t) & (t > 0), t, np.nan)
    return np.where(out > 20.0, np.nan, out).astype(np.float32)


def noisy(depth, seed):
    return apply_depth_noise(depth, NOISE, np.random.default_rng(seed))
