"""Rule-based depth-image obstacle distance for the forward corridor (pure numpy, CPU).

Turns one metric depth frame (H x W, metres, camera optical frame) into the nearest obstacle distance ahead
inside a body-width corridor and a height band. The result feeds `SpeedSeparationGuard(distance=...)` (obstacle.py),
which converts it into a speed cap; `tighten_command` applies a cap to a velocity command and can only reduce it.

Geometry (assumed until the head camera is measured, inventory M03/U22): optical frame x right, y down, z forward;
camera `cam_height` m above the ground, pitched down by `cam_pitch` rad, `cam_forward` m ahead of the body origin.
Points are mapped to body X (forward), lateral (left = -x_opt), Z (up). Pixels with Z outside
[z_min, z_max] (ground, ceiling) or |lateral| > half_width are ignored. Negative obstacles (holes, stair edges
down) are NOT detected by this rule.

Fail-safe: a missing/stale frame returns None (the guard then stops after its own stale_s); a frame that is not a
2-D finite-ish float array, has the wrong shape, or has more than `max_invalid_frac` invalid pixels (NaN, <= 0,
-inf; e.g. covered lens, saturated stereo) returns 0.0 = stop. +inf and values above `max_range` count as "far".
It does NOT replace S1 (api/safety.py) or the vendor's protection; it is a tightening-only input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthGuardConfig:
    fx: float = 380.0
    fy: float = 380.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480
    cam_height: float = 1.2       # m above ground (PLACEHOLDER, measure)
    cam_pitch: float = 0.3        # rad, positive = looking down (PLACEHOLDER, measure)
    cam_forward: float = 0.1      # m ahead of body origin
    half_width: float = 0.35      # corridor half width, m
    z_min: float = 0.12           # ignore returns lower than this above ground
    z_max: float = 1.4            # ignore returns higher than this
    min_range: float = 0.15
    max_range: float = 3.0
    max_invalid_frac: float = 0.5
    min_hits: int = 3             # k-th nearest hit is reported, so isolated noise pixels do not stop the robot
    stride: int = 2               # subsample pixels
    max_age_s: float = 0.3


def _tighten_scalar(cap: float) -> float:
    if cap is None or not math.isfinite(cap):
        return 0.0 if (cap is None or cap != math.inf) else math.inf
    return max(0.0, cap)


def tighten_command(vx: float, vy: float, wz: float, cap: float) -> tuple[float, float, float]:
    """Scale (vx, vy, wz) together so |vx| <= cap. Never increases any component. NaN/None/negative cap -> zeros;
    non-finite command components -> zeros."""
    c = _tighten_scalar(cap)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (vx, vy, wz)):
        return 0.0, 0.0, 0.0
    if c == 0.0:
        return 0.0, 0.0, 0.0
    if abs(vx) <= c or abs(vx) == 0.0:
        return vx, vy, wz
    k = c / abs(vx)
    return vx * k, vy * k, wz * k


class DepthDistance:
    """Callable `distance(direction) -> float | None` for SpeedSeparationGuard. Feed frames with `submit`."""

    def __init__(self, cfg: DepthGuardConfig = DepthGuardConfig(), now_fn=None, rear_unknown: bool = True):
        self.cfg = cfg
        self._now = now_fn
        self.rear_unknown = rear_unknown  # no rear sensor: direction -1 returns None (guard fails closed)
        self._d: float | None = None
        self._t: float | None = None
        self._grid = None

    def _pixel_grid(self):
        if self._grid is None:
            c = self.cfg
            us = np.arange(0, c.width, max(1, c.stride), dtype=np.float32)
            vs = np.arange(0, c.height, max(1, c.stride), dtype=np.float32)
            u, v = np.meshgrid(us, vs)
            self._grid = ((u - c.cx) / c.fx, (v - c.cy) / c.fy)
        return self._grid

    def forward_distance(self, depth) -> float:
        """Nearest obstacle distance ahead (m): inf = nothing in range, 0.0 = invalid frame (stop)."""
        c = self.cfg
        try:
            d = np.asarray(depth, dtype=np.float32)
        except (TypeError, ValueError):
            return 0.0
        if d.ndim != 2 or d.shape != (c.height, c.width):
            return 0.0
        d = d[:: max(1, c.stride), :: max(1, c.stride)]
        invalid = np.isnan(d) | (d <= 0.0) | (d == -np.inf)
        if float(invalid.mean()) > c.max_invalid_frac:
            return 0.0
        xn, yn = self._pixel_grid()
        z = np.where(invalid, 0.0, np.minimum(d, 1e6))
        x_opt, y_opt = xn * z, yn * z
        sp, cp = math.sin(c.cam_pitch), math.cos(c.cam_pitch)
        fwd = c.cam_forward + z * cp - y_opt * sp
        up = c.cam_height - z * sp - y_opt * cp
        keep = (~invalid) & (np.abs(x_opt) <= c.half_width) & (up >= c.z_min) & (up <= c.z_max) \
            & (fwd >= c.min_range) & (fwd <= c.max_range)
        hits = fwd[keep]
        if hits.size < max(1, c.min_hits):
            return math.inf
        k = max(1, c.min_hits) - 1
        return float(np.partition(hits, k)[k])

    def submit(self, depth, now: float) -> float:
        self._d = self.forward_distance(depth)
        self._t = now
        return self._d

    def __call__(self, direction: int) -> float | None:
        if direction < 0 and self.rear_unknown:
            return None
        if self._d is None or self._t is None:
            return None
        if self._now is not None and self._now() - self._t > self.cfg.max_age_s:
            return None
        return self._d
