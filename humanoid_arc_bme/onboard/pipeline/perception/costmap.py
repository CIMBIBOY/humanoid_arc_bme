"""Robot-centred local 2-D costmap built from body-frame point clouds. Stdlib + numpy only.

Frame: body, x forward, y left, z up; metres, seconds. Clouds are (N, 3) in the body frame, so a depth camera
(`sensing.depth_to_points`) and a Livox scan can be concatenated and fed in together.

Limitations (v1, deliberate):
  * Egomotion is out of scope: the grid is always body-centred and is rebuilt in the body frame each update; cells
    age by timestamp only, they are not shifted when the robot moves.
  * No raycasting: observed-free space is NOT inferred from absence of points. A stale obstacle clears only when it
    has not been re-observed for `decay_s` seconds.
  * z band and ranges: z_min=0.10 / z_max=1.60 and the 3 m default range differ from the forward depth guard's
    defaults on purpose. The costmap is a planning map around a walking biped: it must ignore ground returns and
    overhead structure at robot height, and it is body-centred over +-3 m; the forward guard is a last-line stop
    with its own band and ranges tuned for the camera view. Do not unify them without re-measuring both.
  * Never-updated map fails closed: `occupied`/`inflated` report EVERY cell blocked, `nearest_obstacle` and
    `sector_min_distance` return 0.0 (distance zero = blocked). Unknown is never reported as free space.
  * Fail-safe: `healthy(now)` is False if there was never an update or the newest update is older than `max_age_s`.
    A consumer MUST stop (or zero its speed) while the map is unhealthy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from humanoid_arc_bme.onboard.nav.obstacle import sector_min_distance as _nav_sector_min_distance


def _pos(name: str, v: float) -> float:
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0:
        raise ValueError(f"{name}: must be a finite number > 0, got {v!r}")
    return float(v)


@dataclass(frozen=True)
class CostmapConfig:
    size_x: float = 6.0            # total extent along x, centred on the body (m)
    size_y: float = 6.0            # total extent along y (m)
    resolution: float = 0.05       # m per cell
    z_min: float = 0.10            # points below are ground/curb: ignored
    z_max: float = 1.60            # points above are overhang: ignored
    min_points_per_cell: int = 3   # noise rejection
    inflation_radius: float = 0.30  # robot radius + margin (m); 0 disables inflation
    decay_s: float = 1.0           # a cell not re-observed for this long clears
    max_age_s: float = 0.5         # healthy(now) requires the newest update to be at most this old

    def __post_init__(self):
        _pos("size_x", self.size_x)
        _pos("size_y", self.size_y)
        _pos("resolution", self.resolution)
        _pos("decay_s", self.decay_s)
        _pos("max_age_s", self.max_age_s)
        for n in ("z_min", "z_max", "inflation_radius"):
            v = getattr(self, n)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                raise ValueError(f"{n}: must be a finite number, got {v!r}")
        if self.z_min >= self.z_max:
            raise ValueError("z_min: must be < z_max")
        if self.inflation_radius < 0:
            raise ValueError("inflation_radius: must be >= 0")
        if self.size_x < self.resolution or self.size_y < self.resolution:
            raise ValueError("size: must be at least one cell")
        if not isinstance(self.min_points_per_cell, int) or isinstance(self.min_points_per_cell, bool) \
                or self.min_points_per_cell < 1:
            raise ValueError("min_points_per_cell: must be an int >= 1")


class LocalCostmap:
    def __init__(self, cfg: CostmapConfig | None = None):
        self.cfg = cfg or CostmapConfig()
        c = self.cfg
        self.nx = max(1, int(round(c.size_x / c.resolution)))
        self.ny = max(1, int(round(c.size_y / c.resolution)))
        self._x0 = -self.nx * c.resolution / 2.0
        self._y0 = -self.ny * c.resolution / 2.0
        self._last_seen = np.full((self.nx, self.ny), -np.inf)
        self._latest: float | None = None
        r = int(math.ceil(c.inflation_radius / c.resolution))
        d = np.arange(-r, r + 1)
        keep = (d[:, None] ** 2 + d[None, :] ** 2) * c.resolution ** 2 <= c.inflation_radius ** 2 + 1e-9
        self._r = r
        self._offsets = [(int(i), int(j)) for i, j in zip(*np.nonzero(keep))]  # indices into d
        self._d = d
        self._cx = self._x0 + (np.arange(self.nx) + 0.5) * c.resolution
        self._cy = self._y0 + (np.arange(self.ny) + 0.5) * c.resolution

    def update(self, points, stamp: float) -> None:
        """Bin one body-frame cloud (N, 3). Cost is O(N) vectorised. Non-finite rows, points outside the grid or
        outside the [z_min, z_max] band are dropped. Cells with >= min_points_per_cell points get last_seen = stamp;
        cells unseen for more than decay_s (relative to `stamp`) are cleared. An empty cloud is fine (still a
        heartbeat: it refreshes health, and decay still runs)."""
        if not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
            raise ValueError("stamp: must be a finite number")
        if self._latest is not None and stamp < self._latest:
            raise ValueError("stamp: must not go backwards")
        c = self.cfg
        p = np.asarray(points, dtype=np.float64)
        if p.size == 0:
            p = p.reshape(0, 3)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError("points: must have shape (N, 3)")
        # Decay first, so cells re-observed in this very update are not cleared.
        self._last_seen[(stamp - self._last_seen) > c.decay_s] = -np.inf
        ok = np.isfinite(p).all(axis=1)
        p = p[ok]
        p = p[(p[:, 2] >= c.z_min) & (p[:, 2] <= c.z_max)]
        # clip BEFORE the int cast: a 1e200 coordinate is simply out of the grid (no overflow / cast warning)
        lim = 1e6
        fx = np.floor((np.clip(p[:, 0], -lim, lim) - self._x0) / c.resolution)
        fy = np.floor((np.clip(p[:, 1], -lim, lim) - self._y0) / c.resolution)
        ix = np.clip(fx, -1.0, float(self.nx)).astype(np.int64)
        iy = np.clip(fy, -1.0, float(self.ny)).astype(np.int64)
        inb = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        flat = ix[inb] * self.ny + iy[inb]
        counts = np.bincount(flat, minlength=self.nx * self.ny).reshape(self.nx, self.ny)
        self._last_seen[counts >= c.min_points_per_cell] = stamp
        self._latest = stamp

    def healthy(self, now: float) -> bool:
        """False if never updated or the newest update is older than max_age_s (or in the future by > max_age_s)."""
        if self._latest is None or not math.isfinite(now):
            return False
        return abs(now - self._latest) <= self.cfg.max_age_s

    def occupied(self, now: float | None = None) -> np.ndarray:
        """(nx, ny) bool grid, un-inflated; `now` defaults to the newest update stamp. All True before the first update."""
        if self._latest is None:
            return np.ones((self.nx, self.ny), dtype=bool)  # never updated: unknown = blocked
        t = self._latest if now is None else now
        return (t - self._last_seen) <= self.cfg.decay_s

    def inflated(self, now: float | None = None) -> np.ndarray:
        """Occupied grid dilated by a disk of inflation_radius (cells within the radius of an occupied cell)."""
        occ = self.occupied(now)
        r = self._r
        if r == 0:
            return occ
        pad = np.pad(occ, r)
        out = np.zeros_like(occ)
        for i, j in self._offsets:
            out |= pad[i:i + self.nx, j:j + self.ny]
        return out

    def nearest_obstacle(self, heading_corridor_half_width: float, max_range: float,
                         now: float | None = None) -> float | None:
        """Distance ahead (body +x, cell-centre x) to the nearest inflated-occupied cell with |y| <= half width and
        0 < x <= max_range; None if clear. Includes cells already covered by the inflation of obstacles beside or
        behind the robot, which is conservative by design."""
        hw = float(heading_corridor_half_width)
        if not math.isfinite(hw) or hw < 0 or not math.isfinite(max_range) or max_range <= 0:
            raise ValueError("corridor half width must be >= 0 and max_range > 0")
        if self._latest is None:
            return 0.0  # never updated: fail closed (blocked)
        g = self.inflated(now)
        rows = (self._cx > 0) & (self._cx <= max_range)
        cols = np.abs(self._cy) <= hw
        sub = g[np.ix_(rows, cols)]
        hit = sub.any(axis=1)
        if not hit.any():
            return None
        return float(self._cx[rows][np.argmax(hit)])

    def sector_min_distance(self, direction: int, half_width: float = 0.35, max_range: float = 3.0,
                            min_range: float = 0.15, now: float | None = None) -> float:
        """Same contract as nav.obstacle.sector_min_distance (inf when nothing; direction +1 ahead / -1 behind;
        min_range ignores self-hits). That function's signature fits, so it is reused directly on the centres of
        the inflated-occupied cells (no copy of its logic)."""
        if self._latest is None:
            return 0.0  # never updated: fail closed (blocked)
        ii, jj = np.nonzero(self.inflated(now))
        xy = np.stack([self._cx[ii], self._cy[jj]], axis=1)
        return _nav_sector_min_distance(xy, direction, half_width, max_range, min_range)
