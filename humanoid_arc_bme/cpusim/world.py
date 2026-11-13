"""A small analytic world and its RGB-D / instance renderer (numpy ray casting).

World frame: x forward, y left, z up, floor at z = 0. Objects are vertical cylinders (cups, cans) and axis-aligned boxes
(tables, blocks), each with a flat colour and an integer id. `render` returns the optical-z depth image the guard and the
grasp branch expect (pinhole, x right / y down / z forward), an RGB image and an instance-id image.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FLOOR_RGB = (120, 120, 125)
SKY = np.inf


@dataclass
class Cylinder:
    x: float
    y: float
    radius: float
    height: float
    color: tuple[int, int, int]
    label: str = "cup"
    base_z: float = 0.0
    id: int = 0


@dataclass(frozen=True)
class Box:
    x0: float
    x1: float
    y0: float
    y1: float
    z1: float                     # top height (the bottom is the floor)
    color: tuple[int, int, int] = (160, 120, 80)
    label: str = "table"
    id: int = 0


@dataclass
class World:
    boxes: list[Box] = field(default_factory=list)
    cylinders: list[Cylinder] = field(default_factory=list)

    def __post_init__(self):
        self._renumber()

    def _renumber(self) -> None:
        i = 1
        for b in self.boxes:
            object.__setattr__(b, "id", i)
            i += 1
        for c in self.cylinders:
            c.id = i
            i += 1

    def names(self) -> dict[int, str]:
        return {**{b.id: b.label for b in self.boxes}, **{c.id: c.label for c in self.cylinders}}

    def surface_z(self, x: float, y: float, exclude: Cylinder | None = None) -> float:
        """Height of the support under (x, y): the top of a box that covers it, else the floor."""
        z = 0.0
        for b in self.boxes:
            if b.x0 <= x <= b.x1 and b.y0 <= y <= b.y1:
                z = max(z, b.z1)
        return z

    # ------------------------------------------------------------------ rendering
    def render(self, origin, R, fx: float, fy: float, cx: float, cy: float, width: int, height: int):
        """`origin` (3,) camera position and `R` (3, 3) optical-to-world rotation (columns x right, y down, z forward).
        Returns depth (H, W) float32 (inf = no hit), rgb (H, W, 3) uint8, ids (H, W) int32 (0 = floor / nothing)."""
        u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
        d_cam = np.stack(((u - cx) / fx, (v - cy) / fy, np.ones_like(u)), axis=-1).reshape(-1, 3)
        d = d_cam @ np.asarray(R, float).T                      # world directions with optical z = 1: t IS the depth
        o = np.asarray(origin, float)
        n = d.shape[0]
        t_best = np.full(n, np.inf)
        ids = np.zeros(n, dtype=np.int32)
        rgb = np.empty((n, 3), dtype=np.uint8)
        rgb[:] = FLOOR_RGB
        with np.errstate(divide="ignore", invalid="ignore"):
            dz = d[:, 2]
            t_floor = np.where(dz < -1e-9, -o[2] / dz, np.inf)
            hit = t_floor > 1e-6
            t_best[hit] = t_floor[hit]
            for b in self.boxes:
                t0, t1 = np.full(n, -np.inf), np.full(n, np.inf)
                for ax, (lo, hi) in enumerate(((b.x0, b.x1), (b.y0, b.y1), (0.0, b.z1))):
                    inv = 1.0 / d[:, ax]
                    ta, tb = (lo - o[ax]) * inv, (hi - o[ax]) * inv
                    t0, t1 = np.maximum(t0, np.minimum(ta, tb)), np.minimum(t1, np.maximum(ta, tb))
                ok = (t0 <= t1) & (t1 > 1e-6)
                t = np.where(t0 > 1e-6, t0, t1)
                hit = ok & (t < t_best)
                t_best[hit], ids[hit], rgb[hit] = t[hit], b.id, b.color
            for c in self.cylinders:
                ox, oy = o[0] - c.x, o[1] - c.y
                a = d[:, 0] ** 2 + d[:, 1] ** 2
                bq = 2.0 * (ox * d[:, 0] + oy * d[:, 1])
                cq = ox * ox + oy * oy - c.radius ** 2
                disc = bq * bq - 4.0 * a * cq
                sq = np.sqrt(np.maximum(disc, 0.0))
                t_side = (-bq - sq) / (2.0 * a)
                z_side = o[2] + t_side * dz
                side_ok = (disc >= 0) & (t_side > 1e-6) & (z_side >= c.base_z) & (z_side <= c.base_z + c.height)
                z_top = c.base_z + c.height
                t_top = (z_top - o[2]) / dz
                px, py = o[0] + t_top * d[:, 0] - c.x, o[1] + t_top * d[:, 1] - c.y
                top_ok = (dz < -1e-9) & (t_top > 1e-6) & (px * px + py * py <= c.radius ** 2)
                t = np.where(side_ok, t_side, np.inf)
                t = np.where(top_ok & (t_top < t), t_top, t)
                hit = np.isfinite(t) & (t < t_best)
                t_best[hit], ids[hit], rgb[hit] = t[hit], c.id, c.color
        depth = t_best.astype(np.float32)
        return depth.reshape(height, width), rgb.reshape(height, width, 3), ids.reshape(height, width)
