import math

import numpy as np

from humanoid_arc_bme.cpusim.world import Box, Cylinder, World


def cam(origin, pitch_down):
    """Optical frame: x right (= world -y), z forward, pitched down by `pitch_down` rad about the y axis."""
    c, s = math.cos(pitch_down), math.sin(pitch_down)
    fwd, right = np.array([c, 0.0, -s]), np.array([0.0, -1.0, 0.0])
    down = np.cross(fwd, right)
    return np.asarray(origin, float), np.stack([right, down, fwd], axis=1)


def test_floor_depth_matches_analytic_ray():
    o, R = cam((0, 0, 1.2), math.radians(50))
    d, rgb, ids = World().render(o, R, 190, 190, 159.5, 119.5, 320, 240)
    cy = 119
    expected = 1.2 / math.sin(math.radians(50) + math.atan((cy - 119.5) / 190))   # centre row: optical z of the floor hit
    assert abs(float(d[cy, 160]) - expected) < 0.02
    assert (ids == 0).all() and np.isfinite(d).all()


def test_box_and_cylinder_hit_with_ids_and_colours():
    w = World(boxes=[Box(1.0, 1.6, -0.5, 0.5, 0.7)], cylinders=[Cylinder(1.2, 0.0, 0.05, 0.12, (200, 30, 30), "cup", base_z=0.7)])
    o, R = cam((0, 0, 1.2), math.radians(40))
    d, rgb, ids = w.render(o, R, 190, 190, 159.5, 119.5, 320, 240)
    assert set(np.unique(ids)) >= {1, 2}
    red = ids == 2
    assert red.any() and (rgb[red] == (200, 30, 30)).all()
    assert d[red].min() < d[ids == 1].mean()           # the cup is nearer than the average table pixel


def test_surface_z_under_a_box():
    w = World(boxes=[Box(1.0, 2.0, -1.0, 1.0, 0.7)])
    assert w.surface_z(1.5, 0.0) == 0.7 and w.surface_z(0.5, 0.0) == 0.0
