"""B1: behaviour of the rule-based depth guard on SYNTHETIC frames of a forward camera on a walking biped.

Every number here is SIM (synthetic), not MEASURED. A tiny vectorised ray caster renders a ground plane, axis-aligned
boxes and rectangular holes into a metric depth image using the guard's own camera model (default placeholders:
1.2 m high, 0.3 rad down, 0.1 m forward), then `apply_depth_noise` (UNMEASURED placeholder noise) is applied.
Resolution is half of the guard default (320x240, intrinsics halved, stride 1), which samples exactly the pixel grid
the default guard uses at stride 2. Set DEPTH_SCENARIO_VERBOSE=1 and run with -s to print the tables.
Body frame: x forward, y left, z up. Guard is NOT modified; assertions pin its measured behaviour with margins.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from humanoid_arc_bme.onboard.nav.depth_guard import DepthDistance, DepthGuardConfig
from humanoid_arc_bme.onboard.nav.obstacle import SpeedSeparationGuard
from _depth_render import CFG, NOISE, Box, Hole, noisy, render  # noqa: F401  (shared SIM renderer)

V = bool(os.environ.get("DEPTH_SCENARIO_VERBOSE"))
SEEDS = 12


def say(*a):
    if V:
        print(*a)


def guard_on(depth, cfg=CFG):
    dd = DepthDistance(cfg)
    g = SpeedSeparationGuard(dd, a_stop=0.3, d_margin=0.3, v_nominal=0.6)
    d = dd.submit(depth, 0.0)
    return d, g.speed_cap(0.0, 1)


def corridor_box(x, h, w=0.4, y=0.0):
    return Box(x, x + 0.3, y - w / 2, y + w / 2, h)


def test_ground_only_clean_is_clear():
    d, cap = guard_on(render(CFG))
    assert d == math.inf and cap == 0.6


# (1) tall box ---------------------------------------------------------------------------------------------------
def test_tall_box_distance_and_speed_cap():
    rows = []
    for x in (0.5, 0.75, 1.0, 1.5, 2.0, 2.5):
        ds, caps = [], []
        for s in range(SEEDS):
            d, cap = guard_on(noisy(render(CFG, [corridor_box(x, 1.0)]), s))
            ds.append(d)
            caps.append(cap)
        ds = np.array(ds)
        det = float(np.mean(np.isfinite(ds)))
        err = float(np.max(np.abs(ds[np.isfinite(ds)] - x))) if det else float("nan")
        rows.append((x, det, err, max(caps)))
        say(f"SIM tall box x={x}: detect={det:.2f} max|err|={err:.3f} max cap={max(caps):.3f}")
        assert det == 1.0
        assert err < 0.08
        # cap never exceeds the kinematic limit for the TRUE gap (body origin to face = x)
        assert max(caps) <= math.sqrt(2 * 0.3 * max(0.0, x - 0.3)) + 0.05
    # stopping: face closer than margin -> cap 0
    # (box nearer than min_range of the sensor geometry is the blind zone, see low-near test)
    d, cap = guard_on(noisy(render(CFG, [corridor_box(0.35, 1.0)]), 0))
    assert cap <= 0.2


# (2) low box ------------------------------------------------------------------------------------------------------
def test_low_box_detection_threshold():
    res = {}
    for h in (0.10, 0.15, 0.18, 0.2, 0.25, 0.3):
        for x in (0.8, 1.5):
            det = np.mean([np.isfinite(guard_on(noisy(render(CFG, [corridor_box(x, h)]), s))[0])
                           for s in range(SEEDS)])
            res[(h, x)] = float(det)
            say(f"SIM low box h={h} x={x}: detect={det:.2f}")
    assert res[(0.10, 0.8)] == 0.0 and res[(0.10, 1.5)] == 0.0      # below z_min = 0.12: invisible by design
    assert res[(0.3, 0.8)] >= 0.9
    # taller than z_min but still marginal at range (top face spans few pixels)
    assert res[(0.2, 0.8)] >= res[(0.15, 0.8)]


# (3) thin pole ------------------------------------------------------------------------------------------------------
def test_thin_pole_detection_vs_min_hits():
    rows = {}
    for x in (0.5, 1.0, 1.5):
        for mh in (1, 3, 6):
            cfg = DepthGuardConfig(fx=190.0, fy=190.0, cx=160.0, cy=120.0, width=320, height=240, stride=1,
                                   min_hits=mh)
            det = np.mean([np.isfinite(guard_on(noisy(render(cfg, [Box(x, x + 0.03, -0.015, 0.015, 1.0)]), s), cfg)[0])
                           for s in range(SEEDS)])
            rows[(x, mh)] = float(det)
            say(f"SIM 3cm pole x={x} min_hits={mh}: detect={det:.2f}")
    assert rows[(0.5, 3)] >= 0.9               # near: wide enough in pixels
    assert rows[(1.5, 3)] <= rows[(0.5, 3)]    # detection never improves with range
    assert rows[(1.5, 6)] <= rows[(1.5, 1)]    # stricter min_hits cannot detect more


def test_thinner_poles_and_bar_edge_cases():
    """1 cm pole and a 3 cm deep, 0.33 m high, 0.6 m wide slab vs range and min_hits."""
    r = {}
    for name, mk in (("pole1cm", lambda x: Box(x, x + 0.01, -0.005, 0.005, 1.0)),
                     ("slab3cm", lambda x: Box(x, x + 0.03, -0.3, 0.3, 0.33))):
        for x in (0.5, 1.5, 2.5):
            for mh in (3, 8):
                cfg = DepthGuardConfig(fx=190.0, fy=190.0, cx=160.0, cy=120.0, width=320, height=240, stride=1,
                                       min_hits=mh)
                fr = render(cfg, [mk(x)])
                det = np.mean([np.isfinite(guard_on(noisy(fr, s), cfg)[0]) for s in range(SEEDS)])
                r[(name, x, mh)] = float(det)
                say(f"SIM {name} x={x} min_hits={mh}: detect={det:.2f}")
    assert r[("pole1cm", 0.5, 3)] >= 0.9
    assert r[("slab3cm", 0.5, 3)] == 0.0   # near-field blind zone, see test_near_field_blind_zone
    for k in r:
        if k[2] == 8:
            assert r[k] <= r[(k[0], k[1], 3)] + 1e-9


def test_near_field_blind_zone():
    """Camera 1.2 m up, 0.3 rad down, vfov half-angle ~32 deg: the lowest ray lands ~1 m ahead, so a short object just
    in front of the feet is invisible although it is the one the biped will hit first."""
    r = {}
    for x in (0.3, 0.5, 0.7, 0.9, 1.1, 1.3):
        r[x] = float(np.mean([np.isfinite(guard_on(noisy(render(CFG, [Box(x, x + 0.05, -0.2, 0.2, 0.3)]), s))[0])
                              for s in range(SEEDS)]))
        say(f"SIM 0.30 m high 5 cm deep slab at x={x}: detect={r[x]:.2f}")
    assert r[0.3] == 0.0 and r[0.5] == 0.0
    assert r[1.3] == 1.0
    ground_first = 1.2 / math.tan(math.atan(120 / 190.0) + 0.3) + 0.1
    say(f"SIM first visible ground row at ~{ground_first:.2f} m ahead of body")
    assert 0.9 < ground_first < 1.3


# (4) negative obstacle -------------------------------------------------------------------------------------------------
def test_step_down_hole_not_detected_SAFETY_GAP():
    """SAFETY GAP: guard documents that negative obstacles are NOT detected. Pin it so a future fix flips this test."""
    for depth in (0.15, 0.4, 1.0):
        for x in (0.6, 1.2):
            hole = Hole(x, x + 1.5, -0.6, 0.6, depth)
            n = sum(np.isfinite(guard_on(noisy(render(CFG, holes=[hole]), s))[0]) for s in range(SEEDS))
            say(f"SIM hole depth={depth} x={x}: false 'obstacle' reports {n}/{SEEDS} (0 = hole invisible/no stop)")
            assert n == 0, "guard unexpectedly reacts to a hole; update findings and this test"


# (5) body sway ------------------------------------------------------------------------------------------------------
def _sway(rng, pmax=math.radians(5), rmax=math.radians(3)):
    return rng.uniform(-pmax, pmax), rng.uniform(-rmax, rmax)


def test_sway_false_stops_on_empty_floor():
    """Key number. Pitch error tilts the assumed ground plane; ground returns then pass the z_min=0.12 m filter."""
    out = {}
    for label, pm, rm in (("0deg/0deg", 0, 0), ("1deg/0.5deg", 1, 0.5), ("3deg/2deg", 3, 2), ("5deg/3deg", 5, 3)):
        n, det, slow, dmin = 80, 0, 0, math.inf
        rng = np.random.default_rng(7)
        for i in range(n):
            pj, rj = _sway(rng, math.radians(pm), math.radians(rm))
            d, cap = guard_on(noisy(render(CFG, pitch_j=pj, roll_j=rj), i))
            det += math.isfinite(d)
            slow += cap < 0.6
            dmin = min(dmin, d)
        out[label] = (det / n, slow / n)
        say(f"SIM sway {label} on EMPTY floor: phantom-obstacle rate={det/n:.2f} speed-reduced rate={slow/n:.2f} "
            f"nearest phantom={dmin:.2f} m")
    assert out["0deg/0deg"][0] == 0.0
    assert out["5deg/3deg"][0] >= out["1deg/0.5deg"][0]


def test_sway_hides_or_keeps_real_obstacle():
    det = {}
    for label, pm, rm in (("0deg", 0, 0), ("5deg/3deg", 5, 3)):
        rng = np.random.default_rng(11)
        r = []
        for i in range(40):
            pj, rj = _sway(rng, math.radians(pm), math.radians(rm))
            d, _ = guard_on(noisy(render(CFG, [corridor_box(1.2, 0.3)], pitch_j=pj, roll_j=rj), i))
            r.append(np.isfinite(d))
        det[label] = float(np.mean(r))
        say(f"SIM sway {label}: 0.3 m box at 1.2 m detect={det[label]:.2f}")
    assert det["0deg"] >= 0.9
    assert det["5deg/3deg"] >= 0.5


def test_sway_static_pitch_error_threshold_and_low_box_hiding():
    phantom = {}
    for deg in (-8, -5, -3, -1, 0, 1, 3, 5, 8):
        d, _ = guard_on(noisy(render(CFG, pitch_j=math.radians(deg)), 0))
        phantom[deg] = d
        say(f"SIM static pitch error {deg:+d} deg, empty floor: reported distance {d:.2f}")
    assert phantom[0] == math.inf
    assert all(phantom[k] == math.inf for k in (-8, -5, -3, -1, 1))      # camera pitched UP vs assumed: ground looks lower, ignored
    assert phantom[5] < math.inf and phantom[8] < math.inf                                          # camera pitched DOWN >~3 deg vs assumed: ground looks higher than z_min -> phantom obstacle (+pitch_j = down)
    hidden = {}
    for deg in (-8, -5, 0, 5):
        r = [np.isfinite(guard_on(noisy(render(CFG, [corridor_box(1.5, 0.15)], pitch_j=math.radians(deg)), s))[0])
             for s in range(SEEDS)]
        hidden[deg] = float(np.mean(r))
        say(f"SIM 0.15 m box at 1.5 m with pitch error {deg:+d} deg: detect={hidden[deg]:.2f}")
    assert hidden[0] == 1.0
    assert hidden[5] >= 0.9
    assert hidden[-5] == 0.0 and hidden[-8] == 0.0   # low box HIDDEN when camera is pitched up 5 deg


# (6) lens covered -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("fill", [np.nan, 0.0, -np.inf])
def test_covered_lens_stops(fill):
    d, cap = guard_on(np.full((CFG.height, CFG.width), fill, np.float32))
    assert d == 0.0 and cap == 0.0


def test_partially_covered_lens_over_half_stops():
    f = noisy(render(CFG), 0)
    f[:, : int(CFG.width * 0.6)] = np.nan
    assert guard_on(f)[0] == 0.0


# (7) outside corridor --------------------------------------------------------------------------------------------------
def test_obstacle_outside_corridor_ignored():
    for y in (0.7, -0.7, 1.0):
        n = sum(np.isfinite(guard_on(noisy(render(CFG, [corridor_box(1.0, 1.0, y=y)]), s))[0]) for s in range(SEEDS))
        say(f"SIM box at y={y}: stops {n}/{SEEDS}")
        assert n == 0
