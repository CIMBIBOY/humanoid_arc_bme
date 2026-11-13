"""Bounded turn-in-place exception of PipelineRunner (user decision 2026-11-10). SIM results, CPU only.

Soft obstacle stop (`obstacle_stop` the only reason) + full health + no inflated-occupied cell within
`turn_clearance_m` of the body centre -> (0, 0, clamp(wz, +-turn_wz_max)), reason `turn_only`, status stays stop.
Any hard fault -> all zeros incl. wz.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from _depth_render import Box, Hole
from humanoid_arc_bme.onboard.pipeline.perception.costmap import CostmapConfig, LocalCostmap
from humanoid_arc_bme.onboard.pipeline.interfaces import DepthFrame, LocalizerEstimate, yaw_pose
from humanoid_arc_bme.onboard.pipeline.runner import STOP, PipelineRunner, RunnerConfig
from test_pipeline_runner import DT, K, MOUNT, drive, fwd_frame, tick_in, warm

HOLE = Hole(0.5, 1.5, -0.6, 0.6, 0.4)    # step-down 0.5 m ahead: soft obstacle_stop, no costmap cell (below z_min)
WALL = Box(0.55, 0.9, -0.6, 0.6, 0.8)    # solid obstacle ahead: inflated cells near the body
Z = (0.0, 0.0, 0.0)


def stopped_runner(**cfg):
    r = PipelineRunner(RunnerConfig(**cfg) if cfg else None)
    _, t = warm(r)
    return r, t


def _inject(r, t, x, y=0.0):
    cm = r.costmap
    cm._last_seen[int(np.argmin(np.abs(cm._cx - x))), int(np.argmin(np.abs(cm._cy - y)))] = t


def tick(r, t, cmd=(0.2, 0.1, 0.3), holes=(), boxes=(), stop=True, **kw):
    """stop=True: a raw costmap cell 0.60 m ahead (inflated edge 0.30 m = d_margin -> soft obstacle_stop, but outside
    the 0.45 m turn clearance). It is not a depth-geometry hit, so the obstacle memory stays empty."""
    if stop:
        _inject(r, t, 0.60)
    return r.tick(tick_in(t, fwd_frame(t, boxes=boxes, holes=holes, seed=abs(int(t * 10))), cmd=cmd, **kw))


def test_soft_obstacle_stop_with_clear_surroundings_turns_clamped():
    r, t = stopped_runner()
    out = tick(r, t)
    assert out.command_out == (0.0, 0.0, pytest.approx(0.15))          # 0.3 clamped to max_wz / 2
    assert out.safety.level == STOP and set(out.safety.reasons) == {"obstacle_stop", "turn_only"}
    out = tick(r, t + DT, cmd=(0.2, 0.0, -0.3))
    assert out.command_out == (0.0, 0.0, pytest.approx(-0.15))
    out = tick(r, t + 2 * DT, cmd=(0.2, 0.0, 0.05))
    assert out.command_out == (0.0, 0.0, pytest.approx(0.05))          # below the limit: untouched
    out = tick(r, t + 3 * DT, cmd=(0.2, 0.0, 0.0))
    assert out.command_out == Z and out.safety.level == STOP


def test_turn_wz_max_override():
    r, t = stopped_runner(turn_wz_max=0.1)
    assert tick(r, t).command_out == (0.0, 0.0, pytest.approx(0.1))


@pytest.mark.parametrize("cmd", [(0.2, 0.1, 0.3), (-0.2, 0.1, -0.3), (0.0, -0.1, 0.2), (0.05, 0.0, 0.0),
                                 (9.0, 9.0, 9.0)])
def test_translation_always_zero_and_wz_never_exceeds_request(cmd):
    r, t = stopped_runner()
    out = tick(r, t, cmd=cmd)
    assert out.command_out[0] == 0.0 and out.command_out[1] == 0.0
    assert abs(out.command_out[2]) <= min(abs(cmd[2]), 0.15) + 1e-12
    assert out.command_out[2] == 0.0 or math.copysign(1, out.command_out[2]) == math.copysign(1, cmd[2])


def test_turn_clearance_config_changes_the_decision():
    r = PipelineRunner(RunnerConfig(turn_clearance_m=0.2))
    _, t = warm(r)
    _inject(r, t, 0.30)                           # inflated edge 0.0 -> stop; raw 0.30 m > 0.2 clearance
    assert tick(r, t).command_out[2] == pytest.approx(0.15)
    r = PipelineRunner()
    _, t = warm(r)
    _inject(r, t, 0.30)
    assert tick(r, t).command_out == Z            # default 0.45: 0.30 is inside


def test_nan_and_inf_wz_give_zero():
    r, t = stopped_runner()
    for k, w in enumerate((math.nan, math.inf, -math.inf)):
        out = tick(r, t + k * DT, cmd=(0.2, 0.0, w))
        assert out.command_out == Z and "turn_only" not in out.safety.reasons


def _bad_pose(r, t):
    return dict(pose=LocalizerEstimate(t, yaw_pose(0.0, 0.0, 0.0, 0.0), (1.0,) * 6))


def _pose_dropout(r, t):
    return dict(pose=None)          # inside the max age the last pose is held, but it is not "accepted this tick"


def _stale_depth(r, t):
    return dict(fd=DepthFrame(t - 2.0, np.full((240, 320), 5.0, np.float32), K, MOUNT))


def _ground_bad(r, t):
    return dict(fd=DepthFrame(t, np.full((240, 320), np.nan, np.float32), K, MOUNT))


def _sched(r, t):
    r._sched_fault = True
    return {}


def _costmap_unhealthy(r, t):
    r.costmap.healthy = lambda now: False
    return {}


FAULTS = {"bad_pose": _bad_pose, "pose_dropout": _pose_dropout, "stale_depth": _stale_depth,
          "ground_not_ok": _ground_bad, "scheduler": _sched, "costmap_unhealthy": _costmap_unhealthy}


@pytest.mark.parametrize("name", list(FAULTS))
def test_each_hard_fault_zeroes_everything_incl_wz(name):
    r, t = stopped_runner()
    assert tick(r, t).command_out[2] != 0.0                            # baseline: the turn works
    tt = t + DT
    kw = FAULTS[name](r, tt)
    fd = kw.pop("fd", None)
    if fd is not None:
        out = r.tick(tick_in(tt, fd, cmd=(0.2, 0.1, 0.3), **kw))
    else:
        out = tick(r, tt, **kw)
    assert out.command_out == Z and out.safety.level == STOP
    assert "turn_only" not in out.safety.reasons


def test_bad_stamp_zero_and_latched():
    r, t = stopped_runner()
    assert tick(r, t).command_out[2] != 0.0
    assert tick(r, t - 1.0).command_out == Z            # non-monotonic: latched bad_stamp
    out = tick(r, t + DT)
    assert out.command_out == Z and "bad_stamp" in out.safety.reasons    # stays latched with a valid stamp


def test_sticky_hold_from_hard_fault_blocks_turn_then_releases():
    r, t = stopped_runner(hold_ticks=3)
    out = r.tick(tick_in(t, None, cmd=(0.2, 0.0, 0.3)))
    assert "no_forward_depth" in out.safety.reasons and out.command_out == Z
    for k in range(1, 4):                                              # hold window: no turn even with a soft stop
        out = tick(r, t + k * DT)
        assert out.command_out == Z and "turn_only" not in out.safety.reasons, k
    out = tick(r, t + 4 * DT)
    assert out.command_out == (0.0, 0.0, pytest.approx(0.15))          # hold over, soft stop only -> turn again


def test_translation_hold_after_soft_stop_unchanged():
    # obstacle clears: translation stays zero for hold_ticks ("hold"), then resumes; the turn exception does not leak
    r, t = stopped_runner(hold_ticks=3)
    assert tick(r, t).command_out == (0.0, 0.0, pytest.approx(0.15))
    r.costmap._last_seen[:] = -np.inf                  # the obstacle goes away
    for k in range(1, 4):
        out = tick(r, t + k * DT, stop=False)
        assert out.command_out == Z and "hold" in out.safety.reasons and out.safety.level == STOP
    out = tick(r, t + 4 * DT, stop=False)
    assert out.command_out == (0.2, 0.1, 0.3) and out.safety.level == "ok"


def test_reset_clears_hard_hold():
    r, t = stopped_runner(hold_ticks=5)
    r.tick(tick_in(t, None))
    r.reset()
    _, t2 = warm(r, 6, t0=0.0)
    assert tick(r, t2).command_out[2] == pytest.approx(0.15)


@pytest.mark.parametrize("kw", [
    dict(turn_clearance_m=0.0), dict(turn_clearance_m=-0.1), dict(turn_clearance_m=math.nan),
    dict(turn_clearance_m=math.inf), dict(turn_wz_max=0.0), dict(turn_wz_max=-0.1), dict(turn_wz_max=math.nan),
    dict(turn_wz_max=math.inf), dict(turn_wz_max=0.31), dict(turn_wz_max="x"),
])
def test_bad_turn_config_rejected(kw):
    with pytest.raises(ValueError):
        PipelineRunner(RunnerConfig(**kw))


def test_turn_config_defaults():
    c = RunnerConfig()
    assert c.turn_clearance_m == 0.45 and c.turn_wz_max is None and c.turn_wz == pytest.approx(c.max_wz / 2)
    PipelineRunner(RunnerConfig(turn_wz_max=0.3, turn_clearance_m=1.0))


def test_free_floor_unchanged():
    r, t = stopped_runner()
    out = tick(r, t, stop=False)
    assert out.command_out == (0.2, 0.1, 0.3) and "turn_only" not in out.safety.reasons


# ---- RAW-vs-RAW rule (lead, 2026-11-10): default inflation 0.30 m, stop comes from the costmap itself
def test_raw_obstacle_ahead_at_0_6_stops_but_turns():
    r, t = stopped_runner()
    _inject(r, t, 0.60)
    out = tick(r, t, stop=False)
    assert "obstacle_stop" in out.safety.reasons                      # inflated edge at 0.3 = d_margin -> cap 0
    assert out.command_out == (0.0, 0.0, pytest.approx(0.15)) and "turn_only" in out.safety.reasons


def test_raw_obstacle_ahead_at_0_35_no_turn():
    r, t = stopped_runner()
    _inject(r, t, 0.35)
    out = tick(r, t, stop=False)
    assert "obstacle_stop" in out.safety.reasons and out.command_out == Z


def test_raw_obstacle_beside_inside_clearance_no_turn_outside_turns():
    r, t = stopped_runner()
    _inject(r, t, 0.0, 0.3)
    assert tick(r, t).command_out == Z            # hole stop + raw cell 0.3 m to the side
    r, t = stopped_runner()
    _inject(r, t, 0.0, 0.6)
    assert tick(r, t).command_out[2] == pytest.approx(0.15)


def test_step_down_stop_is_conservative_while_the_memory_holds_it():
    # a step-down is remembered by the obstacle memory as a point at ~0 m (existing behaviour), which is inside the
    # turn clearance: no turn. Documented consequence of the raw memory rule.
    r, t = stopped_runner()
    out = tick(r, t, holes=(HOLE,), stop=False)
    assert "obstacle_stop" in out.safety.reasons and out.command_out == Z


@pytest.mark.parametrize("x,turns", [(0.35, False), (0.7, True)])
def test_remembered_obstacle_raw_position(x, turns):
    r, t = stopped_runner()
    r._mem._items.append((x, 0.0, t))                 # map frame; the pose is the origin, yaw 0
    out = tick(r, t)
    assert "obstacle_stop" in out.safety.reasons
    assert (out.command_out[2] != 0.0) == turns
    assert out.command_out[:2] == (0.0, 0.0)


def test_costmap_unhealthy_no_turn_with_default_inflation():
    r, t = stopped_runner()
    r.costmap.healthy = lambda now: False
    assert tick(r, t).command_out == Z


# ---- lidar cases (audit) --------------------------------------------------------------------------------------------
from humanoid_arc_bme.onboard.pipeline.perception.lidar import LidarFrame  # noqa: E402


def _lidar_turn(rows, **cfg):
    r, t = stopped_runner(**cfg)
    t += DT
    pts = np.array([[0.29, 0.35, 0.5]] * 3 + [[0.6, 0.0, 0.5]] * 4 + rows, float)
    return r.tick(tick_in(t, fwd_frame(t, seed=99), cmd=(0.2, 0.0, 0.3), lidar=LidarFrame(t, pts)))


def test_turn_corner_rule_blocks_cell_whose_corner_is_inside_clearance():
    # cluster at true r=0.444 m: its cell centre is ~0.465 m (> 0.45), but the cell corner is inside
    o = _lidar_turn([[0.36, 0.26, 0.5]] * 3)
    assert o.command_out == Z and "turn_only" not in o.safety.reasons and o.safety.level == STOP


def test_turn_cluster_below_min_points_does_not_block_nor_stop():
    o = _lidar_turn([[0.36, 0.26, 0.5]] * 2)
    assert o.command_out[:2] == (0.0, 0.0)           # (0.29,0.35) already stops translation


def test_turn_self_hit_mask_does_not_block_a_turn():
    r, t = stopped_runner()
    t += DT
    pts = np.vstack([np.tile([0.05, 0.0, 0.5], (6, 1)), np.tile([0.3, 0.4, 0.5], (3, 1)), np.tile([0.6, 0.0, 0.5], (4, 1))])   # body returns + clusters outside the turn clearance
    o = r.tick(tick_in(t, fwd_frame(t, seed=99), cmd=(0.2, 0.0, 0.3), lidar=LidarFrame(t, pts)))
    assert "turn_only" in o.safety.reasons and o.command_out == (0.0, 0.0, pytest.approx(0.15))


def test_turn_corner_rule_unit_with_injected_cell():
    r, t = stopped_runner()
    _inject(r, t, 0.60)
    cm = r.costmap
    # a cell centre at ~0.47 m: centre outside 0.45, nearest corner (~0.435 m) inside -> no turn
    _inject(r, t, 0.45, 0.15)
    assert r._turn_wz(0.3, t) is None


def test_stale_lidar_with_require_lidar_means_hard_stop_and_no_turn():
    r, t = stopped_runner(require_lidar=True)
    t += DT
    stale = LidarFrame(t - 1.0, np.tile([0.6, 0.0, 0.5], (5, 1)))
    o = r.tick(tick_in(t, fwd_frame(t, seed=99), cmd=(0.2, 0.0, 0.3), lidar=stale))
    assert "lidar_stale" in o.safety.reasons and o.command_out == Z and "turn_only" not in o.safety.reasons
