"""G1: PipelineRunner end-to-end on CPU with SIM depth (tests/core/_depth_render.py) and a unicycle plant.

SIM results, not MEASURED. Set PIPELINE_VERBOSE=1 and run with -s to print the latency report.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import tracemalloc

import numpy as np
import pytest

from _depth_render import CFG, Box, Hole, noisy, render
from humanoid_arc_bme.onboard.nav.depth_guard import tighten_command
from humanoid_arc_bme.onboard.pipeline.manipulation.grasp_analytic import AnalyticGraspProposer, ReachConstraint
from humanoid_arc_bme.onboard.pipeline.interfaces import DepthFrame, Intrinsics, LocalizerEstimate, Pose, yaw_pose
from humanoid_arc_bme.onboard.pipeline.perception.costmap import LocalCostmap
from humanoid_arc_bme.onboard.pipeline.runner import (
    OK, SLOW, STOP, PipelineRunner, RunnerConfig, SafetyStatus, TickInput)
from humanoid_arc_bme.onboard.pipeline.scheduler import PhaseScheduler
from humanoid_arc_bme.onboard.pipeline.perception.sensing import camera_pose_from_mount
from humanoid_arc_bme.onboard.pipeline.stubs import GroundTruthSemantics

V = bool(os.environ.get("PIPELINE_VERBOSE"))
DT = 0.1
VX = 0.2   # = RunnerConfig.max_vx (g1_limits.yaml): a faster command is clamped
K = Intrinsics(CFG.fx, CFG.fy, CFG.cx, CFG.cy, CFG.width, CFG.height)
MOUNT = camera_pose_from_mount((CFG.cam_forward, 0.0, CFG.cam_height), CFG.cam_pitch)


def fwd_frame(t, boxes=(), holes=(), seed=0, x=0.0):
    """Forward depth as seen from a robot at world x (scene shifted instead of moving the camera)."""
    sb = [Box(b.x0 - x, b.x1 - x, b.y0, b.y1, b.z1) for b in boxes]
    sh = [Hole(h.x0 - x, h.x1 - x, h.y0, h.y1, h.depth) for h in holes]
    return DepthFrame(t, noisy(render(CFG, sb, sh), seed), K, MOUNT)


def pose_at(t, x=0.0, **kw):
    return LocalizerEstimate(t, yaw_pose(x, 0.0, 0.0, 0.0), (1e-4,) * 6, **kw)


def tick_in(t, fd, pose="auto", x=0.0, cmd=(VX, 0.0, 0.0), **kw):
    return TickInput(t, fd, kw.pop("down", None), pose_at(t, x) if isinstance(pose, str) else pose,
                     kw.pop("roll", 0.0), kw.pop("pitch", 0.0), command=cmd, **kw)


def drive(boxes=(), holes=(), seed=0, n=120, appear_tick=0, runner=None):
    """Unicycle plant: x += vx*dt from the runner's output. Boxes/holes appear at `appear_tick`."""
    r = runner or PipelineRunner()
    x = 0.0
    outs = []
    for i in range(n):
        t = i * DT
        on = i >= appear_tick
        fd = fwd_frame(t, boxes if on else (), holes if on else (), seed=seed * 1000 + i, x=x)
        out = r.tick(tick_in(t, fd, x=x))
        x += out.command_out[0] * DT
        outs.append(out)
    return r, x, outs


@pytest.fixture(scope="module")
def free_frame():
    return fwd_frame(0.0, seed=1)


def warm(runner, n=5, x=0.0, t0=0.0):
    out = None
    for i in range(n):
        t = t0 + i * DT
        out = runner.tick(tick_in(t, fwd_frame(t, seed=i), x=x))
    return out, t0 + n * DT


# ---------------------------------------------------------------- (a)
def test_free_floor_passes_unchanged():
    r, x, outs = drive(n=40)
    assert all(o.safety.level == OK and o.safety.reasons == () for o in outs[1:]), [o.safety for o in outs[:5]]
    assert all(o.command_out == (VX, 0.0, 0.0) for o in outs[1:])
    assert x > 0.7
    assert isinstance(outs[-1].safety, SafetyStatus) and outs[-1].clearance.ground_ok


# ---------------------------------------------------------------- (b)
def test_box_appearing_mid_path_stops_with_margin():
    margins = []
    for seed in range(20):
        bx = 1.6 + 0.02 * seed
        box = Box(bx, bx + 0.3, -0.3, 0.3, 0.6)
        r, x, outs = drive(boxes=(box,), seed=seed, n=100, appear_tick=10)
        margins.append(bx - x)
        assert abs(outs[-1].command_out[0]) < 0.05, (seed, outs[-1].command_out)   # creeping at the margin = stopped
        assert abs(outs[-1].command_out[0]) < abs(outs[-10].command_out[0]) + 1e-9
        assert outs[-1].safety.level != OK, (seed, outs[-1].safety)
        assert x > 0.4                                                   # it did walk first
    assert min(margins) >= 0.25, margins
    if V:
        print(f"SIM box stop margin min/mean {min(margins):.2f}/{np.mean(margins):.2f} m")


# ---------------------------------------------------------------- (c)
def test_hole_ahead_stops_robot():
    for seed in range(5):
        h = Hole(3.0, 4.0, -0.6, 0.6, 0.4)
        r, x, outs = drive(holes=(h,), seed=seed, n=100, appear_tick=0)
        assert outs[-1].command_out == (0.0, 0.0, 0.0), seed
        assert outs[-1].safety.level == STOP
        assert x < h.x0 - 0.2, (seed, x)


# ---------------------------------------------------------------- (d)
def _zero_with(reason, out):
    assert out.command_out == (0.0, 0.0, 0.0), out.safety
    assert out.safety.level == STOP
    assert any(reason in s for s in out.safety.reasons), out.safety


def test_bad_inputs_fail_closed(free_frame):
    at = lambda f, ts: dataclasses.replace(f, stamp=ts)  # noqa: E731
    r = PipelineRunner(RunnerConfig(max_stamp_jump_s=10.0))   # the test skips ahead in time to let state decay
    _, t = warm(r)
    _zero_with("no_forward_depth", r.tick(tick_in(t, None)))
    t += 1.0                                                              # let the sticky hold clear in between
    bad = np.full(free_frame.depth.shape, np.nan, np.float32)
    _zero_with("ground_not_ok", r.tick(tick_in(t, DepthFrame(t, bad, K, MOUNT))))
    garbage = np.random.default_rng(0).uniform(-5, 50, free_frame.depth.shape).astype(np.float32)
    garbage[::3] = np.nan
    _zero_with("ground_not_ok", r.tick(tick_in(t + 0.1, DepthFrame(t + 0.1, garbage, K, MOUNT))))
    _zero_with("bad_depth", r.tick(tick_in(t + 0.2, np.zeros((240, 320)))))
    _zero_with("bad_input", r.tick("not a tick"))
    _zero_with("bad_command", r.tick(tick_in(t + 0.3, at(free_frame, t + 0.3), cmd=(math.nan, 0.0, 0.0))))
    _zero_with("bad_command", r.tick(tick_in(t + 0.4, at(free_frame, t + 0.4), cmd=(1.0, 2.0))))
    _zero_with("bad_attitude", r.tick(tick_in(t + 0.5, free_frame, roll=math.nan)))
    t += 3.0                                           # garbage points decay out of the costmap (decay_s) first
    _zero_with("reverse_unsensed", r.tick(tick_in(t + 0.6, at(free_frame, t + 0.6), cmd=(-0.3, 0.0, 0.0))))
    _zero_with("bad_stamp", r.tick(TickInput(math.nan, at(free_frame, t), None, None, 0.0, 0.0)))


def test_pose_faults():
    r = PipelineRunner()
    _zero_with("pose_none", r.tick(tick_in(0.0, fwd_frame(0.0), pose=None)))
    r = PipelineRunner()
    _, t = warm(r)
    # dropout shorter than the gate's max age (0.5 s): hold the last pose, keep going
    out = r.tick(tick_in(t, fwd_frame(t, seed=9), pose=None))
    assert out.command_out == (VX, 0.0, 0.0) and out.safety.level == OK
    # older than max age: zero
    t2 = t + 0.9
    _zero_with("pose_", r.tick(tick_in(t2, fwd_frame(t2, seed=9), pose=None)))
    # jump
    r = PipelineRunner()
    _, t = warm(r)
    _zero_with("pose_jump", r.tick(tick_in(t, fwd_frame(t, seed=9), pose=pose_at(t, x=5.0))))
    # unhealthy localizer
    r = PipelineRunner()
    _, t = warm(r)
    _zero_with("pose_unhealthy", r.tick(tick_in(t, fwd_frame(t, seed=9), pose=pose_at(t, healthy=False))))


def test_stale_costmap_stops():
    class FrozenCostmap(LocalCostmap):
        def update(self, points, stamp):    # never refreshes: health stays False
            return None
    r = PipelineRunner(costmap=FrozenCostmap())
    out, _ = warm(r, 3)
    _zero_with("costmap_unhealthy", out)


def test_sticky_stop_then_recovers():
    r = PipelineRunner(cfg=RunnerConfig(hold_ticks=3))
    _, t = warm(r)
    _zero_with("no_forward_depth", r.tick(tick_in(t, None)))
    levels = []
    for i in range(1, 7):
        o = r.tick(tick_in(t + i * DT, fwd_frame(t + i * DT, seed=i)))
        levels.append(o.safety.level)
        if o.safety.level == STOP:
            assert "hold" in o.safety.reasons and o.command_out == (0.0, 0.0, 0.0)
    assert levels == [STOP, STOP, STOP, OK, OK, OK], levels


# ---------------------------------------------------------------- (e)
def test_tighten_only_never_increases():
    rng = np.random.default_rng(0)
    specials = [math.nan, math.inf, -math.inf, 0.0, -0.0, None, -1.0]
    for _ in range(3000):
        cmd = [float(rng.normal(0, 1)) if rng.random() < 0.85 else float(rng.choice([math.nan, math.inf, 0.0]))
               for _ in range(3)]
        cap = specials[int(rng.integers(len(specials)))] if rng.random() < 0.4 else float(rng.uniform(0, 2))
        out = tighten_command(*cmd, cap)
        assert len(out) == 3
        for o, c in zip(out, cmd):
            assert math.isfinite(o)                                   # NaN/inf never propagate
            assert abs(o) <= abs(c) + 1e-12 if math.isfinite(c) else o == 0.0
            assert o == 0.0 or o * c > 0                              # same sign or zero
        if cap is None or (isinstance(cap, float) and (math.isnan(cap) or cap < 0 or cap == 0)):
            assert out == (0.0, 0.0, 0.0)
    assert tighten_command(1.0, 0.5, 0.2, math.nan) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------- (f) grasp branch
DK = Intrinsics(200.0, 200.0, 80.0, 60.0, 160, 120)
DOWN_POSE = Pose((0.4, 0.0, 0.6), (0.0, 1.0, 0.0, 0.0))     # optical z = body -z (looking straight down)


def down_frame(t=0.0, radius=0.04, height=0.12, cx=0.4, cy=0.0):
    """Floor at depth 0.6 and a vertical cylinder (top disc rendered) of the given radius/height."""
    u, v = np.meshgrid(np.arange(160), np.arange(120))
    top = 0.6 - height
    x = cx - cx + 0.4 + (u - DK.cx) / DK.fx * top
    y = -(v - DK.cy) / DK.fy * top
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    depth = np.where(mask, top, 0.6).astype(np.float32)
    return DepthFrame(t, depth, DK, DOWN_POSE, camera="down"), mask


# The rendered object here is seen from above only (a cylinder top), so the default arm reach filter (no top-down) would
# remove every candidate; this test is about the runner's wiring, not about reachability.
_ANY_REACH = ReachConstraint(allow_top_down=True, cone_half_angle_deg=180)


def make_grasp_runner(**kw):
    state = {}

    def seg(rgb, depth):
        state["called"] = state.get("called", 0) + 1
        return state["mask"].astype(int), {1: "cup"}
    r = PipelineRunner(semantics=kw.pop("semantics", GroundTruthSemantics(seg)),
                       proposer=kw.pop("proposer", AnalyticGraspProposer(seed=0, reach=_ANY_REACH)), **kw)
    return r, state


def test_grasp_branch_returns_candidates_and_does_not_touch_command():
    r, state = make_grasp_runner()
    down, state["mask"] = down_frame(0.0)
    assert r.start_grasp("cup")
    base = PipelineRunner()
    for i in range(3):
        t = i * DT
        fd = fwd_frame(t, seed=i)
        o = r.tick(tick_in(t, fd, down=DepthFrame(t, down.depth, DK, DOWN_POSE)))
        ob = base.tick(tick_in(t, fd))
        assert o.command_out == ob.command_out and o.safety == ob.safety      # base branch unaffected
    assert o.grasp_candidates, o.events
    assert all(math.isfinite(c.score) for c in o.grasp_candidates)
    assert isinstance(o.command_out, tuple) and len(o.command_out) == 3       # only a base velocity command


@pytest.mark.parametrize("which", ["proposer", "semantics"])
def test_grasp_failures_do_not_change_safety(which):
    class Boom:
        def propose(self, *a, **k):
            raise RuntimeError("boom")

        def detect(self, *a, **k):
            raise RuntimeError("boom")
    kw = {"proposer": Boom()} if which == "proposer" else {"semantics": Boom()}
    r, state = make_grasp_runner(**kw)
    down, state["mask"] = down_frame(0.0)
    r.start_grasp("cup")
    base = PipelineRunner()
    for i in range(3):
        t = i * DT
        fd = fwd_frame(t, seed=i)
        o = r.tick(tick_in(t, fd, down=DepthFrame(t, down.depth, DK, DOWN_POSE)))
        ob = base.tick(tick_in(t, fd))
        assert o.command_out == ob.command_out and o.safety == ob.safety
        assert o.grasp_candidates == []
        assert any(e.get("event") == "grasp_error" for e in o.events)
    json.dumps(r.report())


def test_grasp_branch_inactive_without_request():
    r, state = make_grasp_runner()
    down, state["mask"] = down_frame(0.0)
    o = r.tick(tick_in(0.0, fwd_frame(0.0), down=down))
    assert o.grasp_candidates == [] and "called" not in state


# ---------------------------------------------------------------- (g) scheduler
class _Model:
    def __init__(self, name, est=10, fail=False):
        self.name, self.est_bytes, self.loaded, self.fail = name, est, False, fail

    def load(self):
        if self.fail:
            raise RuntimeError("load failed")
        self.loaded = True

    def unload(self):
        self.loaded = False


def test_scheduler_hook_and_failure_stops():
    walk, grasp = _Model("walk"), _Model("grasp")
    s = PhaseScheduler([], {"walk": [walk], "grasp": [grasp]}, 100)
    s.start()
    s.switch("walk")
    r = PipelineRunner(scheduler=s)
    assert r.start_grasp("cup") and s.phase == "grasp" and grasp.loaded and not walk.loaded
    assert r.end_grasp() and s.phase == "walk" and walk.loaded
    bad = _Model("grasp", fail=True)
    s2 = PhaseScheduler([], {"walk": [walk], "grasp": [bad]}, 100)
    s2.start()
    r2 = PipelineRunner(scheduler=s2)
    assert r2.start_grasp("cup") is False and r2.grasp_request is None
    out, _ = warm(r2, 2)
    _zero_with("scheduler_failed", out)
    assert any(e.get("event") == "scheduler_failed" for e in r2.report()["events"])
    assert r2.end_grasp() is True                      # walk phase loads fine -> fault clears
    out, _ = warm(r2, 6, t0=0.2)
    assert out.safety.level == OK


# ---------------------------------------------------------------- (h) soak
def test_soak_memory_and_event_log_bounded():
    cfg = RunnerConfig(max_events=200, timing_window=500)
    r = PipelineRunner(cfg=cfg)
    frames = [fwd_frame(0.0, seed=s).depth for s in range(4)]
    for i in range(50):                                                      # warm caches before measuring
        t = i * DT
        r.tick(tick_in(t, DepthFrame(t, frames[i % 4], K, MOUNT)))
    tracemalloc.start()
    base = tracemalloc.take_snapshot()
    cur0 = tracemalloc.get_traced_memory()[0]
    for i in range(50, 2050):
        t = i * DT
        r.tick(tick_in(t, DepthFrame(t, frames[i % 4], K, MOUNT)))
    cur1 = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    del base
    growth = cur1 - cur0
    assert growth < 5 * 1024 * 1024, growth
    rep = r.report()
    assert rep["ticks"] == 2050 and len(rep["events"]) <= 200 and len(r._events) <= 200
    assert all(rep["timing_ms"][k]["n"] <= 500 for k in rep["timing_ms"])
    json.dumps(rep)


# ---------------------------------------------------------------- (i) latency
def test_latency_report():
    clock_vals = iter(range(10**9))
    r = PipelineRunner()
    for i in range(60):
        t = i * DT
        r.tick(tick_in(t, fwd_frame(t, seed=i % 5)))
    rep = r.report()["timing_ms"]
    if V:
        for k, v in rep.items():
            print(f"SIM CPU latency {k:8s} p50={v['p50']:.1f} p95={v['p95']:.1f} max={v['max']:.1f} ms")
    assert rep["total"]["p50"] < 2000.0          # very generous: this is a CPU smoke bound, not a budget
    assert set(rep) >= {"pose", "depth", "costmap", "guard", "total"}


def test_injected_clock_is_used():
    ticks = iter(float(i) * 0.001 for i in range(10**6))
    r = PipelineRunner(clock=lambda: next(ticks))
    r.tick(tick_in(0.0, fwd_frame(0.0)))
    assert r.report()["timing_ms"]["total"]["max"] > 0.0


# ================================================================ audit regressions (R1-R5)
from humanoid_arc_bme.onboard.pipeline import runner as runner_mod  # noqa: E402

NARROW_BOX = Box(0.6, 1.2, -0.2, 0.2, 0.5)


def _at(frame, ts):
    return dataclasses.replace(frame, stamp=ts)


def _reasons(out):
    return set(out.safety.reasons)


# ---- R1: config validation
BAD_CFG = [
    dict(a_stop=math.nan), dict(a_stop=math.inf), dict(a_stop=0.0), dict(a_stop=-1.0), dict(a_stop="0.3"),
    dict(a_stop=True), dict(a_stop=None),
    dict(d_margin=math.nan), dict(d_margin=-0.1), dict(d_margin=math.inf),
    dict(hold_ticks=math.nan), dict(hold_ticks=True), dict(hold_ticks=1.0), dict(hold_ticks=-1),
    dict(hold_ticks=np.int64(3)), dict(hold_ticks=None),
    dict(corridor_half_width=0.0), dict(corridor_half_width=math.nan), dict(max_range=math.inf),
    dict(points_max_range=-1.0), dict(points_stride=0), dict(points_stride=2.0),
    dict(min_valid_fraction=1.5), dict(min_valid_fraction=math.nan), dict(max_events=0), dict(max_events=2.5),
    dict(timing_window=0), dict(grasp_max_range=0.0), dict(memory_ttl_s=0.0), dict(memory_ttl_s=math.nan),
    dict(memory_lateral_margin=-1.0), dict(max_stamp_jump_s=0.0), dict(max_stamp_jump_s=math.nan),
    dict(max_depth_age_s=0.0), dict(max_depth_age_s=-1.0), dict(max_depth_age_s=math.inf),
    dict(max_vx=0.0), dict(max_vy=math.nan), dict(max_wz=math.inf), dict(max_vx=-0.2),
]


@pytest.mark.parametrize("kw", BAD_CFG, ids=[str(k) for k in BAD_CFG])
def test_r1_bad_config_rejected_and_guard_never_built(kw, monkeypatch):
    built = []
    real = runner_mod.SpeedSeparationGuard

    def spy(*a, **k):
        built.append(k)
        return real(*a, **k)

    monkeypatch.setattr(runner_mod, "SpeedSeparationGuard", spy)
    with pytest.raises(ValueError):
        PipelineRunner(RunnerConfig(**kw))
    assert built == []                     # the runner never hands a bad value to the guard


def test_r1_not_a_config_rejected():
    with pytest.raises(ValueError):
        PipelineRunner({"a_stop": 0.3})  # type: ignore[arg-type]


def test_r1_valid_edge_values_accepted():
    PipelineRunner(RunnerConfig(hold_ticks=0, d_margin=0.0, memory_lateral_margin=0.0, min_valid_fraction=0.0))
    PipelineRunner(RunnerConfig(a_stop=np.float32(0.3), max_vx=1))     # numpy / int floats are fine


def test_r1_nan_hold_ticks_no_longer_disables_sticky_stop():
    r = PipelineRunner(RunnerConfig())
    _, t = warm(r)
    assert r.tick(tick_in(t, None)).safety.level == STOP
    assert r.tick(tick_in(t + DT, fwd_frame(t + DT))).safety.level == STOP     # sticky hold still active


def test_r1_default_envelope_matches_g1_limits_yaml():
    yaml = pytest.importorskip("yaml")
    from pathlib import Path
    d = yaml.safe_load((Path(__file__).resolve().parents[2] / "config" / "g1_limits.yaml").read_text())["base_vel_max"]
    c = RunnerConfig()
    assert (c.max_vx, c.max_vy, c.max_wz) == (d["vx"], d["vy"], d["wz"])


# ---- R2: tick stamp sanity, latch, reset()
def test_r2_forward_glitch_stops_and_does_not_run_memory_ttl():
    r = PipelineRunner()
    o0 = r.tick(tick_in(0.0, fwd_frame(0.0, boxes=(NARROW_BOX,))))
    r.tick(tick_in(DT, fwd_frame(DT)))                                      # box now in the blind zone
    assert len(r._mem) == 1
    out = r.tick(tick_in(1000.0, fwd_frame(1000.0)))
    assert out.command_out == (0.0, 0.0, 0.0) and "bad_stamp" in _reasons(out)
    assert len(r._mem) == 1                                                  # TTL was NOT run on the glitch stamp
    assert o0.command_out[0] <= 0.2
    for i in range(3):                                                       # sane stamps again: still latched
        o = r.tick(tick_in(2 * DT + i * DT, fwd_frame(2 * DT + i * DT)))
        assert o.command_out == (0.0, 0.0, 0.0) and "bad_stamp" in _reasons(o) and o.safety.level == STOP


@pytest.mark.parametrize("second", [1.0, 0.5, 3.5], ids=["equal", "backwards", "jump>1s"])
def test_r2_non_increasing_or_jumping_stamp_is_hard_stop(second):
    r = PipelineRunner()
    r.tick(tick_in(1.0, fwd_frame(1.0)))
    out = r.tick(tick_in(second, fwd_frame(second)))
    assert "bad_stamp" in _reasons(out) and out.command_out == (0.0, 0.0, 0.0)


def test_r2_jump_threshold_is_configurable_and_inclusive():
    r = PipelineRunner(RunnerConfig(max_stamp_jump_s=0.5))
    r.tick(tick_in(0.0, fwd_frame(0.0)))
    assert "bad_stamp" not in _reasons(r.tick(tick_in(0.5, fwd_frame(0.5))))     # exactly the limit
    assert "bad_stamp" in _reasons(r.tick(tick_in(1.1, fwd_frame(1.1))))


def test_r2_nan_stamp_stops_but_does_not_latch():
    r = PipelineRunner()
    _, t = warm(r)
    out = r.tick(TickInput(math.nan, fwd_frame(t), None, pose_at(t), 0.0, 0.0, command=(VX, 0.0, 0.0)))
    assert "bad_stamp" in _reasons(out)
    assert r._stamp_fault is False and r._last_stamp == pytest.approx(t - DT)


def test_r2_gate_reset_alone_is_not_enough_runner_reset_is_the_entry():
    r = PipelineRunner()
    r.tick(tick_in(0.0, fwd_frame(0.0, boxes=(NARROW_BOX,))))
    assert len(r._mem) == 1
    r.tick(tick_in(1000.0, fwd_frame(1000.0)))                               # latch
    r.gate.reset()
    out = r.tick(tick_in(0.1, fwd_frame(0.1)))
    assert "bad_stamp" in _reasons(out) and len(r._mem) == 1                 # gate.reset() left everything stale
    r.reset()
    assert len(r._mem) == 0 and r._last_stamp is None and r._stamp_fault is False and r._hold_left == 0
    assert len(r._pending) == 0 and r.gate.last_accepted is None and r.costmap._latest is None
    assert r._dist == math.inf
    outs = [r.tick(tick_in(0.0 + i * DT, fwd_frame(i * DT, seed=i))) for i in range(6)]   # stamps restart at 0
    assert all("bad_stamp" not in _reasons(o) for o in outs)
    assert outs[-1].safety.level == OK and outs[-1].command_out == (VX, 0.0, 0.0)


def test_r2_reset_clears_sticky_hold_and_pending_events():
    s = _Model("grasp", fail=True)
    sched = PhaseScheduler([], {"walk": [_Model("walk")], "grasp": [s]}, 100)
    sched.start()
    r = PipelineRunner(scheduler=sched)
    r.start_grasp("cup")
    assert len(r._pending) == 1
    _, t = warm(r)
    r.tick(tick_in(t, None))
    assert r._hold_left > 0
    r.reset()
    assert r._hold_left == 0 and len(r._pending) == 0


# ---- R3: depth freshness
def test_r3_stale_forward_depth_is_hard_stop():
    r = PipelineRunner()
    _, t = warm(r)
    fresh = r.tick(tick_in(t, fwd_frame(t, seed=3)))
    assert "depth_stale" not in _reasons(fresh) and fresh.safety.level == OK
    stale = r.tick(tick_in(t + DT, _at(fwd_frame(t + DT, seed=3), t - 1.0)))     # frozen camera
    assert "depth_stale" in _reasons(stale) and stale.command_out == (0.0, 0.0, 0.0) and stale.safety.level == STOP


@pytest.mark.parametrize("skew", [-1.0, 1.0])
def test_r3_future_dated_depth_is_also_stale(skew):
    r = PipelineRunner()
    out = r.tick(tick_in(5.0, _at(fwd_frame(5.0), 5.0 + skew)))
    assert "depth_stale" in _reasons(out)


def test_r3_depth_age_limit_is_configurable():
    r = PipelineRunner(RunnerConfig(max_depth_age_s=0.05))
    out = r.tick(tick_in(1.0, _at(fwd_frame(1.0), 0.9)))
    assert "depth_stale" in _reasons(out)
    r = PipelineRunner(RunnerConfig(max_depth_age_s=0.5))
    assert "depth_stale" not in _reasons(r.tick(tick_in(1.0, _at(fwd_frame(1.0), 0.7))))


def test_r3_stale_down_depth_skips_grasp_with_event_but_no_safety_stop():
    r, state = make_grasp_runner()
    down, state["mask"] = down_frame(0.0)
    assert r.start_grasp("cup")
    base = PipelineRunner()
    for i in range(2):
        t = i * DT
        fd = fwd_frame(t, seed=i)
        o = r.tick(tick_in(t, fd, down=DepthFrame(t - 1.0, down.depth, DK, DOWN_POSE)))   # 1 s old
        ob = base.tick(tick_in(t, fd))
    assert o.grasp_candidates == []
    assert any(e.get("event") == "grasp_skipped" and e.get("reason") == "down_depth_stale" for e in o.events)
    assert o.safety == ob.safety and o.command_out == ob.command_out and "called" not in state


# ---- R4: speed envelope
@pytest.mark.parametrize("cmd,expect", [
    ((1e9, 0.0, 0.0), (0.2, 0.0, 0.0)),
    ((0.0, -1e9, 0.0), (0.0, -0.1, 0.0)),
    ((0.1, 0.0, 99.0), (0.1, 0.0, 0.3)),
    ((0.0, 0.0, -99.0), (0.0, 0.0, -0.3)),
])
def test_r4_absolute_clamp(cmd, expect):
    r = PipelineRunner()
    _, t = warm(r)
    out = r.tick(tick_in(t, fwd_frame(t, seed=1), cmd=cmd))
    assert out.command_out == pytest.approx(expect)
    assert out.safety.level == SLOW and "speed_capped" in _reasons(out)


def test_r4_within_envelope_untouched():
    r = PipelineRunner()
    _, t = warm(r)
    out = r.tick(tick_in(t, fwd_frame(t, seed=1), cmd=(0.15, -0.05, 0.1)))
    assert out.command_out == (0.15, -0.05, 0.1) and out.safety.level == OK


def test_r4_cap_applies_to_planar_speed_not_just_vx():
    r = PipelineRunner()
    r._dist = 0.0
    c = r._limit((0.0, 0.1, 0.2), 0.05)                    # pure lateral motion under a tight cap
    assert math.hypot(c[0], c[1]) == pytest.approx(0.05) and c[2] == pytest.approx(0.1)   # wz scaled together
    c = r._limit((0.2, 0.1, 0.0), 0.1)
    assert math.hypot(c[0], c[1]) == pytest.approx(0.1) and c[0] / c[1] == pytest.approx(2.0)   # direction kept
    assert r._limit((0.2, 0.1, 0.3), 0.0) == (0.0, 0.0, 0.0)     # zero cap zeroes everything incl. wz (user decision)
    assert r._limit((0.2, 0.1, 0.3), math.nan) == (0.0, 0.0, 0.0)
    assert r._limit((0.2, 0.1, 0.3), -1.0) == (0.0, 0.0, 0.0)
    assert r._limit((0.1, 0.05, 0.1), math.inf) == (0.1, 0.05, 0.1)
    for cap in (0.01, 0.1, 5.0):
        o = r._limit((0.2, 0.1, 0.3), cap)
        assert all(abs(a) <= abs(b) + 1e-12 for a, b in zip(o, (0.2, 0.1, 0.3)))


def test_r4_lateral_command_is_guarded_by_obstacle_cap(monkeypatch):
    r = PipelineRunner()
    _, t = warm(r)
    monkeypatch.setattr(r.guard, "speed_cap", lambda *a, **k: 0.03)       # obstacle-limited
    out = r.tick(tick_in(t, fwd_frame(t, seed=2), cmd=(0.0, 0.1, 0.0)))     # pure sideways command
    assert math.hypot(out.command_out[0], out.command_out[1]) == pytest.approx(0.03)
    assert out.safety.level == SLOW and "speed_capped" in _reasons(out)


# ---- R5: bounded pending events, NaN distances
def test_r5_pending_events_are_bounded():
    s = _Model("grasp", fail=True)
    sched = PhaseScheduler([], {"walk": [_Model("walk")], "grasp": [s]}, 100)
    sched.start()
    r = PipelineRunner(RunnerConfig(max_events=20), scheduler=sched)
    for _ in range(5000):
        r.start_grasp("cup")
    assert len(r._pending) == 20
    out = r.tick(tick_in(0.0, fwd_frame(0.0)))
    assert len([e for e in out.events if e.get("event") == "scheduler_failed"]) == 20


@pytest.mark.parametrize("dpos,dneg", [(math.nan, 2.0), (2.0, math.nan), (math.nan, math.nan)])
def test_r5_nan_clearance_distance_is_a_stop(dpos, dneg, monkeypatch):
    orig = runner_mod.analyze_corridor

    def fake(*a, **k):
        real = orig(*a, **k)
        return dataclasses.replace(real, distance_positive=dpos, distance_negative=dneg)

    monkeypatch.setattr(runner_mod, "analyze_corridor", fake)
    r = PipelineRunner()
    _, t = warm(r)
    out = r.tick(tick_in(t, fwd_frame(t, seed=1)))
    assert out.command_out == (0.0, 0.0, 0.0) and "obstacle_stop" in _reasons(out)


# ---- documented by-design behaviour of end_grasp
def test_end_grasp_scheduler_failure_persists_until_a_later_switch_succeeds():
    walk = _Model("walk")
    sched = PhaseScheduler([], {"walk": [walk], "grasp": [_Model("grasp")]}, 100)
    sched.start()
    r = PipelineRunner(scheduler=sched)
    assert r.start_grasp("cup")
    walk.fail = True
    assert r.end_grasp() is False and r.grasp_request is None      # request cleared even though the switch failed
    for i in range(3):                                              # no automatic retry: stays stopped
        out = r.tick(tick_in(i * DT, fwd_frame(i * DT, seed=i)))
        assert "scheduler_failed" in _reasons(out) and out.command_out == (0.0, 0.0, 0.0)
    walk.fail = False
    assert r.end_grasp() is True
    t = 5 * DT
    _, _ = warm(r, 6, t0=t)
    out = r.tick(tick_in(t + 6 * DT, fwd_frame(t + 6 * DT)))
    assert "scheduler_failed" not in _reasons(out)


def test_require_lidar_with_stale_lidar_is_a_hard_stop_lidar_stale():
    from humanoid_arc_bme.onboard.pipeline.perception.lidar import LidarFrame
    r = PipelineRunner(RunnerConfig(require_lidar=True))
    _, t = warm(r)
    t += DT
    stale = LidarFrame(t - 1.0, np.zeros((0, 3)))
    o = r.tick(tick_in(t, fwd_frame(t, seed=3), cmd=(VX, 0.0, 0.3), lidar=stale))
    assert "lidar_stale" in o.safety.reasons and o.command_out == (0.0, 0.0, 0.0)
