"""A3: how much localizer error the pipeline tolerates, and that the gate fails closed. SIM ONLY.

Plant: nav.sim.WalkerPlant ("nominal" lag/delay/sway) driven by the nav PathFollower along a 4 m line. The follower sees
the plant through NoisedLocalizer(TruthLocalizer) -> PoseGate; metrics use the TRUE pose. Set PIPELINE_VERBOSE=1 to print
the tables (copied into docs/plan/pipeline-localizer-tolerance.md).
"""

import math
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

from humanoid_arc_bme.onboard.nav import path as P
from humanoid_arc_bme.onboard.nav.follower import PathFollower
from humanoid_arc_bme.onboard.nav.limits import NavLimits
from humanoid_arc_bme.onboard.nav.odometry import Odometry
from humanoid_arc_bme.onboard.nav.sim import CONDITIONS, WalkerPlant
from humanoid_arc_bme.onboard.pipeline.interfaces import LocalizerEstimate, yaw_pose
from humanoid_arc_bme.onboard.pipeline.perception.pose_gate import PoseGate
from humanoid_arc_bme.onboard.pipeline.stubs import NoisedLocalizer, TruthLocalizer

VERBOSE = os.environ.get("PIPELINE_VERBOSE") == "1"
TOL = 0.15  # m, stated goal tolerance for the study
SEEDS = range(4)
PLANT = replace(CONDITIONS["nominal"], dt=0.01)


def gate(**kw):
    cfg = dict(max_age_s=0.5, max_pos_sigma=0.3, max_yaw_sigma=0.3, max_jump_m=0.3, max_jump_yaw=0.3)
    cfg.update(kw)
    return PoseGate(**cfg)


def est(stamp, x=0.0, y=0.0, yaw=0.0, var=(0.0,) * 6, healthy=True):
    return LocalizerEstimate(stamp, yaw_pose(x, y, 0.0, yaw), var, healthy)


class GatedOdometry(Odometry):
    """What a consumer of a gated localizer does: use the last accepted estimate while fresh, else no pose."""

    def __init__(self, localizer, pg):
        self.loc, self.gate = localizer, pg

    def pose(self, now):
        self.gate.check(self.loc.estimate(now), now)
        e = self.gate.fresh_pose(now)
        return None if e is None else P.Pose2D(e.pose.position[0], e.pose.position[1], e.pose.yaw())


class Cut:
    """Localizer that goes silent from `t_cut` on."""

    def __init__(self, base, t_cut):
        self.base, self.t_cut = base, t_cut

    def estimate(self, stamp):
        return None if stamp >= self.t_cut else self.base.estimate(stamp)


def run(noise, seed, wrap_loc=None, pg=None, length=4.0, hz=20.0):
    plant = WalkerPlant(PLANT, seed=seed)
    truth = TruthLocalizer(lambda t: yaw_pose(plant.x, plant.y, 0.0, plant.yaw))
    loc = NoisedLocalizer(truth, seed=seed, **noise)
    if wrap_loc:
        loc = wrap_loc(loc)
    pg = pg or gate()
    odom = GatedOdometry(loc, pg)
    for i in range(20):  # a consumer waits for a first accepted fix before it starts to move (1 s at 20 Hz)
        odom.pose(-1.0 + 0.05 * i)
    fol = PathFollower(odom, "rpp", NavLimits())
    path = P.line(length)
    fol.follow(path, 0.0)
    seg = fol.plan.segments[0]
    steps_per = int(round(1.0 / hz / PLANT.dt))
    t, k, cmd, cte_max, hist = 0.0, 0, (0.0, 0.0), 0.0, []
    while t < 90.0:
        if k % steps_per == 0:
            bv = fol.step(t)
            cmd = (bv.vx, bv.wz)
            ey = seg.errors(plant.pose(), 0, len(seg))[0]
            cte_max = max(cte_max, abs(ey))
            hist.append((t, plant.v, fol.state))
            if fol.state in ("SUCCEEDED", "FAILED", "STOPPED") and fol.is_still and abs(plant.v) < 0.01:
                break
        plant.step(*cmd)
        t += PLANT.dt
        k += 1
    gerr = math.hypot(plant.x - length, plant.y)
    return dict(state=fol.state, reason=fol.reason, goal_err=gerr, cte_max=cte_max, gate=dict(pg.counts), t=t,
                plant=plant, hist=hist)


def sweep(grid, key):
    rows = []
    for v in grid:
        rs = [run({key: v} if key != "mixed" else v, s) for s in SEEDS]
        n = sum(sum(r["gate"].values()) for r in rs)
        rej = sum(sum(c for k, c in r["gate"].items() if k != "ok") for r in rs)
        rows.append(dict(v=v, goal=sum(r["goal_err"] for r in rs) / len(rs), goal_max=max(r["goal_err"] for r in rs),
                         cte=max(r["cte_max"] for r in rs), rej=rej / max(1, n),
                         ok=sum(r["state"] == "SUCCEEDED" for r in rs), n=len(rs),
                         reasons={k: sum(r["gate"][k] for r in rs) for k in rs[0]["gate"] if k not in ("ok",)}))
    if VERBOSE:
        print(f"\n[{key}] value | mean goal err m | max goal err m | max CTE m | gate reject frac | succeeded")
        for r in rows:
            print(f"{r['v']} | {r['goal']:.3f} | {r['goal_max']:.3f} | {r['cte']:.3f} | {r['rej']:.3f} | {r['ok']}/{r['n']}"
                  f" | {({k: c for k, c in r['reasons'].items() if c})}")
    return rows


# ---------------------------------------------------------------- tolerance study


def test_baseline_no_noise_meets_tolerance():
    r = run({}, 0)
    assert r["state"] == "SUCCEEDED" and r["goal_err"] < 0.05 and r["cte_max"] < 0.1
    assert r["gate"]["ok"] > 0 and sum(c for k, c in r["gate"].items() if k != "ok") == 0


def test_drift_sweep_finds_tolerance_drift():
    rows = sweep([0.0, 0.005, 0.01, 0.02, 0.04, 0.08], "drift_xy")
    by = {r["v"]: r for r in rows}
    assert by[0.0]["goal"] < 0.05
    assert by[0.005]["goal"] < TOL  # small drift is tolerated
    # drift is open loop to the follower: truth goal error grows with drift (generous monotone check on the ends)
    assert by[0.08]["goal"] > by[0.005]["goal"]
    over = [r["v"] for r in rows if r["goal"] > TOL or r["ok"] < r["n"]]
    assert over, "some sweep level must exceed the stated tolerance"
    assert 0.005 < min(over) <= 0.08
    # at the highest level, the reported variance reaches max_pos_sigma and the gate fails closed (rejects)
    assert by[0.08]["rej"] > 0


def test_jitter_sweep():
    rows = sweep([0.0, 0.01, 0.03, 0.06], "jitter_xy")
    by = {r["v"]: r for r in rows}
    assert by[0.01]["goal"] < TOL
    assert by[0.03]["cte"] < 0.8  # never leaves the corridor (follower off_path limit)


def test_dropout_sweep_gate_holds_last_pose_and_still_arrives():
    rows = sweep([0.0, 0.1, 0.3, 0.5], "dropout")
    by = {r["v"]: r for r in rows}
    assert by[0.3]["goal"] < TOL and by[0.3]["ok"] == by[0.3]["n"]
    assert by[0.5]["rej"] > by[0.1]["rej"] > 0
    assert set(k for k, c in by[0.5]["reasons"].items() if c) <= {"none"}


# ---------------------------------------------------------------- safe stop on loss


def test_robot_stops_when_localizer_silent_longer_than_max_age():
    t_cut = 2.0
    r = run({}, 1, wrap_loc=lambda l: Cut(l, t_cut), length=6.0)
    p = r["plant"]
    assert r["state"] == "FAILED" and r["reason"] == "no_pose"
    assert abs(p.v) < 0.01  # velocity is zero at the end
    stopped_at = next(t for t, v, s in r["hist"] if t > t_cut and abs(v) < 0.01)
    assert stopped_at < t_cut + 3.0
    assert p.x < 6.0 - 0.5  # it did not drive on to the goal blind
    if VERBOSE:
        print(f"\nsilent from {t_cut}s: stopped at t={stopped_at:.2f}s, x={p.x:.2f} m, reason={r['reason']}")


def test_short_outage_below_max_age_is_ridden_through():
    r = run({}, 1, wrap_loc=lambda l: _Window(l, 2.0, 2.3))
    assert r["state"] == "SUCCEEDED" and r["goal_err"] < TOL


class _Window:
    def __init__(self, base, t0, t1):
        self.base, self.t0, self.t1 = base, t0, t1

    def estimate(self, stamp):
        return None if self.t0 <= stamp < self.t1 else self.base.estimate(stamp)


def test_gate_catches_relocalization_jump_mid_run():
    class Teleport:
        def __init__(self, base):
            self.base = base

        def estimate(self, stamp):
            e = self.base.estimate(stamp)
            if e is None or stamp < 3.0:
                return e
            x, y, _ = e.pose.position
            return LocalizerEstimate(e.stamp, yaw_pose(x + 1.5, y, 0.0, e.pose.yaw()), e.variance, True, e.source)

    r = run({}, 2, wrap_loc=Teleport, length=6.0)
    assert r["gate"]["jump"] > 0
    assert r["state"] == "FAILED" and r["reason"] == "no_pose"  # held pose ages out; robot stops instead of following the lie
    assert abs(r["plant"].v) < 0.01


# ---------------------------------------------------------------- fail-closed unit tests


def test_gate_accepts_good_and_rejects_bad():
    g = gate()
    assert g.check(est(1.0), 1.1).reason == "ok"
    assert g.check(None, 1.2).reason == "none"
    assert g.check(est(1.2, healthy=False), 1.2).reason == "unhealthy"
    assert g.check(est(0.5), 1.3).reason == "stale"  # 0.8 s old, but also non-monotonic: stale is checked first
    assert g.check(est(1.3, var=(0.25, 0, 0, 0, 0, 0)), 1.3).reason == "variance_too_high"  # sigma 0.5 m
    assert g.check(est(1.3, var=(0, 0, 0, 0, 0, 0.25)), 1.3).reason == "variance_too_high"  # sigma 0.5 rad
    assert g.check(est(1.4, x=5.0), 1.4).reason == "jump"
    assert g.check(est(1.4, yaw=2.0), 1.4).reason == "jump"
    assert g.check(est(0.99), 1.0).reason == "nonmonotonic"  # fresh enough, but before the last accepted (1.0)
    assert g.consecutive_rejects == g.hold_count == 8
    assert g.check(est(1.5, x=0.05), 1.5).accepted and g.consecutive_rejects == 0


def test_rejected_estimate_never_overwrites_last_good():
    g = gate()
    good = est(1.0, x=1.0)
    g.check(good, 1.0)
    for bad, now in ((None, 1.1), (est(1.1, x=9.0), 1.1), (est(1.1, healthy=False), 1.1), (est(0.2), 1.1)):
        assert not g.check(bad, now).accepted
        assert g.last_accepted is good
    assert g.fresh_pose(1.3) is good
    assert g.fresh_pose(1.6) is None  # aged out: fail closed


def test_non_finite_now_and_future_stamp_rejected():
    g = gate()
    g.check(est(1.0), 1.0)
    assert g.check(est(1.1), math.nan).reason == "stale"
    assert g.check(est(2.0), 1.1).reason == "nonmonotonic"  # from the future
    assert g.last_accepted.stamp == 1.0


def test_validation_branches_removed_are_unreachable_through_the_types():
    """P4: the gate no longer re-checks what LocalizerEstimate/Pose guarantee; prove the guarantee."""
    ok = yaw_pose(0, 0, 0, 0)
    with pytest.raises(ValueError):
        LocalizerEstimate(math.nan, ok, (0.0,) * 6)
    with pytest.raises(ValueError):
        LocalizerEstimate(math.inf, ok, (0.0,) * 6)
    with pytest.raises(ValueError):
        LocalizerEstimate(1.0, ok, (math.nan,) * 6)
    with pytest.raises(ValueError):
        LocalizerEstimate(1.0, ok, (-1.0,) + (0.0,) * 5)
    with pytest.raises(ValueError):
        LocalizerEstimate(1.0, ok, (0.0,) * 6, healthy=1)
    with pytest.raises(ValueError):
        yaw_pose(math.nan, 0, 0, 0)
    # anything that is not a LocalizerEstimate (duck types, dicts) fails closed
    fake = SimpleNamespace(stamp=1.0, healthy=True, variance=(0.0,) * 6, pose=ok)
    assert gate().check(fake, 1.0).reason == "none"


def test_wrap_is_the_shared_helper():
    from humanoid_arc_bme.onboard.nav.path import wrap
    from humanoid_arc_bme.onboard.pipeline.perception import pose_gate as pgm
    assert pgm.wrap is wrap and not hasattr(pgm, "_wrap")
    assert gate().check(est(0.0, yaw=3.1), 0.0).accepted and gate().check(est(0.0, yaw=-3.1), 0.0).accepted


def test_p1_first_bogus_estimate_not_adopted():
    g = gate()
    assert g.check(est(0.0, x=1e6), 0.0).reason == "out_of_range"
    assert g.last_accepted is None
    assert g.check(est(0.1, x=0.0), 0.1).accepted  # later good estimate is not poisoned
    assert g.check(est(0.2, y=-2e4), 0.2).reason == "out_of_range"
    g2 = gate()
    assert g2.check(est(0.0, var=(1.0, 0, 0, 0, 0, 0)), 0.0).reason == "variance_too_high"  # first fix variance
    assert g2.last_accepted is None
    with pytest.raises(ValueError):
        gate(max_abs_position=math.nan)


def test_reset_clears_state_and_allows_relocalisation():
    g = gate()
    g.check(est(0.0), 0.0)
    assert g.check(est(0.1, x=9.0), 0.1).reason == "jump"
    g.reset()
    assert g.last_accepted is None and g.consecutive_rejects == 0 and g.hold_count == 0
    assert all(c == 0 for c in g.counts.values())
    assert g.check(est(0.2, x=9.0), 0.2).accepted  # explicit relocalisation


def test_p2_long_outage_does_not_widen_jump_allowance():
    g = gate()
    g.check(est(0.0), 0.0)
    assert g.check(est(10.0, x=9.0), 10.0).reason == "jump"  # was accepted: 0.3 + 1.0 * 10
    g = gate()
    g.check(est(0.0, yaw=0.0), 0.0)
    assert g.check(est(2.0, yaw=math.pi - 0.01), 2.0).reason == "jump"  # was accepted: 0.3 + 1.5 * 2 > pi
    g = gate(max_yaw_rate=100.0)
    g.check(est(0.0), 0.0)
    assert g.check(est(0.4, yaw=2.0), 0.4).reason == "jump"  # yaw allowance capped at pi/2 even for huge rate
    g.reset()
    g.check(est(0.0), 0.0)
    assert g.check(est(10.0, x=0.2), 10.0).accepted  # a stationary robot after a long outage is fine
    g = gate()
    g.check(est(0.0), 0.0)
    assert g.check(est(0.5, x=0.7), 0.5).accepted  # dt up to max_age_s keeps the full allowance (0.3 + 0.5)


def test_p3_clock_going_backwards_is_not_fresh():
    g = gate()
    g.check(est(100.0), 100.0)
    assert g.fresh_pose(100.2) is not None
    assert g.fresh_pose(50.0) is None  # now < stamp
    assert g.fresh_pose(99.999) is None
    assert g.fresh_pose(math.nan) is None


def test_first_estimate_has_no_jump_reference():
    assert gate().check(est(1.0, x=100.0), 1.0).reason == "ok"  # plausible first fix; no reference


def test_jump_allows_speed_times_dt():
    g = gate(max_speed=1.0, max_jump_m=0.1, max_age_s=2.0)
    g.check(est(0.0), 0.0)
    assert g.check(est(1.0, x=1.05), 1.0).accepted  # 1 m/s over 1 s plus floor
    assert g.check(est(1.1, x=2.5), 1.1).reason == "jump"


@pytest.mark.parametrize("bad", [0, -1.0, math.nan, math.inf, None, True, "1"])
@pytest.mark.parametrize("name", ["max_age_s", "max_pos_sigma", "max_yaw_sigma", "max_jump_m", "max_jump_yaw",
                                  "max_speed", "max_yaw_rate"])
def test_config_validation(name, bad):
    with pytest.raises(ValueError):
        gate(**{name: bad})
