"""Operator runtime with fake IO and a fake clock: disarmed start, arming through preflight, deadman, STOP, stale state,
link loss, S1 clamp of the operator command, logs; and the exported E2 ONNX policy in a fake-IO loop (CPU onnxruntime)."""

import json
import os
from pathlib import Path

import pytest

from humanoid_arc_bme.onboard.runtime.controllers import FakeController, FakeLocoClient, PolicyController, VendorLocoController
from humanoid_arc_bme.onboard.runtime.io.fake import FakeRobotIO
from humanoid_arc_bme.onboard.runtime.operator import OperatorLimits, load_operator_limits
from humanoid_arc_bme.onboard.runtime.operator_runtime import OperatorRuntime, RuntimeConfig
from humanoid_arc_bme.onboard.runtime.runlog import RunLog
from humanoid_arc_bme.onboard.api.safety import SafetyLayer, Watchdog, load_limits

ROOT = Path(__file__).resolve().parents[2]
LIMITS_YAML = ROOT / "config" / "g1_limits.yaml"


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FrozenIO(FakeRobotIO):
    """State whose stamp stops advancing after `freeze_at` (stale), or disappears after `drop_at` (link loss)."""

    def __init__(self, clock, freeze_at=None, drop_at=None):
        super().__init__(clock)
        self.freeze_at, self.drop_at, self._frozen = freeze_at, drop_at, None

    def read_state(self):
        t = self.clock()
        if self.drop_at is not None and t >= self.drop_at:
            return None
        if self.freeze_at is not None and t >= self.freeze_at:
            if self._frozen is None:
                self._frozen = super().read_state()
            return self._frozen
        return super().read_state()


def make(tmp_path, io=None, controllers=None, clock=None):
    clock = clock or Clock()
    lim = load_limits(LIMITS_YAML)
    op = load_operator_limits(LIMITS_YAML, lim.base_vel_max)
    io = io or FakeRobotIO(clock)
    fake = FakeController()
    rt = OperatorRuntime(io, controllers or {"fake": fake}, op, SafetyLayer(lim, Watchdog(0.1)), RunLog(tmp_path / "run", wall=lambda: 1.8e9 + clock.t),
                         cfg=RuntimeConfig(), clock=clock, wall=lambda: 1.8e9 + clock.t, sleep=clock.sleep, disk_free=lambda p: 100e9)
    return rt, clock, fake


def run(rt, clock, seconds, hb=(0.0, 0.0, 0.0), hb_hz=10.0, mode="teleop", held=False):
    """`held` = the operator input (pad / HOLD) is pressed in every heartbeat (UI-03: no motion without it)."""
    n = int(round(seconds * 50))
    every = int(round(50 / hb_hz)) if hb_hz else 0
    for k in range(n):
        if every and k % every == 0 and hb is not None:
            rt.heartbeat({"mode": mode, "held": held, "vx": hb[0], "vy": hb[1], "wz": hb[2]} if mode == "teleop"
                         else {"mode": mode, "held": held, **hb})
        rt.tick()
        clock.t += 0.02


def arm(rt):
    code = rt.request_arm()["code"]
    return rt.confirm_arm(f"ARM {code}")


def release_once(rt, clock):
    """After arming the first heartbeat with the input released clears the neutral-first rule (UI-03)."""
    run(rt, clock, 0.2)


def test_starts_disarmed_and_never_commands_without_arming(tmp_path):
    rt, clock, fake = make(tmp_path)
    run(rt, clock, 2.0, hb=(0.2, 0.0, 0.0), held=True)
    assert rt.sup.state == "disarmed" and not [c for c in fake.calls if c[0] == "step"]
    assert rt.last_cmd == (0.0, 0.0, 0.0)
    rt.close()


def test_arming_refused_until_preflight_green_then_ramps_and_s1_clamps(tmp_path):
    rt, clock, fake = make(tmp_path)
    rt.heartbeat({"vx": 0.0})
    rt.tick()
    ok, why = arm(rt)  # loop has not run for a second yet: loop_rate / state_rate not green
    assert not ok and "preflight" in why
    run(rt, clock, 1.2)
    assert rt.status()["preflight_green"], [c for c in rt.status()["preflight"] if c["status"] != "ok"]
    ok, why = arm(rt)
    assert ok, why
    run(rt, clock, 1.0, hb=(0.2, 0.0, 0.0), held=True)  # input held across arming: ignored until released once (UI-03)
    assert all(c[2] == (0.0, 0.0, 0.0) for c in fake.calls if c[0] == "step")
    release_once(rt, clock)
    n0 = len([c for c in fake.calls if c[0] == "step"])
    run(rt, clock, 2.0, hb=(0.2, 0.0, 0.0), held=True)
    steps = [c for c in fake.calls if c[0] == "step"][n0:]
    assert steps and steps[0][2][0] == pytest.approx(0.01)  # motion starts from zero: one ramp step (0.5 m/s^2 * 20 ms)
    assert steps[5][2][0] < 0.1  # ramp: 0.5 m/s^2 -> 0.01 m/s per tick
    s1 = load_limits(LIMITS_YAML).base_vel_max[0]
    assert max(c[2][0] for c in steps) == pytest.approx(min(0.2, s1))  # envelope (= first-walk caps) intersected with S1
    with pytest.raises(ValueError):  # a command above the envelope is refused, not clipped (UI-04)
        rt.heartbeat({"vx": 0.5, "held": True})
    rt.close()
    ev = [json.loads(x) for x in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    assert [e["reason"] for e in ev if e["kind"] == "transition"][-2:] == ["operator_confirmed", "shutdown"]
    assert (tmp_path / "run" / "cmd.jsonl").read_text().strip()


def test_deadman_disarms_and_calls_controller_safe_state(tmp_path):
    rt, clock, fake = make(tmp_path)
    run(rt, clock, 1.2)
    assert arm(rt)[0]
    run(rt, clock, 0.5, hb=(0.2, 0.0, 0.0))
    t_last_hb = clock.t - 0.02 * 5  # heartbeats every 5th tick
    run(rt, clock, 0.5, hb=None)  # UI gone
    assert rt.sup.state == "disarmed" and rt.sup.reason == "deadman_timeout"
    dis = [c for c in fake.calls if c[0] == "disarm"]
    assert dis and dis[0][2] == "deadman_timeout" and dis[0][1] - t_last_hb <= 0.3 + 0.02 + 1e-9
    assert rt.last_cmd == (0.0, 0.0, 0.0)
    rt.close()


def test_stop_is_immediate_latched_and_logged(tmp_path):
    rt, clock, fake = make(tmp_path)
    run(rt, clock, 1.2)
    assert arm(rt)[0]
    run(rt, clock, 0.3, hb=(0.2, 0.0, 0.0))
    rt.stop("ui")
    assert fake.calls[-1][0] == "stop" and rt.sup.stop_latched
    with pytest.raises(PermissionError):
        rt.request_arm()
    rt.reset_stop()
    run(rt, clock, 0.3)
    assert arm(rt)[0]
    rt.close()
    ev = (tmp_path / "run" / "events.jsonl").read_text()
    assert '"stop:ui"' in ev and '"stop_reset"' in ev


@pytest.mark.parametrize("kind,reason", [("freeze", "state_stale"), ("drop", "link_lost")])
def test_stale_state_and_link_loss_disarm(tmp_path, kind, reason):
    clock = Clock()
    io = FrozenIO(clock, freeze_at=clock.t + 1.6 if kind == "freeze" else None, drop_at=clock.t + 1.6 if kind == "drop" else None)
    rt, clock, fake = make(tmp_path, io=io, clock=clock)
    run(rt, clock, 1.2)
    assert arm(rt)[0]
    run(rt, clock, 1.5, hb=(0.1, 0.0, 0.0))
    assert rt.sup.state == "disarmed"
    assert rt.sup.reason == reason if kind == "freeze" else rt.sup.reason in ("state_stale", "link_lost")
    rt.close()


def test_vendor_controller_in_runtime_zero_on_disarm(tmp_path):
    client = FakeLocoClient()
    clock = Clock()
    v = VendorLocoController(client, threaded=False, clock=clock)
    rt, clock, _ = make(tmp_path, controllers={"vendor": v}, clock=clock)
    run(rt, clock, 1.2)
    assert not [c for c in client.calls if c[0] == "SetVelocity"]  # nothing while disarmed
    assert arm(rt)[0]
    release_once(rt, clock)
    n_before = len([c for c in client.calls if c[0] == "SetVelocity"])
    run(rt, clock, 1.0, hb=(0.2, 0.0, 0.1), held=True)
    sv = [c for c in client.calls if c[0] == "SetVelocity"][n_before:]
    assert any(c[1] > 0.05 for c in sv)  # hold-to-run: it really walks while the pad is held
    sv_all = [c for c in client.calls if c[0] == "SetVelocity"]
    assert 9 <= len(sv) <= 11 and all(abs(c[1]) <= 0.2 + 1e-9 for c in sv) and sv[-1][4] == 0.4
    rt.disarm()
    assert client.calls[-1] == ("SetVelocity", 0.0, 0.0, 0.0, 1.0)
    rt.damp()
    assert client.calls[-1] == ("Damp",) and rt.sup.stop_latched
    rt.close()


def test_only_the_arming_client_feeds_the_deadman(tmp_path):
    rt, clock, fake = make(tmp_path)
    run(rt, clock, 1.2)
    code = rt.request_arm()["code"]
    assert rt.confirm_arm(f"ARM {code}", client="laptopA")[0]
    with pytest.raises(PermissionError):
        rt.heartbeat({"vx": 0.2}, client="laptopB")
    for _ in range(20):  # only B keeps sending: A's deadman expires
        try:
            rt.heartbeat({"vx": 0.2}, client="laptopB")
        except PermissionError:
            pass
        rt.tick()
        clock.t += 0.02
    assert rt.sup.reason == "deadman_timeout"
    rt.close()


def test_waypoints_start_on_arm_and_stop_on_disarm(tmp_path):
    pytest.importorskip("humanoid_arc_bme.onboard.nav.runtime_hook")
    from humanoid_arc_bme.onboard.runtime.path_follow import load_nav
    rt, clock, fake = make(tmp_path)
    rt.nav, why = load_nav("humanoid_arc_bme.onboard.nav.runtime_hook", {}, clock)
    assert why == "ok"
    rt.nav.set_waypoints([(1.0, 0.0)])
    run(rt, clock, 1.2, hb={}, mode="waypoints")
    assert not [c for c in fake.calls if c[0] == "step"]  # disarmed: the path has not started
    assert rt.nav.status()["pending_start"]
    assert arm(rt)[0]
    run(rt, clock, 0.2, hb={}, mode="waypoints")  # released once after arming
    run(rt, clock, 2.0, hb={}, mode="waypoints", held=True)  # hold-to-run: the path advances only while HOLD is pressed
    steps = [c for c in fake.calls if c[0] == "step"]
    assert max(c[2][0] for c in steps) > 0.05 and all(c[2][0] <= rt.lim.vx_max + 1e-9 for c in steps)
    rt.disarm()
    assert rt.nav.status()["state"] != "TRACK"  # disarm stopped the follower
    assert arm(rt)[0]
    run(rt, clock, 0.5, hb={}, mode="waypoints")
    assert [c for c in fake.calls if c[0] == "step"][-1][2] == pytest.approx((0.0, 0.0, 0.0))  # no resume without new waypoints
    rt.close()


def test_controller_switch_only_disarmed_and_curvature_mode(tmp_path):
    rt, clock, fake = make(tmp_path, controllers={"fake": FakeController(), "fake2": FakeController()})
    run(rt, clock, 1.2)
    assert arm(rt)[0]
    assert rt.select_controller("fake2") == (False, "disarm first")
    release_once(rt, clock)
    run(rt, clock, 1.0, hb={"v": 0.2, "kappa": 1.0}, mode="curvature", held=True)
    tgt = rt.sup.target
    assert tgt[2] == pytest.approx(1.0 * tgt[0]) and tgt[0] > 0 and abs(tgt[2]) <= rt.lim.wz_max + 1e-9
    rt.disarm()
    assert rt.select_controller("fake2")[0] and rt.controller_name == "fake2"
    rt.close()


# ---------------------------------------------------------------- the real E2 export through onnxruntime (CPU)

MODEL_DIRS = [Path(p) for p in os.environ.get("HUMANOID_ARC_ONNX_DIRS", "").split(os.pathsep) if p] or [
    ROOT / "outputs" / "W6" / "onnx" / d for d in (sorted(os.listdir(ROOT / "outputs" / "W6" / "onnx")) if (ROOT / "outputs" / "W6" / "onnx").is_dir() else [])
] + [ROOT / "outputs" / "W4" / "onnx" / "deployE"]


@pytest.mark.parametrize("model_dir", [d for d in MODEL_DIRS if (d / "policy.onnx").is_file()] or [None])
def test_policy_controller_e2_onnx_fake_io_loop(tmp_path, model_dir):
    pytest.importorskip("onnxruntime")
    if model_dir is None:
        pytest.skip("no exported policy (outputs/W4/onnx/deployE) on this machine")
    clock = Clock()
    io = FakeRobotIO(clock)
    lim = load_limits(LIMITS_YAML)
    safety = SafetyLayer(lim, Watchdog(0.1))
    pc = PolicyController(model_dir, io, safety, height=0.78)
    rt, clock, _ = make(tmp_path, io=io, controllers={pc.name: pc}, clock=clock)
    rt.safety = safety
    run(rt, clock, 1.2)
    assert arm(rt)[0]
    release_once(rt, clock)
    n0 = len(io.sent)
    run(rt, clock, 2.0, hb=(0.2, 0.0, 0.0), held=True)
    sent = io.sent[n0:]
    assert len(sent) >= 99 and all(len(t) == 12 for _n, t in sent)
    import math
    assert all(math.isfinite(x) for _n, t in sent for x in t)
    rng = {n: lim.joint_range(n) for n in sent[0][0]}
    assert all(rng[n][0] - 1e-9 <= x <= rng[n][1] + 1e-9 for names, t in sent for n, x in zip(names, t))
    st = pc.status()
    assert st["infer_ms"] is not None and st["infer_ms"] < 20.0 and pc.policy.last_obs.shape == (pc.policy.layout.obs_dim,)
    rt.stop("ui")
    n1 = len(io.sent)
    run(rt, clock, 0.5, hb=(0.2, 0.0, 0.0))
    assert len(io.sent) == n1  # nothing after STOP
    rt.close()
