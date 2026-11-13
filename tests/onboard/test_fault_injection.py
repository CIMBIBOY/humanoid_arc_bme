"""Fault injection on the fake-IO operator loop (docs/audit/2026-10-30_safety_review.md, UI-01..UI-04, SR-05).

Every case walks the vendor-controller path (FakeLocoClient) at 0.2 m/s, injects ONE fault and proves: the last velocity
call is a zero, the supervisor is disarmed, and it happened within the deadline. Time is a fake clock except for the two
tests that run the real threads. Standard library + pytest only."""

import math
import threading
import time
from pathlib import Path

import pytest

from humanoid_arc_bme.onboard.runtime.controllers import FakeLocoClient, VendorLocoController
from humanoid_arc_bme.onboard.runtime.io.fake import FakeRobotIO
from humanoid_arc_bme.onboard.runtime.operator import load_operator_limits
from humanoid_arc_bme.onboard.runtime.operator_runtime import OperatorRuntime, RuntimeConfig
from humanoid_arc_bme.onboard.runtime.runlog import RunLog
from humanoid_arc_bme.onboard.api.safety import SafetyLayer, Watchdog, load_limits

ROOT = Path(__file__).resolve().parents[2]
LIMITS_YAML = ROOT / "config" / "g1_limits.yaml"
DT = 0.02


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class BadVendor(VendorLocoController):
    """A vendor controller whose step() / fault() can be made to raise (a controller bug, a NaN policy output...)."""

    step_raises = None
    fault_raises = None

    def step(self, now, dt, state, cmd):
        if self.step_raises:
            raise self.step_raises
        return super().step(now, dt, state, cmd)

    def fault(self):
        if self.fault_raises:
            raise self.fault_raises
        return super().fault()


def build(tmp_path, ctrl_cls=BadVendor, io=None):
    clk = Clock()
    client = FakeLocoClient()
    vendor = ctrl_cls(client, threaded=False, clock=clk)
    lim = load_limits(LIMITS_YAML)
    op = load_operator_limits(LIMITS_YAML, lim.base_vel_max)
    rt = OperatorRuntime((io or FakeRobotIO)(clk), {"vendor": vendor}, op, SafetyLayer(lim, Watchdog(0.1)),
                         RunLog(tmp_path / "run", wall=lambda: 1.8e9 + clk.t), cfg=RuntimeConfig(), clock=clk,
                         wall=lambda: 1.8e9 + clk.t, sleep=clk.sleep, disk_free=lambda p: 100e9)
    return rt, clk, vendor, client


def spin(rt, clk, vendor, seconds, hb=None, held=True, pump=True):
    """Control ticks at 50 Hz with the UI heartbeat at 10 Hz (hb = (vx, vy, wz) or None for a dead tab)."""
    for k in range(int(round(seconds * 50))):
        if hb is not None and k % 5 == 0:
            rt.heartbeat({"mode": "teleop", "held": held, "vx": hb[0], "vy": hb[1], "wz": hb[2]})
        rt.tick()
        if pump:
            vendor.pump(clk.t)  # the sender thread, driven by hand so the test is deterministic
        clk.t += DT


def walking(tmp_path, **kw):
    rt, clk, vendor, client = build(tmp_path, **kw)
    spin(rt, clk, vendor, 1.2, hb=(0, 0, 0), held=False)
    code = rt.request_arm()["code"]
    assert rt.confirm_arm(f"ARM {code}")[0]
    spin(rt, clk, vendor, 0.3, hb=(0, 0, 0), held=False)  # released: neutral seen
    spin(rt, clk, vendor, 1.0, hb=(0.2, 0, 0), held=True)
    assert rt.sup.armed and client.calls[-1][1] > 0.05, client.calls[-1]
    return rt, clk, vendor, client


def assert_zero_and_disarmed(rt, client):
    last = [c for c in client.calls if c[0] == "SetVelocity"][-1]
    assert last[1:4] == (0.0, 0.0, 0.0), last
    assert not rt.sup.armed


# ------------------------------------------------------------------------------------------------ exceptions inside a tick

def test_controller_step_raises_nan_policy_like(tmp_path):
    """A NaN policy action makes JointTargets/step raise ValueError: caught in the same tick, fault latched, zero, disarmed."""
    rt, clk, vendor, client = walking(tmp_path)
    vendor.step_raises = ValueError("positions: values must be finite")
    rt.tick()  # must not raise
    assert_zero_and_disarmed(rt, client)
    assert rt.fault and rt.fault.startswith("tick_exception:ValueError") and "finite" in rt.fault_detail
    assert rt.sup.stop_latched
    vendor.step_raises = None
    with pytest.raises(PermissionError):  # arming needs an operator reset of the latch
        rt.request_arm()


def test_controller_fault_method_raises(tmp_path):
    rt, clk, vendor, client = walking(tmp_path)
    vendor.fault_raises = RuntimeError("controller bug")
    rt.tick()
    assert_zero_and_disarmed(rt, client)
    assert "tick_exception" in rt.fault


def test_log_write_raises_disk_full(tmp_path):
    rt, clk, vendor, client = walking(tmp_path)
    rt.runlog.command = lambda rec: (_ for _ in ()).throw(OSError("No space left on device"))
    rt.tick()
    vendor.pump(clk.t + 5.0)  # the sender thread keeps running; it must not resurrect the old velocity
    assert_zero_and_disarmed(rt, client)
    assert "OSError" in rt.fault


def test_state_read_raises_does_not_escape_and_disarms(tmp_path):
    class BadIO(FakeRobotIO):
        fail = False

        def read_state(self):
            if self.fail:
                raise ValueError("quat: quaternion (w, x, y, z) must have unit length")
            return super().read_state()

    rt, clk, vendor, client = walking(tmp_path, io=BadIO)
    rt.io.fail = True
    spin(rt, clk, vendor, 0.6, hb=(0.2, 0, 0))
    assert_zero_and_disarmed(rt, client)
    assert rt.sup.reason in ("state_stale", "link_lost")


def test_fault_is_latched_and_cleared_only_by_reset(tmp_path):
    rt, clk, vendor, client = walking(tmp_path)
    vendor.step_raises = ValueError("boom")
    rt.tick()
    vendor.step_raises = None
    spin(rt, clk, vendor, 1.2, hb=(0, 0, 0), held=False)
    assert rt.fault and not rt.sup.armed
    rt.reset_stop()
    assert rt.fault is None and not rt.sup.stop_latched


# ------------------------------------------------------------------------------------------------ operator side

def test_heartbeat_stops_deadman_zero_within_deadline(tmp_path):
    rt, clk, vendor, client = walking(tmp_path)
    t_last = clk.t
    n = 0
    while rt.sup.armed and n < 100:
        spin(rt, clk, vendor, DT, hb=None)
        n += 1
    assert rt.sup.reason == "deadman_timeout"
    assert clk.t - t_last <= rt.lim.deadman_timeout_s + 0.1 + 1e-9  # last heartbeat was at most 0.1 s before t_last
    assert_zero_and_disarmed(rt, client)


def test_tab_dies_while_pad_held(tmp_path):
    """The browser dies with the pad pressed: the last held=true heartbeat must not keep the robot walking."""
    rt, clk, vendor, client = walking(tmp_path)
    spin(rt, clk, vendor, 1.0, hb=None)
    assert_zero_and_disarmed(rt, client)
    spin(rt, clk, vendor, 1.0, hb=None)
    assert [c for c in client.calls if c[0] == "SetVelocity"][-1][1:4] == (0.0, 0.0, 0.0)


def test_heartbeat_without_hold_never_moves(tmp_path):
    rt, clk, vendor, client = build(tmp_path)
    spin(rt, clk, vendor, 1.2, hb=(0, 0, 0), held=False)
    code = rt.request_arm()["code"]
    assert rt.confirm_arm(f"ARM {code}")[0]
    spin(rt, clk, vendor, 2.0, hb=(0.2, 0, 0), held=False)  # slider at 0.2, no pad held
    assert rt.sup.armed
    assert all(c[1] == 0.0 for c in client.calls if c[0] == "SetVelocity")


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, 9.0, "0.1", None, True])
def test_malformed_heartbeats_do_not_feed_the_deadman(tmp_path, bad):
    rt, clk, vendor, client = walking(tmp_path)
    age0 = rt.sup.heartbeat_age(clk.t)
    clk.t += 0.1
    with pytest.raises(ValueError):
        rt.heartbeat({"mode": "teleop", "held": True, "vx": bad, "vy": 0, "wz": 0})
    assert rt.sup.heartbeat_age(clk.t) == pytest.approx(age0 + 0.1)  # not refreshed


def test_stale_held_value_does_not_start_motion_after_rearm(tmp_path):
    rt, clk, vendor, client = walking(tmp_path)
    rt.disarm()
    spin(rt, clk, vendor, 0.3, hb=(0.2, 0, 0), held=True)
    code = rt.request_arm()["code"]
    assert rt.confirm_arm(f"ARM {code}")[0]
    n0 = len(client.calls)
    spin(rt, clk, vendor, 1.0, hb=(0.2, 0, 0), held=True)  # pad still pressed across arming
    assert all(c[1] == 0.0 for c in client.calls[n0:] if c[0] == "SetVelocity")


# ------------------------------------------------------------------------------------------------ loop dead or stalled

def test_vendor_sender_stops_renewing_when_loop_silent(tmp_path):
    rt, clk, vendor, client = walking(tmp_path)
    t0 = clk.t
    while clk.t - t0 < 0.5:  # control loop gone (no tick, no step); only the sender keeps pumping
        clk.t += 0.01
        vendor.pump(clk.t)
        if client.calls[-1][1] == 0.0:
            break
    assert client.calls[-1][1:4] == (0.0, 0.0, 0.0)
    assert clk.t - t0 <= vendor.step_deadline + 0.05
    n = len(client.calls)
    for _ in range(50):
        clk.t += 0.1
        vendor.pump(clk.t)
    assert all(c[1] == 0.0 for c in client.calls[n:] if c[0] == "SetVelocity")  # never a velocity again


def test_supervisor_disarms_when_ticks_stop(tmp_path):
    """No completed tick for tick_deadline_s (0.2 s): the supervisor (independent of the control thread) disarms."""
    rt, clk, vendor, client = walking(tmp_path)
    t0 = clk.t
    reason = None
    while reason is None and clk.t - t0 < 1.0:
        clk.t += 0.01
        reason = rt.supervise()
    assert reason and reason.startswith("control_loop_stalled")
    assert clk.t - t0 <= rt.cfg.tick_deadline_s + 0.02
    assert_zero_and_disarmed(rt, client)
    assert rt.fault and rt.fault.startswith("watchdog:")


def test_supervisor_works_while_the_control_thread_holds_the_lock(tmp_path):
    rt, clk, vendor, client = walking(tmp_path)
    got = threading.Event()
    done = threading.Event()

    def hog():
        with rt.lock:
            got.set()
            done.wait(5)

    th = threading.Thread(target=hog, daemon=True)
    th.start()
    got.wait(2)
    clk.t += 0.5
    reason = rt.supervise()
    done.set()
    th.join(2)
    assert reason is not None
    last = [c for c in client.calls if c[0] == "SetVelocity"][-1]
    assert last[1:4] == (0.0, 0.0, 0.0)  # the robot side was zeroed without the lock
    rt.tick()  # the pending trip is applied when the loop runs again
    assert not rt.sup.armed and rt.fault


def test_real_threads_hung_tick_is_disarmed_within_the_deadline(tmp_path):
    """Real clock, real supervisor thread, real vendor sender thread: the control loop hangs inside a tick."""
    client = FakeLocoClient()
    vendor = BadVendor(client, threaded=True)
    lim = load_limits(LIMITS_YAML)
    op = load_operator_limits(LIMITS_YAML, lim.base_vel_max)
    rt = OperatorRuntime(FakeRobotIO(time.monotonic), {"vendor": vendor}, op, SafetyLayer(lim, Watchdog(0.1)),
                         RunLog(tmp_path / "run"), cfg=RuntimeConfig(), disk_free=lambda p: 100e9)
    rt.start()
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 4.0 and not rt.status()["preflight_green"]:
            rt.heartbeat({"mode": "teleop", "held": False, "vx": 0, "vy": 0, "wz": 0})
            time.sleep(0.05)
        assert rt.status()["preflight_green"]
        code = rt.request_arm()["code"]
        assert rt.confirm_arm(f"ARM {code}")[0]
        for _ in range(8):
            rt.heartbeat({"mode": "teleop", "held": False, "vx": 0, "vy": 0, "wz": 0})
            time.sleep(0.05)
        for _ in range(20):
            rt.heartbeat({"mode": "teleop", "held": True, "vx": 0.2, "vy": 0, "wz": 0})
            time.sleep(0.05)
        assert max((c[1] for c in client.calls if c[0] == "SetVelocity"), default=0.0) > 0.05

        hang = threading.Event()
        orig = rt.runlog.command
        rt.runlog.command = lambda rec: hang.wait(10)  # the control thread blocks inside the tick, holding the lock
        t_hang = time.monotonic()
        while rt.sup.armed and not rt._pending_trip and time.monotonic() - t_hang < 2.0:
            time.sleep(0.005)
        t_dis = time.monotonic() - t_hang
        # the supervisor cannot take the lock the hung tick holds: the ROBOT side is zeroed at once (emergency_stop, no lock),
        # the supervisor flag follows when the loop runs again (_pending_trip); heartbeats are refused meanwhile (503)
        assert not rt.sup.armed or rt._pending_trip, "still armed 2 s after the control loop hung"
        with pytest.raises(RuntimeError):
            rt.heartbeat({"mode": "teleop", "held": True, "vx": 0.2, "vy": 0, "wz": 0})
        assert t_dis <= rt.cfg.tick_deadline_s + 0.35, f"{t_dis:.3f} s to react"
        time.sleep(0.2)
        assert client.calls[-1][1:4] == (0.0, 0.0, 0.0)
        hang.set()
        rt.runlog.command = orig
    finally:
        rt.close()
        vendor.close()


# ------------------------------------------------------------------------------------------------ safe-state of the vendor sender

def test_vendor_sender_survives_rpc_exceptions(tmp_path):
    class Boom(FakeLocoClient):
        def SetVelocity(self, *a, **k):  # noqa: N802
            raise TimeoutError("rpc timeout")

    clk = Clock()
    v = VendorLocoController(Boom(), threaded=False, clock=clk)
    v.on_arm(clk.t, None)
    for _ in range(30):
        v.step(clk.t, DT, None, (0.2, 0, 0))
        clk.t += DT
    ok, why = v.ready()
    assert not ok and "SetVelocity failed" in why
    assert v.fault()
