"""Regression and property tests from the safety review docs/audit/2026-10-30_safety_review.md.

Each test names the defect id (SR-xx / UI-xx) it checks. All defects fixed in software on 2026-10-30 are now normal tests
(see the "Fixes" section of the review). A test that still fails because of an unfixed defect is marked `xfail(strict=True)`
with the defect id (only SR-12 on the placeholder config/g1_limits.yaml, which is left as it is on purpose: the first walk
uses config/g1_limits_firstwalk.yaml, tested below).
No Isaac, no robot: standard library + the runtime's stdlib-only path.
"""

from __future__ import annotations

import copy
import math

import pytest

from humanoid_arc_bme.onboard.runtime.io.base import RobotIO
from humanoid_arc_bme.onboard.runtime.io.fake import G1_JOINTS
from humanoid_arc_bme.onboard.runtime.io.unitree import state_from_lowstate
from humanoid_arc_bme.onboard.runtime.loop import Runtime
from humanoid_arc_bme.onboard.runtime.telemetry import MemorySink
from humanoid_arc_bme.onboard.runtime.types import StateMeta
from humanoid_arc_bme.onboard.api.contract import Imu, JointTargets, RobotState
from humanoid_arc_bme.onboard.api.safety import SafetyLayer, Watchdog, clamp_joint_targets, limits_from_dict, load_limits

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGS = G1_JOINTS[:12]
KNEE = "left_knee_joint"
DT = 0.02
VMAX = 1.0  # rad/s in the test limits -> 0.02 rad per tick

LIM_DICT = {"validated_on_robot": True, "pos_margin_rad": 0.05, "joint_vel_max_default": VMAX, "wrist_speed_max": 0.2,
            "workspace": {"frame": "pelvis", "min": [-1, -1, -1], "max": [1, 1, 1]},
            "base_vel_max": {"vx": 0.2, "vy": 0.1, "wz": 0.3},
            "joints": {n: {"lower": -3.0, "upper": 3.0} for n in G1_JOINTS}}


def _layer(timeout=0.5, **kw):
    return SafetyLayer(limits_from_dict(LIM_DICT), Watchdog(timeout), **kw)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class SetIO(RobotIO):
    """A simulated robot whose measured joint positions the test sets directly (a joint that does not move, a robot
    pushed by hand, ...). `stamp_offset` shifts the state stamp (negative = old state, positive = future stamp)."""

    simulated = True

    def __init__(self, clock):
        self.clock, self.q = clock, {n: 0.0 for n in G1_JOINTS}
        self.sent: list = []
        self.safe_stops = 0
        self.stamp_offset = 0.0
        self.fail: Exception | None = None

    def read_state(self):
        if self.fail is not None:
            raise self.fail
        st = RobotState(stamp=self.clock() + self.stamp_offset, joint_names=G1_JOINTS, joint_pos=tuple(self.q[n] for n in G1_JOINTS),
                        joint_vel=(0.0,) * len(G1_JOINTS), base_vel=(0.0, 0.0, 0.0), imu=Imu((1, 0, 0, 0), (0, 0, 0), (0, 0, 9.81)),
                        wrist_poses={})
        return st, StateMeta()

    def send_joint_targets(self, names, targets, kp=None, kd=None):
        self.sent.append(dict(zip(names, targets)))

    def safe_stop(self):  # the adapter's safe state (damping / vendor controller); counted, not simulated
        self.safe_stops += 1


class ConstLegPolicy:
    """Leg policy stub: asks for `target` on the knee, 0 elsewhere. `command` is the (vx, vy, wz, height) callable."""

    joint_policy = True
    joint_names = LEGS

    def __init__(self, target=1.0, command=lambda: (0.0, 0.0, 0.0, 0.75)):
        self.target, self.command, self.seen, self.raise_with = target, command, [], None

    def act(self, state, frames=None):
        self.seen.append(self.command())
        if self.raise_with is not None:
            raise self.raise_with
        return JointTargets(LEGS, tuple(self.target if n == KNEE else 0.0 for n in LEGS))


def _rt(policy, io=None, timeout=0.5):
    clk = Clock()
    io = io or SetIO(clk)
    io.clock = clk
    rt = Runtime(io, MemorySink(), policy=policy, safety=_layer(timeout), clock=clk, sleep=clk.sleep)
    return clk, io, rt


# ----------------------------------------------------------------------------------------------------------- pinned (pass today)

def test_pinned_first_command_ever_is_rate_limited_against_measured():
    """Cold start (nothing applied yet): the reference is the measured position."""
    clk, io, rt = _rt(ConstLegPolicy(target=1.0))
    io.q[KNEE] = 0.3
    rt.arm()
    rt.tick()
    assert io.sent[0][KNEE] == pytest.approx(0.3 + VMAX * DT)


def test_pinned_rate_limit_holds_per_tick_against_previous_command():
    layer = _layer()
    st = SetIO(Clock()).read_state()[0]
    prev = 0.0
    for i in range(50):
        layer.command_received(float(i) * DT)
        out, _ = layer.filter_joints(float(i) * DT, DT, st, JointTargets((KNEE,), (2.5,)))
        assert abs(out.positions[0] - prev) <= VMAX * DT + 1e-12
        prev = out.positions[0]


def test_pinned_nan_target_is_rejected_by_the_contract():
    with pytest.raises(ValueError):
        JointTargets((KNEE,), (math.nan,))
    with pytest.raises(ValueError):
        clamp_joint_targets([KNEE], [math.inf], [0.0], DT, limits_from_dict(LIM_DICT))


def test_pinned_disarmed_runtime_never_calls_the_io():
    clk, io, rt = _rt(ConstLegPolicy())
    rt.run(1.0)
    assert io.sent == [] and rt.stats.refused_commands > 0


def test_pinned_repo_limits_refuse_hardware():
    with pytest.raises(RuntimeError):
        load_limits(ROOT / "config" / "g1_limits.yaml").require_validated()


# ----------------------------------------------------------------------------------------------------------- fixed defects

def test_sr01_first_command_after_arm_is_limited_against_measured():
    clk, io, rt = _rt(ConstLegPolicy(target=1.0))
    for _ in range(100):  # 2 s disarmed: S1 filters and remembers targets that are never sent
        rt.tick()
        clk.sleep(DT)
    assert io.sent == []
    rt.arm()
    rt.tick()
    first = io.sent[0][KNEE]
    assert abs(first - io.q[KNEE]) <= VMAX * DT + 1e-9, f"first sent knee target {first} vs measured {io.q[KNEE]}"


def test_sr02_after_a_long_state_gap_the_reference_is_measured():
    clk, io, rt = _rt(ConstLegPolicy(target=0.0), timeout=0.5)
    rt.arm()
    for _ in range(10):
        rt.tick()
        clk.sleep(DT)
    io.stamp_offset = -1.0  # state stream frozen/stale for 3 s (R-1: no policy call, no watchdog feed)
    for _ in range(150):
        rt.tick()
        clk.sleep(DT)
    io.stamp_offset = 0.0
    io.q[KNEE] = 0.8  # meanwhile the knee was moved (vendor damping, a person, a fall)
    rt.policy.target = 0.8  # and the policy now asks to stay there
    rt.tick()
    sent = io.sent[-1][KNEE]
    assert abs(sent - 0.8) <= VMAX * DT + 1e-9, f"sent {sent}: pulled toward the 3 s old target 0.0"


def test_sr03_tracking_error_is_bounded():
    d = copy.deepcopy(LIM_DICT)
    d["tracking_error_max_rad"] = 0.5  # the proposed config key (patch SR-03); ignored by today's loader
    layer = SafetyLayer(limits_from_dict(d), Watchdog(0.5))
    st = SetIO(Clock()).read_state()[0]  # knee measured 0.0 and never moves (blocked, gantry, stuck)
    out = None
    for i in range(200):
        layer.command_received(float(i) * DT)
        out, _ = layer.filter_joints(float(i) * DT, DT, st, JointTargets((KNEE,), (2.5,)))
    assert abs(out.positions[0] - 0.0) <= 0.5, f"|target - measured| = {out.positions[0]:.2f} rad"


def test_sr04_validated_flag_string_false_is_not_true():
    d = copy.deepcopy(LIM_DICT)
    d["validated_on_robot"] = "false"
    try:
        lim = limits_from_dict(d)
    except ValueError:
        return  # refusing a non-bool is also a fix
    assert lim.validated_on_robot is False


def test_sr05_policy_exception_does_not_escape_and_holds():
    clk, io, rt = _rt(ConstLegPolicy(target=0.5))
    rt.arm()
    rt.tick()
    clk.sleep(DT)
    rt.policy.raise_with = ValueError("positions: values must be finite")
    rt.tick()  # must not raise
    assert io.safe_stops == 1 and not rt.armed  # the adapter's safe state was requested and the runtime disarmed
    n = len(io.sent)
    rt.policy.raise_with = None
    clk.sleep(DT)
    rt.tick()
    assert len(io.sent) == n  # the fault is latched: no policy command until an operator resets it


def test_sr05b_state_read_exception_does_not_escape():
    clk, io, rt = _rt(ConstLegPolicy())
    rt.arm()
    rt.tick()
    io.fail = ValueError("quat: quaternion (w, x, y, z) must have unit length")
    rt.tick()


def test_sr06_height_command_is_checked():
    pol = ConstLegPolicy(command=lambda: (0.0, 0.0, 0.0, math.nan))
    clk, io, rt = _rt(pol)
    rt.tick()
    assert all(math.isfinite(c[3]) for c in pol.seen)  # a NaN height never reaches the policy (refused or clipped)
    assert pol.seen or rt.fault is not None


def test_sr08_future_stamped_state_is_not_fresh():
    clk, io, rt = _rt(ConstLegPolicy())
    io.stamp_offset = +5.0
    rt.tick()
    assert rt.policy.seen == [], "policy ran on a state stamped 5 s in the future"


@pytest.mark.xfail(strict=True, reason="SR-12 (placeholder file, kept as is; the first walk uses g1_limits_firstwalk.yaml): config leg speed limits are the URDF hardware maxima (0.4-0.74 rad per 20 ms tick): "
                                       "the S1 rate rule is effectively off for the legs; the first-walk profile must be tighter")
def test_sr12_repo_leg_rate_limit_is_a_real_limit():
    lim = load_limits(ROOT / "config" / "g1_limits.yaml")
    worst = max(lim.joint_vel_limit(n) for n in LEGS) * DT
    assert worst <= 0.2, f"{worst:.2f} rad per tick allowed"


class _Msg:
    pass


def _lowstate(mode_pr=0, tick=1):
    m = _Msg()
    m.mode_pr, m.mode_machine, m.tick = mode_pr, 5, tick
    m.imu_state = _Msg()
    m.imu_state.quaternion, m.imu_state.gyroscope, m.imu_state.accelerometer = (1.0, 0, 0, 0), (0, 0, 0), (0, 0, 9.81)
    motors = []
    for _ in range(35):
        mo = _Msg()
        mo.q, mo.dq, mo.temperature, mo.motorstate = 0.0, 0.0, (30, 30), 0
        motors.append(mo)
    m.motor_state = motors
    return m


def test_sr16_ankle_ab_mode_is_refused():
    with pytest.raises(ValueError):
        state_from_lowstate(_lowstate(mode_pr=1), 0.0)


def test_sr16b_lowstate_health_fields_are_kept():
    _, meta = state_from_lowstate(_lowstate(), 0.0)
    assert "motor_temperature" in meta.extra and "tick" in meta.extra


# ----------------------------------------------------------------------------------------------------------- UI and arming (UI-xx)
# Against the operator runtime written 2026-10-30 (humanoid_arc_bme/onboard/runtime/operator.py, operator_runtime.py, controllers.py, web/).

def _op_runtime(tmp_path, controllers):
    from humanoid_arc_bme.onboard.runtime.io.fake import FakeRobotIO
    from humanoid_arc_bme.onboard.runtime.operator import load_operator_limits
    from humanoid_arc_bme.onboard.runtime.operator_runtime import OperatorRuntime, RuntimeConfig
    from humanoid_arc_bme.onboard.runtime.runlog import RunLog

    clk = Clock()
    lim = load_limits(ROOT / "config" / "g1_limits.yaml")
    op = load_operator_limits(ROOT / "config" / "g1_limits.yaml", lim.base_vel_max)
    rt = OperatorRuntime(FakeRobotIO(clk), controllers, op, SafetyLayer(lim, Watchdog(0.1)),
                         RunLog(tmp_path / "run", wall=lambda: 1.8e9 + clk.t), cfg=RuntimeConfig(), clock=clk,
                         wall=lambda: 1.8e9 + clk.t, sleep=clk.sleep, disk_free=lambda p: 100e9)
    return rt, clk


def _op_run(rt, clk, seconds, vx=0.0):
    for k in range(int(round(seconds * 50))):
        if k % 5 == 0:
            rt.heartbeat({"mode": "teleop", "vx": vx, "vy": 0.0, "wz": 0.0}, client="local")
        rt.tick()
        clk.t += 0.02


def _op_arm(rt):
    code = rt.request_arm()["code"]
    return rt.confirm_arm(f"ARM {code}", client="local")


def test_ui01_tick_exception_does_not_leave_robot_walking(tmp_path):
    from humanoid_arc_bme.onboard.runtime.controllers import FakeLocoClient, VendorLocoController

    vendor = VendorLocoController(FakeLocoClient(), threaded=False, clock=lambda: 0.0)
    rt, clk = _op_runtime(tmp_path, {"vendor": vendor})
    _op_run(rt, clk, 1.0)
    assert _op_arm(rt)[0]
    _op_run(rt, clk, 2.0, vx=0.2)  # walking at 0.2 m/s
    rt.runlog.command = lambda rec: (_ for _ in ()).throw(OSError("No space left on device"))
    try:
        rt.tick()
        raised = False
    except OSError:
        raised = True
    vendor.pump(clk.t + 5.0)  # the sender thread runs on, independent of the dead control thread
    last = vendor.client.calls[-1]
    assert not raised and not rt.sup.armed and last[1:4] == (0.0, 0.0, 0.0), f"raised={raised}, armed={rt.sup.armed}, last call {last}"


def test_ui02_vendor_sender_stops_when_the_loop_stops_feeding_it():
    from humanoid_arc_bme.onboard.runtime.controllers import FakeLocoClient, VendorLocoController

    c = VendorLocoController(FakeLocoClient(), threaded=False, clock=lambda: 0.0)
    c.on_arm(0.0, None)
    c.step(0.0, 0.02, None, (0.2, 0.0, 0.0))  # last step of the control loop
    for k in range(1, 50):  # 5 s of sender-thread pumps without any further step()
        c.pump(0.1 * k)
    assert c.client.calls[-1][1] == 0.0, f"still sending vx={c.client.calls[-1][1]} 5 s after the last step"


def test_ui03_no_motion_after_arm_without_a_neutral_command_first(tmp_path):
    from humanoid_arc_bme.onboard.runtime.controllers import FakeController

    fake = FakeController()
    rt, clk = _op_runtime(tmp_path, {"fake": fake})
    _op_run(rt, clk, 1.0, vx=0.2)  # sliders left at 0.2 before arming
    assert _op_arm(rt)[0]
    _op_run(rt, clk, 1.0, vx=0.2)  # the operator did nothing; the tab keeps sending its slider value
    moving = [c for c in fake.calls if c[0] == "step" and abs(c[2][0]) > 0]
    assert not moving, f"{len(moving)} moving steps after arming without a neutral (zero) command first"


def test_ui04_malformed_heartbeat_does_not_feed_the_deadman():
    from humanoid_arc_bme.onboard.runtime.operator import Supervisor, load_operator_limits

    lim = load_limits(ROOT / "config" / "g1_limits.yaml")
    sup = Supervisor(load_operator_limits(ROOT / "config" / "g1_limits.yaml", lim.base_vel_max))
    sup.heartbeat(0.0, (0.1, 0.0, 0.0))
    with pytest.raises(ValueError):
        sup.heartbeat(0.25, (math.nan, 0.0, 0.0))
    assert sup.heartbeat_age(0.25) == pytest.approx(0.25), "the rejected heartbeat refreshed the deadman"


# ----------------------------------------------------------------------------------------------------------- first-walk config (SR-12)

def test_sr12_firstwalk_limits_are_conservative_and_not_validated():
    fw = load_limits(ROOT / "config" / "g1_limits_firstwalk.yaml")
    assert fw.validated_on_robot is False
    with pytest.raises(RuntimeError):
        fw.require_validated()  # never accepted by the hardware path until step B3.3 has validated it
    worst = max(fw.joint_vel_limit(n) for n in LEGS) * DT
    assert worst <= 0.2, f"{worst:.2f} rad per tick allowed"
    assert fw.base_vel_max == (0.2, 0.1, 0.3)


def test_sr12_firstwalk_file_differs_from_base_only_in_the_documented_keys():
    import yaml
    a = yaml.safe_load((ROOT / "config" / "g1_limits.yaml").read_text())
    b = yaml.safe_load((ROOT / "config" / "g1_limits_firstwalk.yaml").read_text())
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff <= {"joint_vel_max", "tracking_error_max_rad"}, diff
    assert b["validated_on_robot"] is False


def test_firstwalk_envelope_equals_protocol_caps():
    from humanoid_arc_bme.onboard.runtime.operator import load_operator_limits
    for f in ("g1_limits.yaml", "g1_limits_firstwalk.yaml"):
        lim = load_limits(ROOT / "config" / f)
        op = load_operator_limits(ROOT / "config" / f, lim.base_vel_max)
        assert (op.vx_min, op.vx_max, op.vy_max, op.wz_max) == (-0.1, 0.2, 0.1, 0.3), f


def test_damping_state_uses_kd_8_and_zero_kp():
    from humanoid_arc_bme.onboard.runtime.io.unitree import UnitreeG1IO
    import inspect
    assert inspect.signature(UnitreeG1IO.__init__).parameters["damping_kd"].default == 8.0
    import yaml
    assert yaml.safe_load((ROOT / "config" / "robot_runtime.yaml").read_text())["lowcmd"]["damping_kd"] == 8.0


def test_bundle_includes_nav_and_api():
    from humanoid_arc_bme.onboard.runtime.bundle import INCLUDE, files
    assert "humanoid_arc_bme/onboard" in INCLUDE
    names = {str(p.relative_to(ROOT)) for p in files()}
    assert "humanoid_arc_bme/onboard/nav/runtime_hook.py" in names or any(n.startswith("humanoid_arc_bme/onboard/nav/") for n in names)


def test_web_token_compare_is_constant_time_and_strict():
    import inspect
    from humanoid_arc_bme.onboard.runtime.web import server
    src = inspect.getsource(server.WebApp.authorized)
    assert "compare_digest" in src and "==" not in src.replace("==\"", "")  # no plain string equality on the token
