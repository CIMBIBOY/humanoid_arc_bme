"""Controllers with fakes: vendor LocoClient streaming/zero/fault, policy readiness gates, lowcmd builder and gates."""

import math
import types
from pathlib import Path

import pytest

from humanoid_arc_bme.onboard.runtime.controllers import FakeLocoClient, PolicyController, VendorLocoController, list_policies
from humanoid_arc_bme.onboard.runtime.io.base import CommandRefused
from humanoid_arc_bme.onboard.runtime.io.fake import G1_JOINTS, FakeRobotIO
from humanoid_arc_bme.onboard.runtime.io.unitree import N_MOTORS, UnitreeG1IO, fill_lowcmd
from humanoid_arc_bme.onboard.api.safety import SafetyLayer, Watchdog, load_limits

ROOT = Path(__file__).resolve().parents[2]


def vendor(codes=None, **kw):
    c = FakeLocoClient(codes)
    return VendorLocoController(c, threaded=False, **kw), c


def test_vendor_sends_nothing_until_armed_and_streams_at_send_hz():
    v, c = vendor(send_hz=10.0, cmd_duration_s=0.4)
    v.step(0.0, 0.02, None, (0.3, 0.0, 0.0))  # not armed: nothing
    v.pump(0.1)
    assert not [x for x in c.calls if x[0] == "SetVelocity"]
    v.on_arm(1.0, None)
    for k in range(50):  # 1 s at 50 Hz
        v.step(1.0 + k * 0.02, 0.02, None, (0.3, 0.1, -0.2))
    sv = [x for x in c.calls if x[0] == "SetVelocity"]
    assert len(sv) == 10 and sv[-1] == ("SetVelocity", 0.3, 0.1, -0.2, 0.4)


def test_vendor_disarm_and_stop_send_zero_and_stop_streaming():
    v, c = vendor()
    v.on_arm(0.0, None)
    v.step(0.0, 0.02, None, (0.3, 0.0, 0.0))
    v.on_disarm(0.05, "deadman_timeout")
    assert c.calls[-1] == ("SetVelocity", 0.0, 0.0, 0.0, 1.0)
    n = len(c.calls)
    v.step(0.5, 0.02, None, (0.3, 0.0, 0.0))  # a late step after disarm must not restart streaming
    assert all(x[0] != "SetVelocity" or x[1] == 0.0 for x in c.calls[n:])
    v.on_arm(1.0, None)
    v.step(1.0, 0.02, None, (0.2, 0.0, 0.0))
    v.stop(1.01)
    assert c.calls[-1] == ("SetVelocity", 0.0, 0.0, 0.0, 1.0)
    assert v.damp(1.1).startswith("Damp sent") and c.calls[-1] == ("Damp",)


def test_vendor_fault_after_consecutive_errors_and_exceptions():
    v, c = vendor(codes=[3104, 3104, 3104], max_consecutive_errors=3)
    v.on_arm(0.0, None)
    for k in range(3):
        v.step(k * 0.1, 0.1, None, (0.1, 0.0, 0.0))
    assert v.fault() and not v.ready()[0]

    class Boom(FakeLocoClient):
        def SetVelocity(self, *a):  # noqa: N802
            raise TimeoutError("rpc")

    v2 = VendorLocoController(Boom(), threaded=False, max_consecutive_errors=1)
    v2.on_arm(0.0, None)
    v2.step(0.0, 0.1, None, (0.1, 0.0, 0.0))  # must not raise
    assert "TimeoutError" in v2.fault()


def test_vendor_rejects_long_duration():
    with pytest.raises(ValueError):
        VendorLocoController(FakeLocoClient(), threaded=False, cmd_duration_s=5.0)
    with pytest.raises(ValueError):
        VendorLocoController(FakeLocoClient(), threaded=False, send_hz=2.0, cmd_duration_s=0.4)


def test_vendor_threaded_sender_runs_and_closes():
    import time
    c = FakeLocoClient()
    v = VendorLocoController(c, send_hz=20.0, cmd_duration_s=0.4)
    v.on_arm(time.monotonic(), None)
    v.step(time.monotonic(), 0.02, None, (0.1, 0.0, 0.0))
    time.sleep(0.25)
    v.on_disarm(time.monotonic(), "test")
    v.close()
    sv = [x for x in c.calls if x[0] == "SetVelocity"]
    assert len(sv) >= 3 and sv[-1][1:4] == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------- policy readiness gates

def _safety(validated=False):
    import dataclasses
    lim = dataclasses.replace(load_limits(ROOT / "config" / "g1_limits.yaml"), validated_on_robot=validated)
    return SafetyLayer(lim, Watchdog(0.1))


class Stub:
    def __init__(self):
        import numpy as np
        self.np = np

    def run(self, _, feed):
        return [self.np.zeros((1, 12), self.np.float32)]


def _model(tmp_path):
    import json
    lay = {"obs_joint_names": list(G1_JOINTS), "obs_default_pos": [0.0] * 29, "action_joint_names": list(G1_JOINTS[:12]),
           "action_offset": [0.0] * 12, "action_scale": 0.25, "raw_lo": [-1.0] * 12, "raw_hi": [1.0] * 12, "hist": 1,
           "lin_vel": False, "obs_dim": 4 + 76, "action_kp": [50.0] * 12, "action_kd": [2.0] * 12}
    (tmp_path / "layout.json").write_text(json.dumps(lay))
    (tmp_path / "policy.onnx").write_bytes(b"stub")
    return tmp_path


class RealishIO(FakeRobotIO):
    simulated = False


def test_policy_ready_gates_on_real_io(tmp_path):
    d = _model(tmp_path)
    t = [0.0]
    sim = PolicyController(d, FakeRobotIO(lambda: t[0]), _safety(False), session=Stub())
    assert sim.ready()[0]
    real = PolicyController(d, RealishIO(lambda: t[0]), _safety(False), session=Stub())
    ok, why = real.ready()
    assert not ok and "placeholders" in why
    real2 = PolicyController(d, RealishIO(lambda: t[0]), _safety(True), session=Stub())
    ok, why = real2.ready()
    assert not ok and "--allow-lowcmd" in why
    assert list_policies(d) == [d] and list_policies(tmp_path.parent / "nope") == []


def test_policy_step_sends_through_s1_with_layout_gains(tmp_path):
    t = [0.0]
    io = FakeRobotIO(lambda: t[0])
    p = PolicyController(_model(tmp_path), io, _safety(), session=Stub())
    st, _ = io.read_state()
    p.on_arm(0.0, st)
    info = p.step(0.0, 0.02, st, (0.1, 0.0, 0.0))
    assert io.sent and io.sent[-1][0] == G1_JOINTS[:12] and info["infer_ms"] is not None
    assert p.policy.last_obs[:3].tolist() == pytest.approx([0.1, 0.0, 0.0])


# ---------------------------------------------------------------- lowcmd builder + gates (fake SDK)

def _cmd():
    return types.SimpleNamespace(motor_cmd=[types.SimpleNamespace() for _ in range(35)], mode_pr=None, mode_machine=None, crc=None)


def test_fill_lowcmd():
    c = fill_lowcmd(_cmd(), [0.1] * 29, [40.0] * 29, [1.0] * 29, mode_machine=7)
    m = c.motor_cmd[3]
    assert (c.mode_pr, c.mode_machine, m.mode, m.q, m.dq, m.kp, m.kd, m.tau) == (0, 7, 1, 0.1, 0.0, 40.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        fill_lowcmd(_cmd(), [0.1] * 28, [1.0] * 29, [1.0] * 29, 0)
    with pytest.raises(ValueError):
        fill_lowcmd(_cmd(), [math.nan] * 29, [1.0] * 29, [1.0] * 29, 0)
    with pytest.raises(ValueError):
        fill_lowcmd(_cmd(), [0.0] * 29, [-1.0] * 29, [1.0] * 29, 0)


class FakeSDK:
    def __init__(self, modes=("ai",)):
        self.inits, self.written, self.modes = [], [], list(modes)
        sdk = self

        class Sub:
            def __init__(self, topic, typ):
                sdk.sub_topic = topic

            def Init(self, cb, depth):  # noqa: N802
                sdk.cb = cb

        class Pub:
            def __init__(self, topic, typ):
                sdk.pub_topic = topic

            def Init(self):  # noqa: N802
                pass

            def Write(self, cmd):  # noqa: N802
                sdk.written.append(cmd)

        class Crc:
            def Crc(self, cmd):  # noqa: N802
                return 1234

        class Msc:
            def SetTimeout(self, s):  # noqa: N802
                pass

            def Init(self):  # noqa: N802
                pass

            def CheckMode(self):  # noqa: N802
                return 0, {"name": sdk.modes[0] if sdk.modes else ""}

            def ReleaseMode(self):  # noqa: N802
                sdk.modes.pop(0)

        self.ns = types.SimpleNamespace(ChannelFactoryInitialize=lambda *a: self.inits.append(a), ChannelSubscriber=Sub,
                                        ChannelPublisher=Pub, LowState_=object, LowCmd_=object, default_lowcmd=_cmd, CRC=Crc,
                                        MotionSwitcherClient=Msc)


def _lowstate(q=0.05):
    motors = [types.SimpleNamespace(q=q, dq=0.0, temperature=[30, 41]) for _ in range(35)]
    return types.SimpleNamespace(motor_state=motors, mode_machine=5, tick=9,
                                 imu_state=types.SimpleNamespace(quaternion=[1, 0, 0, 0], gyroscope=[0, 0, 0], accelerometer=[0, 0, 9.8]))


def test_unitree_io_read_only_by_default_sends_nothing():
    f = FakeSDK()
    io = UnitreeG1IO("eth0", sdk=f.ns)
    assert f.inits == [(0, "eth0")] and f.sub_topic == "rt/lowstate" and not hasattr(f, "pub_topic")
    f.cb(_lowstate())
    st, meta = io.read_state()
    assert meta.extra["motor_temp_max_c"] == 41 and meta.extra["tick"] == 9
    with pytest.raises(CommandRefused):
        io.send_joint_targets(G1_JOINTS[:1], (0.0,))
    assert io.damp() is False and f.written == []
    with pytest.raises(CommandRefused):
        io.release_vendor_mode()


def test_unitree_io_lowcmd_gates_release_commanding_damping():
    f = FakeSDK(modes=("ai", "ai"))
    io = UnitreeG1IO(None, sdk=f.ns, enable_lowcmd=True, hold_kp=[40.0] * 29, hold_kd=[1.0] * 29, damping_kd=2.0)
    assert f.pub_topic == "rt/lowcmd" and f.written == []  # creating the publisher sends nothing
    f.cb(_lowstate())
    st, _ = io.read_state()
    with pytest.raises(CommandRefused):
        io.begin_commanding(st)  # not released
    assert io.release_vendor_mode(sleep=lambda s: None) and io.vendor_released
    with pytest.raises(CommandRefused):
        io.send_joint_targets(G1_JOINTS[:1], (0.3,))  # released but not armed
    io.begin_commanding(st)
    io.send_joint_targets(G1_JOINTS[:2], (0.3, -0.2), kp=(80.0, 81.0), kd=(3.0, 3.0))
    c = f.written[-1]
    assert c.crc == 1234 and c.mode_machine == 5 and c.motor_cmd[0].q == 0.3 and c.motor_cmd[1].kp == 81.0
    assert c.motor_cmd[5].q == 0.05 and c.motor_cmd[5].kp == 40.0  # uncommanded joints hold their pose at arming
    assert io.damp() and f.written[-1].motor_cmd[0].kp == 0.0 and f.written[-1].motor_cmd[0].kd == 2.0
    with pytest.raises(CommandRefused):
        io.send_joint_targets(G1_JOINTS[:1], (0.3,))  # damp ends commanding
    assert len(f.written) == 2 and N_MOTORS == 29
