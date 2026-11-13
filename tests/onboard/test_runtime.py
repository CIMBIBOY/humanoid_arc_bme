import json
import tarfile
import types

import pytest

from humanoid_arc_bme.onboard.runtime.bundle import build
from humanoid_arc_bme.onboard.runtime.camera import FakeCamera
from humanoid_arc_bme.onboard.runtime.io.base import CommandRefused
from humanoid_arc_bme.onboard.runtime.io.fake import G1_JOINTS, FakeRobotIO
from humanoid_arc_bme.onboard.runtime.io.unitree import UnitreeG1IO, state_from_lowstate
from humanoid_arc_bme.onboard.runtime.loop import Runtime
from humanoid_arc_bme.onboard.runtime.telemetry import JsonlSink, MemorySink
from humanoid_arc_bme.onboard.runtime.types import CameraFrame
from humanoid_arc_bme.onboard.api.contract import ArmCommand, BaseVelocity
from humanoid_arc_bme.onboard.api.safety import SafetyLayer, Watchdog, limits_from_dict


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _runtime(**kw):
    clk = Clock()
    io = FakeRobotIO(clk)
    sink = MemorySink()
    return clk, io, sink, Runtime(io, sink, kw.pop("cameras", []), clock=clk, sleep=clk.sleep, **kw)


def test_observer_mode_publishes_state_and_frames_and_never_commands():
    clk, io, sink, rt = _runtime(cameras=[FakeCamera(Clock())])
    out = rt.run(1.0)
    assert out["ticks"] >= 49 and out["no_state"] == 0 and out["refused_commands"] == 0
    assert len(sink.states) == out["ticks"] and sink.frames and io.sent == []
    assert abs(out["mean_period_ms"] - 20.0) < 0.5


def test_stale_and_missing_state_are_counted():
    clk, io, sink, rt = _runtime()
    io.read_state = lambda: None
    rt.tick()
    assert rt.stats.no_state == 1 and sink.events[0][0] == "no_state"
    real = FakeRobotIO(clk).read_state
    io.read_state = lambda: (lambda st_meta: (st_meta[0].__class__(**{**st_meta[0].to_dict(), "stamp": clk() - 1.0}), st_meta[1]))(real())
    rt.tick()
    assert rt.stats.stale_state == 1


def _safety():
    lim = limits_from_dict({"validated_on_robot": True, "pos_margin_rad": 0.05, "joint_vel_max_default": 1.0, "wrist_speed_max": 0.2,
                            "workspace": {"frame": "pelvis", "min": [-1, -1, -1], "max": [1, 1, 1]},
                            "base_vel_max": {"vx": 0.2, "vy": 0.1, "wz": 0.3}, "joints": {}})
    return SafetyLayer(lim, Watchdog(0.5))


class Mover:
    def act(self, state, frames):
        return BaseVelocity(1.0, 0.0, 0.0), [ArmCommand("left", (0.3, 0.0, 0.0, 1, 0, 0, 0), "pelvis")]


def test_policy_commands_pass_s1_and_are_dropped_while_disarmed():
    clk, io, sink, rt = _runtime(policy=Mover(), safety=_safety())
    rt.tick()
    kinds = [k for k, _ in sink.events]
    assert "s1_base_vel" in kinds and "disarmed_command_dropped" in kinds
    assert rt.stats.refused_commands == 1 and io.sent == [] and rt.armed is False


def test_policy_without_safety_is_refused():
    with pytest.raises(ValueError):
        _runtime(policy=Mover())


def test_unitree_read_only_and_lowstate_conversion():
    motors = [types.SimpleNamespace(q=0.01 * i, dq=0.0) for i in range(35)]
    msg = types.SimpleNamespace(motor_state=motors, mode_machine=5,
                                imu_state=types.SimpleNamespace(quaternion=[1, 0, 0, 0], gyroscope=[0, 0, 0], accelerometer=[0, 0, 9.8]))
    st, meta = state_from_lowstate(msg, 1.0)
    assert st.joint_names == G1_JOINTS and st.joint_pos[28] == pytest.approx(0.28) and meta.mode_machine == 5
    assert meta.base_vel_source == "none" and meta.wrist_pose_source == "none"
    with pytest.raises(ValueError):
        state_from_lowstate(types.SimpleNamespace(motor_state=motors[:10], imu_state=msg.imu_state), 1.0)
    io = UnitreeG1IO.__new__(UnitreeG1IO)  # no SDK needed to check the guard
    with pytest.raises(CommandRefused):
        io.send_joint_targets(("a",), (0.0,))
    with pytest.raises(NotImplementedError):
        io.arm("yes")


def test_camera_frame_validates_size():
    with pytest.raises(ValueError):
        CameraFrame("h", 0.0, 4, 4, "rgb8", b"\x00" * 10)


def test_jsonl_sink_round_trip(tmp_path):
    clk, io, _, _ = _runtime()
    path = tmp_path / "t.jsonl"
    sink = JsonlSink(path)
    rt = Runtime(io, sink, [FakeCamera(Clock())], clock=clk, sleep=clk.sleep)
    rt.run(0.2)
    sink.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert {r["kind"] for r in rows} == {"state", "frame"} and next(r for r in rows if r["kind"] == "state")["state"]["joint_names"][0] == "left_hip_pitch_joint"


def test_bundle_has_runtime_and_contract_but_no_training_code(tmp_path):
    out = str(tmp_path / "b.tar.gz")
    m = build(out)
    names = set(tarfile.open(out).getnames())
    assert "humanoid_arc_bme/onboard/runtime/loop.py" in names and "humanoid_arc_bme/onboard/api/contract.py" in names
    assert not any(n.startswith(("tools/", "learning/", "third_party/", "arc/sim")) for n in names)
    assert set(m["files"]) <= names and "git_dirty" in m
