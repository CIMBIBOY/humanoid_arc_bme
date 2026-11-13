"""OnnxLegPolicy behind the runtime's Policy protocol, through S1, with a stub session (always) and the exported
W4 model through onnxruntime (only where onnxruntime and the export exist; run with the ortenv python)."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from humanoid_arc_bme.onboard.runtime.io.fake import G1_JOINTS, FakeRobotIO
from humanoid_arc_bme.onboard.runtime.loop import Runtime
from humanoid_arc_bme.onboard.runtime.policies import OnnxLegPolicy
from humanoid_arc_bme.onboard.runtime.telemetry import MemorySink
from humanoid_arc_bme.onboard.api.contract import Imu, JointTargets, RobotState
from humanoid_arc_bme.onboard.api.safety import SafetyLayer, Watchdog, load_limits

ROOT = Path(__file__).resolve().parents[2]
LEGS = G1_JOINTS[:12]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class StubSession:
    """Returns a fixed raw action; records the inputs."""

    def __init__(self, raw):
        self.raw, self.inputs = np.asarray(raw, np.float32), []

    def run(self, _, feed):
        self.inputs.append(next(iter(feed.values())))
        return [self.raw[None]]


def _model_dir(tmp_path, hist=2):
    lay = {"obs_joint_names": list(G1_JOINTS), "obs_default_pos": [0.0] * 29, "action_joint_names": list(LEGS),
           "action_offset": [0.1] * 12, "action_scale": 0.25, "raw_lo": [-1.0] * 12, "raw_hi": [1.0] * 12, "hist": hist,
           "lin_vel": False, "obs_dim": 4 + hist * 76}
    (tmp_path / "layout.json").write_text(json.dumps(lay))
    return tmp_path


def test_policy_clamps_decodes_and_feeds_back_the_clamped_action(tmp_path):
    sess = StubSession([3.0] + [0.0] * 11)
    pol = OnnxLegPolicy(_model_dir(tmp_path), command=lambda: (0.5, 0.0, 0.0, 0.75), session=sess)
    st, _ = FakeRobotIO(Clock()).read_state()
    out = pol.act(st)
    assert isinstance(out, JointTargets) and out.names == LEGS
    assert out.positions[0] == pytest.approx(0.1 + 0.25 * 1.0) and out.positions[1] == pytest.approx(0.1)
    pol.act(st)
    assert sess.inputs[1].shape == (1, 4 + 2 * 76) and sess.inputs[1][0, -12] == pytest.approx(1.0)  # newest last action
    assert sess.inputs[1][0, 0] == pytest.approx(0.5)


def test_runtime_joint_path_passes_s1_and_sends_only_when_armed(tmp_path):
    clk = Clock()
    io = FakeRobotIO(clk)
    lim = load_limits(ROOT / "config" / "g1_limits.yaml")
    pol = OnnxLegPolicy(_model_dir(tmp_path), command=lambda: (0.0, 0.0, 0.0, 0.75), session=StubSession([1.0] * 12))
    rt = Runtime(io, MemorySink(), policy=pol, safety=SafetyLayer(lim, Watchdog(0.1)), clock=clk, sleep=clk.sleep)
    rt.tick()
    assert io.sent == [] and rt.stats.refused_commands == 1
    rt.arm()
    rt.tick()
    assert len(io.sent) == 1 and io.sent[0][0] == LEGS
    # speed rule: |target - measured| <= v_max * dt per joint
    st, _ = io.read_state()
    pos = dict(zip(st.joint_names, st.joint_pos))
    for n, t in zip(*io.sent[0]):
        assert abs(t - pos[n]) <= lim.joint_vel_limit(n) * rt.dt + 1e-6


def test_arming_a_real_robot_adapter_is_refused():
    class Real(FakeRobotIO):
        simulated = False

    clk = Clock()
    rt = Runtime(Real(clk), MemorySink(), clock=clk, sleep=clk.sleep)
    with pytest.raises(PermissionError):
        rt.arm()


def test_watchdog_gives_damping_hold(tmp_path):
    lim = load_limits(ROOT / "config" / "g1_limits.yaml")
    s1 = SafetyLayer(lim, Watchdog(0.1))
    st = RobotState(stamp=0.0, joint_names=LEGS, joint_pos=(0.2,) * 12, joint_vel=(0.0,) * 12, base_vel=(0, 0, 0),
                    imu=Imu((1, 0, 0, 0), (0, 0, 0), (0, 0, 9.81)), wrist_poses={})
    out, ev = s1.filter_joints(1.0, 0.02, st, JointTargets(LEGS, (1.0,) * 12))
    assert ev[0].rule == "watchdog" and out.positions == (0.2,) * 12


EXPORTS = [Path(p) for p in os.environ.get("HUMANOID_ARC_ONNX_DIRS", "").split(",") if p] or [
    ROOT / "outputs" / "W4" / "onnx" / n for n in ("deployH", "deployE")]


@pytest.mark.parametrize("d", EXPORTS, ids=lambda p: p.name)
def test_exported_model_matches_torch_actions(d):
    """ORT on the builder's observation vs the torch actor's action recorded in the sim (same steps as the parity test)."""
    pytest.importorskip("onnxruntime")
    if not (d / "policy.onnx").exists() or not (d / "closedloop.npz").exists():
        pytest.skip(f"no export in {d}")
    z = np.load(d / "closedloop.npz")
    names = tuple(json.loads((d / "layout.json").read_text())["all_joint_names"])
    worst = 0.0
    for e in range(min(4, z["obs"].shape[1])):
        pol = OnnxLegPolicy(d, command=lambda: None)
        for t in range(z["obs"].shape[0]):
            if z["dones"][t, e]:
                pol.reset()
            pol.last_action = z["last_action"][t, e].astype(np.float64)  # the env's (clamped) last action
            cmd = tuple(z["command"][t, e])
            pol.command = lambda c=cmd: c
            q = z["root_quat_w"][t, e].astype(np.float64)
            st = RobotState(stamp=0.0, joint_names=names, joint_pos=tuple(z["joint_pos"][t, e]),
                            joint_vel=tuple(z["joint_vel"][t, e]), base_vel=(0, 0, 0),
                            imu=Imu(tuple(q / np.linalg.norm(q)), tuple(z["root_ang_vel_b"][t, e]), (0, 0, 9.81)),
                            wrist_poses={})
            pol.act(st)
            raw = pol.session.run(None, {pol.input_name: pol.last_obs[None]})[0][0]
            worst = max(worst, float(np.abs(raw - z["actions"][t, e]).max()))
    print(f"{d.name}: max |ORT(builder obs) - torch(sim obs)| = {worst:.2e}")
    assert worst < 1e-4


def _st(q):
    return RobotState(stamp=0.0, joint_names=LEGS, joint_pos=(q,) * 12, joint_vel=(0.0,) * 12, base_vel=(0, 0, 0),
                      imu=Imu((1, 0, 0, 0), (0, 0, 0), (0, 0, 9.81)), wrist_poses={})


def test_joint_rate_limit_is_on_the_command_not_the_pd_error():
    lim = load_limits(ROOT / "config" / "g1_limits.yaml")
    knee = "left_knee_joint"
    names = (knee,)
    step = lim.joint_vel_limit(knee) * 0.02
    # a joint that cannot follow (measured stays 0.2) while the policy keeps asking for 1.0 +- small moves: no event
    s1 = SafetyLayer(lim, Watchdog(0.1))
    s1.command_received(0.0)
    out, ev = s1.filter_joints(0.0, 0.02, _st(0.2), JointTargets(names, (0.2 + 0.9 * step,)))
    assert not ev  # first command: against the measured position, within one step
    t = 0.2
    for k in range(1, 6):
        t += 0.3 * step
        s1.command_received(0.02 * k)
        out, ev = s1.filter_joints(0.02 * k, 0.02, _st(0.2), JointTargets(names, (t,)))
        assert not ev and out.positions == (t,)  # PD error grows past one step, the command rate does not
    # a jump in the command is limited to one step from the previous applied target
    s1.command_received(0.2)
    out, ev = s1.filter_joints(0.2, 0.02, _st(0.2), JointTargets(names, (t + 3 * step,)))
    assert ev[0].rule == "joint_speed" and abs(out.positions[0] - (t + step)) < 1e-12
    # the old reference (measured) fires on the PD error alone
    old = SafetyLayer(lim, Watchdog(0.1), joint_rate_reference="measured")
    old.command_received(0.0)
    _, ev = old.filter_joints(0.0, 0.02, _st(0.2), JointTargets(names, (0.2 + 3 * step,)))
    assert ev and ev[0].rule == "joint_speed"
