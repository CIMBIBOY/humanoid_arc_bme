"""R-1 (stale state stops commands), R-4 (leg command passes S1's base clamp), R-7 (no hold to placeholder wrists)."""
import types

from humanoid_arc_bme.onboard.runtime.io.fake import G1_JOINTS, FakeRobotIO
from humanoid_arc_bme.onboard.runtime.loop import Runtime
from humanoid_arc_bme.onboard.runtime.telemetry import MemorySink
from humanoid_arc_bme.onboard.runtime.types import StateMeta
from humanoid_arc_bme.onboard.api.contract import BaseVelocity, JointTargets
from humanoid_arc_bme.onboard.api.safety import SafetyLayer, Watchdog, hold_commands, limits_from_dict

LEGS = G1_JOINTS[:12]


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _safety(timeout=0.5):
    lim = limits_from_dict({"validated_on_robot": True, "pos_margin_rad": 0.05, "joint_vel_max_default": 100.0, "wrist_speed_max": 0.2,
                            "workspace": {"frame": "pelvis", "min": [-1, -1, -1], "max": [1, 1, 1]},
                            "base_vel_max": {"vx": 0.2, "vy": 0.1, "wz": 0.3},
                            "joints": {n: {"lower": -3.0, "upper": 3.0} for n in G1_JOINTS}})
    return SafetyLayer(lim, Watchdog(timeout))


class JointPol:
    joint_policy = True
    joint_names = LEGS

    def __init__(self, command=lambda: (1.0, 1.0, 1.0, 0.75)):
        self.command, self.calls, self.seen = command, 0, []

    def act(self, state, frames=None):
        self.calls += 1
        self.seen.append(self.command())
        return JointTargets(LEGS, (0.0,) * 12)


def test_frozen_state_for_5s_sends_no_commands_and_does_not_feed_watchdog():
    clk = Clock()
    io = FakeRobotIO(clk)
    frozen = io.read_state()
    io.read_state = lambda: frozen
    pol = JointPol()
    sink = MemorySink()
    rt = Runtime(io, sink, policy=pol, safety=_safety(), clock=clk, sleep=clk.sleep)
    rt.arm()
    rt.tick()  # fresh: one command
    assert len(io.sent) == 1
    rt.run(0.3)  # the frozen stamp is still within stale_after_s for the first ticks
    n_calls, n_sent = pol.calls, len(io.sent)
    rt.run(5.0)
    assert pol.calls == n_calls and len(io.sent) == n_sent  # zero further policy calls and commands
    assert rt.stats.stale_state > 240 and rt.stats.events["commands_suspended_stale_state"] == rt.stats.stale_state
    assert rt.safety.watchdog.expired(clk())


def test_missing_state_sends_nothing_and_emits_event():
    clk = Clock()
    io = FakeRobotIO(clk)
    io.read_state = lambda: None
    pol = JointPol()
    rt = Runtime(io, MemorySink(), policy=pol, safety=_safety(), clock=clk, sleep=clk.sleep)
    rt.arm()
    rt.run(1.0)
    assert pol.calls == 0 and io.sent == [] and rt.stats.events["commands_suspended_no_state"] == rt.stats.no_state > 0


def test_leg_policy_base_command_is_clamped_by_s1():
    clk = Clock()
    io = FakeRobotIO(clk)
    pol = JointPol(command=lambda: (1.0, -1.0, 5.0, 0.75))
    rt = Runtime(io, MemorySink(), policy=pol, safety=_safety(), clock=clk, sleep=clk.sleep)
    rt.tick()
    assert pol.seen == [(0.2, -0.1, 0.3, 0.75)]
    assert {"s1_base_vel"} <= set(rt.stats.events)


def test_hold_with_placeholder_wrists_commands_no_arm():
    st, meta = FakeRobotIO(Clock()).read_state()
    assert meta.wrist_pose_source == "none" and st.wrist_poses  # placeholder poses exist
    base, arms = hold_commands(st, "none")
    assert base == BaseVelocity(0.0, 0.0, 0.0) and arms == {}
    assert set(hold_commands(st, "fk")[1]) == {"left", "right"}
    s1 = _safety(0.5)  # watchdog never fed: expired
    _, a, ev = s1.filter(1.0, 0.02, st, None, (), wrist_pose_source="none")
    assert a == {} and ev[0].rule == "watchdog"
    _, a, _ = s1.filter(1.0, 0.02, st, None, (), wrist_pose_source="fk")
    assert set(a) == {"left", "right"}


def test_arm_commands_refused_when_wrist_pose_unknown():
    from humanoid_arc_bme.onboard.api.contract import ArmCommand
    st, _ = FakeRobotIO(Clock()).read_state()
    s1 = _safety()
    s1.command_received(1.0)
    cmd = ArmCommand("left", (0.3, 0.0, 0.0, 1, 0, 0, 0), "pelvis")
    _, a, ev = s1.filter(1.0, 0.02, st, None, [cmd], wrist_pose_source="none")
    assert a == {} and any(e.rule == "wrist_pose_unknown" for e in ev)
