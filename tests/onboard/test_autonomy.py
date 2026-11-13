"""AutonomyLoop beside the real OperatorRuntime (fake robot IO): the runtime keeps arming, deadman and S1; autonomy only
supplies commands and is cancelled by a disarm."""

from types import SimpleNamespace

from humanoid_arc_bme.onboard.runtime.autonomy import AutonomyLoop
from tests.onboard.test_operator_runtime import arm, make, release_once, run


class StubStack:
    def __init__(self, base=(0.15, 0.0, 0.0), arm=None, boom=False):
        self.base, self.arm, self.boom, self.stopped, self.steps = base, arm, boom, [], 0

    def step(self, frame, dt):
        self.steps += 1
        if self.boom:
            raise RuntimeError("boom")
        return SimpleNamespace(base=self.base, arm=self.arm)

    def stop(self, why="stop"):
        self.stopped.append(why)


class StubHub:
    def frame(self, now, state, safe=True):
        return SimpleNamespace(stamp=now, safe=safe) if state is not None else None


class StubCtrl:
    def __init__(self):
        self.set, self.released = [], 0

    def set_upper_body(self, *a):
        self.set.append(a)

    def release_upper_body(self):
        self.released += 1


def drive(rt, clock, auto, seconds, held=True):
    """Runtime at 50 Hz (heartbeat 10 Hz, mode ros), autonomy at 20 Hz."""
    for k in range(int(seconds * 50)):
        if k % 5 == 0:
            rt.heartbeat({"mode": "ros", "held": held})
        if k % 2 == 0:
            auto.tick()
        rt.tick()
        clock.t += 0.02


def test_nothing_is_sent_while_disarmed(tmp_path):
    rt, clock, fake = make(tmp_path)
    stack = StubStack()
    auto = AutonomyLoop(rt, stack, StubHub(), clock=clock)
    run(rt, clock, 1.2)
    drive(rt, clock, auto, 1.0)
    assert stack.steps == 0 and rt.last_cmd == (0.0, 0.0, 0.0)
    rt.close()


def test_armed_and_held_autonomy_moves_the_base_and_s1_still_clamps(tmp_path):
    rt, clock, fake = make(tmp_path)
    stack = StubStack(base=(0.9, 0.0, 0.0))             # far above the S1 limit
    auto = AutonomyLoop(rt, stack, StubHub(), clock=clock)
    run(rt, clock, 1.2)
    assert arm(rt)[0]
    release_once(rt, clock)
    drive(rt, clock, auto, 3.0, held=True)
    vx = rt.last_cmd[0]
    assert 0.05 < vx <= 0.2 + 1e-9, rt.last_cmd
    rt.close()


def test_without_the_hold_input_autonomy_cannot_move_the_robot(tmp_path):
    rt, clock, fake = make(tmp_path)
    auto = AutonomyLoop(rt, StubStack(), StubHub(), clock=clock)
    run(rt, clock, 1.2)
    assert arm(rt)[0]
    release_once(rt, clock)
    drive(rt, clock, auto, 2.0, held=False)
    assert rt.last_cmd == (0.0, 0.0, 0.0)
    rt.close()


def test_disarm_cancels_the_task_and_rearming_does_not_resume_it(tmp_path):
    rt, clock, fake = make(tmp_path)
    stack = StubStack()
    auto = AutonomyLoop(rt, stack, StubHub(), clock=clock)
    run(rt, clock, 1.2)
    assert arm(rt)[0]
    release_once(rt, clock)
    drive(rt, clock, auto, 1.0)
    rt.disarm("test")
    drive(rt, clock, auto, 0.5)
    assert stack.stopped == ["disarmed"]


def test_a_stack_fault_cancels_the_task_and_zeroes_the_command(tmp_path):
    rt, clock, fake = make(tmp_path)
    stack = StubStack(boom=True)
    auto = AutonomyLoop(rt, stack, StubHub(), clock=clock)
    run(rt, clock, 1.2)
    assert arm(rt)[0]
    release_once(rt, clock)
    drive(rt, clock, auto, 1.0)
    assert stack.stopped and "boom" in auto.last_error and rt.last_cmd[0] < 0.02


def test_arm_targets_reach_the_controller_and_are_released_when_absent(tmp_path):
    rt, clock, fake = make(tmp_path)
    ctrl = StubCtrl()
    stack = StubStack(arm=SimpleNamespace(body=(0.0,) * 17, left=(0.0,) * 7, right=(0.0,) * 7))
    auto = AutonomyLoop(rt, stack, StubHub(), controller=ctrl, clock=clock)
    run(rt, clock, 1.2)
    assert arm(rt)[0]
    release_once(rt, clock)
    drive(rt, clock, auto, 0.5)
    assert ctrl.set and ctrl.released == 0
    stack.arm = None
    drive(rt, clock, auto, 0.3)
    assert ctrl.released > 0
