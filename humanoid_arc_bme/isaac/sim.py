"""G1Sim: a world, the simulated G1, its controller link and (optionally) its sensors, advanced in lockstep.

Everything that must happen at the physics rate (publishing rt/lowstate, running the controller, applying rt/lowcmd, sampling the
IMU, the stack's 20 Hz tick) runs in ONE callback before each physics step. That leaves the outer loop free to advance the simulator
the way its sensors need: Isaac's RTX sensors only produce data for a render that happens in the same app update that advances
simulated time, so with sensors one outer step is one lidar slice (several physics steps and one render); without sensors it is one
physics step and nothing is rendered.

Two ways to run the controller side, same messages either way (`lowlevel.py`):
  lockstep   `controller` has `step(LowState, SimState, now) -> LowCmd | None` (the deploy stand-in), called here at the control
             rate. Deterministic and as fast as the machine allows: what sweeps use.
  external   `controller=None`: another process reads rt/lowstate and writes rt/lowcmd through the link (`dds.DdsLink`) at its own
             wall-clock rate, exactly as the real deploy would; pass `realtime=True` so simulated time keeps pace with it.
"""

from __future__ import annotations

import time

from humanoid_arc_bme.isaac.lowlevel import LoopbackLink
from humanoid_arc_bme.isaac.rates import CONTROL_HZ, PHYSICS_HZ


class G1Sim:
    def __init__(self, world, robot, link=None, controller=None, rig=None, physics_hz: int = PHYSICS_HZ, control_hz: int = CONTROL_HZ,
                 realtime: bool = False):
        self.world, self.robot, self.link, self.controller, self.rig = world, robot, link if link is not None else LoopbackLink(), controller, rig
        self.dt, self.decim = 1.0 / physics_hz, physics_hz // control_hz
        self.realtime, self._wall0 = realtime, None
        self.t, self.n = 0.0, 0
        self.state = self.sim_state = None
        self._last_cmd = None
        self._every: list = []                 # [(period in physics steps, fn)]: called before the physics step that starts each period
        self.step_times: list[float] = []      # wall time of each outer step (physics step, or lidar slice with its render)
        self.steps_per_outer = 1 if rig is None else rig.steps_per_slice(self.dt)
        self.on_frame = None                   # called with (self) after each complete 10 Hz sensor frame
        world.add_physics_callback("g1sim", self._before_physics)

    def every(self, seconds: float, fn) -> None:
        """Call `fn(self)` every `seconds` of simulated time, in the physics callback (exact timing whatever the render cadence)."""
        self._every.append((max(1, round(seconds / self.dt)), fn))

    def _before_physics(self, _step_size) -> None:
        for k, fn in self._every:
            if self.n % k == 0:
                fn(self)
        if self.n % self.decim == 0:
            self.state, self.sim_state = self.robot.state(self.t, self.dt * self.decim)
            self.link.write_state(self.state, self.sim_state)
            if self.controller is not None:
                cmd = self.controller.step(self.state, self.sim_state, self.t)
                if cmd is not None:
                    self.link.write_cmd(cmd)
        cmd = self.link.read_cmd()
        fresh = cmd is not None and cmd is not self._last_cmd
        self._last_cmd = cmd
        self.robot.apply(cmd, self.t, fresh)
        if self.rig is not None:
            self.rig.sample_imu(self.t, self.dt)
        self.n += 1
        self.t = self.n * self.dt
        if self.realtime:
            self._wall0 = self._wall0 or (time.perf_counter() - self.t)
            lag = self.t - (time.perf_counter() - self._wall0)
            if lag > 0:
                time.sleep(lag)

    def step(self) -> None:
        """One outer step: a physics step (no sensors), or one lidar slice = several physics steps and a render (with sensors)."""
        a = time.perf_counter()
        if self.rig is None:
            self.world.step(render=False)
        else:
            self.rig.before_render()
            self.world.step(render=True)
            if self.rig.after_render(self.t) and self.on_frame is not None:
                self.on_frame(self)
        self.step_times.append(time.perf_counter() - a)

    def rtf(self) -> float:
        """Real-time factor from the MEDIAN outer-step time: a job the machine guard pauses, or a busy machine, stretches single
        steps by seconds and would make a mean meaningless. With sensors this includes the renders."""
        st = sorted(self.step_times)
        return self.dt * self.steps_per_outer / st[len(st) // 2] if st else float("nan")

    def advance(self, seconds: float, every: float | None = None, fn=None) -> None:
        """Run `seconds` of simulated time; with `every` and `fn`, also call `fn(self)` at that period for the duration."""
        entry = (max(1, round(every / self.dt)), fn) if every and fn is not None else None
        if entry:
            self._every.append(entry)
        end = self.n + round(seconds / self.dt)
        while self.n < end:
            self.step()
        if entry:
            self._every.remove(entry)
