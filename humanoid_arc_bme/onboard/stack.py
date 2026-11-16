"""RobotStack: the whole onboard chain behind two calls, `say(text)` and `step(frame, dt)`.

    sensors -> perception (detector -> scene) -> TaskExecutor -> PipelineRunner (guard, costmap, caps) -> base command
                                                          \\-> arm / hand joint targets

The stack OUTPUTS a tightened base command and arm targets; the runtime (operator loop, GR00T controller, S1) applies
them. The same object runs against the CPU simulator, Isaac and the real robot: only the SensorFrame source differs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from humanoid_arc_bme.onboard.command.grammar import LocalGrammarParser
from humanoid_arc_bme.onboard.command.types import Command
from humanoid_arc_bme.onboard.perception.scene import SceneObject, build_scene
from humanoid_arc_bme.onboard.pipeline.interfaces import DepthFrame, LocalizerEstimate, SemanticsModel
from humanoid_arc_bme.onboard.pipeline.manipulation.grasp_analytic import AnalyticGraspProposer
from humanoid_arc_bme.onboard.pipeline.perception.lidar import LidarFrame
from humanoid_arc_bme.onboard.pipeline.runner import PipelineRunner, RunnerConfig, TickInput
from humanoid_arc_bme.onboard.pipeline.stubs import NullSemantics
from humanoid_arc_bme.onboard.task.arm_task import ArmPickPlace, ArmTargets
from humanoid_arc_bme.onboard.task.executor import ExecutorConfig, TaskExecutor, TaskObservation, TaskStatus

ZERO = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SensorFrame:
    stamp: float
    down_depth: DepthFrame | None
    rgb: np.ndarray | None = None
    lidar: LidarFrame | None = None
    pose: LocalizerEstimate | None = None
    roll: float = 0.0
    pitch: float = 0.0
    joints: dict | None = None           # measured joints by name (waist, arms, hands)
    base_speed: float = 0.0              # m/s, measured (or commanded) planar speed
    safe: bool = True                    # estop clear and the runtime's own safety checks pass


@dataclass(frozen=True)
class StackOutput:
    base: tuple[float, float, float]     # (vx, vy, wz) after the pipeline runner: hand to S1 / the controller
    arm: ArmTargets | None               # joint targets, or None = leave the upper body to the planner
    status: TaskStatus
    safety: str = "ok"                   # runner level: ok | slow | stop
    reasons: tuple[str, ...] = ()
    scene: tuple[SceneObject, ...] = ()
    info: str = ""


def default_runner_config() -> RunnerConfig:
    """The profile for a robot with a down-tilted depth camera and a Livox lidar, no neck camera."""
    return RunnerConfig(guard_camera="down", max_range=1.0, require_lidar=False)


class RobotStack:
    def __init__(self, semantics: SemanticsModel | None = None, arm: ArmPickPlace | None = None,
                 runner: PipelineRunner | None = None, executor_cfg: ExecutorConfig | None = None,
                 parser: LocalGrammarParser | None = None, perception_period_s: float = 0.2, max_scene_age_s: float = 0.6):
        self.semantics = semantics if semantics is not None else NullSemantics()
        self.runner = runner if runner is not None else PipelineRunner(
            default_runner_config(), semantics=self.semantics, proposer=AnalyticGraspProposer(seed=0))
        self.executor = TaskExecutor(arm, executor_cfg)
        self.parser = parser if parser is not None else LocalGrammarParser()
        self.period, self.max_scene_age = float(perception_period_s), float(max_scene_age_s)
        self._scene: tuple[SceneObject, ...] = ()
        self._scene_t = -math.inf
        self._last_percept = -math.inf
        self._safety, self._reasons = "ok", ()
        self._cands: Sequence = ()
        self._last_stamp: float | None = None

    # ---------------------------------------------------------------- perception
    def _perceive(self, f: SensorFrame) -> None:
        if f.down_depth is None or f.stamp - self._last_percept < self.period:
            return
        self._last_percept = f.stamp
        try:
            dets = self.semantics.detect(f.rgb, f.down_depth)
            self._scene = tuple(build_scene(dets, f.rgb, f.down_depth))
            self._scene_t = f.stamp
        except Exception:  # noqa: BLE001 - perception must never take the loop down: no scene = nothing seen
            self._scene, self._scene_t = (), -math.inf

    def _fresh_scene(self, stamp: float) -> tuple[SceneObject, ...]:
        return self._scene if stamp - self._scene_t <= self.max_scene_age else ()

    @staticmethod
    def _xyyaw(f: SensorFrame):
        e = f.pose
        if e is None or not e.healthy:
            return None
        return (e.pose.position[0], e.pose.position[1], e.pose.yaw())

    def _obs(self, f: SensorFrame) -> TaskObservation:
        return TaskObservation(f.stamp, self._xyyaw(f), self._fresh_scene(f.stamp), f.joints, f.base_speed,
                               self._safety == "stop", f.safe, tuple(self._cands))

    @property
    def safety(self) -> str:
        """The pipeline runner's level on the last tick: ok | slow | stop."""
        return self._safety

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(self._reasons)

    def scene(self, stamp: float) -> tuple[SceneObject, ...]:
        """The newest scene if it is fresh at `stamp`, else nothing."""
        return self._fresh_scene(stamp)

    # ---------------------------------------------------------------- operator input
    def parse(self, text: str) -> Command:
        return self.parser.parse(text)

    def say(self, text: str, f: SensorFrame) -> TaskStatus:
        """Parse one sentence and hand it to the executor. Stop is honoured before anything else."""
        return self.command(self.parser.parse(text), f)

    def command(self, cmd: Command, f: SensorFrame) -> TaskStatus:
        self._perceive(f)   # a command sees the newest frame
        return self.executor.submit(cmd, self._obs(f))

    def stop(self, why: str = "stop") -> None:
        self.executor.stop(why)

    def reset_perception(self) -> None:
        """Start perception from "nothing known": runner (map, memory, pose gate, stamp history), scene and grasp candidates. Called when the
        robot is armed again: it may have been carried or pushed while disarmed, and a disarmed gap longer than the runner's stamp-jump
        limit would otherwise latch `bad_stamp` and keep the robot stopped for good."""
        self.runner.reset()
        self._scene, self._scene_t, self._last_percept, self._cands = (), -math.inf, -math.inf, ()
        self._safety, self._reasons = "ok", ()

    # ---------------------------------------------------------------- one control tick
    def step(self, f: SensorFrame, dt: float) -> StackOutput:
        """Never raises. On any internal fault the base command is zero and the arm holds its last targets."""
        try:
            self._perceive(f)
            out = self.executor.tick(self._obs(f), dt)
            self.runner.set_approach(out.approach)
            want = out.want_grasp
            if want != self.runner.grasp_request:
                if want is None:
                    self.runner.end_grasp()
                else:
                    self.runner.start_grasp(want)
            r = self.runner.tick(TickInput(f.stamp, None, f.down_depth, f.pose, f.roll, f.pitch, f.rgb, out.base, f.lidar))
            self._safety, self._reasons = r.safety.level, r.safety.reasons
            self._cands = r.grasp_candidates
            return StackOutput(tuple(r.command_out), out.arm, out.status, r.safety.level, tuple(r.safety.reasons),
                               self._fresh_scene(f.stamp), out.info)
        except Exception as exc:  # noqa: BLE001
            self.executor.stop(f"internal error: {type(exc).__name__}")
            self._safety, self._reasons = "stop", (f"exception:{type(exc).__name__}",)
            return StackOutput(ZERO, self.executor._arm_hold(None), self.executor.status, "stop", self._reasons)
