"""RobotSession: the robot's own software stack around the simulated G1, wired as `runtime/cli.py` wires it on the robot.

    OperatorRuntime (arming, hold-to-run deadman, S1 clamp, supervisor, obstacle stage)
      <- AutonomyLoop <- RobotStack <- SensorHub <- camera / lidar / localizer / hands   (the simulator's sensors)
      -> GrootPlannerController -> planner messages -> the deploy (stand-in) -> rt/lowcmd -> motors

Nothing in `humanoid_arc_bme.onboard` is changed or subclassed: the simulator supplies the same five objects the robot supplies (a
RobotIO, a camera source, a lidar source, a localizer, a hand feed) and a transport for the planner messages. Everything runs on the
simulation clock from G1Sim's physics callback: the runtime at 50 Hz, the heartbeat and the autonomy loop at 20 Hz.
"""

from __future__ import annotations

import math
import time

import numpy as np

from humanoid_arc_bme.isaac.rates import CONTROL_HZ, STACK_HZ
from humanoid_arc_bme.onboard.api.contract import Imu, RobotState
from humanoid_arc_bme.onboard.api.safety import SafetyLayer, Watchdog, load_limits
from humanoid_arc_bme.onboard.g1_layout import MOTOR_JOINTS
from humanoid_arc_bme.onboard.paths import DEFAULT_LIMITS
from humanoid_arc_bme.onboard.pipeline.interfaces import LocalizerEstimate, yaw_pose
from humanoid_arc_bme.onboard.runtime.io.base import CommandRefused, RobotIO
from humanoid_arc_bme.onboard.runtime.types import StateMeta


class SimRobotIO(RobotIO):
    """rt/lowstate as the runtime sees it. The controller (the deploy) owns rt/lowcmd, so joint targets are refused."""
    simulated = True

    def __init__(self, sim):
        self.sim = sim
        self.faults: dict = {}

    def read_state(self):
        st = self.sim.state
        if st is None or self.faults.get("state_lost"):
            return None
        if self.faults.get("state_frozen"):
            st = self.faults.setdefault("_frozen", st)                 # the link still delivers, the content no longer changes
        rs = RobotState(st.stamp, MOTOR_JOINTS, tuple(map(float, st.q)), tuple(map(float, st.dq)), (0.0, 0.0, 0.0),
                        Imu(tuple(map(float, st.quat)), tuple(map(float, st.gyro)), tuple(map(float, st.accel))), {})
        return rs, StateMeta(extra={"tick": st.tick})

    def send_joint_targets(self, names, targets, kp=None, kd=None):
        raise CommandRefused("the GR00T deploy owns rt/lowcmd")


class TruthLocalizer:
    """Ground-truth pose with white noise: the stand-in for a lidar-inertial odometry when none runs. `faults['lost']` makes it report
    an unhealthy estimate, `faults['silent']` makes it stop answering (both are how a real localizer fails)."""

    def __init__(self, robot, noise_m: float = 0.005, noise_rad: float = 0.003, seed: int = 0):
        self.robot, self.noise_m, self.noise_rad = robot, noise_m, noise_rad
        self.rng = np.random.default_rng(seed)
        self.faults: dict = {}

    def estimate(self, stamp: float):
        if self.faults.get("silent"):
            return None
        tr = self.robot.truth()
        n = self.rng.normal(0.0, self.noise_m, 2)
        pose = yaw_pose(tr["pos"][0] + n[0], tr["pos"][1] + n[1], 0.0, tr["yaw"] + self.rng.normal(0.0, self.noise_rad))
        return LocalizerEstimate(stamp, pose, (self.noise_m ** 2,) * 3 + (self.noise_rad ** 2,) * 3, not self.faults.get("lost"), "sim_truth")


class DeployTransport:
    """Planner messages handed to the deploy in-process (lockstep). Same `send` / `close` as the ZMQ transport; keeps a count."""

    def __init__(self, deploy, clock):
        self.deploy, self.clock, self.sent = deploy, clock, 0

    def send(self, msg: bytes) -> None:
        self.sent += 1
        self.deploy.on_message(msg, self.clock())

    def close(self) -> None:
        pass


class RobotSession:
    def __init__(self, sim, robot, rig, deploy_transport, run_dir, mounts, semantics, localizer=None, lidar: bool = True, arm_tasks: bool = True,
                 autonomy_yaml=None, obstacle=None):
        from humanoid_arc_bme.onboard.nav.obstacle_stage import ObstacleStage
        from humanoid_arc_bme.onboard.runtime.autonomy_main import build_arm, build_autonomy, load_config
        from humanoid_arc_bme.onboard.runtime.groot_controller import GrootPlannerController
        from humanoid_arc_bme.onboard.runtime.operator import load_operator_limits
        from humanoid_arc_bme.onboard.runtime.operator_runtime import OperatorRuntime, RuntimeConfig
        from humanoid_arc_bme.onboard.runtime.path_follow import load_nav
        from humanoid_arc_bme.onboard.runtime.runlog import RunLog
        from humanoid_arc_bme.onboard.paths import CONFIG_DIR

        self.sim, self.robot, self.rig = sim, robot, rig
        clock = lambda: sim.t                                           # noqa: E731 - everything below lives on simulation time
        wall0 = time.time()
        limits = load_limits(DEFAULT_LIMITS)
        self.io = SimRobotIO(sim)
        self.transport = deploy_transport
        self.ctrl = GrootPlannerController(deploy_transport, enabled=True, threaded=False, clock=clock, limits=limits)
        # Manual driving (UI teleop, waypoints) passes the runtime's obstacle stage, which needs a distance source: the pipeline runner's own
        # clearance ahead (camera + lidar + memory). Without it the stage reports NO SENSOR and walks the robot into a wall at the S1 speed.
        # No rear sensor: reversing is refused. `self.stack` is set below; the callable reads it lazily.
        self.obstacle = obstacle if obstacle is not None else ObstacleStage(
            distance=lambda d: self.stack.runner.clearance_ahead(sim.t) if d > 0 else None, no_sensor_speed=float(limits.base_vel_max[0]))
        nav, self.nav_why = load_nav("humanoid_arc_bme.onboard.nav.runtime_hook", {}, clock)
        self.rt = OperatorRuntime(self.io, {"groot": self.ctrl}, load_operator_limits(DEFAULT_LIMITS, limits.base_vel_max), SafetyLayer(limits, Watchdog(0.1)),
                                  RunLog(run_dir, wall=lambda: wall0 + sim.t), cfg=RuntimeConfig(), nav=nav, default_controller="groot",
                                  clock=clock, wall=lambda: wall0 + sim.t, sleep=lambda s: None, disk_free=lambda p: 1e12, obstacle=self.obstacle)
        self.localizer = localizer if localizer is not None else TruthLocalizer(robot)
        cfg = load_config(autonomy_yaml or (CONFIG_DIR / "autonomy.yaml"))
        over = mounts.autonomy_overrides("sim")                         # geometry comes from the one mounts file, never from autonomy.yaml
        cfg.update({"pelvis_height_m": over["pelvis_height_m"], "camera": {**cfg["camera"], **over["camera"]}, "lidar": {**cfg.get("lidar", {}), **over["lidar"]}})
        self_r = float(mounts.raw.get("self_view_radius_m", {}).get("sim", 0.0))     # the robot's own feet and hands, for camera and lidar alike
        cfg["runner"] = {**cfg.get("runner", {}), "camera_self_radius_m": self_r, **({"lidar_self_radius_m": self_r} if self_r > 0 else {})}
        self.cfg = cfg
        camera = _Source(lambda: (rig.rgb("d435"), rig.depth("d435")))
        lidar_src = _Source(rig.lidar_frame) if lidar else None
        hands = _Source(lambda: None if sim.state is None else (tuple(map(float, sim.state.hands_q[0])), tuple(map(float, sim.state.hands_q[1]))))
        self.loop = build_autonomy(self.rt, cfg, controller=None, camera=camera, lidar=lidar_src, localizer=self.localizer, hands=hands,
                                   semantics=semantics, arm=build_arm() if arm_tasks else None)
        self.loop.clock = clock
        self.stack = self.loop.stack
        self.mode, self.held, self.teleop = "ros", True, (0.0, 0.0, 0.0)
        self.external_heartbeat = False        # True: a UI sends the operator's heartbeat (hold-to-run) itself, as on the robot
        self.tick_ms: list[float] = []
        self.heartbeat_errors: list[str] = []
        sim.every(1.0 / CONTROL_HZ, lambda s: self.rt.tick())
        sim.every(1.0 / STACK_HZ, self._stack_tick)

    def _stack_tick(self, _sim) -> None:
        a = time.perf_counter()
        if not self.external_heartbeat:
            try:                                                        # the operator's side: the UI heartbeat with the HOLD input
                vx, vy, wz = self.teleop
                self.rt.heartbeat({"mode": self.mode, "held": self.held, "vx": vx, "vy": vy, "wz": wz})
            except Exception as exc:  # noqa: BLE001 - a refused heartbeat is data for the scenario, not a crash
                self.heartbeat_errors.append(f"{type(exc).__name__}: {exc}"[:120])
        self.loop.tick()
        self.tick_ms.append((time.perf_counter() - a) * 1e3)

    # ------------------------------------------------------------------ the operator
    def arm(self) -> tuple[bool, str]:
        """Request and confirm arming, as the UI does; then one released heartbeat (the runtime wants a neutral input first)."""
        code = self.rt.request_arm()["code"]
        ok, why = self.rt.confirm_arm(f"ARM {code}")
        self.rt.heartbeat({"mode": self.mode, "held": False})          # heartbeat only: a second stack tick at the same stamp would (rightly)
        return ok, why                                                  # latch the pipeline runner's `bad_stamp` stop

    def say(self, text: str):
        return self.loop.say(text)

    def run(self, seconds: float, until=None) -> bool:
        """Advance up to `seconds`; stop early when `until(self)` is true (checked at the stack rate). Returns whether it was."""
        end = self.sim.t + seconds
        while self.sim.t < end:
            self.sim.advance(1.0 / STACK_HZ)
            if until is not None and until(self):
                return True
        return False

    def status(self) -> dict:
        st, rs = self.stack.executor.status, self.rt.status()
        return {"t": round(self.sim.t, 2), "task": {"state": st.state, "phase": st.phase, "message": st.message, "command": st.command},
                "safety": self.stack.safety, "reasons": list(self.stack.reasons), "armed": bool(self.rt.sup.armed), "fault": self.rt.fault,
                "last_cmd": [round(float(v), 3) for v in self.rt.last_cmd], "autonomy_error": self.loop.last_error, "runtime_state": rs.get("state")}


class _Source:
    """An object with `latest()`: what SensorHub asks every sensor source for."""

    def __init__(self, fn):
        self.latest = fn
