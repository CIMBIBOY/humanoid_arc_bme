"""SimSession: the whole onboard stack in a closed loop against the CPU world.

    SimSession(world, ...).say("pick up the red cup"); session.run(seconds, until=...)

Per tick (20 Hz): render the down camera (10 Hz) -> SensorFrame (localizer pose, measured joints) -> RobotStack.step ->
GrootPlannerController (S1 limits, rate limit, planner-topic wire format through a FakeTransport) -> the arm targets are
DECODED from the message that would go to the deploy and applied to the kinematic robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from humanoid_arc_bme.cpusim.robot import SimRobot
from humanoid_arc_bme.cpusim.world import World
from humanoid_arc_bme.onboard.api.contract import Imu, RobotState
from humanoid_arc_bme.onboard.api.safety import load_limits
from humanoid_arc_bme.onboard.g1_layout import LEFT_HAND_JOINTS, RIGHT_HAND_JOINTS, UPPER_BODY_JOINTS
from humanoid_arc_bme.onboard.paths import DEFAULT_LIMITS
from humanoid_arc_bme.onboard.pipeline.interfaces import DepthFrame, Intrinsics, LocalizerEstimate, yaw_pose
from humanoid_arc_bme.onboard.pipeline.perception.sensing import camera_pose_from_mount
from humanoid_arc_bme.onboard.pipeline.stubs import GroundTruthSemantics
from humanoid_arc_bme.onboard.runtime.groot_controller import FakeTransport, GrootPlannerController
from humanoid_arc_bme.onboard.stack import RobotStack, SensorFrame, StackOutput

# Mount on the torso (0.43 m above the torso origin, UNMEASURED) of a robot whose pelvis stands 0.78 m high: ~1.2 m above the ground.
MOUNT_POS, MOUNT_PITCH = (0.06, 0.0, 0.78 + 0.43), math.radians(50.0)
K = Intrinsics(fx=190.0, fy=190.0, cx=159.5, cy=119.5, width=320, height=240)
ALL_JOINTS = tuple(UPPER_BODY_JOINTS) + tuple(LEFT_HAND_JOINTS) + tuple(RIGHT_HAND_JOINTS)


@dataclass
class SimSession:
    world: World
    robot: SimRobot
    stack: RobotStack
    dt: float = 0.05
    camera_period: float = 0.1
    pose_noise_m: float = 0.0
    seed: int = 0
    t: float = 0.0
    log: list = field(default_factory=list)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.mount = camera_pose_from_mount(MOUNT_POS, MOUNT_PITCH)
        self.tx = FakeTransport()
        self.ctrl = GrootPlannerController(self.tx, enabled=True, threaded=False, clock=lambda: self.t,
                                           limits=load_limits(DEFAULT_LIMITS))
        self.ctrl.on_arm(self.t, None)
        self._depth = self._rgb = self._ids = None
        self._cam_t = -math.inf
        self.out: StackOutput | None = None

    # ------------------------------------------------------------------ sensors
    def _camera(self) -> None:
        r = self.robot
        c, s = math.cos(r.yaw), math.sin(r.yaw)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        origin = np.array([r.x, r.y, 0.0]) + Rz @ np.asarray(self.mount.position)
        R = Rz @ self.mount.rotation_matrix()
        self._depth, self._rgb, self._ids = self.world.render(origin, R, K.fx, K.fy, K.cx, K.cy, K.width, K.height)
        self._cam_t = self.t

    def _frame(self) -> SensorFrame:
        if self.t - self._cam_t >= self.camera_period - 1e-9:
            self._camera()
        r = self.robot
        n = self.rng.normal(0.0, self.pose_noise_m, 2) if self.pose_noise_m > 0 else (0.0, 0.0)
        est = LocalizerEstimate(self.t, yaw_pose(r.x + n[0], r.y + n[1], 0.0, r.yaw), (1e-4,) * 6)
        depth = DepthFrame(self._cam_t, np.where(np.isfinite(self._depth), self._depth, np.nan).astype(np.float32), K,
                           self.mount, camera="down")
        return SensorFrame(self.t, depth, self._rgb, None, est, 0.0, 0.0, dict(r.q), r.speed(), True)

    def _seg(self, rgb, depth):
        return self._ids, self.world.names()

    def _robot_state(self) -> RobotState:
        q = self.robot.q
        n = len(ALL_JOINTS)
        return RobotState(self.t, ALL_JOINTS, tuple(q[j] for j in ALL_JOINTS), (0.0,) * n, (0.0, 0.0, 0.0),
                          Imu((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), {})

    # ------------------------------------------------------------------ operator
    def say(self, text: str):
        return self.stack.say(text, self._frame())

    # ------------------------------------------------------------------ loop
    def tick(self) -> StackOutput:
        self.t += self.dt
        f = self._frame()
        out = self.stack.step(f, self.dt)
        if out.arm is not None:
            self.ctrl.set_upper_body(out.arm.body, out.arm.left, out.arm.right)
        else:
            self.ctrl.release_upper_body()
        n0 = len(self.tx.sent)
        self.ctrl.step(self.t, self.dt, self._robot_state(), out.base)
        arm = None
        for _topic, d in self.tx.decoded()[n0:]:
            if "upper_body_position" in d:
                arm = (d["upper_body_position"], d["left_hand_joints"], d["right_hand_joints"])
        self.robot.step(self.dt, out.base, arm)
        self.out = out
        r = self.robot
        self.log.append((self.t, r.x, r.y, r.yaw, out.base, out.status.state, out.status.phase, out.safety))
        return out

    def run(self, seconds: float, until=None) -> StackOutput:
        """Run for up to `seconds`; stop early when `until(session)` is true."""
        end = self.t + seconds
        out = self.out
        while self.t < end:
            out = self.tick()
            if until is not None and until(self):
                break
        return out


def make_session(world: World, *, arm: bool = True, **kw) -> SimSession:
    """A session with the default stack: ground-truth semantics standing in for YOLO, down-camera guard, analytic grasp."""
    from humanoid_arc_bme.onboard.task.arm_task import ArmPickPlace

    robot = SimRobot(world)
    holder = {}
    sem = GroundTruthSemantics(lambda rgb, depth: holder["s"]._seg(rgb, depth))
    ap = ArmPickPlace(robot.kin, robot.hands) if arm else None
    stack = RobotStack(semantics=sem, arm=ap)
    s = SimSession(world, robot, stack, **kw)
    holder["s"] = s
    return s
