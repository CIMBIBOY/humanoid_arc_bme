"""STAND-IN for the GR00T/SONIC deploy: the same inputs and outputs, a different policy inside.

The real deploy (`gear_sonic_deploy`, `--input-type zmq_manager`) subscribes the ZMQ `command` and `planner` topics, reads
rt/lowstate and writes rt/lowcmd at about 50 Hz. Its binary and ONNX files are not on this machine, so this class keeps the
interface and the documented behaviour and replaces the controller by NVIDIA's AGILE lower-body policy (TorchScript, the one the
repository's Isaac Lab tasks already walk on). What is kept from the deploy, with the source in docs/plan/groot-behind-s1.md:
  * wire format and fields of both topics (decoded with `groot_controller.decode_message`, the inverse of what the stack sends);
  * a planner message is used once; with none for PLANNER_TIMEOUT (1000 ms) the locomotion resets to IDLE;
  * `movement` is a direction and `facing` a heading in the heading-reference frame fixed when control starts, `speed` in m/s
    (-1 = the mode's default), mode 0 = idle, turning in place = a facing change while idle;
  * optional `upper_body_position` (17) and `left/right_hand_joints` (7 + 7) take the arms; cleared after 1 s of silence;
  * `command` stop = 1 stops the control loop for good: no more rt/lowcmd (what the robot then does is the motor model's business).
What is NOT the deploy (all tagged STAND-IN): the policy and its gains, the speed and turn response (AGILE's own, with bursts
below the commands it follows, instead of the planner network), the default speeds per mode, and one privileged input the real controller does not have: AGILE was trained with
the base linear velocity as an observation, read here from the simulator's sim-only state topic.
"""

from __future__ import annotations

import math

import numpy as np

from humanoid_arc_bme.isaac import g1_model as M
from humanoid_arc_bme.isaac.lowlevel import LowCmd, LowState, SimState
from humanoid_arc_bme.isaac.rates import CONTROL_HZ
from humanoid_arc_bme.onboard.g1_layout import MOTOR_JOINTS, UPPER_BODY_MOTOR_OFFSET
from humanoid_arc_bme.onboard.runtime.groot_controller import MODE_IDLE, decode_message

PLANNER_TIMEOUT_S = 1.0          # zmq_manager.hpp PLANNER_TIMEOUT (OPENED)
UPPER_BODY_TIMEOUT_S = 1.0       # "the deploy clears them after 1 s of silence" (groot_controller.py, from zmq_manager.hpp)
MODE_DEFAULT_SPEED = {1: 0.4, 2: 0.8}     # STAND-IN GUESS: m/s when `speed` <= 0 (the planner's own defaults are not documented in numbers)
MAX_SPEED = 1.0                  # STAND-IN: the AGILE policy is not trusted above this
# AGILE ignores small commands (learned, not coded; measured here in Isaac: it stands for |v| < ~0.15 m/s and for |wz| < ~0.9 rad/s
# at zero speed, tracks 0.2 m/s as 0.18 and 1.0 rad/s as 1.02, and tracks 0.3 rad/s as 0.23 while walking). The planner interface
# promises a speed in m/s and a heading, so below those thresholds the stand-in closes the loop itself: it walks or turns in short
# bursts at the smallest command AGILE follows until the distance (or heading) it owes is made up.
V_TRACK = 0.2                    # m/s: smallest linear command AGILE follows
WZ_TURN = 1.0                    # rad/s: smallest in-place yaw command AGILE follows
K_YAW, WZ_MAX = 2.0, 1.0         # heading controller while walking, rad/s per rad and its cap
LAG_ON_M = 0.04                  # a walking burst starts when the robot is this far behind the commanded travel
YAW_ON, YAW_BRAKE_S = 0.06, 0.25  # a turning burst starts above this heading error (rad) and ends when error < yaw rate * YAW_BRAKE_S
HIP_HEIGHT = 0.72                # AGILE's standing command (humanoid_arc_bme.robots.bases.AGILE_LEGS.idle_command)
ACTION_SCALE = 0.25              # AgileBasedLowerBodyActionCfg.policy_output_scale in the Isaac Lab task

# Joint order of the policy's observation: the body joints in the order Isaac's articulation lists them (breadth first), which is how
# the policy was trained. Checked against the simulator's own dof order by scripts/isaac/g1_loco_gate.py (joint_table).
OBS_JOINTS = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint", "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
    "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint")
ACTION_JOINTS = tuple(n for n in OBS_JOINTS if any(k in n for k in ("_hip_", "_knee_", "_ankle_")))    # 12 leg joints, same order
# STAND-IN upper body while the planner message carries no arm targets ("the planner's own upper-body motion"): arms hanging. At the
# joints' zero the G1's forearms point straight ahead at 0.8 m height, in full view of the down-tilted head camera, and the stack's
# obstacle guard (rightly) stops for them. How SONIC carries the arms on the robot decides what the camera sees there: UNMEASURED.
REST_ARM = {"left_shoulder_roll_joint": 0.15, "right_shoulder_roll_joint": -0.15, "left_elbow_joint": -0.95, "right_elbow_joint": -0.95}
_OBS_IDX = np.array([MOTOR_JOINTS.index(n) for n in OBS_JOINTS])
_ACT_IDX = np.array([MOTOR_JOINTS.index(n) for n in ACTION_JOINTS])
_REST_IDX, _REST_VAL = np.array([MOTOR_JOINTS.index(n) for n in REST_ARM]), np.array(list(REST_ARM.values()))


def yaw_of(q) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def gravity_b(q) -> np.ndarray:
    """Unit gravity direction in the body frame for quaternion (w, x, y, z)."""
    w, x, y, z = q
    return np.array([-2.0 * (x * z - w * y), -2.0 * (y * z + w * x), -(1.0 - 2.0 * (x * x + y * y))])


def wrap(a: float) -> float:
    return math.remainder(a, 2.0 * math.pi)


class DeployStandIn:
    def __init__(self, policy_path: str, running: bool = True, control_hz: float = CONTROL_HZ):
        import torch

        self._torch = torch
        torch.set_num_threads(1)           # a small MLP: thread hand-off cost more than the arithmetic (8 ms per call on 4 shared cores, measured)
        self.policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
        self.dt = 1.0 / control_hz
        self.running = bool(running)       # the operator started control (keyboard `]` on the robot); `command` start = 1 does it too
        self.exited = False                # `command` stop = 1: the control loop is gone for good
        self.kp, self.kd = (np.array(v) for v in M.gains())
        self.q_default = np.array(M.default_pose())
        self.last_action = np.zeros(len(ACTION_JOINTS), dtype=np.float32)
        self.cmd = np.zeros(3)             # (vx, vy, wz) given to the policy, body frame
        self._owed, self._burst, self._turning = 0.0, False, False
        self.yaw0: float | None = None     # heading reference, fixed on the first state after control starts
        self._plan = None                  # the newest unused planner message
        self._plan_t = -math.inf
        self._mode, self._move, self._facing, self._speed, self._height = MODE_IDLE, np.zeros(2), 0.0, -1.0, -1.0
        self._upper: dict | None = None
        self._upper_t = -math.inf
        self.stats = {"planner_msgs": 0, "timeouts": 0, "bad_msgs": 0, "steps": 0}

    def restart(self) -> None:
        """The operator starts the control loop again (a fresh process on the robot): heading reference, mode and filters start over."""
        self.running, self.exited, self.yaw0 = True, False, None
        self.last_action[:] = 0.0
        self.cmd[:] = 0.0
        self._owed, self._burst, self._turning = 0.0, False, False
        self._plan, self._plan_t, self._upper = None, -math.inf, None
        self._mode, self._move, self._facing, self._speed, self._height = MODE_IDLE, np.zeros(2), 0.0, -1.0, -1.0

    # ------------------------------------------------------------------ input: the ZMQ topics
    def on_message(self, raw: bytes, now: float) -> None:
        try:
            topic, d = decode_message(raw)
        except Exception:  # noqa: BLE001 - the deploy logs and drops a message it cannot read
            self.stats["bad_msgs"] += 1
            return
        if topic == "command":
            if d.get("stop"):
                self.exited, self.running = True, False
            elif d.get("start"):
                self.running = not self.exited
            return
        self._plan, self._plan_t = d, now
        self.stats["planner_msgs"] += 1

    def _consume(self, now: float) -> None:
        d, self._plan = self._plan, None                       # a planner message is used once
        if d is not None:
            mv = np.asarray(d["movement"][:2], float)
            n = float(np.linalg.norm(mv))
            self._mode, self._move = int(d["mode"]), (mv / n if n > 1e-6 else np.zeros(2))
            f = d["facing"]
            if math.hypot(f[0], f[1]) > 1e-6:
                self._facing = math.atan2(f[1], f[0])
            self._speed, self._height = float(d.get("speed", -1.0)), float(d.get("height", -1.0))
            if "upper_body_position" in d:
                self._upper = {"body": np.asarray(d["upper_body_position"], float),
                               "left": np.asarray(d["left_hand_joints"], float) if "left_hand_joints" in d else None,
                               "right": np.asarray(d["right_hand_joints"], float) if "right_hand_joints" in d else None}
                self._upper_t = now
        elif now - self._plan_t > PLANNER_TIMEOUT_S and self._mode != MODE_IDLE:
            self._mode, self._move = MODE_IDLE, np.zeros(2)    # PLANNER_TIMEOUT: back to IDLE by itself
            self.stats["timeouts"] += 1
        if self._upper is not None and now - self._upper_t > UPPER_BODY_TIMEOUT_S:
            self._upper = None

    # ------------------------------------------------------------------ one control tick
    def step(self, state: LowState, sim: SimState | None, now: float) -> LowCmd | None:
        """rt/lowstate in, rt/lowcmd out. None = nothing is published (control not started, or stopped)."""
        if not self.running or state is None:
            return None
        self._consume(now)
        yaw = yaw_of(state.quat)
        if self.yaw0 is None:
            self.yaw0 = yaw
        rel = wrap(yaw - self.yaw0)
        speed = 0.0 if self._mode == MODE_IDLE else min(self._speed if self._speed > 0 else MODE_DEFAULT_SPEED.get(self._mode, 0.4), MAX_SPEED)
        c, s = math.cos(-rel), math.sin(-rel)                  # heading-reference frame -> body frame
        u = np.array([c * self._move[0] - s * self._move[1], s * self._move[0] + c * self._move[1]])
        lin = sim.lin_vel_b if sim is not None else np.zeros(3)    # privileged (STAND-IN only), zero when the sim-only topic is absent
        err = wrap(self._facing - rel)
        if speed >= V_TRACK:                                   # AGILE follows this directly
            v, self._owed, self._burst = speed * u, 0.0, False
        elif speed < 1e-3:
            v, self._owed, self._burst = np.zeros(2), 0.0, False
        else:                                                  # slower than AGILE walks: bursts at V_TRACK until the owed travel is made up
            self._owed += (speed - float(lin[:2] @ u)) * self.dt
            self._burst = self._owed > (0.0 if self._burst else LAG_ON_M)
            v = V_TRACK * u if self._burst else np.zeros(2)
        if float(np.linalg.norm(v)) >= V_TRACK:
            wz, self._turning = float(np.clip(K_YAW * err, -WZ_MAX, WZ_MAX)), False
        else:                                                  # standing: turn in place in bursts, stopping early by the braking distance
            self._turning = abs(err) > (abs(float(state.gyro[2])) * YAW_BRAKE_S if self._turning else YAW_ON)
            wz = math.copysign(WZ_TURN, err) if self._turning else 0.0
        self.cmd = np.array([v[0], v[1], wz])
        height = self._height if 0.3 < self._height < 0.9 else HIP_HEIGHT
        q_rel, dq = state.q - self.q_default, state.dq
        obs = np.concatenate([[self.cmd[0], self.cmd[1], self.cmd[2], height], lin, state.gyro, gravity_b(state.quat),
                              q_rel[_OBS_IDX], 0.1 * dq[_OBS_IDX], self.last_action]).astype(np.float32)
        with self._torch.no_grad():
            act = self.policy(self._torch.from_numpy(obs)[None])[0].numpy()
        self.last_action = act.astype(np.float32)
        q = self.q_default.copy()
        q[_ACT_IDX] += ACTION_SCALE * act
        q[_REST_IDX] = _REST_VAL
        hands = None
        if self._upper is not None:
            q[UPPER_BODY_MOTOR_OFFSET:] = self._upper["body"]
            hands = (self._upper["left"], self._upper["right"])
        self.stats["steps"] += 1
        return LowCmd(q, np.zeros(29), np.zeros(29), self.kp.copy(), self.kd.copy(), hands, now)

    def status(self) -> dict:
        return {"running": self.running, "exited": self.exited, "mode": self._mode, "cmd": [round(float(v), 3) for v in self.cmd], **self.stats}
