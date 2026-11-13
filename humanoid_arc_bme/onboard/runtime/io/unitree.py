"""Unitree G1 adapter (unitree_sdk2py over CycloneDDS). Read-only unless built with enable_lowcmd=True AND every gate below.

Topics and message layout are from unitree_sdk2_python's G1 low-level example (OPENED 2026-10-23, re-read in full
2026-10-30 from github.com/unitreerobotics/unitree_sdk2_python example/g1/low_level/g1_low_level_example.py):
`rt/lowstate` (`unitree_hg` LowState_: motor_state[i].q / .dq, imu_state.quaternion (w,x,y,z) / .gyroscope /
.accelerometer, mode_machine) and `rt/lowcmd` (LowCmd_: mode_pr (0 = PR series pitch/roll ankle control), mode_machine
(copied from lowstate), motor_cmd[i].mode (1 enable) / q / dq / kp / kd / tau, crc = CRC().Crc(cmd)). The example first
releases the vendor's motion service with MotionSwitcherClient (CheckMode / ReleaseMode loop) because that service
owns rt/lowcmd while it runs.

Command gates (all needed; nothing here sends at construction):
1. `enable_lowcmd=True` (the CLI's --allow-lowcmd; otherwise no publisher even exists);
2. `release_vendor_mode()` called explicitly (an operator action with a typed confirmation in the UI; Hold step);
3. `begin_commanding(state)` (called by the runtime only when the supervisor armed the policy controller);
`damp()` (kp = 0, kd = damping_kd, q = measured) is the safe state after STOP/disarm while the vendor mode is released;
damping_kd defaults to 8, the vendor's damping command (unitree_rl_gym deploy_real/common/command_helper.py, OPENED).
`state_from_lowstate` refuses mode_pr != 0 (SR-16: AB ankle mode would mislabel the ankle joints).
Never run on a robot: the low-level path is UNVERIFIED until the Hold steps of docs/plan/robot-deploy.md.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from humanoid_arc_bme.onboard.api.contract import Imu, RobotState

from ..types import StateMeta
from .base import CommandRefused, RobotIO
from .fake import G1_JOINTS

N_MOTORS = 29  # G1 29-dof body; motor index i is G1_JOINTS[i] (the example's G1JointIndex order)


def state_from_lowstate(msg, stamp: float) -> tuple[RobotState, StateMeta]:
    """Pure conversion of a LowState_-like object (anything with the attributes above) to the contract. Unit-testable."""
    n = len(G1_JOINTS)
    mode_pr = getattr(msg, "mode_pr", None)
    if mode_pr is not None and int(mode_pr) != 0:  # SR-16: in AB (parallel) mode the ankle q are motor A/B coordinates, not pitch/roll
        raise ValueError(f"lowstate mode_pr={mode_pr}: only PR (0) matches the joint names and S1 limits")
    motors = msg.motor_state
    if len(motors) < n:
        raise ValueError(f"lowstate has {len(motors)} motors, need at least {n}")
    imu = Imu(quat=tuple(msg.imu_state.quaternion), gyro=tuple(msg.imu_state.gyroscope), accel=tuple(msg.imu_state.accelerometer))
    st = RobotState(
        stamp=stamp,
        joint_names=G1_JOINTS,
        joint_pos=tuple(float(motors[i].q) for i in range(n)),
        joint_vel=tuple(float(motors[i].dq) for i in range(n)),
        base_vel=(0.0, 0.0, 0.0),  # placeholder: the G1 has no base velocity sensor (StateMeta.base_vel_source == "none")
        imu=imu,
        wrist_poses={s: (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0) for s in ("left", "right")},  # placeholder, no FK yet
    )
    extra = {}
    per_motor = []
    for i in range(n):  # motor temperature: attribute layout UNVERIFIED on our firmware; read defensively
        t = getattr(motors[i], "temperature", None)
        try:
            vals = [float(x) for x in t] if hasattr(t, "__len__") else ([float(t)] if t is not None else [])
        except (TypeError, ValueError):
            vals = []
        per_motor.append(max(vals) if vals else None)
    temps = [t for t in per_motor if t is not None]
    if temps:
        extra["motor_temperature"] = tuple(per_motor)  # SR-16b: per motor (max of the IDL's temperature[2])
        extra["motor_temp_max_c"] = max(temps)
    if all(hasattr(motors[i], "motorstate") for i in range(n)):  # SR-14/16b: error bits; their meaning is UNKNOWN
        errs = tuple(int(motors[i].motorstate) for i in range(n))
        extra["motor_error"] = errs
        extra["motor_error_any"] = any(errs)
    if mode_pr is not None:
        extra["mode_pr"] = int(mode_pr)
    tick = getattr(msg, "tick", None)
    if tick is not None:
        extra["tick"] = int(tick)
    return st, StateMeta(mode_machine=int(getattr(msg, "mode_machine", -1)), extra=extra)


def fill_lowcmd(cmd, q, kp, kd, mode_machine: int, mode_pr: int = 0):
    """Write one LowCmd_ (or any object with the same attributes): every motor enabled, position q with gains kp/kd,
    dq = 0, tau = 0. Pure; the caller computes the CRC. Raises on a length mismatch or a non-finite value."""
    import math

    if not (len(q) == len(kp) == len(kd) == N_MOTORS):
        raise ValueError(f"lowcmd: need {N_MOTORS} q/kp/kd values")
    for v in (*q, *kp, *kd):
        if not math.isfinite(v):
            raise ValueError("lowcmd: non-finite value")
    if min(kp) < 0 or min(kd) < 0:
        raise ValueError("lowcmd: negative gain")
    cmd.mode_pr = int(mode_pr)
    cmd.mode_machine = int(mode_machine)
    for i in range(N_MOTORS):
        m = cmd.motor_cmd[i]
        m.mode, m.q, m.dq, m.kp, m.kd, m.tau = 1, float(q[i]), 0.0, float(kp[i]), float(kd[i]), 0.0
    return cmd


def load_sdk() -> SimpleNamespace:
    """The unitree_sdk2py names this adapter uses (lazy import: only on the robot / a machine with the SDK)."""
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    ns = SimpleNamespace(ChannelFactoryInitialize=ChannelFactoryInitialize, ChannelSubscriber=ChannelSubscriber,
                         ChannelPublisher=ChannelPublisher, LowState_=LowState_)
    return ns


def load_lowcmd_sdk(ns: SimpleNamespace) -> SimpleNamespace:
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.utils.crc import CRC

    ns.LowCmd_, ns.default_lowcmd, ns.CRC, ns.MotionSwitcherClient = LowCmd_, unitree_hg_msg_dds__LowCmd_, CRC, MotionSwitcherClient
    return ns


class UnitreeG1IO(RobotIO):
    lowcmd_enabled = False  # class defaults: an instance that skipped __init__ is still read-only
    vendor_released = False
    _commanding = False

    def __init__(self, interface: str | None = None, domain: int = 0, clock=time.monotonic, enable_lowcmd: bool = False,
                 hold_kp=None, hold_kd=None, damping_kd: float = 8.0, sdk: SimpleNamespace | None = None, init_channel: bool = True):
        self.sdk = sdk if sdk is not None else load_sdk()
        self.clock, self._latest, self.msg_count = clock, None, 0
        if init_channel:  # once per process; the vendor controller's LocoClient shares it
            if interface:
                self.sdk.ChannelFactoryInitialize(domain, interface)
            else:
                self.sdk.ChannelFactoryInitialize(domain)
        self._sub = self.sdk.ChannelSubscriber("rt/lowstate", self.sdk.LowState_)
        self._sub.Init(self._on_state, 10)
        self.lowcmd_enabled = bool(enable_lowcmd)
        self.vendor_released = False
        self._commanding = False
        self._hold_q = None
        self.hold_kp = tuple(float(x) for x in (hold_kp or [0.0] * N_MOTORS))
        self.hold_kd = tuple(float(x) for x in (hold_kd or [0.0] * N_MOTORS))
        self.damping_kd = float(damping_kd)
        self.published = 0
        if self.lowcmd_enabled:
            if sdk is None:
                load_lowcmd_sdk(self.sdk)
            if len(self.hold_kp) != N_MOTORS or len(self.hold_kd) != N_MOTORS:
                raise ValueError(f"hold_kp/hold_kd: need {N_MOTORS} values")
            self._pub = self.sdk.ChannelPublisher("rt/lowcmd", self.sdk.LowCmd_)
            self._pub.Init()  # a publisher sends nothing until write()
            self._crc = self.sdk.CRC()
            self._msc = self.sdk.MotionSwitcherClient()
            self._msc.SetTimeout(5.0)
            self._msc.Init()

    def _on_state(self, msg):
        self._latest = (msg, self.clock())
        self.msg_count += 1

    def read_state(self):
        if self._latest is None:
            return None
        msg, t = self._latest
        return state_from_lowstate(msg, t)

    def arm(self, acknowledgement: str):
        raise NotImplementedError("arming is the operator runtime's job (humanoid_arc_bme.onboard.runtime.operator.Supervisor + begin_commanding)")

    # ---------------------------------------------------------------- low-level command path (gated)
    def vendor_mode(self) -> str | None:
        """Name of the vendor motion mode that owns rt/lowcmd ('' = released), None without the lowcmd path."""
        if not self.lowcmd_enabled:
            return None
        _status, result = self._msc.CheckMode()
        return (result or {}).get("name", "")

    def release_vendor_mode(self, max_tries: int = 5, sleep=time.sleep) -> bool:
        """HOLD STEP: after this the vendor controller no longer balances the robot (the robot must hang in the gantry).
        Only the UI's typed confirmation calls it. Returns True when CheckMode reports no active mode."""
        if not self.lowcmd_enabled:
            raise CommandRefused("lowcmd path not enabled (--allow-lowcmd)")
        for _ in range(max_tries):
            if not self.vendor_mode():
                self.vendor_released = True
                return True
            self._msc.ReleaseMode()
            sleep(1.0)
        self.vendor_released = not self.vendor_mode()
        return self.vendor_released

    def _publish(self, q, kp, kd) -> None:
        msg, _t = self._latest if self._latest is not None else (None, None)
        if msg is None:
            raise CommandRefused("no lowstate received: mode_machine unknown")
        cmd = fill_lowcmd(self.sdk.default_lowcmd(), q, kp, kd, int(msg.mode_machine))
        cmd.crc = self._crc.Crc(cmd)
        self._pub.Write(cmd)
        self.published += 1

    def begin_commanding(self, state) -> None:
        if not (self.lowcmd_enabled and self.vendor_released):
            raise CommandRefused("joint commands need --allow-lowcmd and a released vendor mode")
        pos = dict(zip(state.joint_names, state.joint_pos))
        self._hold_q = [pos[n] for n in G1_JOINTS]
        self._commanding = True

    def end_commanding(self) -> None:
        self._commanding = False

    def send_joint_targets(self, names, targets, kp=None, kd=None):
        if not (self.lowcmd_enabled and self.vendor_released and self._commanding):
            raise CommandRefused("UnitreeG1IO: not commanding (gates: --allow-lowcmd, vendor released, armed)")
        q, kps, kds = list(self._hold_q), list(self.hold_kp), list(self.hold_kd)
        idx = {n: i for i, n in enumerate(G1_JOINTS)}
        for k, (n, t) in enumerate(zip(names, targets)):
            i = idx[n]
            q[i] = float(t)
            if kp is not None:
                kps[i] = float(kp[k])
            if kd is not None:
                kds[i] = float(kd[k])
        self._publish(q, kps, kds)

    def damp(self) -> bool:
        """Damping state (kp = 0, kd = damping_kd, q = measured). Only while the vendor mode is released (otherwise
        the vendor controller owns the motors and this does nothing). Returns True when a command was published."""
        self._commanding = False
        if not (self.lowcmd_enabled and self.vendor_released) or self._latest is None:
            return False
        msg, _t = self._latest
        q = [float(msg.motor_state[i].q) for i in range(N_MOTORS)]
        self._publish(q, [0.0] * N_MOTORS, [self.damping_kd] * N_MOTORS)
        return True
