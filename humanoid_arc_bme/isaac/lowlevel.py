"""The robot's low-level interface as plain data: what crosses rt/lowcmd, rt/lowstate and the Dex3 topics, in motor order.

The simulator's robot (`robot.SimG1`) consumes `LowCmd` and produces `LowState`; the controller side (the GR00T/SONIC deploy, or
its stand-in) does the opposite. Between the two sits a link: `dds.DdsLink` (CycloneDDS on loopback, Unitree's own message types,
the path the real deploy uses) or `LoopbackLink` below (the same data handed over in-process, for runs that must be deterministic
and faster than real time). Nothing else passes between simulator and controller.

`SimState` is NOT part of the robot's interface: it carries simulator ground truth on a separate sim-only topic (`rt/sim_state`,
after Unitree's own simulator). Only the STAND-IN policy reads it (it was trained with the base velocity as an input); the real
deploy would ignore it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_MOTORS, N_HAND = 29, 7


def _z(n):
    return field(default_factory=lambda: np.zeros(n))


@dataclass
class LowCmd:
    """Per motor: tau = tau + kp (q - q_measured) + kd (dq - dq_measured). `hands` = (left 7, right 7) Dex3 position targets or None."""
    q: np.ndarray = _z(N_MOTORS)
    dq: np.ndarray = _z(N_MOTORS)
    tau: np.ndarray = _z(N_MOTORS)
    kp: np.ndarray = _z(N_MOTORS)
    kd: np.ndarray = _z(N_MOTORS)
    hands: tuple | None = None
    stamp: float = 0.0                     # controller clock, seconds (for latency accounting only)


@dataclass
class LowState:
    """Measured motor state and the pelvis IMU: quaternion (w, x, y, z) world from body, gyro and accelerometer in the body frame."""
    q: np.ndarray = _z(N_MOTORS)
    dq: np.ndarray = _z(N_MOTORS)
    tau: np.ndarray = _z(N_MOTORS)
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    gyro: np.ndarray = _z(3)
    accel: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 9.81]))
    hands_q: tuple = (np.zeros(N_HAND), np.zeros(N_HAND))
    tick: int = 0
    stamp: float = 0.0                     # simulation time, seconds


@dataclass
class SimState:
    """Simulator ground truth (sim-only topic). `lin_vel_b`: base linear velocity in the body frame, m/s."""
    lin_vel_b: np.ndarray = _z(3)
    pos_w: np.ndarray = _z(3)
    stamp: float = 0.0


class LoopbackLink:
    """In-process link: the newest message of each kind, handed over by reference. Same methods as `dds.DdsLink`."""

    def __init__(self):
        self._cmd: LowCmd | None = None
        self._state: LowState | None = None
        self._sim: SimState | None = None

    def write_cmd(self, cmd: LowCmd) -> None:
        self._cmd = cmd

    def read_cmd(self) -> LowCmd | None:
        return self._cmd

    def write_state(self, state: LowState, sim: SimState | None = None) -> None:
        self._state, self._sim = state, sim

    def read_state(self) -> tuple[LowState | None, SimState | None]:
        return self._state, self._sim

    def close(self) -> None:
        pass
