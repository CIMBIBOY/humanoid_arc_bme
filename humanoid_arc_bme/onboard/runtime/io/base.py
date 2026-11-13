"""RobotIO: the only way the runtime reads state from, or sends commands to, a robot."""

from __future__ import annotations

from abc import ABC, abstractmethod

from humanoid_arc_bme.onboard.api.contract import RobotState

from ..types import StateMeta


class CommandRefused(RuntimeError):
    """The adapter will not send commands (read-only adapter, or not armed)."""


class RobotIO(ABC):
    simulated: bool = False  # True only for simulators and fakes: Runtime.arm() refuses anything else
    @abstractmethod
    def read_state(self) -> tuple[RobotState, StateMeta] | None:
        """Latest state, or None when nothing has arrived yet. Never blocks."""

    @abstractmethod
    def send_joint_targets(self, names: tuple, targets: tuple, kp: tuple | None = None, kd: tuple | None = None) -> None:
        """Joint position targets (rad). Adapters that cannot move the robot raise CommandRefused."""

    def close(self) -> None:  # noqa: B027 - optional
        pass
