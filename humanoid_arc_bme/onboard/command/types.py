"""Typed commands: what the grammar (grammar.py) produces from an operator sentence.

Each command is a frozen dataclass that validates itself on construction, so a command that exists is well formed.
Units and frame: distances in metres, angles in radians, body frame with +x forward and +y left (ROS convention).
`Reject` carries the reason shown to the operator and is the outcome for anything the grammar does not accept.

Stdlib only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Union

INTENTS = ("move", "turn", "pick", "place", "stop", "status")
PLACE_SIDES = ("front", "left", "right")


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected a number, got {value!r}")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"{name}: number too large") from None
    if not math.isfinite(number):
        raise ValueError(f"{name}: expected a finite number, got {value!r}")
    return number


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}: expected a non-empty string, got {value!r}")
    return value


@dataclass(frozen=True)
class Move:
    """Walk in the body frame: `forward_m` ahead (+) or back (-), `left_m` to the left (+) or right (-), in metres.

    The grammar gives one axis per command; the type only requires that the move is not zero.
    """

    forward_m: float
    left_m: float

    def __post_init__(self) -> None:
        forward = _finite("forward_m", self.forward_m)
        left = _finite("left_m", self.left_m)
        if forward == 0.0 and left == 0.0:
            raise ValueError("Move: forward_m and left_m are both zero")
        object.__setattr__(self, "forward_m", forward)
        object.__setattr__(self, "left_m", left)


@dataclass(frozen=True)
class Turn:
    """Turn in place by `angle_rad` radians: + is left (counter-clockwise seen from above), - is right."""

    angle_rad: float

    def __post_init__(self) -> None:
        angle = _finite("angle_rad", self.angle_rad)
        if angle == 0.0:
            raise ValueError("Turn: angle_rad is zero")
        object.__setattr__(self, "angle_rad", angle)


@dataclass(frozen=True)
class Pick:
    """Pick up the object called `label` (canonical name, e.g. "cup"), optionally of colour `color` (e.g. "red")."""

    label: str
    color: str | None = None

    def __post_init__(self) -> None:
        _text("label", self.label)
        if self.color is not None:
            _text("color", self.color)


@dataclass(frozen=True)
class Place:
    """Put the held object down at `side`: one of PLACE_SIDES, relative to the robot."""

    side: str = "front"

    def __post_init__(self) -> None:
        if self.side not in PLACE_SIDES:
            raise ValueError(f"Place: side must be one of {PLACE_SIDES}, got {self.side!r}")


@dataclass(frozen=True)
class Stop:
    """Stop all motion now. Any sentence containing a STOP word parses to this."""


@dataclass(frozen=True)
class Status:
    """Report what the robot sees and where it is."""


@dataclass(frozen=True)
class Reject:
    """The sentence was not accepted; `reason` is short, human-readable and safe to show the operator."""

    reason: str

    def __post_init__(self) -> None:
        _text("reason", self.reason)


Command = Union[Move, Turn, Pick, Place, Stop, Status, Reject]

_INTENT_OF = MappingProxyType({Move: "move", Turn: "turn", Pick: "pick", Place: "place",
                               Stop: "stop", Status: "status", Reject: "reject"})


def intent_of(cmd: Command) -> str:
    """Name of the command's kind: one of INTENTS, or "reject" for a Reject."""
    if type(cmd) not in _INTENT_OF:
        raise TypeError(f"not a command: {type(cmd).__name__}")
    return _INTENT_OF[type(cmd)]


@dataclass(frozen=True)
class CommandLimits:
    """Largest distance and angle the grammar accepts. A larger request is rejected, never clamped."""

    max_move_m: float = 3.0
    max_turn_rad: float = math.pi

    def __post_init__(self) -> None:
        for name in ("max_move_m", "max_turn_rad"):
            value = _finite(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name}: must be positive, got {value}")
            object.__setattr__(self, name, value)
