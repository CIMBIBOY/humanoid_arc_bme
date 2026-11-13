"""Rates and delays as parameters, never as constants hidden in code (config/timing.yaml).

`Timing.require(name)` raises while a value is still null (not yet measured on the robot).
`DelayLine` delays a stream by a whole number of steps; `steps_for` converts milliseconds to steps and
rounds to the nearest step (a delay shorter than half a step becomes 0, stated so no one is surprised).

Imports only the standard library (PyYAML lazily, only in `load_timing`).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

FIELDS = ("control_hz", "state_delay_ms", "command_delay_ms", "camera_delay_ms")


@dataclass(frozen=True)
class Timing:
    control_hz: float | None = None
    state_delay_ms: float | None = None
    command_delay_ms: float | None = None
    camera_delay_ms: float | None = None

    def __post_init__(self):
        for f in FIELDS:
            v = getattr(self, f)
            if v is not None and not (isinstance(v, (int, float)) and v >= 0 and (f != "control_hz" or v > 0)):
                raise ValueError(f"{f}: expected a non-negative number (control_hz: positive) or null, got {v!r}")

    def require(self, name: str) -> float:
        if name not in FIELDS:
            raise KeyError(name)
        v = getattr(self, name)
        if v is None:
            raise ValueError(f"{name} has not been measured yet (null in config/timing.yaml)")
        return float(v)

    def steps_for(self, delay_field: str, step_s: float) -> int:
        """Whole simulation/control steps that best match the measured delay."""
        if step_s <= 0:
            raise ValueError("step_s must be positive")
        return int(round(self.require(delay_field) / 1000.0 / step_s))


def load_timing(path) -> Timing:
    import yaml  # lazy

    data = yaml.safe_load(Path(path).read_text())
    unknown = set(data) - set(FIELDS)
    if unknown:
        raise ValueError(f"unknown keys in timing file: {sorted(unknown)}")
    return Timing(**{f: data.get(f) for f in FIELDS})


class DelayLine:
    """Output at step t is the input of step t - k; the first k outputs are `initial`."""

    def __init__(self, k: int, initial=None):
        if not isinstance(k, int) or k < 0:
            raise ValueError("k must be a non-negative integer")
        self.k = k
        self._buf = deque([initial] * k, maxlen=k if k else None)

    def push(self, x):
        if self.k == 0:
            return x
        out = self._buf[0]
        self._buf.append(x)
        return out
