"""Strict numeric validation shared by the onboard modules (pipeline, nav, runtime). Standard library + numpy only.

Every check rejects bool, NaN and inf: a safety value must be a real, finite number, never a truthy flag or a string.
"""

import math

import numpy as np


def is_num(v) -> bool:
    """True for a finite real number (int/float/numpy scalar); False for bool, NaN, inf, str, None."""
    return isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) and math.isfinite(v)


def finite_vec(name: str, values, n: int) -> tuple[float, ...]:
    """`values` as a tuple of n finite floats, or ValueError naming `name`."""
    try:
        out = tuple(float(v) for v in values)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: expected a sequence of numbers") from None
    if len(out) != n:
        raise ValueError(f"{name}: expected {n} values, got {len(out)}")
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name}: values must be finite")
    return out


def finite_number(x, label: str) -> float:
    """Strict: int/float (not bool, not a string such as "nan"), finite; else ValueError."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{label} must be a number, got {type(x).__name__}")
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"{label} must be finite")
    return x
