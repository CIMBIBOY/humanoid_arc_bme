"""Shared helpers for the read-only robot inventory scripts. Stdlib only (runs on the Jetson's system python).

Nothing in this file talks to the robot. It holds the acknowledgement gate, timing statistics and JSON helpers.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import sys

ACK_FLAG = "--i-understand-read-only"


def require_ack(argv: list[str], script: str) -> None:
    """Exit with status 2 unless the explicit read-only acknowledgement flag is present."""
    if ACK_FLAG not in argv:
        sys.stderr.write(
            f"REFUSING TO RUN {script}: pass {ACK_FLAG} to confirm you want this read-only probe.\n"
            f"It only subscribes / reads; see the docstring at the top of {script} for exactly what it touches.\n"
        )
        sys.exit(2)


def utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def percentile(xs: list[float], q: float) -> float | None:
    """Linear-interpolated percentile, q in [0, 100]. None for an empty list."""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _r(x, nd=6):
    return None if x is None else round(float(x), nd)


def rate_stats(times: list[float], window_s: float | None = None) -> dict:
    """Arrival statistics from monotonically recorded arrival times (seconds).

    mean_hz uses (count-1)/(last-first); jitter is the std of the inter-arrival period; gaps_over_2x_median counts
    periods longer than twice the median period (dropped or late messages)."""
    n = len(times)
    out: dict = {"count": n}
    if window_s is not None:
        out["window_s"] = _r(window_s, 3)
        out["rate_over_window_hz"] = _r(n / window_s, 3) if window_s > 0 else None
    if n < 2:
        out.update(mean_hz=None, period_ms={}, gaps_over_2x_median=0, non_monotonic=0)
        return out
    dts = [b - a for a, b in zip(times, times[1:])]
    span = times[-1] - times[0]
    mean = sum(dts) / len(dts)
    std = math.sqrt(sum((d - mean) ** 2 for d in dts) / len(dts))
    med = percentile(dts, 50)
    out["mean_hz"] = _r((n - 1) / span, 3) if span > 0 else None
    out["period_ms"] = {
        "mean": _r(mean * 1e3, 4), "std_jitter": _r(std * 1e3, 4), "min": _r(min(dts) * 1e3, 4),
        "p50": _r(med * 1e3, 4), "p90": _r(percentile(dts, 90) * 1e3, 4), "p99": _r(percentile(dts, 99) * 1e3, 4),
        "max": _r(max(dts) * 1e3, 4),
    }
    out["gaps_over_2x_median"] = sum(1 for d in dts if med and d > 2 * med)
    out["non_monotonic"] = sum(1 for d in dts if d <= 0)
    return out


def counter_stats(values: list[float], times: list[float]) -> dict:
    """For an in-message counter (e.g. LowState.tick): its step sizes and its rate against our arrival clock."""
    n = len(values)
    if n < 2:
        return {"count": n}
    steps = [b - a for a, b in zip(values, values[1:])]
    span_t = times[-1] - times[0]
    return {
        "count": n,
        "first": values[0], "last": values[-1],
        "step_p50": percentile(steps, 50), "step_min": min(steps), "step_max": max(steps),
        "non_increasing": sum(1 for s in steps if s <= 0),
        "units_per_s": _r((values[-1] - values[0]) / span_t, 3) if span_t > 0 else None,
    }


def age_stats(ages_s: list[float]) -> dict:
    """Message age = our wall clock at arrival - the stamp inside the message. Meaningful only with synced clocks."""
    if not ages_s:
        return {"count": 0}
    ms = [a * 1e3 for a in ages_s]
    return {"count": len(ms), "p50_ms": _r(percentile(ms, 50), 3), "p99_ms": _r(percentile(ms, 99), 3),
            "min_ms": _r(min(ms), 3), "max_ms": _r(max(ms), 3)}


def to_plain(obj, depth: int = 0, max_list: int = 64):
    """Convert an IDL / ROS message (or anything with attributes) into JSON-able data, depth- and length-limited."""
    if depth > 6:
        return repr(obj)[:80]
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, (bytes, bytearray)):
        return list(obj[:max_list])
    if isinstance(obj, dict):
        return {str(k): to_plain(v, depth + 1, max_list) for k, v in list(obj.items())[:max_list]}
    if isinstance(obj, (list, tuple)) or type(obj).__name__ in ("ndarray", "array"):
        seq = list(obj)
        res = [to_plain(v, depth + 1, max_list) for v in seq[:max_list]]
        if len(seq) > max_list:
            res.append(f"... {len(seq) - max_list} more")
        return res
    fields = getattr(obj, "__dataclass_fields__", None)
    if fields:
        return {k: to_plain(getattr(obj, k, None), depth + 1, max_list) for k in fields}
    slots = getattr(obj, "__slots__", None)
    if slots and hasattr(obj, "get_fields_and_field_types"):  # ROS 2 message
        return {k: to_plain(getattr(obj, k, None), depth + 1, max_list) for k in obj.get_fields_and_field_types()}
    d = getattr(obj, "__dict__", None)
    if d:
        return {k: to_plain(v, depth + 1, max_list) for k, v in d.items() if not k.startswith("_")}
    return repr(obj)[:200]


def write_json(path: str, data) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=1, sort_keys=False, default=str)
    return path


def robot_iface_from_ip_text(ip_br_text: str, prefix: str = "192.168.123.") -> str | None:
    """Pick the interface carrying a 192.168.123.x address from `ip -br addr` output (documented robot subnet)."""
    for line in ip_br_text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and any(p.startswith(prefix) for p in parts[2:]):
            return parts[0].split("@")[0]
    return None
