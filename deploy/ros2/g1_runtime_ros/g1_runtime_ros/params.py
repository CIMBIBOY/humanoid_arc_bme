"""ROS-free helpers of the wrapper (unit-tested without ROS): parameters -> the CLI's argument namespace, Twist -> command."""

from __future__ import annotations

import argparse

# name -> (default, kind). Launch arguments arrive as strings; kind says how to parse them.
PARAMS = {
    "io": ("fake", str), "iface": ("", str), "models": ("", str), "shadow": ("", str), "camera": ("none", str),
    "lan": (False, bool), "port": (8080, int), "token": ("", str), "allow_lowcmd": (False, bool),
    "config_dir": ("", str), "log_root": ("", str), "foxglove": (-1, int),
    "lan_plain_http_ack": (False, bool), "limits": ("", str),
}


def parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"not a boolean: {v!r}")


def to_args(values: dict) -> argparse.Namespace:
    out = {}
    for k, (default, kind) in PARAMS.items():
        v = values.get(k, default)
        v = parse_bool(v) if kind is bool else kind(v)
        out[k] = v
    if out["io"] not in ("fake", "unitree"):
        raise ValueError(f"io must be fake or unitree, got {out['io']!r}")
    for k in ("iface", "models", "shadow", "token", "config_dir", "log_root", "limits"):
        out[k] = out[k] or None
    out["foxglove"] = None if out["foxglove"] < 0 else out["foxglove"]
    return argparse.Namespace(**out)


def twist_to_cmd(lx: float, ly: float, az: float) -> tuple[float, float, float]:
    """geometry_msgs/Twist (base frame) -> (vx, vy, wz). The runtime clips to the envelope; NaN is rejected there."""
    return float(lx), float(ly), float(az)
