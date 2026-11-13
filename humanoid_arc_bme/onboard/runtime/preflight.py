"""Standalone preflight: versions, network, disk, clock, config, models. Read-only: it never commands the robot.

    python -m humanoid_arc_bme.onboard.runtime preflight [--iface eth0] [--listen 3] [--models DIR] [--config-dir config]

`--listen S` subscribes to rt/lowstate for S seconds (read-only DDS subscriber) and reports the message rate.
Each check is ok / fail / unknown; `required` failures (and unknowns) make the exit code 1. Reuses the Jetson checks of
humanoid_arc_bme.onboard.deploy.preflight (JetPack R36, TensorRT 10.7, MAXN, ROS 2 Humble, 192.168.123.x).
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROBOT_NET_PREFIX = "192.168.123."


@dataclass
class Item:
    name: str
    status: str
    detail: str
    required: bool = True


def _run(argv, timeout=5):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except FileNotFoundError:
        return 127, "", f"{argv[0]} not found"
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


def check_module(name, required=True, version_attr="__version__", want=None) -> Item:
    try:
        m = importlib.import_module(name)
    except Exception as e:  # noqa: BLE001
        return Item(f"py:{name}", "fail", f"not importable ({type(e).__name__}: {e})", required)
    v = getattr(m, version_attr, None)
    if v is None:
        try:
            from importlib.metadata import version
            v = version(name)
        except Exception:  # noqa: BLE001
            v = "?"
    if want and not str(v).startswith(want):
        return Item(f"py:{name}", "fail", f"{v} (want {want}.x)", required)
    return Item(f"py:{name}", "ok", str(v), required)


def check_python() -> Item:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 10)
    return Item("python", "ok" if ok else "fail", f"{sys.version.split()[0]} at {sys.executable}")


def check_onnxruntime() -> Item:
    try:
        import onnxruntime as ort
    except Exception as e:  # noqa: BLE001
        return Item("onnxruntime", "fail", f"not importable ({e}); needed for the policy controller only", required=False)
    return Item("onnxruntime", "ok", f"{ort.__version__}, providers {ort.get_available_providers()}", required=False)


def check_cyclonedds() -> Item:
    try:
        from importlib.metadata import version
        v = version("cyclonedds")
    except Exception:  # noqa: BLE001
        return Item("cyclonedds", "fail", "python package cyclonedds not installed (unitree_sdk2py needs ==0.10.2)")
    home = os.environ.get("CYCLONEDDS_HOME", "unset")
    return Item("cyclonedds", "ok" if v == "0.10.2" else "fail", f"{v} (need 0.10.2), CYCLONEDDS_HOME={home}")


def check_iface(iface: str | None) -> Item:
    code, out, err = _run(["ip", "-4", "-o", "addr"])
    if code != 0:
        return Item("robot_network", "unknown", err.strip() or "ip failed")
    rows = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[2] == "inet":
            rows.append((parts[1], parts[3].split("/")[0]))
    hits = [(i, a) for i, a in rows if a.startswith(ROBOT_NET_PREFIX) and (iface is None or i == iface)]
    if hits:
        return Item("robot_network", "ok", ", ".join(f"{i}={a}" for i, a in hits))
    want = f"interface {iface}" if iface else "any interface"
    return Item("robot_network", "fail", f"{want} has no {ROBOT_NET_PREFIX}x address; have {rows}")


def check_disk(path, min_gb=1.0) -> Item:
    p = Path(os.path.expanduser(str(path)))
    p.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(p).free / 1e9
    return Item("disk", "ok" if free >= min_gb else "fail", f"{free:.1f} GB free in {p} (need {min_gb})")


def check_clock() -> list[Item]:
    yr = time.gmtime().tm_year
    out = [Item("wall_clock", "ok" if yr >= 2026 else "fail", time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), required=False)]
    code, o, _e = _run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"])
    if code == 0 and o.strip() in ("yes", "no"):
        out.append(Item("ntp_synced", "ok" if o.strip() == "yes" else "fail", o.strip(), required=False))
    else:
        out.append(Item("ntp_synced", "unknown", "timedatectl unavailable", required=False))
    t0 = time.monotonic()
    time.sleep(0.01)
    out.append(Item("monotonic_clock", "ok" if time.monotonic() > t0 else "fail", "advances"))
    return out


def check_config(config_dir) -> list[Item]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    out = []
    try:
        from humanoid_arc_bme.onboard.api.safety import load_limits

        from .operator import load_operator_limits

        lim = load_limits(Path(config_dir) / "g1_limits.yaml")
        op = load_operator_limits(Path(config_dir) / "g1_limits.yaml", lim.base_vel_max)
        out.append(Item("limits", "ok", f"operator envelope (S1-narrowed) vx {op.vx_min}..{op.vx_max}, vy {op.vy_max}, wz {op.wz_max}; "
                        f"deadman {op.deadman_timeout_s} s; validated_on_robot={lim.validated_on_robot}"))
    except Exception as e:  # noqa: BLE001
        out.append(Item("limits", "fail", f"{type(e).__name__}: {e}"))
    try:
        import yaml
        yaml.safe_load((Path(config_dir) / "robot_runtime.yaml").read_text())
        out.append(Item("runtime_config", "ok", str(Path(config_dir) / "robot_runtime.yaml")))
    except Exception as e:  # noqa: BLE001
        out.append(Item("runtime_config", "fail", f"{type(e).__name__}: {e}"))
    return out


def check_models(models) -> list[Item]:
    from .controllers import list_policies

    dirs = list_policies(models) if models else []
    if not dirs:
        return [Item("models", "unknown", f"no policy.onnx + layout.json under {models}", required=False)]
    out = []
    for d in dirs:
        try:
            import numpy as np

            from .obs_builders import LowerBodyLayout
            from .policies import ort_session

            lay = LowerBodyLayout.from_json(d / "layout.json")
            s = ort_session(d / "policy.onnx")
            t0 = time.perf_counter()
            y = s.run(None, {s.get_inputs()[0].name: np.zeros((1, lay.obs_dim), np.float32)})[0]
            ms = 1000 * (time.perf_counter() - t0)
            ok = y.shape == (1, len(lay.action_joint_names)) and np.isfinite(y).all()
            out.append(Item(f"model:{d.name}", "ok" if ok else "fail", f"obs {lay.obs_dim} -> act {y.shape[1]}, first run {ms:.1f} ms", required=False))
        except Exception as e:  # noqa: BLE001
            out.append(Item(f"model:{d.name}", "fail", f"{type(e).__name__}: {e}", required=False))
    return out


def listen_lowstate(iface, seconds) -> Item:
    """Read-only: subscribe to rt/lowstate and count messages. Never publishes."""
    try:
        from .io.unitree import UnitreeG1IO

        io = UnitreeG1IO(iface)
    except Exception as e:  # noqa: BLE001
        return Item("lowstate", "fail", f"cannot subscribe ({type(e).__name__}: {e})")
    time.sleep(seconds)
    n = io.msg_count
    got = io.read_state()
    if not n or got is None:
        return Item("lowstate", "fail", f"no rt/lowstate in {seconds} s on {iface}")
    st, meta = got
    return Item("lowstate", "ok", f"{n / seconds:.0f} msg/s, mode_machine {meta.mode_machine}, extra {meta.extra}")


def run_all(iface=None, listen=0.0, models=None, config_dir="config", log_root="~/g1_runs") -> list[Item]:
    items = [check_python()]
    for name, req, want in (("numpy", True, None), ("yaml", True, None)):
        items.append(check_module(name, req, want=want))
    items.append(check_onnxruntime())
    items.append(check_module("unitree_sdk2py", required=True))
    items.append(check_cyclonedds())
    items.append(check_module("rclpy", required=False))
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from humanoid_arc_bme.onboard.deploy import preflight as jet

        for c in (jet.check_jetpack(), jet.check_tensorrt(), jet.check_power_mode(), jet.check_ros()):
            items.append(Item(c.name, c.status, c.detail, required=False))
    except Exception as e:  # noqa: BLE001
        items.append(Item("jetson", "unknown", f"humanoid_arc_bme.onboard.deploy not available ({e})", required=False))
    items.append(check_iface(iface))
    items.append(check_disk(log_root))
    items += check_clock()
    items += check_config(config_dir)
    items += check_models(models)
    if listen > 0:
        items.append(listen_lowstate(iface, listen))
    return items


def report(items) -> tuple[str, int]:
    mark = {"ok": "OK     ", "fail": "FAIL   ", "unknown": "UNKNOWN"}
    lines = [f"{mark[i.status]} {'' if i.required else '(adv) '}{i.name:18s} {i.detail}" for i in items]
    bad = [i for i in items if i.required and i.status != "ok"]
    lines.append(f"{len(items)} checks; required not ok: {len(bad)} {[i.name for i in bad]}")
    return "\n".join(lines), (1 if bad else 0)
