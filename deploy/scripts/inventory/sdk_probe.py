#!/usr/bin/env python3
"""READ-ONLY DDS probe of the G1 through unitree_sdk2_python. It only SUBSCRIBES.

What it does: joins DDS domain `--domain` on the robot network interface, creates one DataReader (ChannelSubscriber)
per state topic in TOPICS, records every arrival for `--seconds`, then writes a JSON report: per topic the rate,
period jitter, gaps, the in-message counter (`tick`), message age where the message has a stamp, and a decoded
summary (joint count, IMU fields, battery fields, hand sensors).

What it never does: create a writer/publisher, call an RPC client (the loco, sport and motion-switcher clients all send
requests, so they are not used, not even their "get" calls), or send anything but the DDS discovery announcements every
participant makes. As a second guard, the SDK's publisher class is replaced by one that raises before any reader is
made.

Usage (on the Jetson, or a laptop on 192.168.123.x):
    python3 sdk_probe.py --i-understand-read-only [--iface auto|eth0] [--domain 0] [--seconds 10] [--out DIR]
                         [--watch-command-topics]
    --watch-command-topics  also READ rt/lowcmd, rt/arm_sdk and the Dex3 cmd topics, to see who is commanding
                            (a reader on a command topic does not command anything)

Topic list sources (tag in each entry): OPENED = read in a Unitree repo on 2026-10-30, see docs/robot-inventory.md;
GUESS = the IDL type exists but the topic name was not found in an opened source (a miss is information too).
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_common import (age_stats, counter_stats, rate_stats, require_ack, robot_iface_from_ip_text,  # noqa: E402
                          to_plain, utc_stamp, write_json)

# (topic, idl package, type name, source tag)
TOPICS = [
    ("rt/lowstate", "unitree_hg", "LowState_", "OPENED unitree_sdk2_python example/g1/low_level/g1_low_level_example.py"),
    ("rt/lf/lowstate", "unitree_hg", "LowState_", "OPENED unitree_ros2 read_low_state_hg.cpp ('lf' = low frequency)"),
    ("rt/secondary_imu", "unitree_hg", "IMUState_", "OPENED unitree_mujoco README: torso IMU, G1 only"),
    ("rt/dex3/left/state", "unitree_hg", "HandState_", "OPENED xr_teleoperate robot_hand_unitree.py"),
    ("rt/dex3/right/state", "unitree_hg", "HandState_", "OPENED xr_teleoperate robot_hand_unitree.py"),
    ("rt/lf/dex3/left/state", "unitree_hg", "HandState_", "OPENED unitree_sdk2 example/g1/dex3/g1_dex3_example.cpp"),
    ("rt/lf/dex3/right/state", "unitree_hg", "HandState_", "OPENED unitree_sdk2 example/g1/dex3/g1_dex3_example.cpp"),
    ("rt/lf/bmsstate", "unitree_hg", "BmsState_", "GUESS: unitree_hg BmsState_ IDL exists, topic name not found"),
    ("rt/bmsstate", "unitree_hg", "BmsState_", "GUESS"),
    ("rt/lf/mainboardstate", "unitree_hg", "MainBoardState_", "GUESS: unitree_hg MainBoardState_ IDL exists"),
    ("rt/sportmodestate", "unitree_go", "SportModeState_", "OPENED for Go2 (unitree_ros2 README); G1 loco example imports the type"),
    ("rt/lf/sportmodestate", "unitree_go", "SportModeState_", "OPENED for Go2"),
    ("rt/odommodestate", "unitree_go", "SportModeState_", "GUESS"),
    ("rt/lf/odommodestate", "unitree_go", "SportModeState_", "GUESS"),
    ("rt/wirelesscontroller", "unitree_go", "WirelessController_", "OPENED unitree_ros2 README '/wirelesscontroller'"),
    ("rt/wireless_controller", "unitree_go", "WirelessController_", "OPENED unitree_mujoco README 'rt/wireless_controller'"),
]
COMMAND_TOPICS = [
    ("rt/lowcmd", "unitree_hg", "LowCmd_", "OPENED g1_low_level_example.py (command topic; READ only here)"),
    ("rt/arm_sdk", "unitree_hg", "LowCmd_", "OPENED g1_arm7_sdk_dds_example.py (command topic; READ only here)"),
    ("rt/dex3/left/cmd", "unitree_hg", "HandCmd_", "OPENED xr_teleoperate (command topic; READ only here)"),
    ("rt/dex3/right/cmd", "unitree_hg", "HandCmd_", "OPENED xr_teleoperate (command topic; READ only here)"),
]
MAX_ARRIVALS = 200_000


# ------------------------------------------------------------------ pure summaries (unit-tested with fakes)
def _f(x):
    try:
        return float(x)
    except Exception:
        return None


def _list(x):
    try:
        return [v for v in x]
    except TypeError:
        return [x]


def summarize_imu(imu) -> dict:
    q = [_f(v) for v in _list(getattr(imu, "quaternion", []))]
    acc = [_f(v) for v in _list(getattr(imu, "accelerometer", []))]
    return {
        "quaternion_wxyz": q,
        "quat_norm": round(sum(v * v for v in q if v is not None) ** 0.5, 5) if q else None,
        "rpy": [_f(v) for v in _list(getattr(imu, "rpy", []))],
        "gyroscope": [_f(v) for v in _list(getattr(imu, "gyroscope", []))],
        "accelerometer": acc,
        "accel_norm": round(sum(v * v for v in acc if v is not None) ** 0.5, 4) if acc else None,
        "temperature": to_plain(getattr(imu, "temperature", None)),
    }


def _motor_temp(m):
    t = getattr(m, "temperature", None)
    vals = [_f(v) for v in _list(t)] if t is not None else []
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def summarize_lowstate(msg) -> dict:
    motors = _list(getattr(msg, "motor_state", []))
    live = [i for i, m in enumerate(motors)
            if any(abs(_f(getattr(m, k, 0)) or 0.0) > 0 for k in ("q", "dq", "tau_est", "vol")) or (_motor_temp(m) or 0) > 0]
    temps = [(_motor_temp(m) or 0.0) for m in motors]
    flags = {i: int(getattr(m, "motorstate", 0) or 0) for i, m in enumerate(motors) if int(getattr(m, "motorstate", 0) or 0)}
    remote = _list(getattr(msg, "wireless_remote", []))
    return {
        "motor_slots": len(motors),
        "motors_live": len(live),
        "live_indices": live,
        "q": [round(_f(getattr(motors[i], "q", 0)) or 0.0, 4) for i in live],
        "motor_temp_max": max(temps) if temps else None,
        "motor_temp_argmax": temps.index(max(temps)) if temps else None,
        "motor_vol": [_f(getattr(motors[i], "vol", None)) for i in live],
        "motorstate_nonzero": flags,
        "mode_pr": to_plain(getattr(msg, "mode_pr", None)),
        "mode_machine": to_plain(getattr(msg, "mode_machine", None)),
        "version": to_plain(getattr(msg, "version", None)),
        "tick": to_plain(getattr(msg, "tick", None)),
        "wireless_remote_nonzero_bytes": sum(1 for b in remote if b),
        "imu": summarize_imu(getattr(msg, "imu_state", None)),
    }


def summarize_hand(msg) -> dict:
    motors = _list(getattr(msg, "motor_state", []))
    press = _list(getattr(msg, "press_sensor_state", []))
    return {
        "motor_count": len(motors),
        "q": [round(_f(getattr(m, "q", 0)) or 0.0, 4) for m in motors],
        "motor_temp_max": max([(_motor_temp(m) or 0.0) for m in motors], default=None),
        "press_sensor_count": len(press),
        "press_values_per_sensor": len(_list(getattr(press[0], "pressure", []))) if press else 0,
        "power_v": _f(getattr(msg, "power_v", None)), "power_a": _f(getattr(msg, "power_a", None)),
        "error": to_plain(getattr(msg, "error", None)),
        "imu": summarize_imu(getattr(msg, "imu_state", None)) if hasattr(msg, "imu_state") else None,
    }


def summarize_bms(msg) -> dict:
    cells = [c for c in _list(getattr(msg, "cell_vol", [])) if c]
    return {
        "soc": to_plain(getattr(msg, "soc", None)), "soh": to_plain(getattr(msg, "soh", None)),
        "current": to_plain(getattr(msg, "current", None)), "bmsvoltage": to_plain(getattr(msg, "bmsvoltage", None)),
        "cells_nonzero": len(cells), "cell_min": min(cells) if cells else None, "cell_max": max(cells) if cells else None,
        "temperature": to_plain(getattr(msg, "temperature", None)), "cycle": to_plain(getattr(msg, "cycle", None)),
        "version": [to_plain(getattr(msg, "version_high", None)), to_plain(getattr(msg, "version_low", None))],
    }


def summarize(type_name: str, msg) -> dict:
    if type_name in ("LowState_", "LowCmd_"):
        s = summarize_lowstate(msg)
        if type_name == "LowCmd_":
            s = {"motor_slots": s["motor_slots"], "mode_pr": s["mode_pr"], "mode_machine": s["mode_machine"]}
        return s
    if type_name in ("HandState_", "HandCmd_"):
        return summarize_hand(msg)
    if type_name == "BmsState_":
        return summarize_bms(msg)
    if type_name == "IMUState_":
        return summarize_imu(msg)
    return {"fields": to_plain(msg, max_list=16)}


def stamp_seconds(msg) -> float | None:
    """Wall-clock stamp inside the message if it has one (SportModeState_.stamp is a TimeSpec sec/nanosec)."""
    st = getattr(msg, "stamp", None)
    if st is None:
        return None
    sec, ns = getattr(st, "sec", None), getattr(st, "nanosec", None)
    if sec is None:
        return None
    return float(sec) + float(ns or 0) * 1e-9


class TopicRecorder:
    """Callback target for one topic. Keeps arrival times cheaply; the decoded summary is made from the kept samples."""

    def __init__(self, topic: str, type_name: str, mono=time.monotonic, wall=time.time):
        self.topic, self.type_name, self.mono, self.wall = topic, type_name, mono, wall
        self.lock = threading.Lock()
        self.t_mono: list[float] = []
        self.ticks: list[float] = []
        self.ages: list[float] = []
        self.first = None
        self.last = None

    def __call__(self, msg):
        tm, tw = self.mono(), self.wall()
        with self.lock:
            if len(self.t_mono) < MAX_ARRIVALS:
                self.t_mono.append(tm)
                tick = getattr(msg, "tick", None)
                if tick is not None:
                    self.ticks.append(float(tick))
                st = stamp_seconds(msg)
                if st is not None and st > 0:
                    self.ages.append(tw - st)
            if self.first is None:
                self.first = msg
            self.last = msg

    def report(self, window_s: float) -> dict:
        with self.lock:
            t, ticks, ages, first, last = list(self.t_mono), list(self.ticks), list(self.ages), self.first, self.last
        r = {"type": self.type_name, "received": bool(t), "rate": rate_stats(t, window_s)}
        if ticks:
            r["tick"] = counter_stats(ticks, t[:len(ticks)])
        if ages:
            r["age"] = age_stats(ages)
        if first is not None:
            r["first_summary"] = summarize(self.type_name, first)
            r["last_summary"] = summarize(self.type_name, last)
            r["first_raw"] = to_plain(first)
        return r


# ------------------------------------------------------------------ SDK glue
class _PublishingRefused:
    def __init__(self, *a, **k):
        raise RuntimeError("sdk_probe is read-only: creating a DDS publisher is refused")


def guard_sdk(channel_mod) -> None:
    """Replace the SDK's publisher class so that nothing in this process can create a writer."""
    channel_mod.ChannelPublisher = _PublishingRefused


def detect_iface() -> str | None:
    try:
        txt = subprocess.run(["ip", "-br", "addr"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    return robot_iface_from_ip_text(txt)


def run_probe(seconds: float, iface: str | None, domain: int, topics, sdk=None, sleep=time.sleep,
              mono=time.monotonic) -> dict:
    if sdk is None:
        import importlib
        sdk = importlib.import_module("unitree_sdk2py.core.channel")
    guard_sdk(sdk)
    import importlib

    if iface:
        sdk.ChannelFactoryInitialize(domain, iface)
    else:
        sdk.ChannelFactoryInitialize(domain)
    recs, subs, skipped = {}, [], {}
    for topic, pkg, tname, src in topics:
        try:
            mod = importlib.import_module(f"unitree_sdk2py.idl.{pkg}.msg.dds_")
            typ = getattr(mod, tname)
        except Exception as e:
            skipped[topic] = f"type {pkg}/{tname} not importable: {type(e).__name__}: {e}"[:200]
            continue
        rec = TopicRecorder(topic, tname)
        try:
            sub = sdk.ChannelSubscriber(topic, typ)
            sub.Init(rec, 0)
        except Exception as e:
            skipped[topic] = f"subscribe failed: {type(e).__name__}: {e}"[:200]
            continue
        recs[topic] = (rec, src)
        subs.append(sub)
    t0 = mono()
    sleep(seconds)
    window = mono() - t0
    topics_out = {}
    for topic, (rec, src) in recs.items():
        r = rec.report(window)
        r["source"] = src
        topics_out[topic] = r
    for s in subs:
        try:
            s.Close()
        except Exception:
            pass
    return {"window_s": round(window, 3), "topics": topics_out, "skipped": skipped,
            "received": sorted(t for t, r in topics_out.items() if r["received"]),
            "silent": sorted(t for t, r in topics_out.items() if not r["received"])}


def _sdk_version() -> dict:
    out = {}
    try:
        import importlib.metadata as md
        for d in ("unitree_sdk2py", "unitree-sdk2py", "cyclonedds"):
            try:
                out[d] = md.version(d)
            except Exception:
                pass
    except Exception:
        pass
    try:
        import unitree_sdk2py
        out["unitree_sdk2py_file"] = getattr(unitree_sdk2py, "__file__", "")
    except Exception as e:
        out["unitree_sdk2py_import_error"] = str(e)[:200]
    return out


def main(argv=None, sdk=None, sleep=time.sleep) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    require_ack(argv, "sdk_probe.py")
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--i-understand-read-only", action="store_true")
    ap.add_argument("--iface", default="auto", help="network interface on 192.168.123.x; 'auto' finds it, 'none' lets the SDK pick")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", default="./inventory_reports")
    ap.add_argument("--watch-command-topics", action="store_true")
    a = ap.parse_args(argv)

    iface = a.iface
    if iface == "auto":
        iface = detect_iface()
    elif iface == "none":
        iface = None
    topics = TOPICS + (COMMAND_TOPICS if a.watch_command_topics else [])
    print(f"[sdk_probe] read-only; domain={a.domain} iface={iface} seconds={a.seconds} topics={len(topics)}")
    started = time.time()
    try:
        res = run_probe(a.seconds, iface, a.domain, topics, sdk=sdk, sleep=sleep)
    except ImportError as e:
        res = {"error": f"unitree_sdk2py not importable: {e}"}
    except Exception as e:  # report, do not hide
        res = {"error": f"{type(e).__name__}: {e}"}
    report = {"probe": "sdk_probe", "version": "1.0", "utc": utc_stamp(), "wall_start": started,
              "args": {"iface": iface, "domain": a.domain, "seconds": a.seconds, "watch_command_topics": a.watch_command_topics},
              "sdk": _sdk_version(), **res}
    path = write_json(os.path.join(a.out, f"sdk_probe_{utc_stamp()}.json"), report)
    if "error" in res:
        print(f"[sdk_probe] ERROR: {res['error']}")
    else:
        for t in res["received"]:
            r = res["topics"][t]["rate"]
            print(f"[sdk_probe] {t:28s} {r['count']:7d} msgs  {r.get('mean_hz')} Hz  p99 period {r['period_ms'].get('p99')} ms")
        print(f"[sdk_probe] silent: {', '.join(res['silent']) or '-'}")
        if res["skipped"]:
            print(f"[sdk_probe] skipped (type missing): {', '.join(res['skipped'])}")
    print(f"[sdk_probe] wrote {path}")
    return 0 if "error" not in res else 1


if __name__ == "__main__":
    sys.exit(main())
