#!/usr/bin/env python3
"""READ-ONLY ROS 2 topic rate probe. It only SUBSCRIBES.

What it does: starts one rclpy node, waits for discovery, lists every topic with its type and the QoS of its
publishers, then subscribes (best-effort, depth 10, serialized/raw so big images and clouds are cheap) to each topic
whose type it can import, for `--seconds`. It writes per topic: rate, period jitter, gaps, message size, and message
age from `header.stamp` (or a Unitree `stamp`) on a sample of messages.

What it never does: create a publisher, call a service or action, or set a parameter.

Usage (source ROS 2 first, plus the unitree_ros2 workspace if you want the unitree_* types decoded):
    source /opt/ros/humble/setup.bash
    python3 ros_topic_probe.py --i-understand-read-only [--seconds 10] [--include REGEX] [--exclude REGEX]
                               [--max-topics 80] [--discovery-s 3] [--out DIR]
"""

from __future__ import annotations

import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_common import age_stats, rate_stats, require_ack, to_plain, utc_stamp, write_json  # noqa: E402

AGE_SAMPLE_EVERY = 10   # deserialize every N-th message for its stamp
AGE_SAMPLE_FIRST = 20   # and the first N


def choose_topics(names_types: list[tuple[str, list[str]]], include: str | None, exclude: str | None,
                  max_topics: int) -> list[tuple[str, str]]:
    """Filter discovered (name, [types]) into (name, type) pairs; ROS internals are skipped unless included."""
    out = []
    inc = re.compile(include) if include else None
    exc = re.compile(exclude) if exclude else None
    for name, types in sorted(names_types):
        if not types:
            continue
        if inc and not inc.search(name):
            continue
        if not inc and name in ("/rosout", "/parameter_events"):
            continue
        if exc and exc.search(name):
            continue
        out.append((name, types[0]))
    return out[:max_topics]


def stamp_of(msg) -> float | None:
    """Seconds from header.stamp (builtin_interfaces/Time) or a top-level `stamp` with sec/nanosec."""
    for holder in (getattr(msg, "header", None), msg):
        st = getattr(holder, "stamp", None) if holder is not None else None
        if st is None:
            continue
        sec = getattr(st, "sec", None)
        if sec is None:
            continue
        return float(sec) + float(getattr(st, "nanosec", 0) or 0) * 1e-9
    return None


def qos_to_dict(info) -> dict:
    q = getattr(info, "qos_profile", None)

    def nm(x):
        return getattr(x, "name", str(x))
    return {
        "node": f"{getattr(info, 'node_namespace', '')}/{getattr(info, 'node_name', '')}".replace("//", "/"),
        "reliability": nm(getattr(q, "reliability", None)), "durability": nm(getattr(q, "durability", None)),
        "history": nm(getattr(q, "history", None)), "depth": getattr(q, "depth", None),
    }


class RawRecorder:
    def __init__(self, cls, deserialize, mono=time.monotonic, wall=time.time):
        self.cls, self.deserialize, self.mono, self.wall = cls, deserialize, mono, wall
        self.t: list[float] = []
        self.sizes: list[int] = []
        self.ages: list[float] = []
        self.first = None
        self.n = 0
        self.decode_errors = 0

    def __call__(self, raw: bytes):
        tm, tw = self.mono(), self.wall()
        self.t.append(tm)
        self.sizes.append(len(raw))
        self.n += 1
        if self.n <= AGE_SAMPLE_FIRST or self.n % AGE_SAMPLE_EVERY == 0:
            try:
                msg = self.deserialize(raw, self.cls)
            except Exception:
                self.decode_errors += 1
                return
            if self.first is None:
                self.first = msg
            st = stamp_of(msg)
            if st is not None and st > 0:
                self.ages.append(tw - st)

    def report(self, window: float) -> dict:
        r = {"received": bool(self.t), "rate": rate_stats(self.t, window)}
        if self.sizes:
            r["bytes"] = {"min": min(self.sizes), "max": max(self.sizes), "mean": round(sum(self.sizes) / len(self.sizes), 1)}
            r["bandwidth_MBps"] = round(sum(self.sizes) / window / 1e6, 4) if window > 0 else None
        if self.ages:
            r["age"] = age_stats(self.ages)
        if self.first is not None:
            r["first_sample"] = to_plain(self.first, max_list=12)
        r["decode_errors"] = self.decode_errors
        return r


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    require_ack(argv, "ros_topic_probe.py")
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--i-understand-read-only", action="store_true")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--discovery-s", type=float, default=3.0)
    ap.add_argument("--include", default=None)
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--max-topics", type=int, default=80)
    ap.add_argument("--out", default="./inventory_reports")
    a = ap.parse_args(argv)

    import rclpy
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    rclpy.init()
    node = rclpy.create_node(f"inventory_ros_probe_{os.getpid()}")
    t_end = time.monotonic() + a.discovery_s
    while time.monotonic() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)
    names_types = node.get_topic_names_and_types()
    chosen = choose_topics(names_types, a.include, a.exclude, a.max_topics)
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                     history=HistoryPolicy.KEEP_LAST, depth=10)
    recs, skipped, pubinfo = {}, {}, {}
    for name, tname in chosen:
        try:
            pubinfo[name] = [qos_to_dict(i) for i in node.get_publishers_info_by_topic(name)]
        except Exception as e:
            pubinfo[name] = [f"error: {e}"]
        try:
            cls = get_message(tname)
        except Exception as e:
            skipped[name] = f"type {tname} not importable (source its workspace): {type(e).__name__}"
            continue
        rec = RawRecorder(cls, deserialize_message)
        node.create_subscription(cls, name, rec, qos, raw=True)
        recs[name] = (rec, tname)
    print(f"[ros_topic_probe] read-only; {len(names_types)} topics discovered, {len(recs)} subscribed, {len(skipped)} type-missing")
    t0 = time.monotonic()
    while time.monotonic() - t0 < a.seconds:
        rclpy.spin_once(node, timeout_sec=0.05)
    window = time.monotonic() - t0
    topics = {}
    for name, (rec, tname) in recs.items():
        r = rec.report(window)
        r["type"] = tname
        r["publishers"] = pubinfo.get(name)
        topics[name] = r
    report = {
        "probe": "ros_topic_probe", "version": "1.0", "utc": utc_stamp(), "window_s": round(window, 3),
        "env": {k: os.environ.get(k) for k in ("ROS_DISTRO", "ROS_DOMAIN_ID", "RMW_IMPLEMENTATION", "ROS_LOCALHOST_ONLY", "CYCLONEDDS_URI")},
        "discovered": [[n, t] for n, t in sorted(names_types)],
        "topics": topics, "skipped": skipped,
        "type_missing_publishers": {n: pubinfo.get(n) for n in skipped},
        "received": sorted(n for n, r in topics.items() if r["received"]),
        "silent": sorted(n for n, r in topics.items() if not r["received"]),
    }
    node.destroy_node()
    rclpy.shutdown()
    path = write_json(os.path.join(a.out, f"ros_topic_probe_{utc_stamp()}.json"), report)
    for n in report["received"]:
        r = topics[n]["rate"]
        print(f"[ros_topic_probe] {n:40s} {r.get('mean_hz')} Hz  p99 period {r['period_ms'].get('p99')} ms")
    print(f"[ros_topic_probe] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
