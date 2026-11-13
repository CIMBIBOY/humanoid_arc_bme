"""One directory per operator run: events.jsonl (every transition and event), state.jsonl (robot state at state_log_hz),
cmd.jsonl (what was commanded, per tick while armed) and telemetry.mcap (the same streams for Foxglove).

A `TelemetrySink`, so the optional live FoxgloveSink can sit next to it in a MultiSink.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path

from .mcap_writer import McapJsonWriter
from .telemetry import TelemetrySink

TRANSITION_SCHEMA = {"type": "object", "properties": {"t": {"type": "number"}, "from": {"type": "string"}, "to": {"type": "string"},
                                                      "reason": {"type": "string"}, "controller": {"type": ["string", "null"]}}}


def new_run_dir(root, wall=time.time) -> Path:
    root = Path(os.path.expanduser(str(root)))
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(wall()))
    d = root / f"run_{stamp}"
    k = 1
    while d.exists():
        k += 1
        d = root / f"run_{stamp}_{k}"
    d.mkdir(parents=True)
    return d


class RunLog(TelemetrySink):
    def __init__(self, run_dir, state_log_hz: float = 10.0, wall=time.time):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.wall = wall
        self._lock = threading.Lock()
        self._ev = open(self.dir / "events.jsonl", "a")
        self._st = open(self.dir / "state.jsonl", "a")
        self._cmd = open(self.dir / "cmd.jsonl", "a")
        self.mcap = McapJsonWriter(self.dir / "telemetry.mcap", profile="g1_runtime")
        self.mcap.channel("/g1/transitions", TRANSITION_SCHEMA)
        self._min_dt = 1.0 / state_log_hz if state_log_hz > 0 else float("inf")
        self._last_state = -1e18
        self.closed = False

    def _ns(self) -> int:
        return int(self.wall() * 1e9)

    def _line(self, f, obj) -> None:
        with self._lock:
            if not self.closed:
                f.write(json.dumps(obj, default=str) + "\n")
                f.flush()

    def transition(self, rec: dict) -> None:
        obj = {"kind": "transition", "wall": self.wall(), **rec}
        self._line(self._ev, obj)
        self.mcap.write("/g1/transitions", obj, self._ns())

    def event(self, kind, detail):
        obj = {"kind": "event", "event": kind, "wall": self.wall(), **(detail or {})}
        self._line(self._ev, obj)
        self.mcap.write("/g1/events", obj, self._ns())

    def state(self, st, meta):
        if st.stamp - self._last_state < self._min_dt - 1e-6:
            return
        self._last_state = st.stamp
        d = st.to_dict()
        obj = {"wall": self.wall(), "state": d, "meta": asdict(meta)}
        self._line(self._st, obj)
        self.mcap.write("/g1/state", {k: d[k] for k in ("stamp", "joint_names", "joint_pos", "joint_vel", "imu")} | {"meta": asdict(meta)}, self._ns())

    def command(self, rec: dict) -> None:
        obj = {"wall": self.wall(), **rec}
        self._line(self._cmd, obj)
        self.mcap.write("/g1/cmd", obj, self._ns())

    def frame(self, fr):
        pass  # images are shown in the UI and go to the optional FoxgloveSink; not written to the run log (size)

    def files(self) -> list[dict]:
        return [{"name": p.name, "bytes": p.stat().st_size} for p in sorted(self.dir.iterdir()) if p.is_file()]

    def close(self):
        with self._lock:
            if self.closed:
                return
            self.closed = True
            for f in (self._ev, self._st, self._cmd):
                f.close()
        self.mcap.close()
