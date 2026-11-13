"""Telemetry sinks: where the operator's viewer and the log files get their data.

Foxglove (WebSocket server for the live viewer + MCAP for replay) is the default viewer because it is a maintained tool with a
documented SDK; our own 3D view, if we ever want one, is just another TelemetrySink. JsonlSink needs nothing installed.
Every sink gets plain dicts (state, meta, events, stats) and CameraFrames, so none of them knows about the robot SDK.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict

from humanoid_arc_bme.onboard.api.contract import RobotState

from .types import CameraFrame, StateMeta


class TelemetrySink(ABC):
    @abstractmethod
    def state(self, st: RobotState, meta: StateMeta) -> None: ...

    @abstractmethod
    def frame(self, fr: CameraFrame) -> None: ...

    @abstractmethod
    def event(self, kind: str, detail: dict) -> None: ...

    def close(self) -> None:  # noqa: B027
        pass


class NullSink(TelemetrySink):
    """Drops everything (counters stay in Runtime.stats). For many simulated runtimes in one process."""

    def state(self, st, meta):
        pass

    def frame(self, fr):
        pass

    def event(self, kind, detail):
        pass


class MemorySink(TelemetrySink):
    def __init__(self):
        self.states, self.frames, self.events = [], [], []

    def state(self, st, meta):
        self.states.append((st, meta))

    def frame(self, fr):
        self.frames.append(fr)

    def event(self, kind, detail):
        self.events.append((kind, detail))


class JsonlSink(TelemetrySink):
    """One JSON object per line; image pixels are not written (only their size and stamp)."""

    def __init__(self, path):
        self._f = open(path, "w")

    def _w(self, kind, obj):
        self._f.write(json.dumps({"kind": kind, **obj}) + "\n")
        self._f.flush()

    def state(self, st, meta):
        self._w("state", {"state": st.to_dict(), "meta": asdict(meta)})

    def frame(self, fr):
        self._w("frame", {"name": fr.name, "stamp": fr.stamp, "width": fr.width, "height": fr.height, "encoding": fr.encoding})

    def event(self, kind, detail):
        self._w("event", {"event": kind, **detail})

    def close(self):
        self._f.close()


class FoxgloveSink(TelemetrySink):
    """Live WebSocket server (host/port) and/or an MCAP file. Imports the foxglove SDK lazily (pip install foxglove-sdk).

    Topics: /g1/state (JSON: joint names/pos/vel, imu, meta), /g1/events (JSON), /camera/<name> (foxglove RawImage).
    The server binds 127.0.0.1 by default: reach it from the laptop through an SSH or Tailscale tunnel rather than exposing
    the robot's network port (the websocket has no authentication).
    """

    _STATE_SCHEMA = {"type": "object", "properties": {"stamp": {"type": "number"}, "joint_names": {"type": "array", "items": {"type": "string"}},
                     "joint_pos": {"type": "array", "items": {"type": "number"}}, "joint_vel": {"type": "array", "items": {"type": "number"}},
                     "imu": {"type": "object"}, "meta": {"type": "object"}}}
    _EVENT_SCHEMA = {"type": "object", "properties": {"kind": {"type": "string"}, "detail": {"type": "object"}}}

    def __init__(self, port: int | None = 8765, host: str = "127.0.0.1", mcap_path: str | None = None):
        import foxglove
        from foxglove import channels, messages

        self._fx, self._channels, self._messages = foxglove, channels, messages
        self._server = foxglove.start_server(name="g1_runtime", host=host, port=port) if port is not None else None
        self._mcap = foxglove.open_mcap(mcap_path, allow_overwrite=True) if mcap_path else None
        self._state_ch = foxglove.Channel("/g1/state", schema=self._STATE_SCHEMA)
        self._event_ch = foxglove.Channel("/g1/events", schema=self._EVENT_SCHEMA)
        self._cams: dict = {}

    def state(self, st, meta):
        d = st.to_dict()
        self._state_ch.log({k: d[k] for k in ("stamp", "joint_names", "joint_pos", "joint_vel", "imu")} | {"meta": asdict(meta)})

    def frame(self, fr):
        ch = self._cams.get(fr.name)
        if ch is None:
            ch = self._cams[fr.name] = self._channels.RawImageChannel(topic=f"/camera/{fr.name}")
        sec = int(fr.stamp)
        ch.log(self._messages.RawImage(
            timestamp=self._messages.Timestamp(sec=sec, nsec=int((fr.stamp - sec) * 1e9)), frame_id=f"camera_{fr.name}",
            width=fr.width, height=fr.height, encoding=fr.encoding, step=fr.width * (3 if fr.encoding != "mono8" else 1), data=fr.data))

    def event(self, kind, detail):
        self._event_ch.log({"kind": kind, "detail": detail})

    def close(self):
        if self._mcap:
            self._mcap.close()
        if self._server:
            self._server.stop()


class MultiSink(TelemetrySink):
    def __init__(self, *sinks):
        self.sinks = sinks

    def state(self, st, meta):
        for s in self.sinks:
            s.state(st, meta)

    def frame(self, fr):
        for s in self.sinks:
            s.frame(fr)

    def event(self, kind, detail):
        for s in self.sinks:
            s.event(kind, detail)

    def close(self):
        for s in self.sinks:
            s.close()
