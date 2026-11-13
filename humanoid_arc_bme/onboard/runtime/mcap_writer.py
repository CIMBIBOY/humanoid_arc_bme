"""A minimal MCAP writer (JSON messages), standard library only, so the robot logs MCAP even without the foxglove SDK.

Writes the records of the MCAP spec (https://mcap.dev/spec, format version 0): magic, Header, Schema, Channel, Message,
DataEnd, Footer, magic. No chunks and no summary section (both optional in the spec): Foxglove and the `mcap` Python
reader read such a file sequentially. Every message is flushed, so a crash loses at most the footer; the file is still
readable record by record. Checked against the official `mcap` reader in tests/runtime/test_runlog.py when installed.
"""

from __future__ import annotations

import json
import struct
import threading

MAGIC = b"\x89MCAP0\r\n"
OP_HEADER, OP_FOOTER, OP_SCHEMA, OP_CHANNEL, OP_MESSAGE, OP_DATA_END = 0x01, 0x02, 0x03, 0x04, 0x05, 0x0F


def _s(text: str) -> bytes:
    b = text.encode("utf-8")
    return struct.pack("<I", len(b)) + b


def _record(op: int, content: bytes) -> bytes:
    return struct.pack("<BQ", op, len(content)) + content


class McapJsonWriter:
    def __init__(self, path, profile: str = "", library: str = "g1_runtime"):
        self._f = open(path, "wb")
        self._lock = threading.Lock()
        self._channels: dict[str, int] = {}
        self._seq: dict[int, int] = {}
        self._closed = False
        self._f.write(MAGIC + _record(OP_HEADER, _s(profile) + _s(library)))
        self._f.flush()

    def channel(self, topic: str, schema: dict | None = None) -> int:
        with self._lock:
            if topic in self._channels:
                return self._channels[topic]
            cid = len(self._channels) + 1
            data = json.dumps(schema or {"type": "object"}).encode()
            self._f.write(_record(OP_SCHEMA, struct.pack("<H", cid) + _s(topic.strip("/").replace("/", ".") or "msg") + _s("jsonschema")
                                  + struct.pack("<I", len(data)) + data))
            self._f.write(_record(OP_CHANNEL, struct.pack("<HH", cid, cid) + _s(topic) + _s("json") + struct.pack("<I", 0)))
            self._channels[topic], self._seq[cid] = cid, 0
            return cid

    def write(self, topic: str, obj: dict, t_ns: int) -> None:
        cid = self._channels.get(topic) or self.channel(topic)
        data = json.dumps(obj, default=str).encode()
        with self._lock:
            if self._closed:
                return
            self._seq[cid] += 1
            self._f.write(_record(OP_MESSAGE, struct.pack("<HIQQ", cid, self._seq[cid], t_ns, t_ns) + data))
            self._f.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._f.write(_record(OP_DATA_END, struct.pack("<I", 0)) + _record(OP_FOOTER, struct.pack("<QQI", 0, 0, 0)) + MAGIC)
            self._f.close()


def read_json_messages(path) -> list[tuple[str, int, dict]]:
    """(topic, log_time_ns, message) for every Message record; for tests and quick looks without the mcap package."""
    with open(path, "rb") as f:
        buf = f.read()
    if not buf.startswith(MAGIC):
        raise ValueError("not an MCAP file")
    i, topics, out = len(MAGIC), {}, []
    while i + 9 <= len(buf):
        op, n = struct.unpack_from("<BQ", buf, i)
        c = buf[i + 9: i + 9 + n]
        i += 9 + n
        if op == OP_CHANNEL:
            cid, _sid = struct.unpack_from("<HH", c, 0)
            tl = struct.unpack_from("<I", c, 4)[0]
            topics[cid] = c[8: 8 + tl].decode()
        elif op == OP_MESSAGE:
            cid, _seq, t, _pt = struct.unpack_from("<HIQQ", c, 0)
            out.append((topics[cid], t, json.loads(c[22:])))
        elif op == OP_FOOTER:
            break
    return out
