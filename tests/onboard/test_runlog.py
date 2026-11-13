"""Run log: transitions go to JSONL and MCAP; the stdlib MCAP file reads back (and with the official reader if installed)."""

import json

import pytest

from humanoid_arc_bme.onboard.runtime.io.fake import FakeRobotIO
from humanoid_arc_bme.onboard.runtime.mcap_writer import MAGIC, McapJsonWriter, read_json_messages
from humanoid_arc_bme.onboard.runtime.runlog import RunLog, new_run_dir


def test_mcap_roundtrip(tmp_path):
    w = McapJsonWriter(tmp_path / "a.mcap")
    w.write("/x", {"a": 1}, 10)
    w.write("/y", {"b": [1, 2]}, 20)
    w.write("/x", {"a": 2}, 30)
    w.close()
    raw = (tmp_path / "a.mcap").read_bytes()
    assert raw.startswith(MAGIC) and raw.endswith(MAGIC)
    assert read_json_messages(tmp_path / "a.mcap") == [("/x", 10, {"a": 1}), ("/y", 20, {"b": [1, 2]}), ("/x", 30, {"a": 2})]


def test_mcap_unclosed_file_is_still_readable(tmp_path):
    w = McapJsonWriter(tmp_path / "b.mcap")
    w.write("/x", {"a": 1}, 1)
    w._f.flush()  # simulated crash: no footer
    assert read_json_messages(tmp_path / "b.mcap") == [("/x", 1, {"a": 1})]


def test_mcap_official_reader(tmp_path):
    mcap_reader = pytest.importorskip("mcap.reader")
    w = McapJsonWriter(tmp_path / "c.mcap")
    for i in range(5):
        w.write("/g1/transitions", {"i": i}, i)
    w.close()
    with open(tmp_path / "c.mcap", "rb") as f:
        msgs = [(ch.topic, json.loads(m.data)) for _s, ch, m in mcap_reader.NonSeekingReader(f).iter_messages()]
    assert msgs == [("/g1/transitions", {"i": i}) for i in range(5)]


def test_runlog_writes_transitions_events_state(tmp_path):
    t = [0.0]
    d = new_run_dir(tmp_path, wall=lambda: 1.8e9)
    log = RunLog(d, state_log_hz=10.0, wall=lambda: 1.8e9 + t[0])
    io = FakeRobotIO(lambda: t[0])
    for k in range(50):  # 1 s at 50 Hz -> 10 state rows
        t[0] = k * 0.02
        log.state(*io.read_state())
    log.transition({"t": 0.5, "from": "disarmed", "to": "armed", "reason": "operator_confirmed", "controller": "fake"})
    log.event("s1_base_vel", {"subject": "vx"})
    log.command({"t": 0.6, "vx": 0.1})
    log.close()
    ev = [json.loads(x) for x in (d / "events.jsonl").read_text().splitlines()]
    assert ev[0]["kind"] == "transition" and ev[0]["to"] == "armed" and ev[1]["event"] == "s1_base_vel"
    assert len((d / "state.jsonl").read_text().splitlines()) == 10
    topics = [m[0] for m in read_json_messages(d / "telemetry.mcap")]
    assert topics.count("/g1/transitions") == 1 and topics.count("/g1/state") == 10 and "/g1/cmd" in topics
    assert {f["name"] for f in log.files()} == {"events.jsonl", "state.jsonl", "cmd.jsonl", "telemetry.mcap"}
