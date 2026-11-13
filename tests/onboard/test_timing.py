"""Timing parameters and the delay line (api/timing.py). No Isaac Sim."""

from pathlib import Path

import pytest

from humanoid_arc_bme.onboard.api.timing import DelayLine, Timing, load_timing

ROOT = Path(__file__).resolve().parents[2]


def test_shipped_file_is_all_null_and_the_loader_refuses_nulls():
    t = load_timing(ROOT / "config" / "timing.yaml")
    for name in ("control_hz", "state_delay_ms", "command_delay_ms", "camera_delay_ms"):
        with pytest.raises(ValueError, match="not been measured"):
            t.require(name)
    with pytest.raises(ValueError):
        t.steps_for("command_delay_ms", 0.02)


def test_rejects_bad_values_and_unknown_keys(tmp_path):
    with pytest.raises(ValueError):
        Timing(control_hz=0)
    with pytest.raises(ValueError):
        Timing(command_delay_ms=-5)
    f = tmp_path / "t.yaml"
    f.write_text("control_hz: 50\nbogus: 1\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_timing(f)


def test_steps_for_rounds_to_nearest_step():
    t = Timing(control_hz=50, command_delay_ms=100)
    assert t.steps_for("command_delay_ms", 0.02) == 5
    assert Timing(command_delay_ms=29).steps_for("command_delay_ms", 0.02) == 1
    assert Timing(command_delay_ms=9).steps_for("command_delay_ms", 0.02) == 0


@pytest.mark.parametrize("k", [0, 1, 3, 7])
def test_impulse_appears_exactly_k_steps_later(k):
    line = DelayLine(k, initial=0)
    out = [line.push(1 if i == 0 else 0) for i in range(k + 4)]
    assert out.index(1) == k
    assert sum(out) == 1


def test_stream_is_shifted_not_altered():
    line = DelayLine(3, initial=-1)
    stream = list(range(10))
    out = [line.push(x) for x in stream]
    assert out == [-1, -1, -1] + stream[:7]
