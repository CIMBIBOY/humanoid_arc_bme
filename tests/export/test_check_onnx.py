"""check_onnx: a good export passes; a flipped or NaN weight fails with exit 1; usage errors exit 2."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import check_onnx

SCRIPT = Path(check_onnx.__file__)


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=120)


def test_good_export_passes(policy_export):
    res = check_onnx.check_policy_pack_dir(policy_export())
    assert res["passed"], res["failures"]
    assert res["max_abs_diff"] < 1e-5
    assert res["runs"] == 500
    assert 0 < res["median_ms"] <= res["p99_ms"]
    assert "CPUExecutionProvider" in res["providers"]


def test_cli_good_export_exits_zero(policy_export):
    proc = run_cli(str(policy_export()))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RESULT: PASS" in proc.stdout


@pytest.mark.parametrize("mode", ["flip", "nan"])
def test_corrupted_onnx_fails(policy_export, corrupt_onnx, mode):
    d = policy_export()
    corrupt_onnx(d / "policy.onnx", mode)
    proc = run_cli(str(d))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "RESULT: FAIL" in proc.stdout
    assert "max |onnx - torch|" in proc.stdout


def test_engine_providers_listed(capsys):
    assert check_onnx.main(["--engine-providers"]) == 0
    assert "onnxruntime" in capsys.readouterr().out


def test_missing_export_is_a_usage_error(tmp_path, capsys):
    assert check_onnx.main([str(tmp_path / "nothing")]) == 2
    assert "no usable export.json" in capsys.readouterr().err


def test_no_directory_argument_is_rejected():
    with pytest.raises(SystemExit):
        check_onnx.main([])
