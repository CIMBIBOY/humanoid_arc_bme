"""deploy/scripts/{install,build,run,preflight,test}.sh: syntax, safe defaults, no sudo, dry-run install writes nothing,
run.sh starts the fake runtime DISARMED on 127.0.0.1, pinned requirements."""

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SR = ROOT / "deploy" / "scripts"
SCRIPTS = ["install.sh", "build.sh", "run.sh", "preflight.sh", "test.sh"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_syntax_strict_mode_and_no_sudo(name):
    p = SR / name
    assert subprocess.run(["bash", "-n", str(p)]).returncode == 0
    text = p.read_text()
    assert "set -euo pipefail" in text and os.access(p, os.X_OK)
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#") or s.startswith("echo"):
            continue
        assert not re.match(r"^(run\s+)?sudo\b", s), f"{name}: {s}"


def test_requirements_are_pinned():
    lines = [ln.split("#")[0].strip() for ln in (SR / "requirements-jetson.txt").read_text().splitlines()]
    pins = [ln for ln in lines if ln]
    assert pins and all(re.fullmatch(r"[A-Za-z0-9_.\-]+==[0-9][0-9A-Za-z.\-]*", p) for p in pins), pins
    assert "onnxruntime==1.23.2" in pins and "numpy==1.26.4" in pins


def test_install_dry_run_writes_nothing(tmp_path):
    env = {**os.environ, "G1_HOME": str(tmp_path / "deps"), "G1_VENV": str(tmp_path / "venv")}
    r = subprocess.run(["bash", str(SR / "install.sh"), "--dry-run"], capture_output=True, text=True, env=env, timeout=60)
    out = r.stdout + r.stderr
    if "MISSING:" in out:
        pytest.skip("build tools missing on this machine: " + out.splitlines()[-2])
    assert r.returncode == 0, out
    assert "--system-site-packages" in out and "cyclonedds==0.10.2" in out and "releases/0.10.x" in out and "814556d" in out
    assert not (tmp_path / "deps").exists() and not (tmp_path / "venv").exists()


def test_run_sh_fake_starts_disarmed_on_localhost(tmp_path):
    env = {**os.environ, "G1_HOME": str(tmp_path / "nodeps"), "G1_MODELS": str(tmp_path / "nomodels")}
    r = subprocess.run(["timeout", "-s", "INT", "4", "bash", str(SR / "run.sh"), "--io", "fake", "--port", "0"],
                       capture_output=True, text=True, env=env, timeout=30, cwd=tmp_path)
    out = r.stdout + r.stderr
    assert "DISARMED" in out and "http://127.0.0.1:" in out and "#token=" in out, out
    assert "0.0.0.0" not in out
    assert "stopped; logs in" in out  # SIGINT -> clean shutdown (disarm, logs closed)


def test_preflight_sh_reports(tmp_path):
    env = {**os.environ, "G1_HOME": str(tmp_path / "nodeps"), "G1_MODELS": str(ROOT / "outputs" / "W4" / "onnx")}
    r = subprocess.run(["bash", str(SR / "preflight.sh"), "--log-root", str(tmp_path / "runs")], capture_output=True, text=True,
                       env=env, timeout=120)
    assert "required not ok" in r.stdout and "robot_network" in r.stdout and r.returncode in (0, 1)
