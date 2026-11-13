"""humanoid_arc_bme.onboard.deploy: parsers, manifest, delivery plan. No robot, no network, no Isaac."""

import json

import pytest

from humanoid_arc_bme.onboard.deploy import deploy, manifest, preflight
from humanoid_arc_bme.onboard.deploy.runner import Result


def test_parse_tegra_release():
    t = "# R36 (release), REVISION: 4.3, GCID: 1, BOARD: generic\n"
    assert preflight.parse_tegra_release(t) == (36, "4.3")
    assert preflight.parse_tegra_release("junk") is None


def _hdr(major, minor, patch=0):
    return f"#define NV_TENSORRT_MAJOR {major}\n#define NV_TENSORRT_MINOR {minor}\n#define NV_TENSORRT_PATCH {patch}\n"


def test_tensorrt_exact_version_only():
    ok = preflight.check_tensorrt({"TensorRT_ROOT": "/t"}, read=lambda p: _hdr(10, 7, 0))
    assert ok.status == "ok"
    for minor in (3, 13):
        assert preflight.check_tensorrt({"TensorRT_ROOT": "/t"}, read=lambda p, m=minor: _hdr(10, m)).status == "fail"
    assert preflight.check_tensorrt({}).status == "fail"


def test_robot_network_detection():
    out = "2: eth0    inet 192.168.123.222/24 brd x scope global eth0\n3: wlan0    inet 10.0.0.5/24 scope global wlan0\n"
    ok = preflight.check_robot_network(lambda c: Result(0, out))
    assert ok.status == "ok" and "192.168.123.222" in ok.detail
    bad = preflight.check_robot_network(lambda c: Result(0, "3: wlan0    inet 10.0.0.5/24 scope global wlan0\n"))
    assert bad.status == "fail"


def test_unknown_is_not_ok():
    assert preflight.check_power_mode(lambda c: Result(127, "", "not found")).status == "unknown"
    assert preflight.check_power_mode(lambda c: Result(0, "NV Power Mode: 15W")).status == "fail"
    assert preflight.check_power_mode(lambda c: Result(0, "NV Power Mode: MAXN")).status == "ok"


@pytest.fixture
def ckpt(tmp_path):
    d = tmp_path / "policy"
    d.mkdir()
    (d / "model.pt").write_bytes(b"weights")
    (d / "cfg.json").write_text("{}")
    return d


def test_manifest_roundtrip_and_tamper(ckpt):
    m = manifest.build(str(ckpt), seed=7)
    assert set(m["files"]) == {"model.pt", "cfg.json"}
    assert manifest.verify(str(ckpt)) == []
    (ckpt / "model.pt").write_bytes(b"changed")
    assert manifest.verify(str(ckpt)) == ["hash differs: model.pt"]
    (ckpt / "model.pt").write_bytes(b"weights")
    (ckpt / "extra.txt").write_text("x")
    assert manifest.verify(str(ckpt)) == ["not in manifest: extra.txt"]


def test_manifest_needs_files_and_missing_manifest(tmp_path):
    with pytest.raises(ValueError):
        manifest.build(str(tmp_path), seed=1)
    assert "cannot read" in manifest.verify(str(tmp_path))[0]


@pytest.mark.parametrize("host,dest", [("a b", "/x/y"), ("h;rm", "/x"), ("h", "/x/../y"), ("h", "/"), ("h", "x y"), ("h", "$(id)")])
def test_unsafe_arguments_rejected(host, dest):
    with pytest.raises(deploy.DeployError):
        deploy.plan("/tmp/p", host, dest)


def test_dry_run_runs_nothing(ckpt):
    manifest.build(str(ckpt), seed=1)
    calls = []
    lines = []
    deploy.deliver(str(ckpt), "unitree@g1", "/home/unitree/policies/p1", runner=lambda *a, **k: calls.append(a) or Result(0, ""), out=lines.append)
    assert calls == [] and all(l.startswith("PLAN ") for l in lines)


def test_execute_order_and_stop_on_failure(ckpt):
    manifest.build(str(ckpt), seed=1)
    seen = []

    def runner(c, timeout=0):
        seen.append(c)
        return Result(1, "", "boom") if "sha256sum" in c[-1] else Result(0, "")

    with pytest.raises(deploy.DeployError, match="boom"):
        deploy.deliver(str(ckpt), "g1", "/p/x", execute=True, runner=runner, out=lambda s: None)
    assert [c[0] for c in seen] == ["ssh", "scp", "ssh"]  # the rename never ran


def test_refuses_tampered_local_checkpoint(ckpt):
    manifest.build(str(ckpt), seed=1)
    (ckpt / "model.pt").write_bytes(b"x")
    with pytest.raises(deploy.DeployError, match="manifest"):
        deploy.deliver(str(ckpt), "g1", "/p/x", out=lambda s: None)
