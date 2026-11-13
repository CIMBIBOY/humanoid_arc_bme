"""ROS 2 wrapper: imports cleanly without rclpy (the node test skips), the core never imports rclpy, parameter parsing,
and a static check of the colcon package layout; a real `colcon build` into tmp when colcon is on PATH (no ROS needed)."""

import ast
import os
import py_compile
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "deploy" / "ros2" / "g1_runtime_ros"
sys.path.insert(0, str(PKG))


def test_package_imports_without_rclpy_and_node_import_skips():
    import importlib.util

    import g1_runtime_ros
    assert g1_runtime_ros.HAVE_RCLPY == (importlib.util.find_spec("rclpy") is not None)
    pytest.importorskip("rclpy")
    import g1_runtime_ros.node  # noqa: F401  (only where ROS 2 is sourced)


def test_core_never_imports_rclpy():
    code = ("import sys; import humanoid_arc_bme.onboard.runtime.cli, humanoid_arc_bme.onboard.runtime.operator_runtime, humanoid_arc_bme.onboard.runtime.web.server, humanoid_arc_bme.onboard.runtime.controllers, "
            "humanoid_arc_bme.onboard.runtime.preflight, g1_runtime_ros, g1_runtime_ros.params; "
            "bad = [m for m in sys.modules if m.split('.')[0] in ('rclpy', 'torch', 'isaaclab', 'onnxruntime', 'unitree_sdk2py')]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env={**os.environ, "PYTHONPATH": f"{ROOT}{os.pathsep}{PKG}"},
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr


def test_params_to_args():
    from g1_runtime_ros.params import to_args, twist_to_cmd
    a = to_args({"io": "unitree", "iface": "eth0", "lan": "true", "port": "9000", "allow_lowcmd": "false", "foxglove": -1})
    assert (a.io, a.iface, a.lan, a.port, a.allow_lowcmd, a.models, a.foxglove) == ("unitree", "eth0", True, 9000, False, None, None)
    with pytest.raises(ValueError):
        to_args({"io": "robot"})
    with pytest.raises(ValueError):
        to_args({"lan": "maybe"})
    assert twist_to_cmd(0.1, 0, -0.2) == (0.1, 0.0, -0.2)


def test_colcon_layout_static():
    x = ET.parse(PKG / "package.xml").getroot()
    assert x.get("format") == "3" and x.findtext("name") == "g1_runtime_ros"
    assert x.find("export").findtext("build_type") == "ament_python"
    deps = {e.text for e in x.findall("exec_depend")}
    assert {"rclpy", "std_msgs", "geometry_msgs", "launch_ros", "python3-numpy", "python3-yaml"} <= deps
    setup = (PKG / "setup.py").read_text()
    tree = ast.parse(setup)
    kw = {k.arg: k.value for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "setup" for k in n.keywords}
    assert ast.literal_eval(kw["version"]) == x.findtext("version")
    pkgs = ast.literal_eval(kw["packages"].elts[0]) if False else [ast.literal_eval(e) if not isinstance(e, ast.Name) else "g1_runtime_ros" for e in kw["packages"].elts]
    for p in pkgs:
        assert (PKG / p.replace(".", "/") / "__init__.py").is_file(), p  # symlinked core packages resolve
    assert "g1_runtime_ros.node:main" in setup and "humanoid_arc_bme.onboard.runtime.cli:main" in setup
    assert (PKG / "resource" / "g1_runtime_ros").is_file()
    assert "script_dir=$base/lib/g1_runtime_ros" in (PKG / "setup.cfg").read_text()
    for link in ("humanoid_arc_bme", "config"):
        assert (PKG / link).is_symlink() and (PKG / link).resolve() == (ROOT / link).resolve()
    launch = PKG / "launch" / "g1_runtime.launch.py"
    py_compile.compile(str(launch), doraise=True)
    from g1_runtime_ros.params import PARAMS
    largs = ast.literal_eval([n for n in ast.parse(launch.read_text()).body if isinstance(n, ast.Assign) and n.targets[0].id == "ARGS"][0].value)
    assert set(largs) == set(PARAMS)
    assert largs["lan"][0] == "false" and largs["allow_lowcmd"][0] == "false" and largs["io"][0] == "fake"  # safe defaults


@pytest.mark.skipif(shutil.which("colcon") is None, reason="colcon not installed")
def test_colcon_build_into_tmp(tmp_path):
    r = subprocess.run(["colcon", "--log-base", str(tmp_path / "log"), "build", "--base-paths", str(ROOT / "deploy" / "ros2"),
                        "--build-base", str(tmp_path / "build"), "--install-base", str(tmp_path / "install")],
                       cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    inst = tmp_path / "install" / "g1_runtime_ros"  # colcon default: isolated install per package
    assert (inst / "lib" / "g1_runtime_ros" / "g1_runtime_node").is_file()
    assert (inst / "share" / "g1_runtime_ros" / "config" / "g1_limits.yaml").is_file()
    assert (inst / "share" / "g1_runtime_ros" / "launch" / "g1_runtime.launch.py").is_file()
    assert list(inst.rglob("humanoid_arc_bme/onboard/runtime/web/static/app.js"))
    assert not list((ROOT / "deploy" / "ros2").rglob("*.egg-info"))  # the build does not write into the source tree
