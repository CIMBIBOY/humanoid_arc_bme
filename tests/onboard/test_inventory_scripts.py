"""Tests for the read-only robot inventory scripts (deploy/scripts/inventory/), with fake SDK / ROS / camera inputs.

Nothing here touches a robot: the SDK, ros2 and the cameras are replaced by fakes, and the shell inventory runs only
a few local, read-only sections against stub commands.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INV = ROOT / "deploy" / "scripts" / "inventory"
DOC = ROOT / "docs" / "robot-inventory.md"
SCRIPTS = ["inventory.sh", "sdk_probe.py", "ros_topic_probe.py", "camera_probe.py", "report_to_facts.py"]
FLAG = "--i-understand-read-only"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), INV / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ gate and static read-only checks
@pytest.mark.parametrize("script", SCRIPTS)
def test_refuses_without_ack_flag(script, tmp_path):
    cmd = ["bash", str(INV / script)] if script.endswith(".sh") else [sys.executable, str(INV / script)]
    if script == "report_to_facts.py":
        cmd.append(str(tmp_path))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=tmp_path)
    assert r.returncode == 2
    assert "REFUSING" in r.stderr and FLAG in r.stderr
    assert list(tmp_path.iterdir()) == []  # nothing written


FORBIDDEN = [
    r"ChannelPublisher\s*\(", r"create_publisher\s*\(", r"\.Write\s*\(", r"\bLocoClient\b", r"\bMotionSwitcherClient\b",
    r"\bSportClient\b", r"\bArmActionClient\b", r"create_client\s*\(", r"ActionClient\s*\(", r"call_async\s*\(",
    r"ros2\s+topic\s+pub", r"ros2\s+service\s+call", r"ros2\s+action\s+send_goal", r"ros2\s+param\s+set",
    r"ros2\s+lifecycle\s+set", r"\bddsperf\b", r"cyclonedds\s+publish",
    r"(?<![\w/])jetson_clocks(?!\w)(?!\s+--show)", r"nvpmodel\s+-m", r"nvpmodel\s+--mode",
    r"systemctl\s+(start|stop|restart|enable|disable|mask|kill|reload)", r"docker\s+(run|start|stop|rm|kill|restart|exec)",
    r"\bip\s+(addr|address|link|route|neigh)\s+(add|del|change|replace|set|flush)", r"rfkill\s+(un)?block",
    r"nmcli\s+.*\b(up|down|modify|add|delete|connect)\b", r"\breboot\b", r"\bshutdown\b(?!\()", r"\bdd\s+if=",
    r"set_option\s*\(", r"hardware_reset\s*\(", r"\bkill\s+-", r"\bpkill\b", r"\bkillall\b", r"\bapt(-get)?\s+install",
    r"pip3?\s+install",
]


@pytest.mark.parametrize("script", SCRIPTS + ["probe_common.py"])
def test_no_command_publishers_or_state_changes(script):
    text = (INV / script).read_text()
    for pat in FORBIDDEN:
        hits = [ln for ln in text.splitlines() if re.search(pat, ln)]
        assert not hits, f"{script}: forbidden pattern {pat!r}: {hits[:3]}"


def test_ros_probe_shutdown_is_rclpy_only():
    # the only 'shutdown' in the ROS probe is rclpy.shutdown() (our own node), never the machine
    text = (INV / "ros_topic_probe.py").read_text()
    assert [ln.strip() for ln in text.splitlines() if "shutdown" in ln] == ["rclpy.shutdown()"]


# ------------------------------------------------------------------ probe_common
def test_rate_stats_and_counters():
    pc = load("probe_common.py")
    t = [i * 0.002 for i in range(501)]  # 500 Hz for 1 s
    t[300] += 0.0  # keep monotonic
    r = pc.rate_stats(t, 1.0)
    assert r["count"] == 501 and abs(r["mean_hz"] - 500) < 1e-6
    assert abs(r["period_ms"]["p50"] - 2.0) < 1e-6 and r["gaps_over_2x_median"] == 0
    gap = t[:100] + [x + 0.05 for x in t[100:]]
    assert pc.rate_stats(gap)["gaps_over_2x_median"] == 1
    c = pc.counter_stats([10, 12, 14, 14], [0, 0.002, 0.004, 0.006])
    assert c["non_increasing"] == 1 and c["step_p50"] == 2
    assert pc.age_stats([0.001, 0.003])["p50_ms"] == 2.0
    assert pc.robot_iface_from_ip_text("lo UNKNOWN 127.0.0.1/8\neth0 UP 192.168.123.164/24 fe80::1/64\n") == "eth0"
    assert pc.robot_iface_from_ip_text("eno1 UP 192.168.0.5/24\n") is None


# ------------------------------------------------------------------ sdk_probe with a fake unitree_sdk2py
class _NS(types.SimpleNamespace):
    pass


def _lowstate(tick):
    motors = [_NS(q=0.1 * i if i < 29 else 0.0, dq=0.0, ddq=0.0, tau_est=0.0, temperature=[30 + (i < 29), 31] if i < 29 else [0, 0],
                  vol=24.5 if i < 29 else 0.0, sensor=[0, 0], motorstate=(7 if i == 3 else 0)) for i in range(35)]
    imu = _NS(quaternion=[1.0, 0.0, 0.0, 0.0], gyroscope=[0.0, 0.0, 0.01], accelerometer=[0.0, 0.0, 9.81], rpy=[0.0, 0.0, 0.0], temperature=40)
    return _NS(version=[1, 0], mode_pr=0, mode_machine=5, tick=tick, imu_state=imu, motor_state=motors,
               wireless_remote=[0] * 40, reserve=[0] * 4, crc=123)


def _hand():
    return _NS(motor_state=[_NS(q=0.2, dq=0.0, tau_est=0.0, temperature=[28, 28]) for _ in range(7)],
               press_sensor_state=[_NS(pressure=[0.0] * 12, temperature=[25.0] * 12) for _ in range(9)],
               imu_state=_NS(quaternion=[1, 0, 0, 0], gyroscope=[0, 0, 0], accelerometer=[0, 0, 9.8], rpy=[0, 0, 0], temperature=0),
               power_v=24.0, power_a=0.3, system_v=24.0, device_v=5.0, error=[0, 0], reserve=[0, 0])


@pytest.fixture
def fake_sdk(monkeypatch):
    subs = {}
    calls = {"init": None, "publishers": 0}

    class ChannelSubscriber:
        def __init__(self, name, typ):
            self.name, self.typ = name, typ

        def Init(self, handler=None, queueLen=0):
            subs[self.name] = handler

        def Close(self):
            pass

    class ChannelPublisher:  # would be used by a non-read-only script; the probe must replace it before anything
        def __init__(self, *a, **k):
            calls["publishers"] += 1

    def ChannelFactoryInitialize(domain=0, iface=None):
        calls["init"] = (domain, iface)

    channel = types.ModuleType("unitree_sdk2py.core.channel")
    channel.ChannelSubscriber, channel.ChannelPublisher, channel.ChannelFactoryInitialize = ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
    mods = {"unitree_sdk2py": types.ModuleType("unitree_sdk2py"), "unitree_sdk2py.core": types.ModuleType("unitree_sdk2py.core"),
            "unitree_sdk2py.core.channel": channel, "unitree_sdk2py.idl": types.ModuleType("unitree_sdk2py.idl")}
    for pkg, names in (("unitree_hg", ["LowState_", "IMUState_", "HandState_", "BmsState_", "MainBoardState_", "LowCmd_", "HandCmd_"]),
                       ("unitree_go", ["SportModeState_"])):  # WirelessController_ deliberately missing -> 'skipped'
        for sub in (f"unitree_sdk2py.idl.{pkg}", f"unitree_sdk2py.idl.{pkg}.msg"):
            mods[sub] = types.ModuleType(sub)
        dds = types.ModuleType(f"unitree_sdk2py.idl.{pkg}.msg.dds_")
        for n in names:
            setattr(dds, n, type(n, (), {}))
        mods[f"unitree_sdk2py.idl.{pkg}.msg.dds_"] = dds
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    return channel, subs, calls


def test_sdk_probe_with_fake_sdk(fake_sdk, tmp_path):
    sp = load("sdk_probe.py")
    channel, subs, calls = fake_sdk

    def feed(seconds):  # stands in for time.sleep: deliver 500 Hz lowstate and 100 Hz hand state for ~0.2 s
        t_end = time.monotonic() + 0.2
        k = 0
        while time.monotonic() < t_end:
            subs["rt/lowstate"](_lowstate(1000 + 2 * k))
            if k % 5 == 0:
                subs["rt/dex3/left/state"](_hand())
            k += 1
            time.sleep(0.002)

    rc = sp.main([FLAG, "--iface", "eth9", "--seconds", "0.2", "--out", str(tmp_path), "--watch-command-topics"], sdk=channel, sleep=feed)
    assert rc == 0
    assert calls["init"] == (0, "eth9") and calls["publishers"] == 0
    with pytest.raises(RuntimeError):
        channel.ChannelPublisher("rt/lowcmd", object)  # guard installed
    rep = json.loads(next(tmp_path.glob("sdk_probe_*.json")).read_text())
    assert "rt/lowstate" in rep["received"] and "rt/dex3/left/state" in rep["received"]
    assert "rt/lowcmd" in rep["silent"] and "rt/wirelesscontroller" in rep["skipped"]
    ls = rep["topics"]["rt/lowstate"]
    assert ls["rate"]["count"] > 20 and ls["rate"]["mean_hz"] > 50
    fs = ls["first_summary"]
    assert fs["motor_slots"] == 35 and fs["motors_live"] == 29 and fs["mode_machine"] == 5
    assert fs["motorstate_nonzero"] == {"3": 7} or fs["motorstate_nonzero"] == {3: 7}
    assert abs(fs["imu"]["accel_norm"] - 9.81) < 1e-3 and fs["imu"]["quat_norm"] == 1.0
    assert ls["tick"]["step_p50"] == 2 and ls["tick"]["non_increasing"] == 0
    hs = rep["topics"]["rt/dex3/left/state"]["first_summary"]
    assert hs["motor_count"] == 7 and hs["press_sensor_count"] == 9 and hs["press_values_per_sensor"] == 12


def test_sdk_probe_summaries_and_stamp():
    sp = load("sdk_probe.py")
    b = sp.summarize_bms(_NS(cell_vol=[3900, 3950] + [0] * 38, soc=87, soh=99, current=-1200, bmsvoltage=[51000, 0, 0],
                             temperature=[30] * 12, cycle=12, version_high=1, version_low=2))
    assert b["soc"] == 87 and b["cells_nonzero"] == 2 and b["cell_min"] == 3900
    assert sp.stamp_seconds(_NS(stamp=_NS(sec=10, nanosec=500_000_000))) == 10.5
    assert sp.stamp_seconds(_NS()) is None


# ------------------------------------------------------------------ ros_topic_probe pure parts
def test_ros_probe_topic_choice_and_stamp():
    rp = load("ros_topic_probe.py")
    nt = [("/rosout", ["rcl_interfaces/msg/Log"]), ("/lowstate", ["unitree_hg/msg/LowState"]),
          ("/camera/color/image_raw", ["sensor_msgs/msg/Image"]), ("/x", [])]
    assert [n for n, _ in rp.choose_topics(nt, None, None, 10)] == ["/camera/color/image_raw", "/lowstate"]
    assert [n for n, _ in rp.choose_topics(nt, None, "camera", 10)] == ["/lowstate"]
    assert [n for n, _ in rp.choose_topics(nt, "rosout", None, 10)] == ["/rosout"]
    msg = _NS(header=_NS(stamp=_NS(sec=5, nanosec=250_000_000)))
    assert rp.stamp_of(msg) == 5.25
    rec = rp.RawRecorder(cls=object, deserialize=lambda raw, cls: msg)
    for _ in range(25):
        rec(b"\x00" * 100)
    r = rec.report(1.0)
    assert r["rate"]["count"] == 25 and r["bytes"]["mean"] == 100 and r["age"]["count"] >= 20


# ------------------------------------------------------------------ camera_probe
def test_camera_helpers(tmp_path):
    cp = load("camera_probe.py")
    sysfs = tmp_path / "v4l"
    for i, name in ((0, "Intel(R) RealSense(TM) Depth Ca"), (2, "USB HDR Camera"), (10, "x")):
        d = sysfs / f"video{i}"
        d.mkdir(parents=True)
        (d / "name").write_text(name + "\n")
        (d / "index").write_text("0\n")
    nodes = cp.list_v4l2_nodes(str(sysfs), "/dev")
    assert [n["node"] for n in nodes] == ["/dev/video0", "/dev/video2", "/dev/video10"]
    assert cp.is_realsense_name(nodes[0]["name"]) and not cp.is_realsense_name(nodes[1]["name"])
    assert cp.fps_from_hw_ms([0.0, 33.333, 66.666, 100.0]) == pytest.approx(30.0, rel=1e-3)
    intr = _NS(width=640, height=480, fx=600.0, fy=601.0, ppx=320.0, ppy=240.0, model="distortion.brown_conrady", coeffs=[0, 0, 0, 0, 0])
    assert cp.intrinsics_to_dict(intr)["fy"] == 601.0
    p = cp.write_ppm(str(tmp_path / "a.ppm"), [[(255, 0, 0), (0, 255, 0)]])
    assert Path(p).read_bytes().startswith(b"P6 2 1 255\n")


def test_camera_probe_v4l2_with_fake_cv2(tmp_path):
    cp = load("camera_probe.py")
    np = pytest.importorskip("numpy")

    class Cap:
        def __init__(self, node, api):
            self.node = node

        def isOpened(self):
            return self.node == "/dev/video2"

        def read(self):
            time.sleep(0.001)
            return True, np.zeros((4, 6, 3), dtype="uint8")

        def get(self, prop):
            return {3: 6, 4: 4, 5: 30.0, 6: 0x47504A4D}.get(prop, 0)

        def release(self):
            pass

    fake_cv2 = types.SimpleNamespace(VideoCapture=Cap, CAP_V4L2=200, CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4,
                                     CAP_PROP_FPS=5, CAP_PROP_FOURCC=6)
    nodes = [{"node": "/dev/video0", "name": "Intel RealSense"}, {"node": "/dev/video1", "name": "meta"}, {"node": "/dev/video2", "name": "USB HDR Camera"}]
    res = cp.probe_v4l2(nodes, str(tmp_path), frames=5, skip_realsense=True, max_nodes=4, cv2=fake_cv2)
    assert "skipped" in res[0] and res[1]["opened"] is False
    assert res[2]["width"] == 6 and res[2]["fourcc"] == "MJPG" and res[2]["frames_read"] == 5


def test_camera_probe_main_without_devices(tmp_path):
    r = subprocess.run([sys.executable, str(INV / "camera_probe.py"), FLAG, "--out", str(tmp_path), "--no-realsense",
                        "--sysfs-root", str(tmp_path / "none")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    rep = json.loads(next(tmp_path.glob("camera_probe_*/camera_probe.json")).read_text())
    assert rep["v4l2_nodes"] == [] and "realsense" not in rep


# ------------------------------------------------------------------ inventory.sh against stub tools
def _stub(bin_dir: Path, name: str, body: str):
    p = bin_dir / name
    p.write_text("#!/usr/bin/env bash\n" + body + "\n")
    p.chmod(0o755)


def test_inventory_sh_sections_with_stubs(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir, "ros2", 'case "$1 $2" in\n'
          '"topic list") echo "/lowstate [unitree_hg/msg/LowState]"; echo "/lf/lowstate [unitree_hg/msg/LowState]"; '
          'echo "/utlidar/cloud [sensor_msgs/msg/PointCloud2]";;\n'
          '"topic info") echo "Type: x"; echo "Publisher count: 1";;\n'
          '"node list") echo "/fake_node";;\n"service list") echo "/fake_srv [std_srvs/srv/Empty]";;\n'
          '"pkg list") echo "unitree_hg";;\n*) echo "stub $*";;\nesac')
    _stub(bin_dir, "cyclonedds", 'echo "stub cyclonedds $*"')
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", ROS_DISTRO="humble", ROS_DOMAIN_ID="0",
               RMW_IMPLEMENTATION="rmw_cyclonedds_cpp")
    out = tmp_path / "reports"
    r = subprocess.run(["bash", str(INV / "inventory.sh"), FLAG, "--out", str(out), "--no-ping", "--timeout", "5",
                        "--sections", "1,4,5,7"], capture_output=True, text=True, timeout=180, env=env)
    assert r.returncode == 0, r.stderr
    folders = [p for p in out.iterdir() if p.is_dir()]
    assert len(folders) == 1 and (out / (folders[0].name + ".tar.gz")).exists()
    f = folders[0]
    for name in ("_meta.txt", "_exit_codes.txt", "01_os_release.txt", "04_ip_addr.txt", "05_timedatectl.txt",
                 "07_ros2_topics.txt", "07_ros2_topic_info.txt"):
        assert (f / name).exists(), name
    assert not (f / "08_pip_list.txt").exists() and not (f / "09_lsusb.txt").exists()  # sections not asked for
    assert "/lowstate [unitree_hg/msg/LowState]" in (f / "07_ros2_topics.txt").read_text()

    # and the facts step on top of it, with fake probe outputs placed in the folder
    sdk = {"args": {"iface": "eth0", "domain": 0}, "received": ["rt/lowstate", "rt/dex3/left/state"], "silent": ["rt/lf/bmsstate"],
           "topics": {"rt/lowstate": {"received": True, "rate": {"mean_hz": 498.7, "period_ms": {"p99": 2.6, "std_jitter": 0.2}, "gaps_over_2x_median": 0},
                                      "first_summary": {"motor_slots": 35, "motors_live": 29, "mode_machine": 5, "mode_pr": 0,
                                                        "imu": {"quat_norm": 1.0, "accel_norm": 9.8}}},
                      "rt/dex3/left/state": {"received": True, "rate": {"mean_hz": 100.0}, "first_summary": {"motor_count": 7, "press_sensor_count": 9}},
                      "rt/lf/bmsstate": {"received": False, "rate": {}}}}
    (f / "sdk_probe_20261106T100000Z.json").write_text(json.dumps(sdk))
    rf = subprocess.run([sys.executable, str(INV / "report_to_facts.py"), FLAG, str(f)], capture_output=True, text=True, timeout=60)
    assert rf.returncode == 0, rf.stderr
    md = (f / "robot_facts_summary.md").read_text()
    assert "| U14 | rt/lowstate rate and jitter seen on the dev computer | **resolved** |" in md
    assert "| U15 |" in md and "| U20 |" in md
    y = (f / "robot_facts.candidates.yaml").read_text()
    assert '"state.lowstate_hz":' in y and "498.7" in y and "validated: false" in y
    assert "/utlidar/cloud" in y  # LiDAR topic candidate picked up from the ROS list
    yaml = pytest.importorskip("yaml")
    assert yaml.safe_load(y)["facts"]["state.motors_live"]["value"] == 29


# ------------------------------------------------------------------ report_to_facts parsing on Jetson-like text
def _w(folder: Path, name: str, out: str):
    (folder / f"{name}.txt").write_text(f"$ cmd\n---\n{out}")


def test_report_to_facts_jetson_like(tmp_path):
    rt = load("report_to_facts.py")
    f = tmp_path / "inv_ubuntu_20261106T100000Z"
    f.mkdir()
    _w(f, "01_model", "NVIDIA Jetson Orin NX Engine Board\x00\ncompatible\n")
    _w(f, "01_nv_tegra_release", "# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, EABI: aarch64, DATE: Wed Jan  8 01:49:37 UTC 2025\n")
    _w(f, "01_os_release", 'PRETTY_NAME="Ubuntu 22.04.5 LTS"\n')
    _w(f, "01_uname", "Linux ubuntu 5.15.148-tegra #1 SMP PREEMPT Tue Jan 7 17:14:21 PST 2025 aarch64 aarch64 aarch64 GNU/Linux\n")
    _w(f, "01_dpkg_nvidia", "nvidia-l4t-core 36.4.3-20250107174145\ntensorrt 10.3.0.30-1+cuda12.5\n")
    _w(f, "01_cuda_trt", "Cuda compilation tools, release 12.6, V12.6.68\n")
    _w(f, "02_nvpmodel", "NV Power Mode: MAXN\n0\n")
    _w(f, "03_memory", "MemTotal:       15655792 kB\n")
    _w(f, "03_df", "Filesystem Type Size Used Avail Use% Mounted on\n/dev/nvme0n1p1 ext4 116G 40G 70G 37% /\n")
    _w(f, "04_ip_addr", "lo UNKNOWN 127.0.0.1/8\neth0 UP 192.168.123.164/24 fe80::1/64\nwlan0 DOWN\n\nfull\n")
    _w(f, "04_ip_neigh", "192.168.123.161 dev eth0 lladdr aa:bb:cc:dd:ee:01 REACHABLE\n192.168.123.120 dev eth0 lladdr aa:bb:cc:dd:ee:02 STALE\n")
    _w(f, "04_ping_192.168.123.161", "3 packets transmitted, 3 received, 0% packet loss\n")
    _w(f, "05_timedatectl", "NTPSynchronized=no\nTimezone=Asia/Shanghai\n")
    _w(f, "05_time_services", "chrony inactive disabled\nsystemd-timesyncd active enabled\n")
    _w(f, "06_services_running", "unitree_video.service loaded active running x\nssh.service loaded active running y\n")
    facts = rt.build_facts(str(f), {"sdk": None, "ros": None, "camera": None})
    v = {k: d["value"] for k, d in facts.d.items()}
    assert v["compute.module"] == "NVIDIA Jetson Orin NX Engine Board"
    assert v["software.l4t"] == "R36.4.3" and v["software.jetpack"].startswith("6.2")
    assert v["software.tensorrt"].startswith("10.3") and v["software.cuda"] == "12.6"
    assert v["compute.ram_gb"] == pytest.approx(14.93, abs=0.01) and v["compute.nvpmodel"] == "MAXN"
    assert v["network.robot_iface"] == "eth0" and v["network.robot_ip"] == "192.168.123.164/24"
    assert v["network.reach_192_168_123_161"] is True
    assert v["lidar.candidate_neighbours"] == ["192.168.123.120 aa:bb:cc:dd:ee:02"]
    assert v["time.sync_service"] == ["systemd-timesyncd"] and v["time.ntp_synchronized"] == "no"
    assert v["vendor.services"] == ["unitree_video.service"]
    unk = rt.resolve_unknowns(facts)
    assert unk["U02"]["status"] == "resolved" and unk["U24"]["status"] == "partial" and unk["U14"]["status"] == "open"


def test_unknown_ids_match_the_doc():
    rt = load("report_to_facts.py")
    doc = DOC.read_text()
    doc_ids = set(re.findall(r"^\| (U\d\d) \|", doc, re.M))
    assert doc_ids == set(rt.UNKNOWNS), f"doc-only: {doc_ids - set(rt.UNKNOWNS)}, script-only: {set(rt.UNKNOWNS) - doc_ids}"
