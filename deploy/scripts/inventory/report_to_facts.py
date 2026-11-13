#!/usr/bin/env python3
"""Turn an inventory report folder into robot-fact candidates (YAML) and a markdown summary. Offline; reads files only.

Inputs: the folder written by inventory.sh (inv_<host>_<utc>/ or its .tar.gz unpacked), and the JSON files of
sdk_probe.py, ros_topic_probe.py and camera_probe.py. Those are looked for inside the folder and next to it (newest
of each kind wins), or given with --probe PATH (repeatable).

Outputs (into the report folder unless --out-dir): robot_facts.candidates.yaml (every fact with value, source file,
tag MEASURED), robot_facts_summary.md (the facts plus which UNKNOWN ids of docs/robot-inventory.md are now
resolved, partly resolved or still open). With --write-config PATH it also writes the YAML there (refuses to overwrite
an existing file unless --force); review it before committing.

Usage:
    python3 report_to_facts.py --i-understand-read-only REPORT_DIR [--probe FILE]... [--out-dir DIR]
                               [--write-config config/robot_facts.yaml [--force]]
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_common import require_ack, robot_iface_from_ip_text, utc_stamp  # noqa: E402

# The UNKNOWN register. Ids and wording must match the table in docs/robot-inventory.md (a test checks it).
# Each entry: question, and the fact keys that answer it (all present -> resolved, some -> partial).
UNKNOWNS = {
    "U01": ("Jetson module variant and RAM", ["compute.module", "compute.ram_gb"]),
    "U02": ("L4T / JetPack release on the dev computer", ["software.l4t", "software.jetpack"]),
    "U03": ("Ubuntu release and kernel", ["software.os", "software.kernel"]),
    "U04": ("CUDA / TensorRT / cuDNN versions installed", ["software.cuda", "software.tensorrt"]),
    "U05": ("ROS 2 distro installed and the RMW configured", ["software.ros_distro", "dds.rmw_implementation"]),
    "U06": ("unitree_sdk2py and cyclonedds (python) installed, versions", ["python.unitree_sdk2py", "python.cyclonedds"]),
    "U07": ("onnxruntime / TensorRT python bindings and their GPU providers", ["python.onnxruntime", "python.tensorrt"]),
    "U08": ("pyrealsense2 / OpenCV in python", ["python.pyrealsense2", "python.cv2"]),
    "U09": ("power mode (nvpmodel) and clocks", ["compute.nvpmodel"]),
    "U10": ("disk: NVMe model, size, free space", ["storage.root_size", "storage.root_avail", "storage.nvme_model"]),
    "U11": ("robot-network interface name and the dev computer's IP", ["network.robot_iface", "network.robot_ip"]),
    "U12": ("motion computer 192.168.123.161 reachable from the dev computer", ["network.reach_192_168_123_161"]),
    "U13": ("DDS domain id and CycloneDDS config used on the dev computer", ["dds.ros_domain_id", "dds.cyclonedds_uri"]),
    "U14": ("rt/lowstate rate and jitter seen on the dev computer", ["state.lowstate_hz", "state.lowstate_p99_ms"]),
    "U15": ("LowState motor slots and live motors (29-DoF body?)", ["state.motor_slots", "state.motors_live"]),
    "U16": ("mode_machine / mode_pr values reported by LowState", ["state.mode_machine", "state.mode_pr"]),
    "U17": ("IMU fields: quaternion order/norm, accel magnitude, rpy", ["state.imu_quat_norm", "state.imu_accel_norm"]),
    "U18": ("torso secondary IMU topic (rt/secondary_imu) present", ["state.secondary_imu_hz"]),
    "U19": ("battery state topic and fields", ["state.bms_topic"]),
    "U20": ("Dex3 state topic names, motor and pressure-sensor counts", ["hands.state_topics", "hands.motor_count", "hands.press_sensor_count"]),
    "U21": ("SportModeState / odometry topic on the G1", ["state.sport_topics"]),
    "U22": ("head camera: model, interface, resolution, fps", ["camera.devices"]),
    "U23": ("RealSense depth camera present, intrinsics, IMU", ["camera.realsense_model", "camera.realsense_color_intrinsics"]),
    "U24": ("LiDAR model, address and topic", ["lidar.model", "lidar.ros_topics", "lidar.candidate_neighbours"]),
    "U25": ("clock sync on the dev computer (NTP / chrony)", ["time.ntp_synchronized", "time.sync_service"]),
    "U26": ("vendor services running on the dev computer", ["vendor.services"]),
    "U27": ("who publishes rt/lowcmd / rt/arm_sdk while the vendor controller runs", ["commands.lowcmd_seen"]),
    "U28": ("microphone array capture device", ["audio.capture_devices"]),
    "U29": ("real-time kernel and CPU governor", ["compute.realtime_kernel", "compute.cpu_governor"]),
    "U30": ("ROS 2 topics the robot exposes on the dev computer", ["ros.topic_count", "ros.unitree_topics"]),
}

# JetPack <-> L4T. R36.4.4 = 6.2.1 is [V] on this workstation (docs/plan/B-robot-g1.md); the others are from NVIDIA's
# release list as remembered, not opened: the dpkg `nvidia-jetpack` version, when present, overrides this table.
L4T_TO_JETPACK = {"R36.4.4": "6.2.1", "R36.4.3": "6.2", "R36.4.0": "6.1", "R36.3.0": "6.0", "R35.6.0": "5.1.4",
                  "R35.5.0": "5.1.3", "R35.4.1": "5.1.2", "R35.3.1": "5.1.1", "R35.2.1": "5.1", "R35.1.0": "5.0.2"}


# ------------------------------------------------------------------ parsing helpers (tested)
def body(folder: str, name: str) -> str | None:
    """Output part of an inventory file (after the '---' line), or None if the file is missing."""
    p = os.path.join(folder, name if name.endswith(".txt") else name + ".txt")
    if not os.path.exists(p):
        return None
    with open(p, errors="replace") as f:
        txt = f.read()
    return txt.split("\n---\n", 1)[1] if "\n---\n" in txt else txt


def exit_codes(folder: str) -> dict:
    out = {}
    p = os.path.join(folder, "_exit_codes.txt")
    if os.path.exists(p):
        for line in open(p):
            parts = line.split()
            if len(parts) == 2:
                out[parts[0]] = int(parts[1])
    return out


def parse_l4t(text: str | None) -> str | None:
    m = re.search(r"#\s*R(\d+)\s*\(release\),\s*REVISION:\s*([\d.]+)", text or "")
    if not m:
        return None
    rev = m.group(2)
    if rev.count(".") == 0:
        rev += ".0"
    return f"R{m.group(1)}.{rev}"


def parse_os(text: str | None) -> str | None:
    m = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', text or "", re.M)
    return m.group(1) if m else None


def parse_meminfo_gb(text: str | None) -> float | None:
    m = re.search(r"MemTotal:\s+(\d+)\s+kB", text or "")
    return round(int(m.group(1)) / 1024 / 1024, 2) if m else None


def parse_dpkg(text: str | None) -> dict:
    out = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and not line.startswith("$"):
            out[parts[0]] = parts[1]
    return out


def parse_nvcc(text: str | None) -> str | None:
    m = re.search(r"release\s+([\d.]+)", text or "")
    return m.group(1) if m else None


def parse_nvpmodel(text: str | None) -> str | None:
    m = re.search(r"NV Power Mode:\s*(\S+)", text or "")
    return m.group(1) if m else None


def parse_df_root(text: str | None) -> tuple[str | None, str | None, str | None]:
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) >= 7 and parts[-1] == "/":
            return parts[0], parts[2], parts[4]
    return None, None, None


def parse_nvme_model(text: str | None) -> str | None:
    m = re.search(r"nvme\d+ model=([^\n]*?) fw=", text or "")
    return m.group(1).strip() if m and m.group(1).strip() else None


def ping_ok(text: str | None) -> bool | None:
    if text is None:
        return None
    m = re.search(r"(\d+) (?:packets )?received", text)
    return bool(m and int(m.group(1)) > 0)


def parse_env(text: str | None) -> dict:
    out = {}
    for line in (text or "").splitlines():
        if "=" in line and not line.startswith("$"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def parse_ros_topics(text: str | None) -> list[tuple[str, str]]:
    out = []
    for line in (text or "").splitlines():
        m = re.match(r"^(/\S+)\s+\[([^\]]+)\]", line.strip())
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def parse_timedatectl(text: str | None) -> dict:
    out = {}
    for line in (text or "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def parse_time_services(text: str | None) -> list[str]:
    return [line.split()[0] for line in (text or "").splitlines() if len(line.split()) >= 2 and line.split()[1] == "active"]


def parse_services(text: str | None) -> list[str]:
    names = []
    for line in (text or "").splitlines():
        parts = line.split()
        if parts and parts[0].endswith(".service"):
            names.append(parts[0])
    return names


def parse_neigh(text: str | None, known=("192.168.123.161", "192.168.123.164"), prefix="192.168.123.") -> list[str]:
    """Hosts on the robot subnet other than the two documented computers (LiDAR / switch / laptop candidates)."""
    out = []
    for line in (text or "").splitlines():
        parts = line.split()
        if parts and parts[0].startswith(prefix) and parts[0] not in known and "FAILED" not in line:
            mac = parts[parts.index("lladdr") + 1] if "lladdr" in parts else "?"
            out.append(f"{parts[0]} {mac}")
    return out


def parse_arecord(text: str | None) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.startswith("card ")]


def parse_python_reports(folder: str) -> list[dict]:
    res = []
    for p in sorted(glob.glob(os.path.join(folder, "08_python_*.txt"))):
        b = body(folder, os.path.basename(p))
        try:
            j = json.loads(b[b.index("{"):])
            j["_file"] = os.path.basename(p)
            res.append(j)
        except Exception:
            continue
    return res


def newest(paths: list[str]) -> str | None:
    paths = [p for p in paths if os.path.exists(p)]
    return max(paths, key=os.path.getmtime) if paths else None


def find_probes(folder: str, extra: list[str]) -> dict:
    found = {}
    pats = {"sdk": "sdk_probe_*.json", "ros": "ros_topic_probe_*.json", "camera": "camera_probe*/camera_probe.json"}
    for kind, pat in pats.items():
        cands = glob.glob(os.path.join(folder, pat)) + glob.glob(os.path.join(os.path.dirname(os.path.abspath(folder)), pat))
        cands += [p for p in extra if re.search(pat.replace("*", ".*").replace("/", "[/\\\\]") + "$", p)]
        found[kind] = newest(cands)
    return found


# ------------------------------------------------------------------ fact building
class Facts:
    def __init__(self):
        self.d: dict = {}

    def put(self, key: str, value, source: str):
        if value is None or value == [] or value == {}:
            return
        self.d[key] = {"value": value, "source": source, "tag": "MEASURED"}

    def has(self, key: str) -> bool:
        return key in self.d


def build_facts(folder: str, probes: dict) -> Facts:
    F = Facts()
    model_lines = [ln.strip("\x00 ") for ln in (body(folder, "01_model") or "").splitlines()
                   if ln.strip("\x00 ") and "No such file" not in ln]
    F.put("compute.module", model_lines[0] if model_lines else None, "01_model.txt")
    F.put("compute.ram_gb", parse_meminfo_gb(body(folder, "03_memory")), "03_memory.txt")
    l4t = parse_l4t(body(folder, "01_nv_tegra_release"))
    F.put("software.l4t", l4t, "01_nv_tegra_release.txt")
    dp = parse_dpkg(body(folder, "01_dpkg_nvidia"))
    jp = dp.get("nvidia-jetpack")
    if jp:
        F.put("software.jetpack", jp.split("+")[0], "01_dpkg_nvidia.txt (nvidia-jetpack)")
    elif l4t in L4T_TO_JETPACK:
        F.put("software.jetpack", L4T_TO_JETPACK[l4t] + " (from L4T table; nvidia-jetpack package absent)", "01_nv_tegra_release.txt")
    F.put("software.os", parse_os(body(folder, "01_os_release")), "01_os_release.txt")
    un = body(folder, "01_uname") or ""
    F.put("software.kernel", (un.split()[2] if len(un.split()) > 2 else None), "01_uname.txt")
    F.put("compute.realtime_kernel", ("PREEMPT_RT" in un or "realtime: 1" in un) if un else None, "01_uname.txt")
    gov = sorted(set(re.findall(r"gov=(\S+)", body(folder, "01_cpu_governor") or "")))
    F.put("compute.cpu_governor", gov, "01_cpu_governor.txt")
    F.put("software.cuda", parse_nvcc(body(folder, "01_cuda_trt")) or dp.get("cuda-toolkit") or dp.get("nvidia-l4t-cuda"), "01_cuda_trt.txt / 01_dpkg_nvidia.txt")
    trt = dp.get("tensorrt") or dp.get("libnvinfer10") or dp.get("libnvinfer8") or next((v for k, v in dp.items() if k.startswith("libnvinfer")), None)
    F.put("software.tensorrt", trt, "01_dpkg_nvidia.txt")
    F.put("software.cudnn", next((v for k, v in dp.items() if "cudnn" in k), None), "01_dpkg_nvidia.txt")
    F.put("compute.nvpmodel", parse_nvpmodel(body(folder, "02_nvpmodel")), "02_nvpmodel.txt")
    jc = body(folder, "02_jetson_clocks_show")
    if jc and "skipped" not in jc:
        F.put("compute.jetson_clocks_show", jc.strip().splitlines()[:6], "02_jetson_clocks_show.txt")
    dev, size, avail = parse_df_root(body(folder, "03_df"))
    F.put("storage.root_device", dev, "03_df.txt")
    F.put("storage.root_size", size, "03_df.txt")
    F.put("storage.root_avail", avail, "03_df.txt")
    F.put("storage.nvme_model", parse_nvme_model(body(folder, "03_nvme")), "03_nvme.txt")

    ipbr = body(folder, "04_ip_addr") or ""
    iface = robot_iface_from_ip_text(ipbr)
    F.put("network.robot_iface", iface, "04_ip_addr.txt")
    if iface:
        m = re.search(rf"^{re.escape(iface)}\S*\s+\S+\s+.*?(192\.168\.123\.\d+/\d+)", ipbr, re.M)
        F.put("network.robot_ip", m.group(1) if m else None, "04_ip_addr.txt")
    F.put("network.interfaces", [ln.split()[0] + " " + " ".join(ln.split()[2:3]) for ln in ipbr.split("\n\n")[0].splitlines() if ln.strip() and not ln.startswith("$")], "04_ip_addr.txt")
    for ip in ("192.168.123.161", "192.168.123.164"):
        F.put(f"network.reach_{ip.replace('.', '_')}", ping_ok(body(folder, f"04_ping_{ip}")), f"04_ping_{ip}.txt")
    F.put("lidar.candidate_neighbours", parse_neigh(body(folder, "04_ip_neigh")), "04_ip_neigh.txt (other hosts on 192.168.123.x; LiDAR candidates)")
    # lidar.model is never measured here: it needs the label / manual (B0.6); U24 stays at most 'partial'.

    env = parse_env(body(folder, "07_env"))
    rosenv = body(folder, "07_ros_env") or ""
    m = re.search(r"ROS_DISTRO=(\S*)", rosenv)
    F.put("software.ros_distro", (m.group(1) if m and m.group(1) else env.get("ROS_DISTRO")), "07_ros_env.txt / 07_env.txt")
    m = re.search(r"RMW_IMPLEMENTATION=(\S*)", rosenv)
    F.put("dds.rmw_implementation", (m.group(1) if m and m.group(1) else env.get("RMW_IMPLEMENTATION")) or "unset (ROS 2 default)", "07_ros_env.txt")
    m = re.search(r"ROS_DOMAIN_ID=(\S*)", rosenv)
    F.put("dds.ros_domain_id", (m.group(1) if m and m.group(1) else env.get("ROS_DOMAIN_ID")) or "unset (0)", "07_ros_env.txt")
    F.put("dds.cyclonedds_uri", env.get("CYCLONEDDS_URI") or "unset", "07_env.txt")

    topics = parse_ros_topics(body(folder, "07_ros2_topics"))
    if body(folder, "07_ros2_topics") is not None:
        F.put("ros.topic_count", len(topics), "07_ros2_topics.txt")
        F.put("ros.unitree_topics", [f"{n} [{t}]" for n, t in topics if "unitree" in t or re.search(r"lowstate|lowcmd|sportmode|dex3|arm_sdk|bms|wireless", n)] or ["none"], "07_ros2_topics.txt")
        lid = [f"{n} [{t}]" for n, t in topics if "PointCloud" in t or "livox" in t.lower() or "lidar" in n.lower()]
        F.put("lidar.ros_topics", lid or ["none"], "07_ros2_topics.txt")
        cams = [f"{n} [{t}]" for n, t in topics if "sensor_msgs/msg/Image" in t or "CompressedImage" in t]
        F.put("camera.ros_image_topics", cams or ["none"], "07_ros2_topics.txt")

    tdc = parse_timedatectl(body(folder, "05_timedatectl"))
    F.put("time.ntp_synchronized", tdc.get("NTPSynchronized"), "05_timedatectl.txt")
    F.put("time.timezone", tdc.get("Timezone"), "05_timedatectl.txt")
    F.put("time.sync_service", parse_time_services(body(folder, "05_time_services")) or ["none active"], "05_time_services.txt")
    ch = body(folder, "05_chrony") or ""
    m = re.search(r"System time\s*:\s*([\d.]+ seconds \w+)", ch)
    F.put("time.chrony_offset", m.group(1) if m else None, "05_chrony.txt")

    svcs = parse_services(body(folder, "06_services_running"))
    interest = [s for s in svcs if re.search(r"unitree|ros|dds|docker|realsense|livox|lidar|teleimager|image|camera|motion|robot|sport|ota|tailscale|chrony", s, re.I)]
    F.put("vendor.services", interest or ["none matching"], "06_services_running.txt")
    F.put("vendor.services_running_count", len(svcs) or None, "06_services_running.txt")
    dock = body(folder, "06_docker") or ""
    if "CONTAINER ID" in dock:
        F.put("vendor.docker_containers", [ln for ln in dock.splitlines()[1:] if ln.strip() and "REPOSITORY" not in ln][:20], "06_docker.txt")

    F.put("audio.capture_devices", parse_arecord(body(folder, "09_audio")), "09_audio.txt")
    usb = body(folder, "09_lsusb") or ""
    F.put("usb.camera_like", [ln.strip() for ln in usb.splitlines() if re.search(r"RealSense|Intel Corp.*0b|Camera|camera|UVC|Video|Webcam|8086:0b", ln)], "09_lsusb.txt")
    rs = body(folder, "09_rs_enumerate_short") or ""
    F.put("camera.rs_enumerate", [ln.strip() for ln in rs.splitlines() if "Intel RealSense" in ln][:4], "09_rs_enumerate_short.txt")

    pys = parse_python_reports(folder)
    if pys:
        sysp = next((p for p in pys if p.get("executable", "").startswith("/usr/bin/")), pys[0])
        F.put("python.system_version", sysp.get("version", "").split()[0], sysp["_file"])
        for mod, key in (("unitree_sdk2py", "python.unitree_sdk2py"), ("cyclonedds", "python.cyclonedds"), ("onnxruntime", "python.onnxruntime"),
                         ("tensorrt", "python.tensorrt"), ("pyrealsense2", "python.pyrealsense2"), ("cv2", "python.cv2"),
                         ("rclpy", "python.rclpy"), ("numpy", "python.numpy"), ("torch", "python.torch")):
            per = {}
            for p in pys:
                mi = p.get("modules", {}).get(mod, {})
                if mi.get("ok"):
                    v = mi.get("version") or p.get("dists", {}).get(mod) or "installed"
                    if mod == "onnxruntime":
                        v = f"{v} providers={mi.get('providers')}"
                    if mod == "unitree_sdk2py":
                        v = f"{p.get('dists', {}).get('unitree_sdk2py') or p.get('dists', {}).get('unitree-sdk2py') or v} at {mi.get('file')}"
                    per[p["executable"]] = v
            F.put(key, per or "not importable in any inspected python", ",".join(p["_file"] for p in pys))

    # ---- probes
    if probes.get("sdk"):
        s = json.load(open(probes["sdk"]))
        src = os.path.basename(probes["sdk"])
        tp = s.get("topics", {})
        F.put("dds.sdk_probe_iface", s.get("args", {}).get("iface"), src)
        F.put("dds.sdk_probe_domain", s.get("args", {}).get("domain"), src)
        F.put("dds.sdk_topics_received", s.get("received"), src)
        F.put("dds.sdk_topics_silent", s.get("silent"), src)
        ls = tp.get("rt/lowstate", {})
        if ls.get("received"):
            F.put("state.lowstate_hz", ls["rate"].get("mean_hz"), src)
            F.put("state.lowstate_p99_ms", ls["rate"].get("period_ms", {}).get("p99"), src)
            F.put("state.lowstate_jitter_ms", ls["rate"].get("period_ms", {}).get("std_jitter"), src)
            F.put("state.lowstate_gaps", ls["rate"].get("gaps_over_2x_median"), src)
            fs = ls.get("first_summary", {})
            F.put("state.motor_slots", fs.get("motor_slots"), src)
            F.put("state.motors_live", fs.get("motors_live"), src)
            F.put("state.mode_machine", fs.get("mode_machine"), src)
            F.put("state.mode_pr", fs.get("mode_pr"), src)
            F.put("state.lowstate_version", fs.get("version"), src)
            imu = fs.get("imu", {})
            F.put("state.imu_quat_norm", imu.get("quat_norm"), src)
            F.put("state.imu_accel_norm", imu.get("accel_norm"), src)
            F.put("state.motor_temp_max", fs.get("motor_temp_max"), src)
            if ls.get("tick"):
                F.put("state.tick_units_per_s", ls["tick"].get("units_per_s"), src)
        else:
            F.put("state.lowstate_hz", 0 if "rt/lowstate" in tp else None, src)
        si = tp.get("rt/secondary_imu", {})
        if "rt/secondary_imu" in tp:
            F.put("state.secondary_imu_hz", si["rate"].get("mean_hz") if si.get("received") else "silent", src)
        bms = [t for t in ("rt/lf/bmsstate", "rt/bmsstate") if tp.get(t, {}).get("received")]
        if any(t in tp for t in ("rt/lf/bmsstate", "rt/bmsstate")):
            F.put("state.bms_topic", bms or "none of the guessed names (rt/lf/bmsstate, rt/bmsstate) received", src)
            if bms:
                F.put("state.bms_first", tp[bms[0]].get("first_summary"), src)
        hand_topics = [t for t in tp if "dex3" in t and t.endswith("/state")]
        if hand_topics:
            rec = [t for t in hand_topics if tp[t].get("received")]
            F.put("hands.state_topics", {t: tp[t]["rate"].get("mean_hz") for t in rec} or "none received", src)
            if rec:
                F.put("hands.motor_count", tp[rec[0]].get("first_summary", {}).get("motor_count"), src)
                F.put("hands.press_sensor_count", tp[rec[0]].get("first_summary", {}).get("press_sensor_count"), src)
        sport = [t for t in tp if "sportmode" in t or "odommode" in t]
        if sport:
            F.put("state.sport_topics", {t: tp[t]["rate"].get("mean_hz") for t in sport if tp[t].get("received")} or "none received", src)
        if "rt/lowcmd" in tp or "rt/arm_sdk" in tp:
            F.put("commands.lowcmd_seen", {t: tp[t]["rate"].get("mean_hz") if tp[t].get("received") else "silent" for t in ("rt/lowcmd", "rt/arm_sdk") if t in tp}, src)
        if s.get("error"):
            F.put("dds.sdk_probe_error", s["error"], src)

    if probes.get("ros"):
        r = json.load(open(probes["ros"]))
        src = os.path.basename(probes["ros"])
        F.put("ros.probe_rates_hz", {n: v["rate"].get("mean_hz") for n, v in r.get("topics", {}).items() if v.get("received")}, src)
        F.put("ros.probe_type_missing", sorted(r.get("skipped", {})), src)
        if not F.has("ros.topic_count"):
            F.put("ros.topic_count", len(r.get("discovered", [])), src)

    if probes.get("camera"):
        c = json.load(open(probes["camera"]))
        src = os.path.relpath(probes["camera"], os.path.dirname(os.path.dirname(probes["camera"])))
        devs = []
        rsd = c.get("realsense", {}).get("devices", [])
        for d in rsd:
            st = d.get("stream", {})
            devs.append(f"RealSense {d.get('name')} sn={d.get('serial_number')} fw={d.get('firmware_version')} usb={d.get('usb_type_descriptor')} color_hw_fps={st.get('color_hw_fps')}")
        for v in c.get("v4l2", []):
            if v.get("opened"):
                devs.append(f"V4L2 {v.get('node')} '{v.get('name')}' {v.get('width')}x{v.get('height')} {v.get('fourcc')} {v.get('arrival', {}).get('mean_hz')} Hz")
        F.put("camera.devices", devs or ["none opened"], src)
        if rsd:
            d = rsd[0]
            F.put("camera.realsense_model", d.get("name"), src)
            F.put("camera.realsense_has_imu", d.get("has_motion_module"), src)
            streams = d.get("stream", {}).get("streams", {})
            F.put("camera.realsense_color_intrinsics", streams.get("stream.color", {}).get("intrinsics"), src)
            F.put("camera.realsense_depth_intrinsics", streams.get("stream.depth", {}).get("intrinsics"), src)
            F.put("camera.realsense_depth_scale_m", d.get("stream", {}).get("depth_scale_m"), src)
            F.put("camera.realsense_extrinsics_depth_to_color", d.get("stream", {}).get("extrinsics_depth_to_color"), src)
            F.put("camera.realsense_timestamp_domain", d.get("stream", {}).get("color_timestamp_domain"), src)
            if d.get("stream_error"):
                F.put("camera.realsense_stream_error", d["stream_error"], src)
        elif c.get("realsense", {}).get("available") is not None:
            F.put("camera.realsense_model", "none found", src)
    return F


def resolve_unknowns(F: Facts) -> dict:
    out = {}
    for uid, (q, keys) in UNKNOWNS.items():
        have = [k for k in keys if F.has(k)]
        status = "resolved" if len(have) == len(keys) else ("partial" if have else "open")
        out[uid] = {"question": q, "status": status, "facts": have, "missing": [k for k in keys if k not in have]}
    return out


# ------------------------------------------------------------------ writers
def _y(v, ind: int) -> str:
    pad = "  " * ind
    if isinstance(v, dict):
        if not v:
            return " {}"
        return "".join(f"\n{pad}{json.dumps(str(k))}:{_y(x, ind + 1)}" for k, x in v.items())
    if isinstance(v, list):
        if not v:
            return " []"
        return "".join(f"\n{pad}-{_y(x, ind + 1)}" for x in v)
    if isinstance(v, bool):
        return " true" if v else " false"
    if v is None:
        return " null"
    if isinstance(v, (int, float)):
        return f" {v}"
    return " " + json.dumps(str(v))


def to_yaml(header: str, data: dict) -> str:
    """Minimal YAML writer (stdlib only; strings JSON-quoted, which YAML accepts)."""
    lines = "".join(f"# {h}\n" for h in header.splitlines())
    return lines + "".join(f"{k}:{_y(v, 1)}\n" for k, v in data.items())


def summary_md(folder: str, F: Facts, unk: dict, probes: dict, codes: dict) -> str:
    L = [f"# Robot facts from `{os.path.basename(os.path.abspath(folder))}`", "",
         f"Generated {utc_stamp()} by deploy/scripts/inventory/report_to_facts.py. All values MEASURED on the robot's dev computer;",
         "review before copying anything into config/. Probe files used: "
         + ", ".join(f"{k}={os.path.basename(v) if v else 'none'}" for k, v in probes.items()), "",
         "## UNKNOWN register (docs/robot-inventory.md)", "", "| id | question | status | missing |", "|---|---|---|---|"]
    for uid, u in unk.items():
        L.append(f"| {uid} | {u['question']} | **{u['status']}** | {', '.join(u['missing']) or '-'} |")
    n = {s: sum(1 for u in unk.values() if u["status"] == s) for s in ("resolved", "partial", "open")}
    L += ["", f"resolved {n['resolved']}, partial {n['partial']}, open {n['open']} of {len(unk)}.", "", "## Facts", "",
          "| key | value | source |", "|---|---|---|"]
    for k, v in sorted(F.d.items()):
        val = json.dumps(v["value"]) if not isinstance(v["value"], str) else v["value"]
        if len(val) > 160:
            val = val[:157] + "..."
        L.append(f"| {k} | {val.replace('|', '/')} | {v['source']} |")
    failed = sorted(k for k, c in codes.items() if c != 0)
    L += ["", "## Inventory commands that returned non-zero", "", (", ".join(failed) or "none") +
          " (127 = tool not installed, 124 = timeout; open the matching .txt for the message)", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    require_ack(argv, "report_to_facts.py")
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--i-understand-read-only", action="store_true")
    ap.add_argument("report_dir")
    ap.add_argument("--probe", action="append", default=[])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--write-config", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    folder = a.report_dir
    if not os.path.isdir(folder):
        print(f"not a folder: {folder} (unpack the .tar.gz first: tar xzf inv_*.tar.gz)", file=sys.stderr)
        return 1
    probes = find_probes(folder, a.probe)
    F = build_facts(folder, probes)
    unk = resolve_unknowns(F)
    codes = exit_codes(folder)
    out_dir = a.out_dir or folder
    os.makedirs(out_dir, exist_ok=True)
    meta = body(folder, "_meta") or ""
    header = ("Robot fact CANDIDATES, generated by deploy/scripts/inventory/report_to_facts.py.\n"
              f"Source report: {os.path.basename(os.path.abspath(folder))}\n"
              "Every value is MEASURED on the robot; 'validated' stays false until you have reviewed it.")
    data = {"validated": False, "generated_utc": utc_stamp(),
            "report": {"folder": os.path.basename(os.path.abspath(folder)),
                       "inventory_version": (re.search(r"version: (.*)", meta).group(1) if "version:" in meta else None),
                       "probes": {k: (os.path.basename(v) if v else None) for k, v in probes.items()}},
            "facts": F.d,
            "unknowns": {k: v["status"] for k, v in unk.items()}}
    ytxt = to_yaml(header, data)
    yp = os.path.join(out_dir, "robot_facts.candidates.yaml")
    open(yp, "w").write(ytxt)
    mp = os.path.join(out_dir, "robot_facts_summary.md")
    open(mp, "w").write(summary_md(folder, F, unk, probes, codes))
    if a.write_config:
        if os.path.exists(a.write_config) and not a.force:
            print(f"refusing to overwrite {a.write_config} (use --force after reviewing)", file=sys.stderr)
            return 1
        open(a.write_config, "w").write(ytxt)
        print(f"[report_to_facts] wrote {a.write_config}")
    n = {s: sum(1 for u in unk.values() if u["status"] == s) for s in ("resolved", "partial", "open")}
    print(f"[report_to_facts] {len(F.d)} facts; UNKNOWNs resolved {n['resolved']}, partial {n['partial']}, open {n['open']}")
    print(f"[report_to_facts] wrote {yp}\n[report_to_facts] wrote {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
