"""Read-only checks of the dev computer against the stack the plan requires (B2.5-B2.7).

Expected: JetPack 6.2 (Jetson Linux R36), TensorRT exactly 10.7, MAXN power mode, Docker,
ROS 2 Humble, the robot's 192.168.123.x network, Tailscale. Each check returns ok, fail or
unknown; unknown means "could not be read here", never a pass.
"""

import os
import re
from dataclasses import dataclass

from .runner import run

EXPECTED_TENSORRT = (10, 7)
ROBOT_NET_PREFIX = "192.168.123."


@dataclass
class Check:
    name: str
    status: str  # ok | fail | unknown
    detail: str


def parse_tegra_release(text):
    """'# R36 (release), REVISION: 4.3 ...' -> (36, '4.3'); None if not parseable."""
    m = re.search(r"R(\d+)\s*\(release\),\s*REVISION:\s*([\d.]+)", text)
    return (int(m.group(1)), m.group(2)) if m else None


def parse_tensorrt_header(text):
    """Read NV_TENSORRT_MAJOR/MINOR/PATCH from NvInferVersion.h -> (10, 7, 0) or None."""
    vals = {}
    for key in ("MAJOR", "MINOR", "PATCH"):
        m = re.search(rf"#define\s+NV_TENSORRT_{key}\s+(\d+)", text)
        if not m:
            return None
        vals[key] = int(m.group(1))
    return vals["MAJOR"], vals["MINOR"], vals["PATCH"]


def parse_ipv4_addrs(ip_a_oneline):
    """`ip -4 -o addr` output -> list of (ifname, address)."""
    out = []
    for line in ip_a_oneline.splitlines():
        m = re.match(r"\d+:\s+(\S+)\s+inet\s+(\d+\.\d+\.\d+\.\d+)", line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def check_jetpack(read=lambda p: open(p).read()):
    try:
        text = read("/etc/nv_tegra_release")
    except OSError:
        return Check("jetson_linux", "unknown", "/etc/nv_tegra_release not present (not a Jetson)")
    rel = parse_tegra_release(text)
    if rel is None:
        return Check("jetson_linux", "unknown", f"unparseable: {text[:80]!r}")
    ok = rel[0] == 36
    return Check("jetson_linux", "ok" if ok else "fail", f"R{rel[0]} rev {rel[1]} (JetPack 6.x is R36)")


def check_tensorrt(env=os.environ, read=lambda p: open(p).read()):
    root = env.get("TensorRT_ROOT")
    if not root:
        return Check("tensorrt", "fail", "TensorRT_ROOT is not set")
    try:
        text = read(os.path.join(root, "include", "NvInferVersion.h"))
    except OSError:
        return Check("tensorrt", "fail", f"{root}/include/NvInferVersion.h not readable")
    v = parse_tensorrt_header(text)
    if v is None:
        return Check("tensorrt", "unknown", "version macros not found in header")
    ok = v[:2] == EXPECTED_TENSORRT
    return Check("tensorrt", "ok" if ok else "fail", f"{v[0]}.{v[1]}.{v[2]} (required exactly {EXPECTED_TENSORRT[0]}.{EXPECTED_TENSORRT[1]})")


def check_power_mode(runner=run):
    r = runner(["nvpmodel", "-q"])
    if r.code != 0:
        return Check("power_mode", "unknown", r.err.strip() or "nvpmodel unavailable")
    return Check("power_mode", "ok" if "MAXN" in r.out else "fail", r.out.strip().replace("\n", " "))


def check_tool(name, argv, runner=run):
    r = runner(argv)
    if r.code == 127:
        return Check(name, "fail", f"{argv[0]} not found")
    return Check(name, "ok" if r.code == 0 else "fail", (r.out or r.err).strip().splitlines()[0] if (r.out or r.err).strip() else f"exit {r.code}")


def check_ros(env=os.environ, exists=os.path.isdir):
    distro = env.get("ROS_DISTRO")
    if distro == "humble" or exists("/opt/ros/humble"):
        return Check("ros2_humble", "ok", f"ROS_DISTRO={distro or 'unset'}, /opt/ros/humble={exists('/opt/ros/humble')}")
    return Check("ros2_humble", "unknown", "not on the host; the plan uses the Docker environment (B2.7), check inside it")


def check_robot_network(runner=run):
    r = runner(["ip", "-4", "-o", "addr"])
    if r.code != 0:
        return Check("robot_network", "unknown", "ip command failed")
    hits = [(i, a) for i, a in parse_ipv4_addrs(r.out) if a.startswith(ROBOT_NET_PREFIX)]
    if hits:
        return Check("robot_network", "ok", ", ".join(f"{i}={a}" for i, a in hits))
    return Check("robot_network", "fail", f"no interface in {ROBOT_NET_PREFIX}x")


def check_tailscale(runner=run):
    r = runner(["tailscale", "status"])
    if r.code == 127:
        return Check("tailscale", "fail", "tailscale not installed")
    return Check("tailscale", "ok" if r.code == 0 else "fail", (r.out or r.err).strip().splitlines()[0] if (r.out or r.err).strip() else f"exit {r.code}")


def run_all(runner=run, env=os.environ):
    return [
        check_jetpack(),
        check_tensorrt(env),
        check_power_mode(runner),
        check_tool("docker", ["docker", "--version"], runner),
        check_ros(env),
        check_robot_network(runner),
        check_tailscale(runner),
    ]


def format_report(checks):
    mark = {"ok": "OK     ", "fail": "FAIL   ", "unknown": "UNKNOWN"}
    lines = [f"{mark[c.status]} {c.name:15s} {c.detail}" for c in checks]
    bad = sum(c.status == "fail" for c in checks)
    lines.append(f"{len(checks)} checks, {bad} failed, {sum(c.status == 'unknown' for c in checks)} unknown")
    return "\n".join(lines)
