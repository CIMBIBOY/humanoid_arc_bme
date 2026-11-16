"""The robot's DDS interface on the loopback interface: rt/lowstate, rt/lowcmd and the Dex3 topics with Unitree's own types.

`DdsLink("sim")` is the simulated robot's end (publishes state, subscribes commands); `DdsLink("controller")` is the deploy's end.
Both have the methods of `lowlevel.LoopbackLink`, so the simulator does not know which link it has.

SAFETY. The real G1 listens for rt/lowcmd on its own network (DDS domain 0; Unitree's simulator uses domain 1). A simulated command
must never reach it, so this module refuses anything but the loopback interface: the CycloneDDS configuration names `lo` as the
only interface, turns multicast off, lists 127.0.0.1 as the only peer, sets a discovery tag no other DDS program has, and takes a domain on which nothing on this machine listens. With `trace_file` CycloneDDS logs every packet it sends; `trace_destinations()` extracts the addresses so
a test can assert that all of them are 127.0.0.1 (scripts/isaac/dds_loopback_check.py). Note: CycloneDDS binds its RECEIVE sockets to
0.0.0.0 (it has no option to bind them to one address in this version), so a host that knew the port could send to them; nothing is
ever sent to such a host, and a participant without the tag is ignored. The lease is long (300 s) and discovery is re-announced every
second: a process stopped for a while by a job guard (it happened: 40 s) must not lose its peer for the rest of the run.

Needs `cyclonedds` and `unitree_sdk2py` (not installed with the package). If they live outside site-packages, list their folders in
$HUMANOID_ARC_DDS_PATHS (os.pathsep separated).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

from humanoid_arc_bme.isaac.lowlevel import N_HAND, N_MOTORS, LowCmd, LowState, SimState

INTERFACE = "lo"
TAG = "humanoid-arc-sim"         # DDSI domain tag: participants with another tag (or none: ROS 2, a robot) are ignored at discovery
CANDIDATE_DOMAINS = range(60, 100)   # above 100 the DDS port range runs into the kernel's ephemeral ports
_CONFIG = """<CycloneDDS><Domain Id="any"><General><Interfaces><NetworkInterface name="{iface}" multicast="false"/></Interfaces>
<AllowMulticast>false</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>40</MaxAutoParticipantIndex>
<Tag>{tag}</Tag><Peers><Peer address="127.0.0.1"/></Peers><SPDPInterval>1s</SPDPInterval><LeaseDuration>300s</LeaseDuration></Discovery>{trace}</Domain></CycloneDDS>"""


def domain_ports(domain: int) -> range:
    return range(7400 + 250 * domain, 7400 + 250 * (domain + 1))      # DDSI-RTPS well-known ports of a domain


def pick_domain(candidates=CANDIDATE_DOMAINS) -> int:
    """The first candidate domain on which NOTHING on this machine listens (other users run ROS 2 and DDS here: domain 77 was taken).
    Both ends of a link must be given the same result; `$HUMANOID_ARC_DDS_DOMAIN` carries it to a child process."""
    env = os.environ.get("HUMANOID_ARC_DDS_DOMAIN")
    if env:
        if not env.isdigit():
            raise ValueError("HUMANOID_ARC_DDS_DOMAIN must be a non-negative integer")
        return int(env)
    out = subprocess.run(["ss", "-H", "-uan"], capture_output=True, text=True, timeout=10).stdout
    used = {int(line.split()[3].rsplit(":", 1)[1]) for line in out.splitlines() if line.split()[3].rsplit(":", 1)[1].isdigit()}
    for d in candidates:
        if not used & set(domain_ports(d)):
            return d
    raise RuntimeError("no free DDS domain among the candidates")


for _p in filter(None, os.environ.get("HUMANOID_ARC_DDS_PATHS", "").split(os.pathsep)):       # also where pyzmq may live (Isaac's Python)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _imports():
    from cyclonedds.domain import Domain, DomainParticipant
    from cyclonedds.pub import DataWriter
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic
    from unitree_sdk2py.idl import default as D
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC
    return locals()


class DdsLink:
    def __init__(self, side: str, domain: int | None = None, trace_file: str | None = None, interface: str = INTERFACE):
        if side not in ("sim", "controller"):
            raise ValueError("side: sim | controller")
        if interface != "lo":
            raise ValueError("the simulated robot's DDS is loopback only (a simulated rt/lowcmd must never reach a real robot)")
        self.domain = domain = pick_domain() if domain is None else int(domain)
        if domain in (0, 1):                                   # checked AFTER the environment variable and the picker had their say
            raise ValueError("domains 0 (real robot) and 1 (Unitree's simulator) are not used here")
        m = self._m = _imports()
        trace = f"<Tracing><Verbosity>finest</Verbosity><OutputFile>{trace_file}</OutputFile></Tracing>" if trace_file else ""   # every packet, for the check
        self._domain = m["Domain"](domain, _CONFIG.format(iface=interface, trace=trace, tag=TAG))
        self._dp = m["DomainParticipant"](domain)
        self.side, self.crc, self.crc_errors = side, m["CRC"](), 0
        types = {"rt/lowstate": m["LowState_"], "rt/lowcmd": m["LowCmd_"], "rt/sim_state": m["String_"],
                 "rt/dex3/left/cmd": m["HandCmd_"], "rt/dex3/right/cmd": m["HandCmd_"],
                 "rt/dex3/left/state": m["HandState_"], "rt/dex3/right/state": m["HandState_"]}
        out = [t for t in types if (t.endswith("state")) == (side == "sim")]
        self._topics = {t: m["Topic"](self._dp, t, ty) for t, ty in types.items()}
        self._w = {t: m["DataWriter"](self._dp, self._topics[t]) for t in out}
        self._r = {t: m["DataReader"](self._dp, self._topics[t]) for t in types if t not in out}
        self._last: dict = {}
        self._cmd: LowCmd | None = None
        self._hands = [None, None]

    def _newest(self, topic: str):
        got = [x for x in self._r[topic].take(N=32) if type(x).__name__ != "InvalidSample"]   # disposal / liveliness notices carry no data
        if got:
            self._last[topic] = got[-1]
            return got[-1], True
        return self._last.get(topic), False

    # ------------------------------------------------------------------ controller -> robot
    def write_cmd(self, cmd: LowCmd) -> None:
        m = self._m["D"].unitree_hg_msg_dds__LowCmd_()
        m.mode_pr, m.mode_machine = 0, 0
        for i in range(N_MOTORS):
            c = m.motor_cmd[i]
            c.mode, c.q, c.dq, c.tau, c.kp, c.kd = 1, float(cmd.q[i]), float(cmd.dq[i]), float(cmd.tau[i]), float(cmd.kp[i]), float(cmd.kd[i])
        m.crc = self.crc.Crc(m)
        self._w["rt/lowcmd"].write(m)
        for side, q in zip(("left", "right"), cmd.hands if cmd.hands is not None else (None, None)):
            if q is not None:
                h = self._m["D"].unitree_hg_msg_dds__HandCmd_()
                for i in range(N_HAND):
                    h.motor_cmd[i].mode, h.motor_cmd[i].q = 1, float(q[i])
                self._w[f"rt/dex3/{side}/cmd"].write(h)

    def read_cmd(self) -> LowCmd | None:
        """The newest command (a NEW object only when a new message arrived). A message with a wrong CRC is dropped, as on the robot."""
        msg, fresh = self._newest("rt/lowcmd")
        for k, side in enumerate(("left", "right")):
            h, hf = self._newest(f"rt/dex3/{side}/cmd")
            if hf:
                self._hands[k] = np.array([h.motor_cmd[i].q for i in range(N_HAND)])
        if fresh:
            if self.crc.Crc(msg) != msg.crc:
                self.crc_errors += 1
                return self._cmd
            mc = msg.motor_cmd
            self._cmd = LowCmd(*(np.array([getattr(mc[i], f) for i in range(N_MOTORS)]) for f in ("q", "dq", "tau", "kp", "kd")),
                               None if self._hands[0] is None and self._hands[1] is None else tuple(self._hands))
        return self._cmd

    # ------------------------------------------------------------------ robot -> controller
    def write_state(self, state: LowState, sim: SimState | None = None) -> None:
        m = self._m["D"].unitree_hg_msg_dds__LowState_()
        m.tick = int(state.tick)
        for i in range(N_MOTORS):
            s = m.motor_state[i]
            s.q, s.dq, s.tau_est = float(state.q[i]), float(state.dq[i]), float(state.tau[i])
        m.imu_state.quaternion = [float(v) for v in state.quat]            # (w, x, y, z), as on the robot
        m.imu_state.gyroscope = [float(v) for v in state.gyro]
        m.imu_state.accelerometer = [float(v) for v in state.accel]
        m.crc = self.crc.Crc(m)
        self._w["rt/lowstate"].write(m)
        for side, q in zip(("left", "right"), state.hands_q):
            h = self._m["D"].unitree_hg_msg_dds__HandState_()
            for i in range(N_HAND):
                h.motor_state[i].q = float(q[i])
            self._w[f"rt/dex3/{side}/state"].write(h)
        if sim is not None:                                                # sim-only topic, never present on a robot
            self._w["rt/sim_state"].write(self._m["String_"](json.dumps({"lin_vel_b": [float(v) for v in sim.lin_vel_b],
                                                                           "pos_w": [float(v) for v in sim.pos_w], "stamp": sim.stamp, "state_stamp": state.stamp})))

    def read_state(self) -> tuple[LowState | None, SimState | None]:
        msg, _ = self._newest("rt/lowstate")
        if msg is None:
            return None, None
        ms = msg.motor_state
        hands = []
        for side in ("left", "right"):
            h, _ = self._newest(f"rt/dex3/{side}/state")
            hands.append(np.zeros(N_HAND) if h is None else np.array([h.motor_state[i].q for i in range(N_HAND)]))
        s, _ = self._newest("rt/sim_state")
        d = json.loads(s.data) if s is not None else None
        st = LowState(np.array([ms[i].q for i in range(N_MOTORS)]), np.array([ms[i].dq for i in range(N_MOTORS)]),
                      np.array([ms[i].tau_est for i in range(N_MOTORS)]), np.array(msg.imu_state.quaternion, float),
                      np.array(msg.imu_state.gyroscope, float), np.array(msg.imu_state.accelerometer, float), tuple(hands), int(msg.tick),
                      d["state_stamp"] if d else 0.0)
        return st, (SimState(np.array(d["lin_vel_b"]), np.array(d["pos_w"]), d["stamp"]) if d else None)

    def close(self) -> None:
        self._w.clear()
        self._r.clear()


def trace_destinations(trace_file: str) -> dict:
    """Addresses CycloneDDS sent packets to, from its own trace: {address: count}. The check asserts they are all loopback."""
    import collections
    import re
    c: collections.Counter = collections.Counter()
    with open(trace_file, errors="replace") as f:
        for line in f:
            for addr in re.findall(r"udp/(\[?[0-9a-fA-F:.]+\]?):\d+", line):
                c[addr] += 1
    return dict(c)


def socket_report(pid: int | None = None) -> dict:
    """UDP sockets of a process (`ss -uanp`): {"local": [addresses], "outside_loopback": [those not in 127.0.0.0/8 or ::1]}."""
    pid = pid or os.getpid()
    out = subprocess.run(["ss", "-H", "-uanp"], capture_output=True, text=True, timeout=10).stdout
    local = sorted({line.split()[3] for line in out.splitlines() if f"pid={pid}," in line})
    return {"pid": pid, "local": local, "outside_loopback": [a for a in local if not (a.startswith("127.") or a.startswith("[::1]"))]}
