"""Check that the simulated robot's DDS stays on the loopback interface. No Kit; two processes.

    python3 scripts/isaac/dds_loopback_check.py [--out result.json]

1. A `sim` link and a `controller` link in two processes exchange rt/lowstate and rt/lowcmd (values and CRC arrive intact).
2. Every packet either process sends goes to 127.0.0.1 (CycloneDDS's own packet trace; the receive sockets are listed too: CycloneDDS
   binds them to 0.0.0.0).
3. A listener on the same domain and topics bound to a NON-loopback interface, without the discovery tag, receives nothing.
4. DdsLink refuses any interface other than `lo` and the domains 0 and 1.
There is no kernel packet capture here (it needs root on this machine); 2 and 3 are the evidence that nothing leaves the host.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402

from humanoid_arc_bme.isaac import dds  # noqa: E402
from humanoid_arc_bme.isaac.lowlevel import LowCmd, LowState, SimState  # noqa: E402


def controller(q, trace, domain):
    link = dds.DdsLink("controller", domain, trace_file=trace)
    got, t0 = None, time.time()
    while time.time() - t0 < 8.0:
        st, sim = link.read_state()
        if st is not None and st.tick >= 5:
            got = (st, sim)
            break
        time.sleep(0.02)
    for k in range(50):
        link.write_cmd(LowCmd(np.arange(29) * 0.01, np.zeros(29), np.ones(29) * 0.5, np.full(29, 100.0), np.full(29, 2.5), (np.full(7, 0.1), np.full(7, -0.1))))
        time.sleep(0.02)
    q.put({"state_q3": None if got is None else float(got[0].q[3]), "quat": None if got is None else [float(v) for v in got[0].quat],
           "sim_lin": None if got is None or got[1] is None else [float(v) for v in got[1].lin_vel_b], "sockets": dds.socket_report()})


def outside_listener(q, iface, domain):
    """Same domain and topic, but on a real network interface: must stay silent."""
    m = dds._imports()
    cfg = f'<CycloneDDS><Domain Id="any"><General><Interfaces><NetworkInterface name="{iface}"/></Interfaces></General></Domain></CycloneDDS>'
    try:
        dom = m["Domain"](domain, cfg)
        dp = m["DomainParticipant"](domain)
        r1 = m["DataReader"](dp, m["Topic"](dp, "rt/lowcmd", m["LowCmd_"]))
        r2 = m["DataReader"](dp, m["Topic"](dp, "rt/lowstate", m["LowState_"]))
        n, t0 = 0, time.time()
        while time.time() - t0 < 6.0:
            n += sum(type(x).__name__ != "InvalidSample" for x in r1.take(N=32) + r2.take(N=32))
            time.sleep(0.05)
        q.put({"iface": iface, "received": n})
        del dom
    except Exception as exc:  # noqa: BLE001
        q.put({"iface": iface, "received": 0, "error": f"{type(exc).__name__}: {exc}"[:200]})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--outside-iface", default=None, help="a non-loopback interface for the silent-listener test (default: the first one that is up)")
    a = ap.parse_args()
    iface = a.outside_iface
    if iface is None:
        import subprocess
        for line in subprocess.run(["ip", "-br", "addr"], capture_output=True, text=True).stdout.splitlines():
            p = line.split()
            if p[0] != "lo" and p[1] == "UP" and len(p) > 2 and "." in p[2]:
                iface = p[0].split("@")[0]
                break
    ctx = mp.get_context("spawn")
    q1, q2 = ctx.Queue(), ctx.Queue()
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="dds_check_"))
    domain = dds.pick_domain()
    pc = ctx.Process(target=controller, args=(q1, str(tmp / "controller.trace"), domain))
    po = ctx.Process(target=outside_listener, args=(q2, iface, domain))
    pc.start()
    po.start()
    link = dds.DdsLink("sim", domain, trace_file=str(tmp / "sim.trace"))
    cmd, t0, tick = None, time.time(), 0
    while time.time() - t0 < 7.0:
        tick += 1
        st = LowState(q=np.arange(29) * 0.1, quat=np.array([0.5, 0.5, 0.5, 0.5]), tick=tick, stamp=tick * 0.02)
        link.write_state(st, SimState(np.array([0.1, 0.2, 0.3]), np.zeros(3), tick * 0.02))
        c = link.read_cmd()
        cmd = c if c is not None else cmd
        time.sleep(0.02)
    rc, ro = q1.get(timeout=10), q2.get(timeout=10)
    pc.join(5)
    po.join(5)
    mine = dds.socket_report()
    dest = {n: dds.trace_destinations(str(tmp / f"{n}.trace")) for n in ("sim", "controller")}
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    loop_only = all(a.startswith("127.") for d in dest.values() for a in d) and all(dest.values())
    refusals = []
    for kw in ({"interface": "eno1"}, {"domain": 0}, {"domain": 1}):
        try:
            dds.DdsLink("sim", **kw)
            refusals.append(False)
        except ValueError:
            refusals.append(True)
    res = {"domain": domain, "state_received_by_controller": rc["state_q3"] is not None and abs(rc["state_q3"] - 0.3) < 1e-6 and rc["quat"] == [0.5, 0.5, 0.5, 0.5],
           "sim_state_received": rc["sim_lin"] == [0.1, 0.2, 0.3],
           "cmd_received_by_sim": cmd is not None and abs(float(cmd.q[5]) - 0.05) < 1e-6 and float(cmd.kp[0]) == 100.0 and cmd.hands is not None
           and abs(float(cmd.hands[1][0]) + 0.1) < 1e-6, "crc_errors": link.crc_errors,
           "packet_destinations": dest, "all_packets_to_loopback": loop_only,
           "receive_sockets_sim": mine["local"], "receive_sockets_controller": rc["sockets"]["local"], "outside_listener": ro, "refuses_other_interface_and_domains": refusals}
    res["listener_built"] = iface is not None and "error" not in ro
    res["pass"] = bool(res["listener_built"] and res["state_received_by_controller"] and res["sim_state_received"] and res["cmd_received_by_sim"] and loop_only and ro["received"] == 0 and all(refusals) and link.crc_errors == 0)
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
