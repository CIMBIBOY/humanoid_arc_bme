"""The deploy stand-in as its own process: ZMQ planner topics in, rt/lowstate in, rt/lowcmd out, at 50 Hz of wall-clock time.

    python -m humanoid_arc_bme.isaac.standin_main --zmq tcp://127.0.0.1:5556 --domain <d> [--policy agile_locomotion.pt]

This is where the real `gear_sonic_deploy --input-type zmq_manager` would run instead; the simulator side cannot tell the difference.
The process ends when it receives `command` stop = 1 (the deploy's stop-control), like the real control loop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zmq", default="tcp://127.0.0.1:5556", help="the planner publisher's endpoint (the stack binds it; this side connects)")
    ap.add_argument("--domain", type=int, default=None, help="DDS domain of the simulator (default: $HUMANOID_ARC_DDS_DOMAIN; one of the two must be set: "
                    "picking a free domain here would pick a different one from the simulator's and the two would never meet)")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--max-seconds", type=float, default=3600.0)
    a = ap.parse_args()
    if not a.zmq.startswith(("tcp://127.", "tcp://localhost", "ipc://")):
        ap.error("the stand-in talks to a local planner publisher only")
    import zmq

    from humanoid_arc_bme.isaac.dds import DdsLink
    from humanoid_arc_bme.isaac.rates import CONTROL_HZ
    from humanoid_arc_bme.isaac.standin import DeployStandIn
    from humanoid_arc_bme.robots.assets import AGILE_LOCOMOTION_POLICY

    import os
    if a.domain is None and not os.environ.get("HUMANOID_ARC_DDS_DOMAIN"):
        ap.error("--domain (or $HUMANOID_ARC_DDS_DOMAIN) is required")
    deploy = DeployStandIn(a.policy or str(AGILE_LOCOMOTION_POLICY))
    link = DdsLink("controller", a.domain)
    sub = zmq.Context.instance().socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(a.zmq)
    period, t0, n, last_tick, last_print = 1.0 / CONTROL_HZ, time.monotonic(), 0, -1, 0.0
    print(json.dumps({"standin": "ready", "domain": link.domain, "zmq": a.zmq}), flush=True)
    while not deploy.exited and time.monotonic() - t0 < a.max_seconds:
        now = time.monotonic()
        while True:
            try:
                deploy.on_message(sub.recv(zmq.NOBLOCK), now)
            except zmq.Again:
                break
        state, sim = link.read_state()
        if state is not None and state.tick != last_tick:              # act on fresh robot state only (a silent robot gets no commands)
            last_tick = state.tick
            cmd = deploy.step(state, sim, now)
            if cmd is not None:
                link.write_cmd(cmd)
        if now - last_print > 2.0:
            last_print = now
            print(json.dumps({"t": round(now - t0, 1), **deploy.status()}), flush=True)
        n += 1
        time.sleep(max(0.0, t0 + n * period - time.monotonic()))
    print(json.dumps({"standin": "exit", **deploy.status()}), flush=True)
    link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
