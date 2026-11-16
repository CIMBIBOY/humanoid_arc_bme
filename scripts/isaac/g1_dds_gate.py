"""Gate WP3 over the real interfaces: the simulated G1 in Isaac, the controller in ANOTHER process, DDS and ZMQ on loopback.

    python.sh scripts/isaac/g1_dds_gate.py --out result.json --python <interpreter with cyclonedds, unitree_sdk2py, torch, zmq>

Isaac publishes rt/lowstate and applies rt/lowcmd (DdsLink, simulated time paced to wall-clock time); the deploy stand-in runs as a
child process at its own 50 Hz; the stack's GrootPlannerController publishes planner messages on a real ZMQ PUB socket. The child is
started and killed by PID. Phases: connect, stand, walk, stop, sender silent (planner timeout), stop-control (the controller exits).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--python", default=sys.executable, help="interpreter for the stand-in process")
ap.add_argument("--speed", type=float, default=0.2)
args, _ = ap.parse_known_args()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid_arc_bme.isaac.app import make_app  # noqa: E402

app = make_app("g1_dds_gate")

import numpy as np  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> dict:
    from isaacsim.core.api import World

    from humanoid_arc_bme.isaac import dds, g1_model
    from humanoid_arc_bme.isaac.rates import PHYSICS_HZ, STACK_HZ
    from humanoid_arc_bme.isaac.robot import SimG1
    from humanoid_arc_bme.isaac.sim import G1Sim
    from humanoid_arc_bme.onboard.api.safety import load_limits
    from humanoid_arc_bme.onboard.paths import DEFAULT_LIMITS
    from humanoid_arc_bme.onboard.runtime.groot_controller import GrootPlannerController, ZmqPubTransport, build_command_message
    from humanoid_arc_bme.robots.assets import G1_29DOF_USD

    world = World(physics_dt=1.0 / PHYSICS_HZ, rendering_dt=1.0 / PHYSICS_HZ, stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
    robot = SimG1(str(G1_29DOF_USD))
    world.reset()
    robot.start()
    domain, port = dds.pick_domain(), free_port()
    link = dds.DdsLink("sim", domain)
    sim = G1Sim(world, robot, link=link, controller=None, realtime=True)
    tx = ZmqPubTransport(f"tcp://127.0.0.1:{port}")
    ctrl = GrootPlannerController(tx, enabled=True, threaded=False, clock=lambda: sim.t, limits=load_limits(DEFAULT_LIMITS))
    child = subprocess.Popen([args.python, "-m", "humanoid_arc_bme.isaac.standin_main", "--zmq", f"tcp://127.0.0.1:{port}", "--domain", str(domain),
                              "--max-seconds", "300"], cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             env={**{k: v for k, v in os.environ.items() if k not in ("HUMANOID_ARC_DDS_PATHS", "PYTHONHOME", "PYTHONPATH",
                                                                                         "LD_LIBRARY_PATH", "LD_PRELOAD")},
                                  "PYTHONPATH": str(ROOT)})        # the child is a plain Python: none of Kit's library paths
    res = {"domain": domain, "zmq_port": port, "child_pid": child.pid, "phases": {}}
    want = {"cmd": (0.0, 0.0, 0.0), "send": True}
    lat, last_pub = [], [0.0]

    def tick(s):
        if want["send"]:
            ctrl.step(s.t, 1.0 / STACK_HZ, None, want["cmd"])

    def run(seconds):
        prev = sim._last_cmd
        for _ in range(round(seconds * PHYSICS_HZ)):
            if sim.n % (PHYSICS_HZ // STACK_HZ) == 0:
                tick(sim)
            if sim.n % sim.decim == 0:
                last_pub[0] = time.perf_counter()
            sim.step()
            if sim._last_cmd is not prev:                                     # a new rt/lowcmd arrived: time since the last state publish
                prev = sim._last_cmd
                lat.append(time.perf_counter() - last_pub[0])
        return robot.truth()

    try:
        t0 = time.time()
        while sim._last_cmd is None and time.time() - t0 < 60.0:              # wait for the controller process (model load, DDS discovery)
            run(0.1)
        res["phases"]["connect"] = {"first_lowcmd_after_s": round(time.time() - t0, 2), "robot_fell_while_waiting": robot.truth()["fallen"],
                                    "pass": sim._last_cmd is not None}
        if sim._last_cmd is None:
            child.terminate()
            raise RuntimeError("no rt/lowcmd from the controller process; its output:\n" + (child.communicate(timeout=10)[0] or "")[-1500:])
        # The robot hung passive while the controller started: put it back on its feet, as an operator does before starting control.
        robot.art.set_world_poses(np.array([[0.0, 0.0, g1_model.ROOT_HEIGHT]]), np.array([[1.0, 0.0, 0.0, 0.0]]))
        robot.art.set_velocities(np.zeros((1, 6)))
        q0 = np.zeros(len(robot.dof_names))
        q0[robot.i_motor] = g1_model.default_pose()
        robot.art.set_joint_positions(q0[None])
        robot.art.set_joint_velocities(np.zeros((1, len(robot.dof_names))))
        ctrl.on_arm(sim.t, None)
        a = run(2.0)
        b = run(5.0)
        res["phases"]["stand"] = {"pelvis_z": round(b["pos"][2], 3), "drift_m": round(math.dist(a["pos"][:2], b["pos"][:2]), 3), "pass": bool(not b["fallen"] and math.dist(a["pos"][:2], b["pos"][:2]) < 0.10)}
        want["cmd"] = (args.speed, 0.0, 0.0)
        run(2.0)
        b = robot.truth()
        c = run(5.0)
        v = math.dist(b["pos"][:2], c["pos"][:2]) / 5.0
        res["phases"]["walk"] = {"commanded": args.speed, "achieved": round(v, 3), "pass": bool(not c["fallen"] and 0.5 * args.speed <= v <= 1.5 * args.speed)}
        want["cmd"] = (0.0, 0.0, 0.0)
        a = robot.truth()
        b = run(2.0)
        res["phases"]["stop"] = {"distance_after_idle_m": round(math.dist(a["pos"][:2], b["pos"][:2]), 3), "pass": bool(not b["fallen"] and math.dist(a["pos"][:2], b["pos"][:2]) < 0.30)}
        want["cmd"] = (args.speed, 0.0, 0.0)
        run(3.0)
        want["send"] = False                                                  # the planner publisher goes silent
        a = robot.truth()
        b = run(2.5)
        c = run(1.0)
        res["phases"]["timeout"] = {"distance_after_silence_m": round(math.dist(a["pos"][:2], b["pos"][:2]), 3),
                                    "still_moving_m_per_s": round(math.dist(b["pos"][:2], c["pos"][:2]), 3),
                                    "pass": bool(not c["fallen"] and math.dist(b["pos"][:2], c["pos"][:2]) < 0.05)}
        tx.send(build_command_message(False, True, True))                      # stop-control: the controller process must end
        t_exit, t_passive = None, None
        for k in range(60):
            run(0.05)
            if child.poll() is not None and t_exit is None:
                t_exit = (k + 1) * 0.05
            if robot.passive and t_passive is None:
                t_passive = (k + 1) * 0.05
        res["phases"]["exit"] = {"controller_process_ended_after_s": t_exit, "motors_passive_after_s": t_passive, "exit_code": child.poll(),
                                 "pass": bool(t_exit is not None and t_passive is not None)}
        la = np.array(lat) * 1e3
        res["lowstate_to_lowcmd_ms"] = {"n": int(la.size), "median": round(float(np.median(la)), 2), "p95": round(float(np.percentile(la, 95)), 2),
                                        "max": round(float(la.max()), 2)} if la.size else None
        res["rtf_physics"] = round(sim.rtf(), 2)
        res["sim_seconds"] = round(sim.t, 1)
        res["crc_errors"] = link.crc_errors
        res["sockets"] = dds.socket_report()["local"]
    finally:
        if child.poll() is None:
            child.terminate()                                                 # by PID: the process this script started
            try:
                child.wait(5)
            except subprocess.TimeoutExpired:
                child.kill()
        res["child_log_tail"] = (child.stdout.read() or "").splitlines()[-6:]
        tx.close()
        link.close()
    res["pass"] = all(p["pass"] for p in res["phases"].values())
    return res


FAILED = False
try:
    r = main()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(r, indent=1))
    print("GATE_DONE", json.dumps(r), flush=True)
except Exception:
    print("GATE_TRACEBACK", traceback.format_exc(), flush=True)
    FAILED = True
finally:
    app.close()
if FAILED:
    sys.exit(1)
