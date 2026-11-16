"""Gates WP2 and WP3 in Isaac Sim: the G1 stands, then walks under the deploy stand-in through the planner interface.

    python.sh scripts/isaac/g1_loco_gate.py --out result.json [--ground flat|carpet] [--phases stand,walk,turn,stop,timeout,exit]

Lockstep: GrootPlannerController (the stack's real adapter, S1 limits, wire format) -> planner messages -> DeployStandIn ->
LowCmd -> motor model -> PhysX at the policy's rate. Every phase reports numbers; `pass` flags use the bounds written next to them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--ground", default="flat", choices=("flat", "carpet"))
ap.add_argument("--phases", default="stand,walk,stop,turn,slow,timeout,exit")
ap.add_argument("--speed", type=float, default=0.2)
args, _ = ap.parse_known_args()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid_arc_bme.isaac.app import make_app  # noqa: E402

app = make_app("g1_loco_gate")

import numpy as np  # noqa: E402

FRICTION = {"flat": (1.0, 1.0), "carpet": (0.9, 0.8)}     # flat = Isaac Lab's default plane; carpet = the scenes' floor material (worldgen tables)


def main() -> dict:
    from isaacsim.core.api import World

    from humanoid_arc_bme.isaac import g1_model, standin as S
    from humanoid_arc_bme.isaac.rates import PHYSICS_HZ, STACK_HZ
    from humanoid_arc_bme.isaac.robot import SimG1
    from humanoid_arc_bme.isaac.sim import G1Sim
    from humanoid_arc_bme.onboard.api.contract import Imu, RobotState
    from humanoid_arc_bme.onboard.api.safety import load_limits
    from humanoid_arc_bme.onboard.g1_layout import LEFT_HAND_JOINTS, MOTOR_JOINTS, RIGHT_HAND_JOINTS
    from humanoid_arc_bme.onboard.paths import DEFAULT_LIMITS
    from humanoid_arc_bme.onboard.runtime.groot_controller import FakeTransport, GrootPlannerController, build_command_message
    from humanoid_arc_bme.robots.assets import AGILE_LOCOMOTION_POLICY, G1_29DOF_USD

    world = World(physics_dt=1.0 / PHYSICS_HZ, rendering_dt=1.0 / PHYSICS_HZ, stage_units_in_meters=1.0)
    mu_s, mu_d = FRICTION[args.ground]
    world.scene.add_default_ground_plane(static_friction=mu_s, dynamic_friction=mu_d, restitution=0.0)
    robot = SimG1(str(G1_29DOF_USD), floor_pose=(0.3, 0.2, math.radians(37.0)))      # non-identity pose: catches world-frame mistakes
    world.reset()
    robot.start()
    deploy = S.DeployStandIn(str(AGILE_LOCOMOTION_POLICY))
    sim = G1Sim(world, robot, controller=deploy)
    tx = FakeTransport()
    ctrl = GrootPlannerController(tx, enabled=True, threaded=False, clock=lambda: sim.t, limits=load_limits(DEFAULT_LIMITS))
    names = tuple(MOTOR_JOINTS) + tuple(LEFT_HAND_JOINTS) + tuple(RIGHT_HAND_JOINTS)
    want = {"cmd": (0.0, 0.0, 0.0), "send": True}
    seen = {"n": 0}

    def tick(s):
        """The stack side at 20 Hz: the controller adapter turns (vx, vy, wz) into planner messages; they reach the stand-in at once."""
        if want["send"] and s.state is not None:
            st = s.state
            rs = RobotState(s.t, names, tuple(map(float, st.q)) + tuple(map(float, st.hands_q[0])) + tuple(map(float, st.hands_q[1])),
                            (0.0,) * len(names), (0.0, 0.0, 0.0), Imu(tuple(map(float, st.quat)), tuple(map(float, st.gyro)), tuple(map(float, st.accel))), {})
            ctrl.step(s.t, 1.0 / STACK_HZ, rs, want["cmd"])
        for raw in tx.sent[seen["n"]:]:
            deploy.on_message(raw, s.t)
        seen["n"] = len(tx.sent)

    def run(seconds):
        sim.advance(seconds, 1.0 / STACK_HZ, tick)
        return robot.truth()

    res = {"ground": args.ground, "friction": [mu_s, mu_d], "physics_hz": PHYSICS_HZ, "dof_names": list(robot.dof_names), "phases": {}}
    body = [n for n in robot.dof_names if "_hand_" not in n]
    res["joint_table"] = {"motor_joints_all_in_asset": True, "obs_order_matches_simulator": body == list(S.OBS_JOINTS),
                          "asset_body_dof_order": body}
    phases = args.phases.split(",")
    ctrl.on_arm(sim.t, None)
    t0 = robot.truth()

    if "stand" in phases:                                                     # WP2 / S1: stand 10 s on the idle command
        a = run(2.0)                                                          # warm-up block: let the spawn settle before measuring
        b = run(10.0)
        res["phases"]["stand"] = {"pelvis_z": round(b["pos"][2], 3), "tilt_deg": round(b["tilt_deg"], 2), "drift_m": round(math.dist(a["pos"][:2], b["pos"][:2]), 3),
                                  "yaw_drift_deg": round(math.degrees(S.wrap(b["yaw"] - a["yaw"])), 2), "spawn_settle_m": round(math.dist(t0["pos"][:2], a["pos"][:2]), 3),
                                  "pass": bool(not b["fallen"] and math.dist(a["pos"][:2], b["pos"][:2]) < 0.10)}       # bound: 0.10 m in 10 s
    if "walk" in phases:                                                      # commanded forward velocity
        a = robot.truth()
        want["cmd"] = (args.speed, 0.0, 0.0)
        run(2.0)                                                              # acceleration
        b = robot.truth()
        c = run(6.0)
        fwd = (math.cos(a["yaw"]), math.sin(a["yaw"]))
        d = (c["pos"][0] - b["pos"][0], c["pos"][1] - b["pos"][1])
        v = (d[0] * fwd[0] + d[1] * fwd[1]) / 6.0
        lat = (-d[0] * fwd[1] + d[1] * fwd[0])
        res["phases"]["walk"] = {"commanded": args.speed, "achieved": round(v, 3), "lateral_m": round(lat, 3), "yaw_change_deg": round(math.degrees(S.wrap(c["yaw"] - a["yaw"])), 2),
                                 "tilt_deg": round(c["tilt_deg"], 2), "pass": bool(not c["fallen"] and 0.5 * args.speed <= v <= 1.5 * args.speed)}   # bound: within 50 %
    if "stop" in phases:                                                      # IDLE: how long and how far until it stands
        a = robot.truth()
        want["cmd"] = (0.0, 0.0, 0.0)
        t_stop, pos_prev, still_t = None, a["pos"], 0.0
        for k in range(100):                                                  # 5 s in 0.05 s slices
            cur = run(0.05)
            sp = math.dist(cur["pos"][:2], pos_prev[:2]) / 0.05
            pos_prev = cur["pos"]
            still_t = still_t + 0.05 if sp < 0.05 else 0.0
            if still_t >= 0.5 and t_stop is None:
                t_stop = (k + 1) * 0.05 - 0.5
        b = robot.truth()
        res["phases"]["stop"] = {"time_to_still_s": None if t_stop is None else round(t_stop, 2), "distance_after_idle_m": round(math.dist(a["pos"][:2], b["pos"][:2]), 3),
                                 "pass": bool(not b["fallen"] and t_stop is not None)}
    if "turn" in phases:                                                      # turn in place: a facing change while idle
        a = robot.truth()
        want["cmd"] = (0.0, 0.0, 0.3)
        trace = []
        for k in range(14):                                                   # 5 s turning, then 2 s to settle; yaw sampled every 0.5 s
            if k == 10:
                want["cmd"] = (0.0, 0.0, 0.0)
            c = run(0.5)
            trace.append([round(math.degrees(S.wrap(c["yaw"] - a["yaw"])), 1), round(math.degrees(ctrl.yaw), 1), round(float(deploy.cmd[2]), 2)])
        dyaw = math.degrees(S.wrap(c["yaw"] - a["yaw"]))
        res["phases"]["turn"] = {"commanded_deg": round(math.degrees(0.3 * 5.0), 1), "achieved_deg": round(dyaw, 1), "drift_m": round(math.dist(a["pos"][:2], c["pos"][:2]), 3),
                                 "trace_deg_[robot, facing_sent, wz_cmd]": trace, "pass": bool(not c["fallen"] and 0.6 * 86 <= dyaw <= 1.4 * 86)}
    if "slow" in phases:                                                      # a speed below what the policy follows: honoured on average
        a = robot.truth()
        want["cmd"] = (0.08, 0.0, 0.0)
        c = run(10.0)
        want["cmd"] = (0.0, 0.0, 0.0)
        run(1.0)
        d = math.dist(a["pos"][:2], c["pos"][:2]) / 10.0
        res["phases"]["slow"] = {"commanded": 0.08, "achieved_mean": round(d, 3), "pass": bool(not c["fallen"] and 0.5 * 0.08 <= d <= 1.5 * 0.08)}
    if "timeout" in phases:                                                   # the sender dies while walking: IDLE after PLANNER_TIMEOUT
        want["cmd"] = (args.speed, 0.0, 0.0)
        run(3.0)
        n_to = deploy.stats["timeouts"]
        want["send"] = False                                                  # no more planner messages (a crashed stack sends nothing)
        a = robot.truth()
        t_idle = None
        for k in range(80):
            run(0.05)
            if deploy.stats["timeouts"] > n_to and t_idle is None:
                t_idle = (k + 1) * 0.05
        b = robot.truth()
        res["phases"]["timeout"] = {"idle_after_s": t_idle, "distance_after_silence_m": round(math.dist(a["pos"][:2], b["pos"][:2]), 3), "mode_after": deploy.status()["mode"],
                                    "pass": bool(t_idle is not None and 0.95 <= t_idle <= 1.15 and not b["fallen"])}
        want["send"] = True
        ctrl.on_arm(sim.t, None)                                              # the stack re-arms: streaming resumes from idle
        want["cmd"] = (0.0, 0.0, 0.0)
        run(1.0)
    if "exit" in phases:                                                      # the deploy's control loop stops: no more rt/lowcmd
        deploy.on_message(build_command_message(False, True, True), sim.t)
        a = robot.truth()
        t_passive = t_fall = None
        for k in range(100):
            cur = run(0.05)
            if robot.passive and t_passive is None:
                t_passive = (k + 1) * 0.05
            if cur["fallen"] and t_fall is None:
                t_fall = (k + 1) * 0.05
        res["phases"]["exit"] = {"motors_passive_after_s": t_passive, "fallen_after_s": t_fall, "final_pelvis_z": round(robot.truth()["pos"][2], 3),
                                 "note": "motor model goes passive (damping) after LOWCMD_TIMEOUT_S; the real robot's behaviour is UNKNOWN",
                                 "pass": bool(t_passive is not None and t_passive <= g1_model.LOWCMD_TIMEOUT_S + 0.1)}
    res["rtf_physics"] = round(sim.rtf(), 2)
    res["step_ms_median"] = round(sim.dt / sim.rtf() * 1e3, 3)
    res["deploy"] = deploy.status()
    res["pass"] = bool(res["phases"]) and all(p["pass"] for p in res["phases"].values()) and res["joint_table"]["obs_order_matches_simulator"]
    return res


FAILED = False
try:
    r = main()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(r, indent=1))
    print("GATE_DONE", json.dumps({k: v for k, v in r.items() if k != "dof_names"}), flush=True)
except Exception:
    print("GATE_TRACEBACK", traceback.format_exc(), flush=True)
    FAILED = True
finally:
    app.close()
if FAILED:
    sys.exit(1)
