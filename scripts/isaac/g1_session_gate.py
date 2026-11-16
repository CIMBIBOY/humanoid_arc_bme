"""Gate WP5 in Isaac Sim: the unchanged onboard stack drives the simulated G1 inside a generated scene.

    python.sh scripts/isaac/g1_session_gate.py --scene <scene dir> --out result.json [--commands "go forward 1 meters;turn left 90 degrees"]

scene.json + scene.usda -> G1 at a spawn pose of the manifest -> sensors -> SensorHub -> RobotStack -> OperatorRuntime (armed, HOLD held)
-> GrootPlannerController -> deploy stand-in -> motors. The operator's sentences go through the stack's own grammar. Reports, per
command, the task's final state, what the robot did on the ground truth, the safety levels seen, and the timing of the whole loop.
The heading at the spawn point is chosen among 8 so that the robot has room ahead (the manifest's yaw is random).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--commands", default="go forward 1 meters;turn left 90 degrees;go forward 0.5 meters")
ap.add_argument("--run-dir", default=None)
ap.add_argument("--seed", type=int, default=0)
args, _ = ap.parse_known_args()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid_arc_bme.isaac.app import make_app  # noqa: E402

app = make_app("g1_session_gate", lidar=True)

import numpy as np  # noqa: E402


def main() -> dict:
    from isaacsim.core.api import World

    from humanoid_arc_bme.isaac import scene as SC, standin as S
    from humanoid_arc_bme.isaac.rates import PHYSICS_HZ
    from humanoid_arc_bme.isaac.robot import SimG1
    from humanoid_arc_bme.isaac.sensors import SensorRig
    from humanoid_arc_bme.isaac.session import DeployTransport, RobotSession
    from humanoid_arc_bme.isaac.sim import G1Sim
    from humanoid_arc_bme.onboard.pipeline.stubs import GroundTruthSemantics
    from humanoid_arc_bme.onboard.sensors import mounts as MT
    from humanoid_arc_bme.robots.assets import AGILE_LOCOMOTION_POLICY, G1_29DOF_USD

    mounts = MT.load_mounts()
    t0 = time.perf_counter()
    world = World(physics_dt=1.0 / PHYSICS_HZ, rendering_dt=SensorRig.slice_dt_of(mounts), stage_units_in_meters=1.0)
    man = SC.load_scene(args.scene)
    if abs(man["physics"]["dt"] - 1.0 / PHYSICS_HZ) > 1e-9:
        raise RuntimeError(f"scene physics dt {man['physics']['dt']} is not the robot's {1.0 / PHYSICS_HZ}")
    obstacles = SC.obstacles_2d(man)
    x, y, _ = SC.spawn_pose(man)
    yaw = max((k * math.pi / 4 for k in range(8)), key=lambda a: SC.free_run(obstacles, x, y, a, 0.35))
    robot = SimG1(str(G1_29DOF_USD), floor_pose=(x, y, yaw))
    rig = SensorRig(robot.prim_path, mounts, world, seed=args.seed, segmentation=True)
    world.reset()
    robot.start()
    rig.start(robot.art)
    deploy = S.DeployStandIn(str(AGILE_LOCOMOTION_POLICY))
    sim = G1Sim(world, robot, controller=deploy, rig=rig)

    def seg(_rgb, _depth):
        got = rig.segmentation("d435")
        return (np.zeros(_depth.depth.shape, np.int32), {}) if got is None else (got[0], {k: v.split("#")[0] for k, v in got[1].items()})

    run_dir = Path(args.run_dir or Path(args.out).with_suffix("")).resolve()
    ses = RobotSession(sim, robot, rig, DeployTransport(deploy, lambda: sim.t), run_dir, mounts, GroundTruthSemantics(seg))
    res = {"scene": man["scene_id"], "spawn": [round(x, 3), round(y, 3), round(yaw, 3)], "free_ahead_m": round(SC.free_run(obstacles, x, y, yaw, 0.35), 2),
           "load_s": round(time.perf_counter() - t0, 2), "nav_module": ses.nav_why, "commands": []}
    ses.run(3.0)                                                              # warm-up: sensors deliver, the robot settles on its feet
    tr = robot.truth()
    res["settle"] = {"pelvis_z": round(tr["pos"][2], 3), "moved_m": round(math.dist(tr["pos"][:2], (x, y)), 3), "fallen": tr["fallen"],
                     "frame_ready": ses.loop.latest is not None, "status": ses.status()}
    # what the pipeline runner sees ahead of the standing robot (diagnosis of obstacle stops: the same functions the runner calls)
    from humanoid_arc_bme.onboard.pipeline.perception.ground import analyze_corridor, fit_ground, gravity_align
    from humanoid_arc_bme.onboard.pipeline.perception.lidar import self_exclusion_keep
    from humanoid_arc_bme.onboard.pipeline.perception.sensing import depth_to_points
    last = {}
    runner = ses.stack.runner
    orig_tick = runner.tick
    runner.tick = lambda inp: last.__setitem__("out", orig_tick(inp)) or last["out"]
    f = ses.loop.latest
    if f is None or f.down_depth is None:
        raise RuntimeError("no sensor frame after the warm-up")
    c = runner.cfg
    pts = depth_to_points(f.down_depth, max_range=c.points_max_range, stride=c.points_stride)
    al = gravity_align(pts, f.roll, f.pitch)
    plane = fit_ground(al)
    keep = self_exclusion_keep(al, plane, c.camera_self_radius_m)
    cl = analyze_corridor(al[keep], plane, corridor_half_width=c.corridor_half_width, max_range=c.max_range, valid_fraction=1.0, min_valid_fraction=c.min_valid_fraction)
    fin = al[np.isfinite(al).all(axis=1)]
    h = plane.height(fin) if plane is not None else fin[:, 2]
    cor = (np.abs(fin[:, 1]) < c.corridor_half_width) & (fin[:, 0] > 0) & (fin[:, 0] < c.max_range)
    high = cor & (h > 0.05)
    res["runner_view"] = {"roll_pitch_deg": [round(math.degrees(f.roll), 2), round(math.degrees(f.pitch), 2)], "self_radius": c.camera_self_radius_m,
                          "points": int(len(fin)), "masked_as_self": int((~keep).sum()), "plane": None if plane is None else str(plane)[:160],
                          "x_range": [round(float(fin[:, 0].min()), 2), round(float(fin[:, 0].max()), 2)],
                          "height_percentiles_in_corridor": [round(float(v), 3) for v in np.percentile(h[cor], [1, 50, 99])] if cor.any() else None,
                          "points_above_5cm_in_corridor": int(high.sum()),
                          "nearest_high_point_xyh": [round(float(v), 3) for v in (*fin[high][np.argmin(fin[high][:, 0])][:2], h[high][np.argmin(fin[high][:, 0])])] if high.any() else None,
                          "radius_of_high_points_p50_max": [round(float(v), 3) for v in np.percentile(np.hypot(fin[high][:, 0], fin[high][:, 1]), [50, 100])] if high.any() else None,
                          "clearance": {k: (round(float(v), 3) if isinstance(v, float) else v) for k, v in vars(cl).items() if isinstance(v, (int, float, bool, str))}}
    ok, why = ses.arm()
    res["arm"] = {"ok": bool(ok), "why": why}
    min_clear = math.inf
    for text in [c.strip() for c in args.commands.split(";") if c.strip()]:
        a = robot.truth()
        st = ses.say(text)
        levels, t_start = set(), sim.t

        def done(s):
            nonlocal min_clear
            levels.add(s.stack.safety)
            p = robot.truth()["pos"]
            min_clear = min(min_clear, SC.clearance(obstacles, p[0], p[1]))
            return s.stack.executor.status.state in ("succeeded", "failed")

        finished = ses.run(90.0, until=done) if st is not None and st.state == "running" else True
        ses.run(1.0)
        b = robot.truth()
        d = (b["pos"][0] - a["pos"][0], b["pos"][1] - a["pos"][1])
        fin = ses.stack.executor.status
        seen = None
        o = last.get("out")
        fr = ses.loop.latest
        if o is not None and fr is not None and fr.down_depth is not None and o.clearance is not None and math.isfinite(o.clearance.distance_positive):
            # what is the camera's nearest obstacle? the same points, put into the world with the true pose, against the manifest
            pts_ = depth_to_points(fr.down_depth, max_range=c.points_max_range, stride=c.points_stride)
            al_ = gravity_align(pts_, fr.roll, fr.pitch)
            al_ = al_[np.isfinite(al_).all(axis=1)]
            pl_ = fit_ground(al_)
            hh = pl_.height(al_) if pl_ is not None else al_[:, 2]
            m_ = (np.abs(al_[:, 1]) < c.corridor_half_width) & (al_[:, 0] > 0) & (hh > 0.05) & (np.hypot(al_[:, 0], al_[:, 1]) >= c.camera_self_radius_m)
            if m_.any():
                p_ = al_[m_][np.argmin(al_[m_][:, 0])]
                wx, wy = b["pos"][0] + math.cos(b["yaw"]) * p_[0] - math.sin(b["yaw"]) * p_[1], b["pos"][1] + math.sin(b["yaw"]) * p_[0] + math.cos(b["yaw"]) * p_[1]
                near = min(man["objects"], key=lambda ob: math.hypot(ob["pose"][0] - wx, ob["pose"][1] - wy))
                seen = {"body_xyh": [round(float(p_[0]), 2), round(float(p_[1]), 2), round(float(hh[m_][np.argmin(al_[m_][:, 0])]), 2)], "world_xy": [round(wx, 2), round(wy, 2)],
                        "nearest_object": [near["id"], near["class"], near.get("door", {}).get("state"), round(math.hypot(near["pose"][0] - wx, near["pose"][1] - wy), 2)],
                        "true_clearance_there_m": round(SC.clearance(obstacles, wx, wy), 2)}
        res["commands"].append({"text": text, "camera_nearest": seen, "runner": None if o is None else {"camera_m": o.clearance.distance_positive if o.clearance else None, "lidar_m": o.lidar.distance if o.lidar else None, "lidar_hits": o.lidar.n_hits if o.lidar else None}, "accepted": None if st is None else st.state, "final": fin.state, "phase": fin.phase, "message": fin.message,
                                "finished": bool(finished), "duration_s": round(sim.t - t_start, 2),
                                "forward_m": round(d[0] * math.cos(a["yaw"]) + d[1] * math.sin(a["yaw"]), 3),
                                "lateral_m": round(-d[0] * math.sin(a["yaw"]) + d[1] * math.cos(a["yaw"]), 3),
                                "turned_deg": round(math.degrees(S.wrap(b["yaw"] - a["yaw"])), 1), "safety_levels": sorted(levels),
                                "reasons": list(ses.stack.reasons), "fallen": b["fallen"]})
    out = last.get("out")
    if out is not None:
        res["last_tick"] = {"safety": [out.safety.level, list(out.safety.reasons)], "lidar": None if out.lidar is None else {k: (round(float(v), 3) if isinstance(v, float) else v) for k, v in vars(out.lidar).items() if isinstance(v, (int, float, bool, str))},
                            "clearance": None if out.clearance is None else {k: (round(float(v), 3) if isinstance(v, float) else v) for k, v in vars(out.clearance).items() if isinstance(v, (int, float, bool, str))},
                            "last_event": {k: v for k, v in (out.events[-1] if out.events else {}).items() if k != "timing_ms"}}
    res["min_clearance_m"] = round(min_clear, 3)
    res["timing"] = {"sim_seconds": round(sim.t, 1), "rtf_median": round(sim.rtf(), 2), "wall_s": round(time.perf_counter() - t0, 1),
                     "stack_tick_ms_median": round(float(np.median(ses.tick_ms)), 2), "stack_tick_ms_p95": round(float(np.percentile(ses.tick_ms, 95)), 2),
                     "slice_ms_median": round(sim.dt * sim.steps_per_outer / sim.rtf() * 1e3, 1)}
    res["final"] = ses.status()
    res["heartbeat_errors"] = ses.heartbeat_errors[:5]
    res["deploy"] = deploy.status()
    res["pass"] = bool(res["arm"]["ok"] and res["commands"] and not robot.truth()["fallen"] and all(c["final"] == "succeeded" for c in res["commands"]))
    return res


FAILED = False
try:
    r = main()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(r, indent=1, default=str))
    print("GATE_DONE", json.dumps(r, default=str), flush=True)
except Exception:
    print("GATE_TRACEBACK", traceback.format_exc(), flush=True)
    FAILED = True
finally:
    app.close()
if FAILED:
    sys.exit(1)
