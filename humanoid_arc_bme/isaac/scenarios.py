"""Scripted scenarios S1-S7 for the simulated G1 in a scene described by a manifest. Needs Kit.

One `Bench` = one scene, one robot, one `RobotSession`. Scenarios run one after another; between them the robot is put back by
restoring poses and joint states (never by reloading the scene) and the deploy stand-in is restarted, as an operator would restart
the controller. Every scenario returns {"pass": bool, "reason": str, ...metrics}; `pass` is decided on simulator ground truth with the
bounds named in BOUNDS (all GUESSES until the robot gives numbers). `na` marks a scenario the scene cannot host (no open door, no
nameable object in reach): it is reported, not counted as a failure.

The detector here is the simulator's segmentation (`GroundTruthSemantics`): a STAND-IN for YOLO, which has no weights on this
machine. The localizer is ground truth plus noise: a STAND-IN for DLIO. Both sit behind the stack's own interfaces.
"""

from __future__ import annotations

import math
import time

import numpy as np

from humanoid_arc_bme.isaac import g1_model, scene as SC, standin as S
from humanoid_arc_bme.isaac.rates import PHYSICS_HZ

BOUNDS = {
    "stand_drift_m": 0.10,            # S1: pelvis drift over 10 s
    "goal_tol_m": 0.35,               # S2: distance to the goal at the end
    "body_radius_m": 0.25,            # plan-view radius of the robot for "touched furniture" (pelvis closer than this to an obstacle)
    "door_cross_m": 0.5,              # S3: pelvis this far beyond the door plane
    "door_push_open_deg": 10.0,       # S3: an OPEN door may not be moved more than this (an ajar one is pushed: reported)
    "stop_latency_s": 1.0,            # S4 / S7: zero command and standing still within this of the stop
    "obstacle_gap_m": 0.10,           # S6: free gap left between the robot's body and the obstacle
    "fault_stop_s": 2.0,              # S7: the base command is zero within this of the fault
}
GRAMMAR_LABELS = ("cup", "bottle", "bowl", "box", "can")
# The pipeline runner's obstacle geometry. "stock" is the stack as shipped: a 0.35 m half corridor plus 0.30 m costmap inflation, so
# anything within 0.65 m of the path stops the robot and a 0.9 m door (jambs 0.45 m from the centre line) can never be passed
# (measured: S2 and S3 blocked). "fitted" is a GUESS sized to the simulated G1: half width with hanging arms about 0.27 m, 0.10 m of
# margin, and the turn-in-place clearance to match. Neither is validated on the robot.
GUARD_PROFILES = {"stock": None, "fitted": {"corridor_half_width": 0.27, "inflation_radius": 0.10, "turn_clearance_m": 0.37}}


class Bench:
    def __init__(self, scene_dir, run_dir, seed: int = 0, guard: str = "fitted", record_dir=None, semantics=None, realtime: bool = False):
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid

        from humanoid_arc_bme.isaac.robot import SimG1
        from humanoid_arc_bme.isaac.sensors import SensorRig
        from humanoid_arc_bme.isaac.session import DeployTransport, RobotSession
        from humanoid_arc_bme.isaac.sim import G1Sim
        from humanoid_arc_bme.onboard.pipeline.stubs import GroundTruthSemantics
        from humanoid_arc_bme.onboard.sensors import mounts as MT
        from humanoid_arc_bme.robots.assets import AGILE_LOCOMOTION_POLICY, G1_29DOF_USD

        t0 = time.perf_counter()
        self.mounts = MT.load_mounts()
        self.world = World(physics_dt=1.0 / PHYSICS_HZ, rendering_dt=SensorRig.slice_dt_of(self.mounts), stage_units_in_meters=1.0)
        self.man = SC.load_scene(scene_dir)
        if abs(self.man["physics"]["dt"] - 1.0 / PHYSICS_HZ) > 1e-9:
            raise RuntimeError("scene physics rate differs from the robot's")
        self.obstacles = SC.obstacles_2d(self.man)
        x, y, _ = SC.spawn_pose(self.man)
        self.home = (x, y, self.best_heading(x, y))
        self.robot = SimG1(str(G1_29DOF_USD), floor_pose=self.home)
        self.rig = SensorRig(self.robot.prim_path, self.mounts, self.world, seed=seed, segmentation=True)
        self.box = DynamicCuboid("/World/Intruder", position=np.array([100.0, 100.0, 0.45]), scale=np.array([0.45, 0.45, 0.9]), mass=5.0,
                                 color=np.array([0.9, 0.5, 0.1]))             # S6: a chair-sized box, parked far away until it "appears"
        self.world.reset()
        self.robot.start()
        self.rig.start(self.robot.art)
        self.deploy = S.DeployStandIn(str(AGILE_LOCOMOTION_POLICY))
        self.sim = G1Sim(self.world, self.robot, controller=self.deploy, rig=self.rig, realtime=realtime)

        def seg(_rgb, depth):
            got = self.rig.segmentation("d435")
            return (np.zeros(depth.depth.shape, np.int32), {}) if got is None else (got[0], {k: v.split("#")[0] for k, v in got[1].items()})

        self.ses = RobotSession(self.sim, self.robot, self.rig, DeployTransport(self.deploy, lambda: self.sim.t), run_dir, self.mounts,
                                semantics if semantics is not None else GroundTruthSemantics(seg))
        self.guard = guard
        prof = GUARD_PROFILES[guard]
        if prof is not None:                                               # same runner class, other numbers; nothing in the stack is edited
            import dataclasses

            from humanoid_arc_bme.onboard.pipeline.perception.costmap import CostmapConfig, LocalCostmap
            from humanoid_arc_bme.onboard.pipeline.runner import PipelineRunner
            old = self.ses.stack.runner
            cfg = dataclasses.replace(old.cfg, corridor_half_width=prof["corridor_half_width"], turn_clearance_m=prof["turn_clearance_m"])
            self.ses.stack.runner = PipelineRunner(cfg, semantics=self.ses.stack.semantics, proposer=old.proposer if hasattr(old, "proposer") else None,
                                                   costmap=LocalCostmap(CostmapConfig(inflation_radius=prof["inflation_radius"])))
        self.need = 0.45 if guard == "fitted" else 0.70                       # lateral room a straight walk needs under this profile
        self.grammar = self.ses.stack.parser
        self.load_s = time.perf_counter() - t0
        self._doors: dict = {}
        self.min_clear = math.inf
        self.reset_times: list[float] = []
        self.sim.every(0.05, self._watch)
        self.ses.localizer.rng = np.random.default_rng(seed + 1)                # the pose noise follows the run's seed too
        self.record_dir, self._rec_n = record_dir, 0
        if record_dir is not None:
            from pathlib import Path
            Path(record_dir).mkdir(parents=True, exist_ok=True)
            self.sim.on_frame = self._record

    def _record(self, _sim) -> None:
        """One file per 10 Hz sensor frame: what a driver would deliver (images, raw lidar with per-point time, IMU samples) plus the
        simulator's truth (pose, segmentation labels). For offline work: a detector's detection rate, a lidar-inertial odometry replay."""
        rig, tr = self.rig, self.truth()
        raw = rig.lidar_raw()
        imu = rig.imu_samples()
        out = {"t": self.sim.t, "truth_pos": np.array(tr["pos"]), "truth_quat": np.array(tr["quat"]),
               "imu_t": np.array([x[0] for x in imu]), "imu_gyro": np.array([x[1] for x in imu]), "imu_accel": np.array([x[2] for x in imu])}
        if raw is not None:
            out.update({"lidar_xyz": raw[0].astype(np.float32), "lidar_t": raw[1], "lidar_frame_end": raw[2]})
        for name in rig.cam:
            if rig.rgb(name) is not None:
                out[f"{name}_rgb"] = rig.rgb(name)
            d = rig.depth(name)
            if d is not None:
                out[f"{name}_depth"] = d.depth
            sg = rig.segmentation(name)
            if sg is not None:
                out[f"{name}_seg"] = sg[0].astype(np.int32)
                out[f"{name}_seg_labels"] = np.array([f"{k}={v}" for k, v in sorted(sg[1].items())])
        np.savez_compressed(f"{self.record_dir}/frame_{self._rec_n:05d}.npz", **out)
        self._rec_n += 1

    # ------------------------------------------------------------------ helpers
    def best_heading(self, x: float, y: float, radius: float = 0.35) -> float:
        return max((k * math.pi / 8 for k in range(16)), key=lambda a: SC.free_run(self.obstacles, x, y, a, radius))

    def _watch(self, _sim) -> None:
        p = self.robot.truth()["pos"]
        self.min_clear = min(self.min_clear, SC.clearance(self.obstacles, p[0], p[1]))

    def truth(self) -> dict:
        return self.robot.truth()

    def reset(self, pose=None, settle: float = 1.5) -> dict:
        """Put the robot back on its feet at `pose` (x, y, yaw), restart the controller, clear faults and re-arm. Pose restore only."""
        a = time.perf_counter()
        x, y, yaw = pose or self.home
        art, ses = self.robot.art, self.ses
        if ses.rt.sup.armed:
            ses.rt.disarm("scenario reset")
        ses.stack.stop("scenario reset")
        ses.stack.parser = self.grammar
        for f in (self.rig.faults, ses.io.faults, ses.localizer.faults):
            f.clear()
        art.set_world_poses(np.array([[x, y, g1_model.ROOT_HEIGHT]]), np.array([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]]))
        art.set_velocities(np.zeros((1, 6)))
        q0 = np.zeros(len(self.robot.dof_names))
        q0[self.robot.i_motor] = g1_model.default_pose()
        art.set_joint_positions(q0[None])
        art.set_joint_velocities(np.zeros((1, len(q0))))
        self.box.set_world_pose(np.array([100.0, 100.0, 0.45]))
        self.box.set_linear_velocity(np.zeros(3))
        self.deploy.restart()
        ses.mode, ses.teleop, ses.held = "ros", (0.0, 0.0, 0.0), True
        ses.run(settle)
        ses.rt.reset_stop()
        ses.stack.runner.reset()                                   # the robot was moved: the runner's map, memory and pose gate start over
        ses.run(0.5)
        ok, why = ses.arm()
        self.min_clear = math.inf
        self.reset_times.append(time.perf_counter() - a)
        return {"armed": bool(ok), "why": why, "fallen": self.truth()["fallen"]}

    def moving(self, window: float = 0.3) -> float:
        a = self.truth()["pos"]
        self.ses.run(window)
        b = self.truth()["pos"]
        return math.dist(a[:2], b[:2]) / window

    def time_to_still(self, limit: float = 5.0) -> tuple[float | None, float]:
        """(seconds until the robot stands still for 0.3 s, distance travelled meanwhile)."""
        a, t0 = self.truth()["pos"], self.sim.t
        while self.sim.t - t0 < limit:
            if self.moving(0.3) < 0.04:
                return self.sim.t - t0 - 0.3, math.dist(a[:2], self.truth()["pos"][:2])
        return None, math.dist(a[:2], self.truth()["pos"][:2])

    def say_and_wait(self, text: str, limit: float = 60.0, until=None) -> dict:
        ses, a, t0, levels = self.ses, self.truth(), self.sim.t, set()
        st = ses.say(text)

        def done(s):
            levels.add(s.stack.safety)
            return s.stack.executor.status.state in ("succeeded", "failed") or (until is not None and until(s))

        if st is not None and st.state == "running":
            ses.run(limit, until=done)
        b, fin = self.truth(), ses.stack.executor.status
        d = (b["pos"][0] - a["pos"][0], b["pos"][1] - a["pos"][1])
        return {"text": text, "accepted": None if st is None else st.state, "final": fin.state, "message": fin.message or (st.message if st else ""),
                "duration_s": round(self.sim.t - t0, 2), "forward_m": round(d[0] * math.cos(a["yaw"]) + d[1] * math.sin(a["yaw"]), 3),
                "turned_deg": round(math.degrees(S.wrap(b["yaw"] - a["yaw"])), 1), "safety": sorted(levels), "reasons": list(ses.stack.reasons)}

    def door_angle(self, obj: dict) -> float | None:
        from isaacsim.core.prims import Articulation

        try:
            if obj["id"] not in self._doors:
                a = Articulation(prim_paths_expr=f"/World/Scene{obj['prim_path'][len('/World'):]}/asset", name=f"door_{obj['id']}")
                a.initialize()
                self._doors[obj["id"]] = a
            return float(np.asarray(self._doors[obj["id"]].get_joint_positions())[0][0])
        except Exception:  # noqa: BLE001 - scoring only
            return None

    # ------------------------------------------------------------------ S1: spawn, stand 10 s
    def s1_stand(self) -> dict:
        r = self.reset()
        a = self.truth()
        e0 = self.ses.localizer.estimate(self.sim.t)
        self.ses.run(10.0)
        b = self.truth()
        e1 = self.ses.localizer.estimate(self.sim.t)
        drift = math.dist(a["pos"][:2], b["pos"][:2])
        est = math.dist(e0.pose.position[:2], e1.pose.position[:2])
        ok = r["armed"] and not b["fallen"] and drift < BOUNDS["stand_drift_m"] and est < BOUNDS["stand_drift_m"]
        return {"pass": bool(ok), "reason": "" if ok else ("fell" if b["fallen"] else "not armed" if not r["armed"] else "drift"), "drift_m": round(drift, 3),
                "localizer_drift_m": round(est, 3), "localizer": e1.source, "pelvis_z": round(b["pos"][2], 3), "tilt_deg": round(b["tilt_deg"], 1)}

    # ------------------------------------------------------------------ S2: walk to a goal across the room
    def s2_goal(self) -> dict:
        x, y, _ = self.home
        yaw0 = self.home[2]
        best = max(((SC.free_run(self.obstacles, x, y, yaw0 + k * math.pi / 8, self.need), k) for k in range(-6, 7) if k != 0), key=lambda t: t[0])
        dist = min(best[0] - 0.8, 2.5)
        if dist < 0.8:
            return {"pass": None, "na": True, "reason": f"no free line of 1.6 m with {self.need} m of room on each side from the spawn pose"}
        turn = best[1] * 22.5
        goal = (x + math.cos(yaw0 + math.radians(turn)) * dist, y + math.sin(yaw0 + math.radians(turn)) * dist)
        self.reset()
        c1 = self.say_and_wait(f"turn {'left' if turn > 0 else 'right'} {abs(turn):.0f} degrees", 40.0)
        c2 = self.say_and_wait(f"go forward {dist:.1f} meters", 60.0)
        b = self.truth()
        err = math.dist(b["pos"][:2], goal)
        touched = self.min_clear < BOUNDS["body_radius_m"]
        ok = (not b["fallen"]) and err < BOUNDS["goal_tol_m"] and not touched
        return {"pass": bool(ok), "reason": "" if ok else ("fell" if b["fallen"] else "touched furniture" if touched else f"{c2['final']}: {c2['message']}" if c2["final"] != "succeeded" else "missed the goal"),
                "goal_error_m": round(err, 3), "goal_distance_m": round(dist, 2), "turn_deg": turn, "min_clearance_m": round(self.min_clear, 3), "commands": [c1, c2]}

    # ------------------------------------------------------------------ S3: through an open or ajar door
    def s3_door(self) -> dict:
        doors = [o for o in self.man["objects"] if o["class"] == "door" and o.get("door", {}).get("state") in ("open", "ajar")]
        if not doors:
            return {"pass": None, "na": True, "reason": "no open or ajar door in this scene"}
        doors.sort(key=lambda o: o["door"]["state"] != "open")              # an open door first
        out = None
        for d in doors:
            n, c = d["door"]["swing_normal"], d["pose"][:2]
            for side in (-1.0, 1.0):                                        # start on the side the door does not swing into first
                start = (c[0] + side * n[0] * 1.3, c[1] + side * n[1] * 1.3)
                if SC.clearance(self.obstacles, *start) > 0.45:
                    out = (d, n, c, side, start)
                    break
            if out:
                break
        if out is None:
            return {"pass": None, "na": True, "reason": "no free standing place in front of an open door"}
        d, n, c, side, start = out
        yaw = math.atan2(-side * n[1], -side * n[0])
        self.reset((start[0], start[1], yaw))
        q0 = self.door_angle(d)
        cmd = self.say_and_wait("go forward 2.2 meters", 70.0)
        b = self.truth()
        crossed = -side * ((b["pos"][0] - c[0]) * n[0] + (b["pos"][1] - c[1]) * n[1])      # metres beyond the door plane, in the walking direction
        q1 = self.door_angle(d)
        moved = None if q0 is None or q1 is None else abs(math.degrees(q1 - q0))
        if d["door"]["state"] == "open" and moved is None:
            return {"pass": False, "reason": "the door's joint angle could not be read: cannot tell whether the robot pushed it", "door": d["id"]}
        ok = (not b["fallen"]) and crossed >= BOUNDS["door_cross_m"] and (d["door"]["state"] != "open" or moved < BOUNDS["door_push_open_deg"])
        why = "" if ok else ("fell" if b["fallen"] else f"stopped {crossed:+.2f} m from the door plane ({cmd['final']}: {cmd['message']}; {cmd['reasons']})" if crossed < BOUNDS["door_cross_m"] else "pushed the open door")
        return {"pass": bool(ok), "reason": why, "door": d["id"], "door_state": d["door"]["state"], "clear_width_m": d["door"]["clear_width"], "approach_side": "push" if side < 0 else "swing",
                "beyond_door_plane_m": round(crossed, 3), "door_moved_deg": None if moved is None else round(moved, 1), "min_clearance_m": round(self.min_clear, 3), "command": cmd}

    # ------------------------------------------------------------------ S4: gateway command, map goal, velocity teleop, stop
    def s4_teleop(self) -> dict:
        import os
        import sys

        for p in filter(None, os.environ.get("HUMANOID_ARC_DDS_PATHS", "").split(os.pathsep)):
            if p not in sys.path:
                sys.path.insert(0, p)                                       # where httpx lives for Isaac's Python
        from fastapi.testclient import TestClient

        from humanoid_arc_bme.onboard.runtime.command_gateway import StackCommandService, create_app

        ses = self.ses
        res: dict = {}
        self.reset()
        token = "sim-token"
        client = TestClient(create_app(StackCommandService(ses.stack, lambda: ses.loop.latest, ses.loop.lock), token, clock=lambda: self.sim.t))
        hdr = {"Authorization": f"Bearer {token}"}
        r = client.post("/api/command", json={"text": "go forward 2 meters"}, headers=hdr)
        j = r.json()
        res["command"] = {"http": r.status_code, "accepted": j.get("accepted"), "needs_confirm": j.get("needs_confirm")}
        unauth = client.post("/api/command", json={"text": "go forward 1 meters"}).status_code
        if j.get("needs_confirm"):
            c = client.post("/api/confirm", json={"id": j["id"]}, headers=hdr)
            res["confirm"] = {"http": c.status_code, "state": c.json().get("status", {}).get("state")}
        ses.run(3.0)
        v = self.moving(0.4)
        t_http = time.perf_counter()
        s = client.post("/api/stop")                                        # STOP needs no token
        http_ms = (time.perf_counter() - t_http) * 1e3
        t_still, d_stop = self.time_to_still()
        res["gateway_stop"] = {"http": s.status_code, "speed_before_mps": round(v, 3), "http_ms": round(http_ms, 2), "time_to_still_s": None if t_still is None else round(t_still, 2),
                               "distance_m": round(d_stop, 3), "task_after": ses.stack.executor.status.state}
        ok1 = res["command"]["accepted"] and unauth in (401, 403) and v > 0.08 and t_still is not None and t_still <= BOUNDS["stop_latency_s"]
        # goal on the map: the operator UI's waypoint panel (runtime navigation module, start-frame goal 1 m ahead)
        self.reset()
        a = self.truth()
        if ses.rt.nav is None:
            res["map_goal"] = {"error": f"no navigation module: {ses.nav_why}"}
            ok2 = False
        else:
            with ses.rt.lock:
                ses.rt.nav.set_waypoints([(1.0, 0.0)])
            ses.mode = "waypoints"
            ses.run(25.0, until=lambda s_: not ses.rt.nav.status().get("active", True) or math.dist(self.truth()["pos"][:2], a["pos"][:2]) > 1.3)
            ses.run(2.0)
            b = self.truth()
            fwd = (b["pos"][0] - a["pos"][0]) * math.cos(a["yaw"]) + (b["pos"][1] - a["pos"][1]) * math.sin(a["yaw"])
            res["map_goal"] = {"goal_m": 1.0, "forward_m": round(fwd, 3), "nav": {k: v_ for k, v_ in ses.rt.nav.status().items() if isinstance(v_, (int, float, str, bool))}}
            ok2 = 0.6 <= fwd <= 1.4 and not b["fallen"]
        # velocity teleoperation with the HOLD input, then the operator's STOP
        self.reset()
        ses.mode, ses.teleop = "teleop", (0.2, 0.0, 0.0)
        a = self.truth()
        ses.run(4.0)
        v = self.moving(0.4)
        ses.held = False                                                    # releasing HOLD alone must stop the robot (hold-to-run)
        t_rel, d_rel = self.time_to_still()
        ses.held = True
        ses.run(1.5)
        v2 = self.moving(0.4)
        ses.rt.stop("scenario")
        t_stop, _ = self.time_to_still()
        ses.run(1.0)
        res["velocity"] = {"commanded": 0.2, "speed_mps": round(v, 3), "release_hold_time_to_still_s": None if t_rel is None else round(t_rel, 2), "release_distance_m": round(d_rel, 3),
                           "speed_after_hold_again_mps": round(v2, 3), "stop_time_to_still_s": None if t_stop is None else round(t_stop, 2), "armed_after_stop": bool(ses.rt.sup.armed),
                           "last_cmd_after_stop": list(ses.rt.last_cmd)}
        ok3 = v > 0.08 and t_rel is not None and t_rel <= BOUNDS["stop_latency_s"] and t_stop is not None and t_stop <= BOUNDS["stop_latency_s"] and not ses.rt.sup.armed
        # manual driving toward an obstacle: the runtime's obstacle stage must stop the robot, whoever drives
        x0, y0, yaw0 = self.home
        yawf = self.best_heading(x0, y0, self.need)
        if SC.free_run(self.obstacles, x0, y0, yawf, self.need) >= 2.4:
            self.reset((x0, y0, yawf))
            p = self.truth()["pos"]
            bx, by_ = p[0] + math.cos(yawf) * 1.8, p[1] + math.sin(yawf) * 1.8
            self.box.set_world_pose(np.array([bx, by_, 0.45]))
            self.box.set_linear_velocity(np.zeros(3))
            ses.mode, ses.teleop = "teleop", (0.2, 0.0, 0.0)
            gap = math.inf
            for _ in range(160):                                    # up to 16 s at 0.2 m/s
                ses.run(0.1)
                q = self.truth()["pos"]
                gap = min(gap, math.dist(q[:2], (bx, by_)) - 0.225 - BOUNDS["body_radius_m"])
                if gap < 1.2 and max(abs(c) for c in ses.rt.last_cmd) < 1e-6:     # it has set off and the stage has stopped it
                    break
            ses.run(1.5)
            moved_box = math.dist(np.asarray(self.box.get_world_pose()[0], float)[:2], (bx, by_))
            res["teleop_obstacle"] = {"min_gap_m": round(gap, 3), "box_moved_m": round(moved_box, 3), "stage": (ses.rt.obstacle_info or {}).get("reason"), "speed_after": round(self.moving(0.4), 3)}
            ok4 = gap >= BOUNDS["obstacle_gap_m"] and moved_box < 0.03
        else:
            res["teleop_obstacle"] = {"na": "no 2.4 m free line from the spawn pose"}
            ok4 = True                                              # not applicable here: reported as such, not counted as evidence
        ok = bool(ok1 and ok2 and ok3 and ok4 and not self.truth()["fallen"])
        return {"pass": ok, "reason": "" if ok else "; ".join(n for n, k in (("gateway command/stop", ok1), ("map goal", ok2), ("velocity/hold/stop", ok3), ("teleop obstacle stop", ok4)) if not k), **res}

    # ------------------------------------------------------------------ S5: "go to the <object>", with and without Jev
    def s5_object(self) -> dict:
        from humanoid_arc_bme.onboard.command.composite import CompositeParser
        from humanoid_arc_bme.onboard.command.jev import FakeJev

        by = {o["id"]: o for o in self.man["objects"]}
        cand = []
        for o in self.man["objects"]:
            word = SC.LABEL_OF.get(o.get("role") or "")
            if word in GRAMMAR_LABELS and o.get("support"):
                ox, oy = o["pose"][0], o["pose"][1]
                # the down camera sees a 0.9 m high top only from about 0.6 m: stand as near as the furniture allows
                for dist in (0.65, 0.8, 0.95):
                    spots = [(SC.clearance(self.obstacles, ox + math.cos(a) * dist, oy + math.sin(a) * dist), a) for a in (k * math.pi / 8 for k in range(16))]
                    cl, a = max(spots)
                    if cl > 0.32:
                        cand.append((-dist, cl, o, word, a, dist))
                        break
        if not cand:
            return {"pass": None, "na": True, "reason": "no nameable object (cup, bottle, bowl, box, can) with a free standing place within 0.95 m"}
        _, _, o, word, a, dist = max(cand, key=lambda t: t[:2])
        ox, oy = o["pose"][0], o["pose"][1]
        start = (ox + math.cos(a) * dist, oy + math.sin(a) * dist, a + math.pi)
        res = {"target": o["id"], "label": word, "target_height_m": round(o["pose"][2], 2), "host": by[o["support"]]["class"], "start_distance_m": dist}
        ses = self.ses
        # 1. without Jev: the grammar does not know "go to": a defined refusal, and the robot does not move
        self.reset(start)
        p0 = self.truth()["pos"]
        st = ses.say(f"go to the {word}")
        ses.run(2.0)
        moved = math.dist(p0[:2], self.truth()["pos"][:2])
        res["without_jev"] = {"state": None if st is None else st.state, "message": None if st is None else st.message, "moved_m": round(moved, 3)}
        refused = st is not None and st.state != "running" and moved < 0.05
        seen = sorted({s_.label for s_ in ses.stack.scene(self.sim.t)})
        res["detector_sees"] = seen
        # 2. with Jev (the fake): the classifier names the intent, the grammar supplies the object, the stack approaches it
        fake = FakeJev()
        ses.stack.parser = CompositeParser(self.grammar, classifier=fake)
        d0 = math.dist(self.truth()["pos"][:2], (ox, oy))
        cmd = self.say_and_wait(f"go to the {word}", 45.0)
        ses.stack.stop("scenario end")
        ses.run(1.0)
        d1 = math.dist(self.truth()["pos"][:2], (ox, oy))
        res["with_jev"] = {**cmd, "classifier_calls": len(fake.calls), "distance_before_m": round(d0, 2), "distance_after_m": round(d1, 2)}
        ses.stack.parser = self.grammar
        found = word in seen
        approached = cmd["accepted"] == "running" and (d1 < d0 - 0.15 or cmd["final"] == "succeeded")
        defined = cmd["final"] == "failed" and bool(cmd["message"])
        ok = bool(refused and len(fake.calls) == 1 and (approached or defined) and not self.truth()["fallen"])
        res["outcome"] = "approached" if approached else "defined refusal" if defined else "undefined"
        return {"pass": ok, "reason": "" if ok else ("moved without understanding" if not refused else "classifier not asked once" if len(fake.calls) != 1 else "no approach and no defined refusal"),
                "target_found_by_detector": found, **res}

    # ------------------------------------------------------------------ S6: an obstacle appears in the path
    def s6_obstacle(self) -> dict:
        x, y, yaw = self.home
        yaw = self.best_heading(x, y, self.need)
        free = SC.free_run(self.obstacles, x, y, yaw, self.need)
        if free < 2.4:
            return {"pass": None, "na": True, "reason": f"only {free:.1f} m free ahead of the spawn pose ({self.need} m of room on each side)"}
        self.reset((x, y, yaw))
        ses, a = self.ses, self.truth()
        st = ses.say(f"go forward {min(free - 0.8, 2.5):.1f} meters")
        ses.run(20.0, until=lambda s_: math.dist(self.truth()["pos"][:2], a["pos"][:2]) > 0.3)
        p = self.truth()
        if st is None or st.state != "running" or math.dist(p["pos"][:2], a["pos"][:2]) <= 0.3:
            return {"pass": False, "reason": f"the robot never set off (command {None if st is None else st.state}: {None if st is None else st.message})"}
        bx, by_ = p["pos"][0] + math.cos(p["yaw"]) * 1.1, p["pos"][1] + math.sin(p["yaw"]) * 1.1
        self.box.set_world_pose(np.array([bx, by_, 0.45]))
        self.box.set_linear_velocity(np.zeros(3))
        t0, min_gap, levels = self.sim.t, math.inf, set()

        def watch(s_):
            nonlocal min_gap
            q = self.truth()["pos"]
            min_gap = min(min_gap, math.dist(q[:2], (bx, by_)) - 0.225 - BOUNDS["body_radius_m"])
            levels.add(s_.stack.safety)
            return s_.stack.executor.status.state in ("succeeded", "failed")

        ses.run(25.0, until=watch)
        t_end = self.sim.t - t0
        ses.run(1.0)
        bpos = np.asarray(self.box.get_world_pose()[0], float)
        box_moved = math.dist(bpos[:2], (bx, by_))
        fin = ses.stack.executor.status
        b = self.truth()
        ok = bool(min_gap >= BOUNDS["obstacle_gap_m"] and box_moved < 0.03 and not b["fallen"])
        return {"pass": ok, "reason": "" if ok else ("fell" if b["fallen"] else "touched or pushed the obstacle" if box_moved >= 0.03 or min_gap < 0 else "stopped too close"),
                "accepted": None if st is None else st.state, "obstacle_appeared_at_m": 1.1, "min_gap_m": round(min_gap, 3), "obstacle_moved_m": round(box_moved, 3),
                "final": fin.state, "message": fin.message, "safety": sorted(levels), "reasons": list(ses.stack.reasons), "seconds_to_end": round(t_end, 2)}

    # ------------------------------------------------------------------ S7: faults end in a defined safe state
    def s7_faults(self) -> dict:
        from humanoid_arc_bme.onboard.command.composite import CompositeParser
        from humanoid_arc_bme.onboard.command.jev import FakeJev
        from humanoid_arc_bme.onboard.runtime.groot_controller import build_command_message

        ses, out = self.ses, {}
        x, y, _ = self.home
        yaw = self.best_heading(x, y, self.need)
        room = SC.free_run(self.obstacles, x, y, yaw, self.need)
        go = f"go forward {min(room - 0.8, 2.0):.1f} meters"

        def walking_then(inject, name, expect_disarm=False, limit=3.5):
            self.reset((x, y, yaw))
            st = ses.say(go)
            ses.run(2.5)
            for _ in range(10):                                     # perception restarts on arming: give the robot time to set off before the fault
                v0 = self.moving(0.3)
                if v0 > 0.1:
                    break
            inject()
            t0, t_zero = self.sim.t, None
            while self.sim.t - t0 < limit:
                ses.run(0.1)
                if t_zero is None and max(abs(v) for v in ses.rt.last_cmd) < 1e-6:
                    t_zero = self.sim.t - t0
            b = self.truth()
            still = self.moving(0.3) < 0.05
            rec = {"walking_before_mps": round(v0, 3), "accepted": None if st is None else st.state, "zero_command_after_s": None if t_zero is None else round(t_zero, 2),
                   "standing_still": bool(still), "fallen": b["fallen"], "safety": ses.stack.safety, "reasons": list(ses.stack.reasons), "task": ses.stack.executor.status.state,
                   "task_message": ses.stack.executor.status.message, "armed": bool(ses.rt.sup.armed), "runtime_fault": ses.rt.fault}
            if expect_disarm:
                rec["pass"] = bool(t_zero is not None and t_zero <= BOUNDS["fault_stop_s"] and not ses.rt.sup.armed)
            else:
                rec["pass"] = bool(v0 > 0.05 and t_zero is not None and t_zero <= BOUNDS["fault_stop_s"] and still and not b["fallen"])
            out[name] = rec

        if room < 2.0:
            return {"pass": None, "na": True, "reason": f"only {room:.1f} m free ahead of the spawn pose ({self.need} m of room on each side)"}
        walking_then(lambda: self.rig.faults.__setitem__("d435_dropout", True), "depth_dropout")
        walking_then(lambda: self.rig.faults.__setitem__("d435_stale", True), "stale_frames")
        walking_then(lambda: ses.localizer.faults.__setitem__("lost", True), "localizer_unhealthy")
        walking_then(lambda: ses.localizer.faults.__setitem__("silent", True), "localizer_silent")
        walking_then(lambda: ses.io.faults.__setitem__("state_lost", True), "lowstate_lost", expect_disarm=True)
        # Jev outage: the sentence the grammar does not know is refused, nothing moves, and STOP still works
        fake = FakeJev(mode="timeout")
        self.reset((x, y, yaw))
        ses.stack.parser = CompositeParser(self.grammar, classifier=fake)
        p0 = self.truth()["pos"]
        st = ses.say("go to the cup")
        ses.run(1.5)
        moved_unknown = math.dist(p0[:2], self.truth()["pos"][:2])
        st2 = ses.say(go)
        ses.run(2.5)
        v = self.moving(0.3)
        ses.say("stop")
        t_still, _ = self.time_to_still()
        out["jev_outage"] = {"unknown_sentence": None if st is None else [st.state, st.message], "classifier_calls": len(fake.calls), "parser_error": ses.stack.parser.last_error,
                             "grammar_command_still_runs": None if st2 is None else st2.state, "speed_mps": round(v, 3), "stop_time_to_still_s": None if t_still is None else round(t_still, 2),
                             "moved_on_unknown_m": round(moved_unknown, 3)}
        out["jev_outage"]["pass"] = bool(st is not None and st.state != "running" and moved_unknown < 0.05 and len(fake.calls) == 1 and st2 is not None and st2.state == "running" and v > 0.05
                                         and t_still is not None and t_still <= BOUNDS["stop_latency_s"])
        ses.stack.parser = self.grammar
        # policy exit, last (the robot goes down): the deploy's control loop stops, rt/lowcmd stops, the runtime must notice and disarm
        self.reset((x, y, yaw))
        ses.say(go)
        ses.run(2.5)
        self.deploy.on_message(build_command_message(False, True, True), self.sim.t)
        t0, t_dis, t_fall = self.sim.t, None, None
        while self.sim.t - t0 < 6.0:
            ses.run(0.1)
            if t_dis is None and not ses.rt.sup.armed:
                t_dis = self.sim.t - t0
            if t_fall is None and self.truth()["fallen"]:
                t_fall = self.sim.t - t0
        out["policy_exit"] = {"motors_passive": self.robot.passive, "fallen_after_s": None if t_fall is None else round(t_fall, 2), "disarmed_after_s": None if t_dis is None else round(t_dis, 2),
                              "runtime_fault": ses.rt.fault, "last_cmd": list(ses.rt.last_cmd), "note": "what the real robot does when the deploy exits is UNKNOWN; here the motors go passive",
                              "pass": bool(t_dis is not None and self.robot.passive)}
        ok = all(v_["pass"] for v_ in out.values())
        return {"pass": bool(ok), "reason": "" if ok else "not safe on: " + ", ".join(k for k, v_ in out.items() if not v_["pass"]), "faults": out}

    # ------------------------------------------------------------------ all of them
    def run_all(self, which=("s1", "s2", "s3", "s4", "s5", "s6", "s7")) -> dict:
        table = {"s1": self.s1_stand, "s2": self.s2_goal, "s3": self.s3_door, "s4": self.s4_teleop, "s5": self.s5_object, "s6": self.s6_obstacle, "s7": self.s7_faults}
        res = {}
        for k in which:
            a, t_sim = time.perf_counter(), self.sim.t
            try:
                r = table[k]()
            except Exception as exc:  # noqa: BLE001 - one scenario's crash is that scenario's failure
                import traceback
                r = {"pass": False, "reason": f"exception: {type(exc).__name__}: {exc}"[:300], "traceback": traceback.format_exc()[-800:]}
            r["wall_s"], r["sim_s"] = round(time.perf_counter() - a, 1), round(self.sim.t - t_sim, 1)
            res[k] = r
        return res

    def timing(self) -> dict:
        ses, sim = self.ses, self.sim
        ms = np.array(ses.tick_ms) if ses.tick_ms else np.array([0.0])
        return {"scene_load_s": round(self.load_s, 2), "reset_s_median": round(float(np.median(self.reset_times)), 2) if self.reset_times else None,
                "resets": len(self.reset_times), "sim_seconds": round(sim.t, 1), "rtf_median": round(sim.rtf(), 2),
                "slice_ms_median": round(sim.dt * sim.steps_per_outer / sim.rtf() * 1e3, 1), "stack_tick_ms": {"median": round(float(np.median(ms)), 2),
                                                                                                       "p95": round(float(np.percentile(ms, 95)), 2), "max": round(float(ms.max()), 2)}}
