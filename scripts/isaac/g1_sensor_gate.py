"""Gate WP4 in Isaac Sim: each simulated sensor sees known geometry where it is, and the mounts match Unitree's asset.

    python.sh scripts/isaac/g1_sensor_gate.py --out result.json

World: carpet floor (z = 0), a wall whose face is at x = 2.5 m, a 0.6 m wide, 1.5 m tall pillar on the robot's LEFT at (1.2, 1.2). The G1 stands at the
origin under the stand-in policy. Checks, all against simulator ground truth:
  mounts   every ASSET frame of config/sensor_mounts.yaml equals the asset's frame (position in the pelvis frame, pitch);
  cameras  true depth back-projected through the intrinsics, the optical-frame conversion and the camera's true pose lands on the
           floor, the wall or the box; the noise model's error at the measured ranges; minimum range and dropout behave as configured;
  lidar    returns land on the same surfaces, the box is on the left (azimuth sign), the floor is below (elevation sign), a frame
           spans 0.1 s of per-point time in 4 slices;
  imu      specific force is gravity (9.81 m/s^2) and the gyro is quiet for a standing robot.
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
ap.add_argument("--out", required=True)
args, _ = ap.parse_known_args()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid_arc_bme.isaac.app import make_app  # noqa: E402

app = make_app("g1_sensor_gate", lidar=True)

import numpy as np  # noqa: E402

WALL_X, BOX, BOX_H = 2.5, (1.2, 1.2, 0.6), 1.5   # wall face; pillar centre x, y, edge length; pillar height (taller than the lidar: its lowest ray
#                                                  points about 13 degrees down, so it sees nothing low and near)
TOL = 0.04                                  # m: a back-projected point counts as "on a known surface" within this


def on_known_surface(p: np.ndarray) -> np.ndarray:
    """True for world points on the floor, the wall face or the box (the robot's own body is not a known surface)."""
    half = BOX[2] / 2 + TOL
    floor = np.abs(p[:, 2]) < TOL
    wall = np.abs(p[:, 0] - WALL_X) < TOL
    box = (np.abs(p[:, 0] - BOX[0]) < half) & (np.abs(p[:, 1] - BOX[1]) < half) & (p[:, 2] > -TOL) & (p[:, 2] < BOX_H + TOL)
    return floor | wall | box


def main() -> dict:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid

    from humanoid_arc_bme.isaac import standin as S
    from humanoid_arc_bme.isaac.rates import PHYSICS_HZ
    from humanoid_arc_bme.isaac.robot import SimG1
    from humanoid_arc_bme.isaac.sensors import SensorRig
    from humanoid_arc_bme.isaac.sim import G1Sim
    from humanoid_arc_bme.onboard.sensors import mounts as MT
    from humanoid_arc_bme.robots.assets import AGILE_LOCOMOTION_POLICY, G1_29DOF_USD

    mounts = MT.load_mounts()
    world = World(physics_dt=1.0 / PHYSICS_HZ, rendering_dt=SensorRig.slice_dt_of(mounts), stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane(static_friction=0.9, dynamic_friction=0.8, restitution=0.0)
    FixedCuboid("/World/wall", position=np.array([WALL_X + 0.05, 0.0, 1.2]), scale=np.array([0.1, 8.0, 2.4]), color=np.array([0.8, 0.8, 0.75]))
    FixedCuboid("/World/box", position=np.array([BOX[0], BOX[1], BOX_H / 2]), scale=np.array([BOX[2], BOX[2], BOX_H]), color=np.array([0.8, 0.1, 0.1]))
    robot = SimG1(str(G1_29DOF_USD))
    rig = SensorRig(robot.prim_path, mounts, world, seed=0)
    res = {"mounts": {}, "cameras": {}, "lidar": {}, "imu": {}}

    # ---- mounts against the asset (default pose, before physics)
    p_pel, R_pel = rig.usd_pose(f"{robot.prim_path}/pelvis")
    for name, f in mounts.raw["frames"].items():
        path = f"{robot.prim_path}/{f['parent']}/{f['asset_frame'] or name + '_mount'}"
        p, R = rig.usd_pose(path)
        rel = R_pel.T @ (p - p_pel)
        x_axis = R_pel.T @ R[:, 0]
        pitch = math.degrees(math.asin(-x_axis[2]))
        res["mounts"][name] = {"tag": f["tag"], "config_xyz": f["xyz_pelvis"], "asset_xyz": [round(float(v), 4) for v in rel], "config_pitch": f["pitch_down_deg"],
                               "asset_pitch": round(pitch, 2), "pass": bool(np.allclose(rel, f["xyz_pelvis"], atol=0.001) and abs(pitch - f["pitch_down_deg"]) < 0.15)}
    world.reset()
    robot.start()
    rig.start(robot.art)
    sim = G1Sim(world, robot, controller=S.DeployStandIn(str(AGILE_LOCOMOTION_POLICY)), rig=rig)
    frames = [0]
    sim.on_frame = lambda s: frames.__setitem__(0, frames[0] + 1)
    t_wall = time.perf_counter()
    sim.advance(3.0)
    wall_s = time.perf_counter() - t_wall
    res["timing"] = {"sim_seconds": 3.0, "frames": frames[0], "slice_ms_median": round(sim.dt * sim.steps_per_outer / sim.rtf() * 1e3, 2), "slices": len(sim.step_times),
                     "physics_steps_per_slice": sim.steps_per_outer, "rtf_with_sensors_wall": round(3.0 / wall_s, 2), "rtf_with_sensors_median": round(sim.rtf(), 2)}
    truth = robot.truth()
    res["robot"] = {"pelvis_z": round(truth["pos"][2], 3), "tilt_deg": round(truth["tilt_deg"], 2), "fallen": truth["fallen"]}

    # ---- cameras: true depth through intrinsics + optical conversion + true pose must land on known surfaces
    for name, c in rig.cam.items():
        K, d = c["K"], c["depth_true"]
        if d is None:
            res["cameras"][name] = {"pass": False, "error": "no frame"}
            continue
        u, v = np.meshgrid(np.arange(K.width), np.arange(K.height))
        ok = np.isfinite(d) & (d > 0.05) & (d < 20)
        z = d[ok]
        p_opt = np.stack([(u[ok] - K.cx) / K.fx * z, (v[ok] - K.cy) / K.fy * z, z], axis=-1)
        p_usd = p_opt @ MT.OPTICAL_FROM_USD.T                          # the conversion is its own inverse
        pc, Rc = rig.frame_world_pose(c["prim"])
        pw = p_usd @ Rc.T + pc
        known = on_known_surface(pw)
        near_body = np.linalg.norm(pw[:, :2] - np.array(truth["pos"][:2]), axis=1) < 0.45      # the robot sees its own feet and hands
        frac = float(known[~near_body].mean()) if (~near_body).any() else 0.0
        noisy = c["depth"]
        both = ok & np.isfinite(noisy)
        err = (noisy[both] - d[both])
        cfg = c["cfg"]
        expect = np.sqrt(np.mean((mounts.depth_noise().sigma0 + mounts.depth_noise().sigma_k * d[both] ** 2) ** 2))
        cy, cx = K.height // 2, K.width // 2
        centre = pw.reshape(-1, 3)[np.flatnonzero((u[ok] == cx) & (v[ok] == cy))]
        res["cameras"][name] = {
            "resolution": [K.width, K.height], "fx": round(K.fx, 2), "valid_fraction": round(float(ok.mean()), 3),
            "on_known_surface_fraction": round(frac, 4), "self_view_fraction": round(float(near_body.mean()), 3),
            "centre_ray_hits_world": [round(float(x), 3) for x in centre[0]] if len(centre) else None, "depth_range_m": [round(float(z.min()), 3), round(float(z.max()), 3)],
            "noise_rms_m": round(float(np.sqrt(np.mean(err ** 2))), 4), "noise_rms_expected_m": round(float(expect), 4),
            "dropout_fraction": round(float(1.0 - both.sum() / max(1, ok.sum())), 4), "below_min_range_invalid": bool(not np.isfinite(noisy[ok & (d < cfg["min_range_m"])]).any()),
            "rgb_shape": list(c["rgb"].shape) if c["rgb"] is not None else None, "rgb_mean": round(float(c["rgb"].mean()), 1) if c["rgb"] is not None else None,
            "pass": bool(frac > 0.97 and 0.6 * expect < np.sqrt(np.mean(err ** 2)) < 1.6 * expect and c["rgb"] is not None and c["rgb"].mean() > 5)}
        # with the NOMINAL mount (what the stack uses) and no attitude correction: how far the floor appears from z = 0
        if name == "d435":
            from humanoid_arc_bme.onboard.pipeline.perception.sensing import depth_to_points
            from humanoid_arc_bme.onboard.pipeline.interfaces import DepthFrame
            pb = depth_to_points(DepthFrame(0.0, d, K, c["pose"], camera="down"), max_range=10.0)
            pb = pb[np.isfinite(pb).all(axis=1)]
            res["cameras"][name]["floor_z_with_nominal_mount_m"] = {"median": round(float(np.median(pb[:, 2])), 3), "note": "body leans forward under the stand-in; the stack corrects with IMU roll/pitch"}

    # ---- lidar
    raw = rig.lidar_raw()
    if raw is None:
        res["lidar"] = {"pass": False, "error": "no frame"}
    else:
        pts, times, t_end = raw
        pl, Rl = rig.frame_world_pose(rig.lidar["prim"])
        pw = pts @ Rl.T + pl
        near_body = np.linalg.norm(pw[:, :2] - np.array(truth["pos"][:2]), axis=1) < 0.45
        known = on_known_surface(pw)
        fwd = pw[(np.abs(pw[:, 1]) < 0.3) & (pw[:, 2] > 0.3) & (pw[:, 0] > 1.0)]
        left_box = ((np.abs(pw[:, 0] - BOX[0]) < 0.34) & (np.abs(pw[:, 1] - BOX[1]) < 0.34) & (pw[:, 2] > 0.05)).sum()
        right_mirror = ((np.abs(pw[:, 0] - BOX[0]) < 0.34) & (np.abs(pw[:, 1] + BOX[1]) < 0.34) & (pw[:, 2] > 0.05)).sum()
        floor = pw[np.abs(pw[:, 2]) < TOL]
        dts = np.diff(rig.lidar["render_time_ns"][-8:]) * 1e-9
        res["lidar"] = {"points_per_frame": int(len(pts)), "on_known_surface_fraction": round(float(known[~near_body].mean()), 4), "self_hits_fraction": round(float(near_body.mean()), 3),
                        "wall_ahead_x_median": round(float(np.median(fwd[:, 0])), 3) if len(fwd) else None, "wall_expected_x": WALL_X,
                        "box_points_left": int(left_box), "box_points_if_mirrored": int(right_mirror), "floor_points": int(len(floor)),
                        "nearest_floor_return_m": round(float(np.linalg.norm(floor[:, :2] - pl[:2], axis=1).min()), 2) if len(floor) else None,
                        "time_span_s": round(float(times.max() - times.min()), 4), "frame_end_minus_last_point_s": round(float(t_end - times.max()), 4),
                        "render_period_s": [round(float(x), 4) for x in dts], "range_m": [round(float(np.linalg.norm(pts, axis=1).min()), 2), round(float(np.linalg.norm(pts, axis=1).max()), 2)],
                        "elevation_deg": [round(float(np.degrees(np.arcsin(pts[:, 2] / np.linalg.norm(pts, axis=1))).min()), 1),
                                          round(float(np.degrees(np.arcsin(pts[:, 2] / np.linalg.norm(pts, axis=1))).max()), 1)]}
        L = res["lidar"]
        L["pass"] = bool(L["on_known_surface_fraction"] > 0.95 and L["wall_ahead_x_median"] is not None and abs(L["wall_ahead_x_median"] - WALL_X) < 0.05
                         and left_box > 20 and right_mirror == 0 and len(floor) > 50 and 0.07 < L["time_span_s"] <= 0.1001)
    # ---- IMU
    s = rig.imu_samples()[-200:]
    acc, gyr = np.array([x[2] for x in s]), np.array([x[1] for x in s])
    res["imu"] = {"rate_hz": round(len(s) / (s[-1][0] - s[0][0] + sim.dt), 1), "specific_force_norm": round(float(np.linalg.norm(acc.mean(axis=0))), 3),
                  "specific_force_mean": [round(float(v), 3) for v in acc.mean(axis=0)], "gyro_mean": [round(float(v), 4) for v in gyr.mean(axis=0)],
                  "accel_std": round(float(acc.std(axis=0).mean()), 3)}
    res["imu"]["pass"] = bool(abs(res["imu"]["specific_force_norm"] - 9.81) < 0.3 and np.abs(gyr.mean(axis=0)).max() < 0.05)
    res["pass"] = bool(all(v["pass"] for v in res["mounts"].values() if v["tag"] == "ASSET") and all(v["pass"] for v in res["cameras"].values())
                       and res["lidar"]["pass"] and res["imu"]["pass"] and not truth["fallen"])
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
