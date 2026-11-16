"""The simulated G1's sensors, built from config/sensor_mounts.yaml: two RealSense-like RGB-D cameras, a Mid-360-like RTX lidar
and its IMU. Needs Kit.

    app = make_app(..., lidar=True)                       # Kit must know the lidar profile folder at start-up
    world = World(physics_dt=1 / 200, rendering_dt=SensorRig.slice_dt_of(mounts))   # one render = one lidar slice
    rig = SensorRig(robot_prim_path, mounts, world)       # before world.reset(); cameras and lidar are children of the asset's own frames
    rig.start(robot.art)                                  # after world.reset()
    sim = G1Sim(world, robot, controller=..., rig=rig)    # sim.step() = one lidar slice: physics steps, IMU samples, one render
    rig.depth("d435") / rig.rgb("head_rgbd") / rig.lidar_frame() / rig.lidar_raw() / rig.imu_samples()

Time base. Isaac's RTX lidar advances by the app's rendering period at every render and returns the returns fired in that period with a
per-point time offset. A frame (10 Hz) is therefore rendered in `subframes` slices: the sensor is where the robot is at each slice, so a
walking robot distorts the scan as on the real unit, and each point keeps its own time. The cameras are read on the last slice only.
The lidar pattern is an approximation of the Mid-360 (isaac/lidar_profiles); the depth error model is the stack's own `DepthNoise`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from humanoid_arc_bme.isaac.robot import quat_rotate_inv
from humanoid_arc_bme.onboard.pipeline.interfaces import DepthFrame
from humanoid_arc_bme.onboard.pipeline.perception.lidar import LidarFrame
from humanoid_arc_bme.onboard.pipeline.perception.sensing import apply_depth_noise
from humanoid_arc_bme.onboard.sensors import mounts as MT

PROFILES = Path(__file__).resolve().parent / "lidar_profiles"
GRAVITY = 9.81


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def spherical_to_xyz(az_deg, el_deg, dist) -> np.ndarray:
    """Lidar returns (azimuth about +z from +x, elevation above the xy plane, range) -> (N, 3) in the sensor frame (x forward, z up)."""
    az, el, d = np.radians(az_deg), np.radians(el_deg), np.asarray(dist, float)
    return np.stack([d * np.cos(el) * np.cos(az), d * np.cos(el) * np.sin(az), d * np.sin(el)], axis=-1)


class SensorRig:
    def __init__(self, robot_prim_path: str, mounts: MT.Mounts, world, seed: int = 0, cameras: tuple = ("d435", "head_rgbd"), lidar: bool = True,
                 segmentation: bool = False):
        import omni.kit.commands
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom

        self.m, self.world, self.rng = mounts, world, np.random.default_rng(seed)
        stage = omni.usd.get_context().get_stage()
        self.robot_path = robot_prim_path
        lc = mounts.raw["lidar"]
        self.frame_dt, self.nsub = 1.0 / lc["rate_hz"], int(lc["subframes"])
        self.slice_dt = self.frame_dt / self.nsub
        self._rep, self.cam, self.faults = rep, {}, {}
        for name in cameras:
            f, c = mounts.frame(name), mounts.raw["cameras"][name]
            if f["asset_frame"]:                                       # Unitree's own frame in the asset: the camera sits exactly on it
                base = f"{robot_prim_path}/{f['parent']}/{f['asset_frame']}"
                if not stage.GetPrimAtPath(base):
                    raise RuntimeError(f"asset has no frame {base}")
            else:                                                      # a mount the asset does not have: a new frame under the parent link
                base = f"{robot_prim_path}/{f['parent']}/{name}_mount"
                parent_w = UsdGeom.Xformable(stage.GetPrimAtPath(f"{robot_prim_path}/{f['parent']}")).ComputeLocalToWorldTransform(0)
                pelvis_w = UsdGeom.Xformable(stage.GetPrimAtPath(f"{robot_prim_path}/pelvis")).ComputeLocalToWorldTransform(0)
                R = MT.link_rotation_for_pitch(f["pitch_down_deg"])
                T_pelvis = Gf.Matrix4d(Gf.Matrix3d(*R.T.flatten().tolist()), Gf.Vec3d(*map(float, f["xyz_pelvis"])))    # USD: row vectors
                local = T_pelvis * pelvis_w * parent_w.GetInverse()
                x = UsdGeom.Xform.Define(stage, base)
                x.ClearXformOpOrder()
                x.AddTransformOp().Set(local)
            cam = UsdGeom.Camera.Define(stage, f"{base}/camera")
            K = mounts.intrinsics(name)
            ap = 20.955
            cam.CreateHorizontalApertureAttr(ap)
            cam.CreateVerticalApertureAttr(ap * K.height / K.width)
            cam.CreateFocalLengthAttr(K.fx * ap / K.width)
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 100.0))
            xf = UsdGeom.Xformable(cam.GetPrim())
            xf.ClearXformOpOrder()
            q = MT.quat_of(MT.USD_CAMERA_IN_LINK)
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[0], q[1], q[2], q[3]))
            rp = rep.create.render_product(str(cam.GetPath()), (K.width, K.height))
            a_rgb, a_d = rep.AnnotatorRegistry.get_annotator("rgb"), rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            a_rgb.attach([rp])
            a_d.attach([rp])
            a_seg = None
            if segmentation:                                           # ground-truth labels (simulation only): a stand-in detector reads them
                a_seg = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False})
                a_seg.attach([rp])
            self.cam[name] = {"prim": str(cam.GetPath()), "rp": rp, "rgb_a": a_rgb, "depth_a": a_d, "seg_a": a_seg, "K": K, "cfg": c,
                              "pose": mounts.camera_pose(name, "sim"), "rgb": None, "depth": None, "depth_true": None, "seg": None, "stamp": -math.inf}
        # The asset's head shell is one closed visual mesh around the lidar and the D435 frames. RTX lidar rays do not cull back faces, so
        # the shell swallowed every ray above about 7 degrees (measured). The real unit looks out through the head's openings; how much
        # of its field of view the real shell clips is UNMEASURED. Here the shell is hidden from all sensors.
        inst = stage.GetPrimAtPath(f"{robot_prim_path}/torso_link/visuals")
        if inst and inst.IsInstance():
            inst.SetInstanceable(False)                                # the visuals are instanced: un-instance to edit one child
        shell = stage.GetPrimAtPath(f"{robot_prim_path}/torso_link/visuals/head_link")
        if shell:
            UsdGeom.Imageable(shell).MakeInvisible()
        self.lidar = None
        if lidar:
            lf = mounts.frame("mid360")
            parent = f"{robot_prim_path}/{lf['parent']}/{lf['asset_frame']}"
            _, prim = omni.kit.commands.execute("IsaacSensorCreateRtxLidar", path="/lidar", parent=parent, config=lc["profile"], force_camera_prim=True)
            xf = UsdGeom.Xformable(prim)                               # identity in the asset's lidar frame (x forward, z up)
            xf.ClearXformOpOrder()
            rp = rep.create.render_product(str(prim.GetPath()), (1, 1))
            gmo = rep.AnnotatorRegistry.get_annotator("GenericModelOutput")
            gmo.attach([rp])
            self.lidar = {"prim": str(prim.GetPath()), "rp": rp, "gmo": gmo, "cfg": lc, "pose": mounts.lidar_pose("sim"), "slices": [], "frame": None,
                          "raw": None, "stamp": -math.inf, "render_time_ns": []}
        ic = mounts.raw["imu"]
        self.imu_cfg = ic
        self._gyro_bias = self.rng.normal(0.0, ic["gyro_bias_rad_s"], 3)
        self._accel_bias = self.rng.normal(0.0, ic["accel_bias_m_s2"], 3)
        self._imu: list = []
        self._imu_prev_v = None
        self.art = None
        self._slice = 0

    @staticmethod
    def slice_dt_of(mounts: MT.Mounts) -> float:
        """The rendering period the World must be created with: one lidar slice."""
        return 1.0 / mounts.raw["lidar"]["rate_hz"] / int(mounts.raw["lidar"]["subframes"])

    # ------------------------------------------------------------------ set-up
    def start(self, art) -> None:
        """`art`: the robot's Articulation (isaacsim.core.prims), initialised. Link poses and velocities are read from PhysX through
        it: an articulation link has no rigid-body view of its own, and USD transforms are not a reliable source while simulating."""
        import omni.usd
        from pxr import UsdGeom

        physics_dt = self.world.get_physics_dt()
        if abs(self.slice_dt / physics_dt - round(self.slice_dt / physics_dt)) > 1e-6:
            raise RuntimeError("a lidar slice must be a whole number of physics steps")
        if abs(self.world.get_rendering_dt() - self.slice_dt) > 1e-9:  # changing it now would invalidate every physics view (measured)
            raise RuntimeError(f"create the World with rendering_dt={self.slice_dt} (one render = one lidar slice); it is {self.world.get_rendering_dt()}")
        self.art = art
        names = list(art.body_names)
        stage = omni.usd.get_context().get_stage()
        self._static = {}                                              # sensor prim -> (link index, local position, local rotation) fixed in its link
        paths = [c["prim"] for c in self.cam.values()] + ([self.lidar["prim"]] if self.lidar else [])
        lf = self.m.frame("mid360")
        self._imu_frame = f"{self.robot_path}/{lf['parent']}/{lf['asset_frame']}"
        for path in paths + [self._imu_frame] + [f"{self.robot_path}/{n}" for n in ("pelvis", "torso_link")]:
            link = path[len(self.robot_path) + 1:].split("/")[0]
            m_p = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)
            m_l = UsdGeom.Xformable(stage.GetPrimAtPath(f"{self.robot_path}/{link}")).ComputeLocalToWorldTransform(0)
            rel = m_p * m_l.GetInverse()                              # both read from USD at the same instant: only the ratio is used
            R = np.array([[rel[j][i] for j in range(3)] for i in range(3)])
            self._static[path] = (names.index(link), np.array(rel.ExtractTranslation()), R)

    def _link_state(self, idx: int):
        """(position, quaternion wxyz, linear velocity, angular velocity) of a robot link in the world, from PhysX."""
        tr = np.asarray(self.art._physics_view.get_link_transforms())[0][idx]
        ve = np.asarray(self.art._physics_view.get_link_velocities())[0][idx]
        return tr[:3].astype(float), np.array([tr[6], tr[3], tr[4], tr[5]], float), ve[:3].astype(float), ve[3:].astype(float)

    def frame_world_pose(self, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
        """(position, rotation matrix) of a sensor frame in the world (ground truth: for checks and scoring, never for the stack)."""
        idx, p_l, R_l = self._static[prim_path]
        p, q, _, _ = self._link_state(idx)
        R = quat_to_matrix(q)
        return p + R @ p_l, R @ R_l

    def usd_pose(self, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
        """(position, rotation matrix) of a prim as authored in USD (the robot's default pose before simulation starts)."""
        import omni.usd
        from pxr import UsdGeom

        m = UsdGeom.Xformable(omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)).ComputeLocalToWorldTransform(0)
        return np.array(m.ExtractTranslation()), np.array([[m[j][i] for j in range(3)] for i in range(3)])     # USD stores row vectors

    # ------------------------------------------------------------------ per lidar slice (one stepped and rendered app update)
    def steps_per_slice(self, physics_dt: float) -> int:
        return max(1, round(self.slice_dt / physics_dt))

    def before_render(self) -> None:
        self._slice += 1

    def after_render(self, t: float) -> bool:
        """After the app update, at simulation time `t`: collect the lidar slice and, on the last slice, the frame. True = new frame.
        The cameras render on every slice (switching a render product off and on stalls Kit for about 100 ms: measured) and are
        read on the last one."""
        if self.lidar is not None:
            self._read_lidar_slice(t)
        if self._slice % self.nsub:
            return False
        for name, c in self.cam.items():
            self._read_camera(name, c, t)
        if self.lidar is not None:
            self._finish_lidar_frame(t)
        return True

    # ------------------------------------------------------------------ cameras
    def _read_camera(self, name: str, c: dict, t: float) -> None:
        if self.faults.get(f"{name}_stale"):
            return                                                     # the driver stopped delivering: the old frame and stamp stay
        rgba, d = c["rgb_a"].get_data(), c["depth_a"].get_data()
        if d is None or getattr(d, "size", 0) == 0:
            return
        d = np.asarray(d, dtype=np.float32).reshape(c["K"].height, c["K"].width)
        c["depth_true"] = d.copy()
        cfg = c["cfg"]
        out = apply_depth_noise(d, self.m.depth_noise(), self.rng)
        with np.errstate(invalid="ignore"):
            out[np.isfinite(d) & (d < cfg["min_range_m"])] = np.nan      # nearer than the stereo pair resolves: no depth
            out[np.isfinite(d) & (d > cfg["max_range_m"])] = np.inf
        if self.faults.get(f"{name}_dropout"):
            out[:] = np.nan                                            # covered or failed imager: every pixel invalid
        c["depth"], c["stamp"] = out, t
        if rgba is not None and getattr(rgba, "size", 0):
            img = np.asarray(rgba).reshape(c["K"].height, c["K"].width, -1)[:, :, :3]
            c["rgb"] = self._auto_exposure(c, img)
        if c["seg_a"] is not None:
            sd = c["seg_a"].get_data()
            if sd is not None and getattr(sd.get("data"), "size", 0):
                labels = {int(k): v.get("class", "") for k, v in sd["info"]["idToLabels"].items()}
                c["seg"] = (np.asarray(sd["data"]).reshape(c["K"].height, c["K"].width).astype(np.int32),
                            {k: v for k, v in labels.items() if "#" in v})      # only objects labelled `<word>#<object id>` by scene.load_scene

    AE_TARGET, AE_MAX_GAIN, AE_SMOOTH = 105.0, 8.0, 0.3        # GUESS: a RealSense auto-exposure holds the mean grey level near mid-range

    def _auto_exposure(self, c: dict, img: np.ndarray) -> np.ndarray:
        """The cameras auto-expose like the real ones: one gain per frame brings the mean level to AE_TARGET (never darkens), smoothed over
        frames. Scenes lit for the eye ("evening" preset) otherwise give black images that no detector can use."""
        mean = float(img.mean()) + 1e-3
        want = min(self.AE_MAX_GAIN, max(1.0, self.AE_TARGET / mean))
        g = c["ae_gain"] = c.get("ae_gain", want) * (1.0 - self.AE_SMOOTH) + want * self.AE_SMOOTH
        return np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)

    def segmentation(self, name: str = "d435"):
        """(ids (H, W) int32, {id: "<word>#<object id>"}) of the newest frame, or None. Ground truth: not available on a robot."""
        return self.cam[name]["seg"]

    def depth(self, name: str = "d435") -> DepthFrame | None:
        c = self.cam[name]
        if c["depth"] is None:
            return None
        return DepthFrame(c["stamp"], c["depth"], c["K"], c["pose"], camera="down" if name == "d435" else name)

    def rgb(self, name: str = "d435"):
        return self.cam[name]["rgb"]

    # ------------------------------------------------------------------ lidar
    def _read_lidar_slice(self, t: float) -> None:
        from isaacsim.sensors.rtx import get_gmo_data

        L = self.lidar
        buf = L["gmo"].get_data()                                      # keep `buf` alive: the parsed struct points into it
        if buf is None or getattr(buf, "size", 0) == 0:
            return
        g = get_gmo_data(buf)
        n = int(g.numElements)
        L["render_time_ns"] = (L["render_time_ns"] + [int(g.timestampNs)])[-8:]
        if n == 0 or L["render_time_ns"][-1] == (L["render_time_ns"][-2] if len(L["render_time_ns"]) > 1 else None):
            return                                                     # nothing fired, or the same slice again (never with synchronous rendering)
        az, el, dist = np.array(g.x[:n], float), np.array(g.y[:n], float), np.array(g.z[:n], float)
        off = np.array(g.timeOffSetNs[:n], float) * 1e-9
        ok = (dist >= L["cfg"]["min_range_m"]) & (dist <= L["cfg"]["max_range_m"])
        dist = dist + self.rng.normal(0.0, L["cfg"]["range_noise_m"], n)
        pts = spherical_to_xyz(az[ok], el[ok], dist[ok])
        L["slices"].append((pts, (t - self.slice_dt) + off[ok]))       # the slice covers the period that ends at this render

    def _finish_lidar_frame(self, t: float) -> None:
        L = self.lidar
        slices, L["slices"] = L["slices"], []
        if self.faults.get("lidar_off") or not slices:
            return
        pts = np.concatenate([s[0] for s in slices])
        times = np.concatenate([s[1] for s in slices])
        L["raw"] = (pts, times, t)                                     # sensor frame, per-point time: what a driver hands to DLIO
        pose = L["pose"]
        L["frame"] = LidarFrame(t, pts @ quat_to_matrix(pose.quat).T + np.asarray(pose.position))   # body frame with the nominal mount, no deskew
        L["stamp"] = t

    def lidar_frame(self) -> LidarFrame | None:
        return None if self.lidar is None else self.lidar["frame"]

    def lidar_raw(self):
        """(points (N, 3) in the lidar frame, per-point times (N,), frame end time) of the newest frame, or None."""
        return None if self.lidar is None else self.lidar["raw"]

    # ------------------------------------------------------------------ IMU (at the lidar frame)
    def sample_imu(self, t: float, dt: float) -> None:
        idx, p_l, _ = self._static[self._imu_frame]
        _, q, lin, ang = self._link_state(idx)
        v = lin + np.cross(ang, quat_to_matrix(q) @ p_l)               # velocity of the IMU point (lever arm from the link origin)
        acc_w = np.zeros(3) if self._imu_prev_v is None else (v - self._imu_prev_v) / max(dt, 1e-6)
        self._imu_prev_v = v
        c = self.imu_cfg
        gyro = quat_rotate_inv(q, ang) + self._gyro_bias + self.rng.normal(0.0, c["gyro_noise_rad_s"], 3)
        accel = quat_rotate_inv(q, acc_w + np.array([0.0, 0.0, GRAVITY])) + self._accel_bias + self.rng.normal(0.0, c["accel_noise_m_s2"], 3)
        self._imu.append((t, gyro, accel))

    def imu_samples(self, clear: bool = True) -> list:
        """[(t, gyro (3) rad/s, specific force (3) m/s^2)] in the lidar frame's parent link axes since the last call."""
        out = self._imu
        if clear:
            self._imu = []
        return out
