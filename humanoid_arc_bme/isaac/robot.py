"""The simulated G1: Unitree's 29-DoF asset with Dex3 hands in an Isaac Sim world, driven only through `LowCmd`.

Needs Kit (create the SimulationApp first). Uses the plain Isaac Sim core API on CPU PhysX with the numpy backend: one robot, no
Isaac Lab. The robot is added to whatever stage is open; the scene around it is somebody else's business.
"""

from __future__ import annotations

import math

import numpy as np

from humanoid_arc_bme.isaac import g1_model as M
from humanoid_arc_bme.isaac.lowlevel import LowCmd, LowState, SimState
from humanoid_arc_bme.onboard.g1_layout import LEFT_HAND_JOINTS, MOTOR_JOINTS, RIGHT_HAND_JOINTS


def quat_rotate_inv(q, v):
    """Vector `v` (world) expressed in the frame of quaternion q = (w, x, y, z) (body)."""
    w, x, y, z = q
    u = np.array([x, y, z])
    return v - 2.0 * w * np.cross(u, v) + 2.0 * np.cross(u, np.cross(u, v))


class SimG1:
    def __init__(self, usd_path: str, prim_path: str = "/World/Robot", floor_pose=(0.0, 0.0, 0.0)):
        """Reference the robot asset at `floor_pose` = (x, y, yaw). Call `start()` after `world.reset()`."""
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(str(usd_path))
        x, y, yaw = floor_pose
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), M.ROOT_HEIGHT))
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
        # Contact resolution as in the Isaac Lab robot configuration: a link that starts a few millimetres inside the floor is pushed
        # out at 1 m/s at most. Without the cap PhysX launched the robot upward at 2.6 m/s on the first step (measured).
        from pxr import Sdf, Usd, UsdPhysics
        for p in Usd.PrimRange(prim):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                p.AddAppliedSchema("PhysxRigidBodyAPI")
                p.CreateAttribute("physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float).Set(M.MAX_DEPENETRATION_VELOCITY)
        self.prim_path, self.spawn = prim_path, (float(x), float(y), float(yaw))
        self.art = None
        self.tick = 0
        self._last_cmd_t = -math.inf
        self._passive = True
        self._gains = None

    # ------------------------------------------------------------------ set-up
    def start(self) -> None:
        from isaacsim.core.prims import Articulation

        self.art = Articulation(prim_paths_expr=self.prim_path, name="g1")
        self.art.initialize()
        names = list(self.art.dof_names)
        self.dof_names = names
        missing = [n for n in MOTOR_JOINTS + LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS if n not in names]
        if missing:
            raise RuntimeError(f"robot asset lacks joints the motor table names: {missing}")
        self.i_motor = np.array([names.index(n) for n in MOTOR_JOINTS])
        self.i_hand = (np.array([names.index(n) for n in LEFT_HAND_JOINTS]), np.array([names.index(n) for n in RIGHT_HAND_JOINTS]))
        self.i_all = np.concatenate([self.i_motor, *self.i_hand])
        self.art.set_armatures(np.array([M.armature(names)]))
        self.art.set_solver_position_iteration_counts(np.array([M.SOLVER_ITERATIONS[0]]))
        self.art.set_solver_velocity_iteration_counts(np.array([M.SOLVER_ITERATIONS[1]]))
        self.art.set_enabled_self_collisions(np.array([False]))
        q0 = np.zeros(len(names))
        q0[self.i_motor] = M.default_pose()
        self.art.set_joint_positions(q0[None])
        self.art.set_joint_velocities(np.zeros((1, len(names))))
        self.art.set_joint_position_targets(q0[None])
        hk, hd = M.gains(M.HAND_JOINTS)
        self._hand_gains = (np.array(hk), np.array(hd))
        self._set_gains(np.zeros(29), np.full(29, M.PASSIVE_KD))        # passive until the first command arrives

    def _set_gains(self, kp, kd) -> None:
        key = (kp.tobytes(), kd.tobytes())
        if key == self._gains:
            return
        self._gains = key
        k, d = np.zeros(len(self.dof_names)), np.zeros(len(self.dof_names))
        k[self.i_motor], d[self.i_motor] = kp, kd
        k[np.concatenate(self.i_hand)], d[np.concatenate(self.i_hand)] = self._hand_gains
        self.art.set_gains(kps=k[None], kds=d[None])

    # ------------------------------------------------------------------ the low-level interface
    def apply(self, cmd: LowCmd | None, now: float, fresh: bool) -> None:
        """Apply the newest command at simulation time `now`. `fresh` = a new command arrived since the last call; when none has for
        LOWCMD_TIMEOUT_S the motors go passive (damping only), which is what a silent controller leaves behind."""
        if cmd is not None and fresh:
            self._last_cmd_t = now
        if cmd is None or now - self._last_cmd_t > M.LOWCMD_TIMEOUT_S:
            if not self._passive:
                self._passive = True
                self._set_gains(np.zeros(29), np.full(29, M.PASSIVE_KD))
                self.art.set_joint_efforts(np.zeros((1, len(self.dof_names))))
            return
        self._passive = False
        self._set_gains(np.asarray(cmd.kp, float), np.asarray(cmd.kd, float))
        n = len(self.dof_names)
        q, dq, tau = np.zeros(n), np.zeros(n), np.zeros(n)
        q[self.i_motor], dq[self.i_motor], tau[self.i_motor] = cmd.q, cmd.dq, cmd.tau
        cur = self.art.get_joint_positions()[0]
        for idx, target in zip(self.i_hand, cmd.hands if cmd.hands is not None else (None, None)):
            q[idx] = cur[idx] if target is None else target            # no hand command: hold where the fingers are
        self.art.set_joint_position_targets(q[None])
        self.art.set_joint_velocity_targets(dq[None])
        self.art.set_joint_efforts(tau[None])

    @property
    def passive(self) -> bool:
        return self._passive

    def state(self, now: float, dt: float) -> tuple[LowState, SimState]:
        q, dq = self.art.get_joint_positions()[0], self.art.get_joint_velocities()[0]
        try:
            tau = self.art.get_measured_joint_efforts()[0]
        except Exception:  # noqa: BLE001 - not every backend reports efforts
            tau = np.zeros_like(q)
        pos, quat = self.art.get_world_poses()
        pos, quat = np.asarray(pos[0], float), np.asarray(quat[0], float)
        vel = np.asarray(self.art.get_velocities()[0], float)              # world frame: linear (3), angular (3)
        lin_b, gyro = quat_rotate_inv(quat, vel[:3]), quat_rotate_inv(quat, vel[3:])
        prev = getattr(self, "_prev_lin_w", vel[:3])
        self._prev_lin_w = vel[:3].copy()
        accel = quat_rotate_inv(quat, (vel[:3] - prev) / max(dt, 1e-6) + np.array([0.0, 0.0, 9.81]))   # specific force, as an accelerometer reads
        self.tick += 1
        st = LowState(q[self.i_motor].copy(), dq[self.i_motor].copy(), tau[self.i_motor].copy(), quat.copy(), gyro, accel,
                      (q[self.i_hand[0]].copy(), q[self.i_hand[1]].copy()), self.tick, now)
        return st, SimState(lin_b, pos.copy(), now)

    # ------------------------------------------------------------------ ground truth (scoring only, never fed to the stack)
    def truth(self) -> dict:
        pos, quat = self.art.get_world_poses()
        w, x, y, z = (float(v) for v in quat[0])
        up_z = 1.0 - 2.0 * (x * x + y * y)
        return {"pos": [float(v) for v in pos[0]], "quat": [w, x, y, z], "yaw": math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
                "tilt_deg": math.degrees(math.acos(max(-1.0, min(1.0, up_z)))), "fallen": bool(pos[0][2] < 0.45 or up_z < math.cos(math.radians(45)))}
