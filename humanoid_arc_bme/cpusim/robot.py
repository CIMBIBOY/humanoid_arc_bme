"""Kinematic G1 for the CPU simulator: a lagged base, lagged arm and Dex3 joints, and a held-object model.

The base integrates the commanded (vx, vy, wz) through a first-order lag. Arm, waist and hand joints follow their targets
with a first-order lag too (the targets come from the real GrootPlannerController path, so S1's clamp and rate limit are
exercised). A hand that closes with an object inside its pocket stalls at `grasp_closure`, which is what the task layer
reads as "object in hand"; opening it drops the object onto the surface below. Forward kinematics is the repository's own.
"""

from __future__ import annotations

import math

import numpy as np

from humanoid_arc_bme.cpusim.world import Cylinder, World
from humanoid_arc_bme.onboard.g1_layout import LEFT_ARM_JOINTS, LEFT_HAND_JOINTS, RIGHT_ARM_JOINTS, RIGHT_HAND_JOINTS, UPPER_BODY_JOINTS
from humanoid_arc_bme.onboard.manipulation.dex3 import Dex3Hand
from humanoid_arc_bme.onboard.manipulation.kinematics import G1ArmKinematics
from humanoid_arc_bme.onboard.pipeline.interfaces import Pose

REST_TOOL = {"left": (0.25, 0.22, -0.20), "right": (0.25, -0.22, -0.20)}   # pelvis frame: arms relaxed in front


class SimRobot:
    def __init__(self, world: World, pose=(0.0, 0.0, 0.0), pelvis_height: float = 0.78, base_tau: float = 0.25,
                 joint_tau: float = 0.15, pocket_m: float = 0.07, grasp_closure: float = 0.55):
        self.world = world
        self.x, self.y, self.yaw = pose
        self.H, self.base_tau, self.joint_tau = pelvis_height, base_tau, joint_tau
        self.pocket, self.grasp_closure = pocket_m, grasp_closure
        self.kin = {s: G1ArmKinematics(s) for s in ("left", "right")}
        self.hands = {s: Dex3Hand(s) for s in ("left", "right")}
        self.vel = (0.0, 0.0, 0.0)
        self.q = {n: 0.0 for n in UPPER_BODY_JOINTS}
        for s in ("left", "right"):
            names = LEFT_ARM_JOINTS if s == "left" else RIGHT_ARM_JOINTS
            res = self.kin[s].ik(Pose(REST_TOOL[s]), np.zeros(7), ori_weight=0.0)
            for n, v in zip(names, res.q):
                self.q[n] = float(v)
        hn = {"left": LEFT_HAND_JOINTS, "right": RIGHT_HAND_JOINTS}
        for s, names in hn.items():
            for n, v in zip(names, self.hands[s].closure_to_joints(0.0)):
                self.q[n] = float(v)
        self.target = dict(self.q)
        self._closure = {"left": 0.0, "right": 0.0}
        self.held: Cylinder | None = None
        self.held_side: str | None = None

    # ------------------------------------------------------------------ state
    def tool_world(self, side: str) -> np.ndarray:
        names = LEFT_ARM_JOINTS if side == "left" else RIGHT_ARM_JOINTS
        p = np.asarray(self.kin[side].fk(np.array([self.q[n] for n in names])).position)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([self.x + c * p[0] - s * p[1], self.y + s * p[0] + c * p[1], self.H + p[2]])

    def speed(self) -> float:
        return math.hypot(self.vel[0], self.vel[1])

    # ------------------------------------------------------------------ dynamics
    def step(self, dt: float, base_cmd, arm=None) -> None:
        """`arm`: (body17, left7, right7) joint targets or None (hold the current targets)."""
        a = 1.0 - math.exp(-dt / self.base_tau)
        self.vel = tuple(v + (c - v) * a for v, c in zip(self.vel, base_cmd))
        vx, vy, wz = self.vel
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (c * vx - s * vy) * dt
        self.y += (s * vx + c * vy) * dt
        self.yaw = math.remainder(self.yaw + wz * dt, 2 * math.pi)
        if arm is not None:
            body, left, right = arm
            for n, v in zip(UPPER_BODY_JOINTS, body):
                self.target[n] = float(v)
            for n, v in zip(LEFT_HAND_JOINTS, left):
                self.target[n] = float(v)
            for n, v in zip(RIGHT_HAND_JOINTS, right):
                self.target[n] = float(v)
        k = 1.0 - math.exp(-dt / self.joint_tau)
        for n in UPPER_BODY_JOINTS:
            self.q[n] += (self.target[n] - self.q[n]) * k
        for s_ in ("left", "right"):
            self._step_hand(s_, k)
        self._update_object()

    def _step_hand(self, side: str, k: float) -> None:
        names = LEFT_HAND_JOINTS if side == "left" else RIGHT_HAND_JOINTS
        want = self.hands[side].joints_to_closure(np.array([self.target[n] for n in names]))
        limit = self.grasp_closure if (self.held is not None and self.held_side == side) else 1.0
        self._closure[side] += (min(want, limit) - self._closure[side]) * k
        for n, v in zip(names, self.hands[side].closure_to_joints(self._closure[side])):
            self.q[n] = float(v)

    def _commanded_closure(self, side: str) -> float:
        names = LEFT_HAND_JOINTS if side == "left" else RIGHT_HAND_JOINTS
        return self.hands[side].joints_to_closure(np.array([self.target[n] for n in names]))

    def _update_object(self) -> None:
        if self.held is not None:
            side = self.held_side
            if self._commanded_closure(side) < 0.3:        # opened: the object drops onto the surface below
                h = self.held
                h.base_z = self.world.surface_z(h.x, h.y)
                self.held = self.held_side = None
                return
            t = self.tool_world(side)
            self.held.x, self.held.y, self.held.base_z = float(t[0]), float(t[1]), float(t[2]) - self.held.height / 2
            return
        for side in ("left", "right"):
            if self._commanded_closure(side) >= 0.9:
                t = self.tool_world(side)
                for c in self.world.cylinders:
                    ctr = np.array([c.x, c.y, c.base_z + c.height / 2])
                    if float(np.linalg.norm(ctr - t)) <= self.pocket:
                        self.held, self.held_side = c, side
                        return
