"""Scripted pick-and-place for the pick-place task (tasks/pick_place).

Produces the env's full action vector each step:
    [left wrist pos(3) quat(4), right wrist pos(3) quat(4), 14 hand joints, base command]
The left arm holds its reset pose, the right arm follows a timed waypoint
sequence, and the base gets its plug-in's idle command (robots/bases.py:
`[0, 0, 0, 0.72]` stand still for `agile_legs`, nothing for `fixed`).
Vectorized over envs: each env runs its own phase clock and restarts on reset.
"""

import torch

import isaaclab.utils.math as math_utils

from humanoid_arc_bme.robots.bases import get_base

# Grasp pocket of the right tri-hand, in the wrist_yaw_link frame: fingers
# extend along +x and curl toward +y, the thumb extends along +y and curls
# toward +x (from the IK URDF).
GRASP_OFFSET = (0.12, 0.045, 0.0)
APPROACH_HEIGHT = 0.15
# The hand rotates from hanging-down to fingers-forward at this pose, raised
# and pulled back toward the robot, so the fingertips don't sweep the object.
READY_BACK = 0.12
READY_HEIGHT = 0.20
# Default place target: same table, shifted from wherever the object actually
# was -- self-contained per object, not a hand-picked world coordinate, so
# this scales to any number of tables without per-table tuning. Shifted
# *forward* rather than sideways: tasks/pick_place/env_cfg.py's tables are 0.6m
# along forward vs 0.3m wide, so forward has roughly double the clearance,
# and the pick point already sits off-center on the narrow axis. A sideways
# 0.25m shift there pushed the object off the table edge (confirmed
# on-machine: repeated "object dropped" resets, one per stop, every lap --
# humanoid_arc_bme docs/audit.md item 4; FINDINGS.md M11). Kept short because forward is at the edge of the
# right arm's reach: measured single-table error 7mm at 0.10, 24mm at 0.15,
# 60mm at 0.20 (the wrist stops short and the box slides on release).
DEFAULT_PLACE_OFFSET = (0.10, 0.0, 0.0)

RIGHT_HAND_CLOSED = {
    "right_hand_index_0_joint": 0.9,
    "right_hand_index_1_joint": 0.9,
    "right_hand_middle_0_joint": 0.9,
    "right_hand_middle_1_joint": 0.9,
    "right_hand_thumb_0_joint": 0.0,
    "right_hand_thumb_1_joint": -0.6,
    "right_hand_thumb_2_joint": -0.9,
}

# (name, duration [s], right-wrist waypoint, hand closed)
PHASES = (
    ("settle", 1.0, "rest", 0.0),
    ("raise", 1.5, "ready_pick", 0.0),
    ("reach", 1.5, "above_pick", 0.0),
    ("descend", 1.5, "pick", 0.0),
    ("grasp", 1.0, "pick", 1.0),
    ("lift", 1.5, "above_pick", 1.0),
    ("transport", 2.0, "above_place", 1.0),
    ("lower", 1.5, "place", 1.0),
    ("release", 1.0, "place", 0.0),
    ("retreat", 1.0, "above_place", 0.0),
    ("withdraw", 1.0, "ready_place", 0.0),
    ("home", 1.5, "rest", 0.0),
)
WAYPOINTS = ("rest", "ready_pick", "above_pick", "pick", "above_place", "place", "ready_place")


class ScriptedPickPlace:
    """Drives one object through pick-place on one robot. `object_name` is a
    key into `env.scene`, so one env with N tables/objects uses N instances
    rather than a per-table subclass."""

    def __init__(
        self, env, object_name: str = "object", place_offset=DEFAULT_PLACE_OFFSET, object_home=None, speed: float = 1.0
    ):
        self.env = env
        self.robot = env.scene["robot"]
        self.object = env.scene[object_name]
        self.n = env.num_envs
        self.device = env.device
        self.dt = env.step_dt

        self._pelvis = self.robot.find_bodies("pelvis")[0][0]
        self._left = self.robot.find_bodies("left_wrist_yaw_link")[0][0]
        self._right = self.robot.find_bodies("right_wrist_yaw_link")[0][0]

        hand_names = env.cfg.actions.upper_body_ik.hand_joint_names
        self._hand_closed = torch.zeros(len(hand_names), device=self.device)
        for i, name in enumerate(hand_names):
            self._hand_closed[i] = RIGHT_HAND_CLOSED.get(name, 0.0)

        # speed > 1 shortens every phase (the demo env spawns settled, so "settle" can shrink too).
        self._durations = torch.tensor([p[1] / speed for p in PHASES], device=self.device)
        self._phase_wp = torch.tensor([WAYPOINTS.index(p[2]) for p in PHASES], device=self.device)
        self._phase_closed = torch.tensor([p[3] for p in PHASES], device=self.device)
        self._grasp_offset = torch.tensor(GRASP_OFFSET, device=self.device).expand(self.n, 3)
        self._up = torch.tensor((0.0, 0.0, APPROACH_HEIGHT), device=self.device)
        self._place_offset = torch.tensor(place_offset, device=self.device).expand(self.n, 3)
        base = get_base(env.cfg.robot_base)
        self._base_idle = torch.tensor(base.idle_command, device=self.device).expand(self.n, base.command_dim)

        self.phase = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.t = torch.zeros(self.n, device=self.device)
        self._rest_left = torch.zeros(self.n, 7, device=self.device)
        self._rest_right = torch.zeros(self.n, 7, device=self.device)
        self._q_grasp = torch.zeros(self.n, 4, device=self.device)
        self._pick = torch.zeros(self.n, 3, device=self.device)
        self._place = torch.zeros(self.n, 3, device=self.device)
        # Where the OBJECT (not the wrist) should end up: _place, like _pick,
        # is a wrist/IK target, offset from the object by _grasp_offset (see
        # reset() below) -- comparing a final object position against _place
        # directly is off by that offset (~13cm here). Callers checking
        # placement success should use this instead.
        self.place_target = torch.zeros(self.n, 3, device=self.device)
        self._ready = torch.zeros(self.n, 3, device=self.device)
        # Fixed per-env "home" position for the object, captured once here
        # (before anything moves it) rather than read live in reset(): the
        # place target is derived from this, not from wherever the object
        # currently sits, so repeated laps return it to the same spot
        # instead of drifting further sideways by `place_offset` each time.
        # Pass `object_home` (env frame) when the object's start is randomized,
        # so the place target stays at a fixed goal (the pick-place demo env).
        if object_home is None:
            self._object_home = self.object.data.root_pos_w - self.env.scene.env_origins
        else:
            self._object_home = torch.tensor(object_home, device=self.device).expand(self.n, 3)
        self.reset(torch.arange(self.n, device=self.device))

    def phase_name(self, env_id: int = 0) -> str:
        return PHASES[int(self.phase[env_id])][0]

    def reset(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        pelvis_pos, pelvis_quat = self._pelvis_pose()
        for body, rest in ((self._left, self._rest_left), (self._right, self._rest_right)):
            pos, quat = self._body_pose(body)
            rel_pos, rel_quat = math_utils.subtract_frame_transforms(pelvis_pos, pelvis_quat, pos, quat)
            rest[env_ids] = torch.cat([rel_pos, rel_quat], dim=-1)[env_ids]

        q_grasp = math_utils.yaw_quat(pelvis_quat)
        self._q_grasp[env_ids] = q_grasp[env_ids]
        offset_w = math_utils.quat_apply(q_grasp, self._grasp_offset)
        obj_pos = self.object.data.root_pos_w - self.env.scene.env_origins
        place_shift_w = math_utils.quat_apply(q_grasp, self._place_offset)
        place_target = self._object_home + place_shift_w
        self._pick[env_ids] = (obj_pos - offset_w)[env_ids]
        self._place[env_ids] = (place_target - offset_w)[env_ids]
        self.place_target[env_ids] = place_target[env_ids]
        back = torch.tensor((-READY_BACK, 0.0, 0.0), device=self.device).expand(self.n, 3)
        ready = math_utils.quat_apply(q_grasp, back)
        ready[:, 2] += READY_HEIGHT
        self._ready[env_ids] = ready[env_ids]
        self.phase[env_ids] = 0
        self.t[env_ids] = 0.0

    def compute(self) -> torch.Tensor:
        left = self._from_pelvis(self._rest_left)
        wps_pos, wps_quat = self._waypoints()

        cur = self._phase_wp[self.phase]
        prev = self._phase_wp[(self.phase - 1).clamp(min=0)]
        s = self._smoothstep(self.t / self._durations[self.phase]).unsqueeze(-1)
        idx = torch.arange(self.n, device=self.device)
        right_pos = torch.lerp(wps_pos[prev, idx], wps_pos[cur, idx], s)
        right_quat = self._nlerp(wps_quat[prev, idx], wps_quat[cur, idx], s)

        prev_closed = self._phase_closed[(self.phase - 1).clamp(min=0)]
        closed = torch.lerp(prev_closed, self._phase_closed[self.phase], s.squeeze(-1))
        hand = closed.unsqueeze(-1) * self._hand_closed

        self._advance()
        return torch.cat([left, right_pos, right_quat, hand, self._base_idle], dim=-1)

    def _advance(self):
        self.t += self.dt
        done = (self.t >= self._durations[self.phase]) & (self.phase < len(PHASES) - 1)
        self.phase[done] += 1
        self.t[done] = 0.0

    def _waypoints(self):
        rest = self._from_pelvis(self._rest_right)
        pos = torch.stack([
            rest[:, :3],
            self._pick + self._ready,
            self._pick + self._up,
            self._pick,
            self._place + self._up,
            self._place,
            self._place + self._ready,
        ])
        quat = torch.stack([rest[:, 3:]] + [self._q_grasp] * (len(WAYPOINTS) - 1))
        return pos, quat

    def _from_pelvis(self, rel: torch.Tensor) -> torch.Tensor:
        pelvis_pos, pelvis_quat = self._pelvis_pose()
        pos, quat = math_utils.combine_frame_transforms(pelvis_pos, pelvis_quat, rel[:, :3], rel[:, 3:])
        return torch.cat([pos, quat], dim=-1)

    def _pelvis_pose(self):
        return self._body_pose(self._pelvis)

    def _body_pose(self, body_idx: int):
        state = self.robot.data.body_link_state_w[:, body_idx]
        return state[:, :3] - self.env.scene.env_origins, state[:, 3:7]

    @staticmethod
    def _smoothstep(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    @staticmethod
    def _nlerp(q0: torch.Tensor, q1: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        q1 = torch.where((q0 * q1).sum(-1, keepdim=True) < 0, -q1, q1)
        return math_utils.normalize(torch.lerp(q0, q1, s))
