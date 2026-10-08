"""Continuous multi-table tour: walk to a stop, pick-place, walk to the next,
forever. One `ScriptedPickPlace` instance per stop (cheap, and each keeps its
own held-arm-pose state); `navigator.navigate()` drives the walk between them.
"""

import torch

from humanoid_arc_bme.envs.g1_table_tour.env_cfg import STOPS
from humanoid_arc_bme.policies.navigator import navigate
from humanoid_arc_core.skills.scripted_pick_place import PHASES, ScriptedPickPlace

HOME_PHASE = [p[0] for p in PHASES].index("home")
HAND_OPEN = 0.0


class TableTourController:
    def __init__(self, env, speed: float = 1.0):
        if env.cfg.robot_base != "agile_legs":
            raise ValueError("the tour walks between tables: it needs the agile_legs base")
        self.env = env
        self.robot = env.scene["robot"]
        self.n = env.num_envs
        self.device = env.device
        self.num_stops = len(STOPS)

        self.stops = [ScriptedPickPlace(env, object_name=f"object_{s.name}", speed=speed) for s in STOPS]
        self._headings = torch.tensor([s.heading for s in STOPS], device=self.device)
        self._targets = torch.tensor([s.position for s in STOPS], device=self.device)
        self._pelvis = self.robot.find_bodies("pelvis")[0][0]
        self._hand_dim = self.stops[0]._hand_closed.numel()

        self.stop_idx = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.navigating = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self.laps = torch.zeros(self.n, dtype=torch.long, device=self.device)

    def current_stop_name(self, env_id: int = 0) -> str:
        return STOPS[int(self.stop_idx[env_id])].name

    def status(self, env_id: int = 0) -> str:
        if self.navigating[env_id]:
            return f"walk -> {self.current_stop_name(env_id)}"
        return f"{self.current_stop_name(env_id)}:{self.stops[int(self.stop_idx[env_id])].phase_name(env_id)}"

    def reset(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        # Only stop 0's targets are (re)computed here, from the pose the env
        # resets to (its spawn pose, which is stop 0's arrival pose by
        # construction -- see g1_table_tour.env_cfg.layout_square). Stops
        # 1..N-1 are reset lazily, exactly when each env actually arrives at
        # them (below) -- their pick/place/grasp targets depend on the
        # pelvis pose *at that stop*, which isn't known any earlier.
        self.stop_idx[env_ids] = 0
        self.navigating[env_ids] = True
        self.stops[0].reset(env_ids)

    def compute(self) -> torch.Tensor:
        pelvis_pos = self.robot.data.body_link_pos_w[:, self._pelvis, :2] - self.env.scene.env_origins[:, :2]
        pelvis_quat = self.robot.data.body_link_quat_w[:, self._pelvis]
        target_xy = self._targets[self.stop_idx]
        target_heading = self._headings[self.stop_idx]
        lower_body, arrived = navigate(pelvis_pos, pelvis_quat, target_xy, target_heading)

        idx = torch.arange(self.n, device=self.device)
        actions = torch.zeros(self.n, 7 + 7 + self._hand_dim + 4, device=self.device)  # left, right, hand, lower body
        for i, stop in enumerate(self.stops):
            at_stop = self.stop_idx == i
            if not at_stop.any():
                continue
            left = stop._from_pelvis(stop._rest_left)
            right = stop._from_pelvis(stop._rest_right)
            actions[at_stop, :7] = left[at_stop]
            actions[at_stop, 7:14] = right[at_stop]
            actions[at_stop, 14 : 14 + self._hand_dim] = HAND_OPEN
        actions[:, -4:] = lower_body

        # Arrived while navigating -> reset this stop's grasp targets from
        # the pose we just arrived at (see the note in reset() above), then
        # start its pick-place sequence.
        start_pickplace = self.navigating & arrived
        for i, stop in enumerate(self.stops):
            ids = (start_pickplace & (self.stop_idx == i)).nonzero().flatten()
            if len(ids):
                stop.reset(ids)
        self.navigating[start_pickplace] = False

        pickplace_out = torch.zeros_like(actions)
        current_phase = torch.stack([stop.phase for stop in self.stops], dim=1)[idx, self.stop_idx]
        for i, stop in enumerate(self.stops):
            # Note: this runs stop i's full interpolation for all n envs
            # every step, even those elsewhere, and keeps its own idle
            # phase clock ticking in the background -- harmless (reset()
            # above always re-zeros phase/t on the next real arrival) but
            # wasteful; fine at this num_envs, worth masking properly (or
            # batching all stops' math together) before scaling way up.
            running = (~self.navigating) & (self.stop_idx == i)
            if not running.any():
                continue
            pickplace_out[running] = stop.compute()[running]
        actions = torch.where((~self.navigating).unsqueeze(-1), pickplace_out, actions)

        # A stop's phase machine holds at "home" once done (ScriptedPickPlace
        # ._advance clamps there); ~self.navigating guards this so it fires
        # exactly once per stop, not every step it's held at "home".
        finished = (~self.navigating) & (current_phase == HOME_PHASE)
        if finished.any():
            wrapped = (self.stop_idx[finished] + 1) == self.num_stops
            self.laps[finished] += wrapped.long()
            self.stop_idx[finished] = (self.stop_idx[finished] + 1) % self.num_stops
            self.navigating[finished] = True

        return actions
