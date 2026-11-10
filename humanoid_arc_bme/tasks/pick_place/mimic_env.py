"""Isaac Lab Mimic API for the pick-place demo env.

Built on IsaacLab's LocomanipulationG1MimicEnv (wrist poses from the policy
obs, actions as absolute wrist poses), with three fixes for this env:

- The stock env returns only the 28 upper-body values, so the base's idle
  command (robots/bases.py) is appended here, as the scripted demos do: for
  `agile_legs` the 4-dim stand-still command (32 total), for `fixed` nothing.
- The 14 hand joints are interleaved left/right (pink_controller_cfg's
  hand_joint_names), but the stock split takes them as 7 left then 7 right.
  Split by name instead, or one hand's joints follow the other arm's timing.
- get_subtask_term_signals is implemented, so annotate_demos.py --auto works.
  "raised" ends the right arm's first subtask (see mimic_env_cfg.py for why
  the raise is its own subtask), "grasp" the second.
"""

from collections.abc import Sequence

import torch

from isaaclab_mimic.envs.pinocchio_envs.locomanipulation_g1_mimic_env import LocomanipulationG1MimicEnv

from humanoid_arc_bme.robots.bases import get_base

from .env_cfg import OBJECT_POS, OBJECT_SIZE, TABLE_SIZE

NUM_POSE = 14  # left wrist pos+quat, right wrist pos+quat
NUM_HAND = 14
# The object counts as grasped once it is this far above its resting height.
GRASP_LIFT = 0.03
# The right wrist counts as raised (scripted "raise" phase done, hand above
# and behind the object) above this height, env frame. It rests at ~0.84 and
# peaks at ~0.95 (measured on the scripted demos).
RAISED_WRIST_Z = 0.94


class G1PickPlaceMimicEnv(LocomanipulationG1MimicEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        names = self.cfg.actions.upper_body_ik.hand_joint_names
        self._hand_idx = {
            side: torch.tensor([i for i, n in enumerate(names) if n.startswith(side)], device=self.device)
            for side in ("left", "right")
        }
        self._base_idle = torch.tensor(get_base(self.cfg.robot_base).idle_command, device=self.device)
        self._rest_z = TABLE_SIZE[2] + OBJECT_SIZE[2] / 2
        assert abs(self._rest_z - (OBJECT_POS[2] - 0.002)) < 1e-6

    def target_eef_pose_to_action(self, target_eef_pose_dict, gripper_action_dict, action_noise_dict=None, env_id=0):
        pose_and_hands = super().target_eef_pose_to_action(
            target_eef_pose_dict, {"left": torch.zeros(0, device=self.device), "right": torch.zeros(0, device=self.device)},
            action_noise_dict, env_id,
        )
        hand = torch.zeros(NUM_HAND, device=self.device)
        for side, idx in self._hand_idx.items():
            hand[idx] = gripper_action_dict[side].to(self.device)
        return torch.cat((pose_and_hands[:NUM_POSE], hand, self._base_idle))

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        hand = actions[..., NUM_POSE : NUM_POSE + NUM_HAND]
        return {side: hand[..., idx] for side, idx in self._hand_idx.items()}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        obj_z = self.scene["object"].data.root_pos_w[env_ids, 2] - self.scene.env_origins[env_ids, 2]
        wrist_z = self.obs_buf["policy"]["right_eef_pos"][env_ids, 2]
        return {"raised": wrist_z > RAISED_WRIST_Z, "grasp": obj_z - self._rest_z > GRASP_LIFT}
