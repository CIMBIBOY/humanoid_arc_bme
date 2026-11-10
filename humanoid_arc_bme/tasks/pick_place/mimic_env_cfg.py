"""Mimic data-generation config for the pick-place demo env.

Right arm, three subtasks (signals in mimic_env.py):
1. settle + raise, replayed in the env frame (ends on "raised"). Not
   object-relative: at rest the right fingers hang 2-3cm from the object, so
   shifting the rest pose with the object (up to ~3.6cm) knocked it over.
2. reach + grasp + lift, relative to the object (ends on "grasp"), so it
   adapts to wherever the object starts. The shift is taken up by
   interpolating from the raised hand, well above the object.
3. transport + place + withdraw, replayed in the env frame: the place goal is
   fixed, so this segment must not move with the object's start pose as an
   object-relative segment would. The interpolation into it is long enough
   (0.5s) not to jerk the held object.
Left arm: one subtask, env frame (it only holds its rest pose).

The rest follows IsaacLab's LocomanipulationG1MimicEnvCfg.
"""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from .env_cfg import G1PickPlaceDemoEnvCfg

# Gaussian noise on generated actions (DART-style). 0.003 is the v2 decision (plan.md, problem B);
# the v1 dataset datasets/g1_pick_place_generated.hdf5 was generated with 0.001.
ACTION_NOISE = 0.003


@configclass
class G1PickPlaceMimicEnvCfg(G1PickPlaceDemoEnvCfg, MimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "g1_pick_place_scripted"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 1000
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_relative = False
        self.datagen_config.generation_joint_pos = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # One source demo per generated demo, picked at the first subtask
        # (generation_select_src_per_subtask/per_arm=False). That subtask is in
        # the env frame, so the pick is random, not nearest to the object.
        self.subtask_configs["right"] = [
            SubTaskConfig(
                object_ref=None,
                subtask_term_signal="raised",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                action_noise=ACTION_NOISE,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal="grasp",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=ACTION_NOISE,
                num_interpolation_steps=10,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
            SubTaskConfig(
                object_ref=None,
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                action_noise=ACTION_NOISE,
                num_interpolation_steps=25,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
        ]
        self.subtask_configs["left"] = [
            SubTaskConfig(
                object_ref=None,
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                action_noise=ACTION_NOISE,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
        ]
