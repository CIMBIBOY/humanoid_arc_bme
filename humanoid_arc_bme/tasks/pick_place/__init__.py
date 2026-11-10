"""Gym ids for IsaacLab's dataset tools (record/replay, Mimic, robomimic).

String entry points only, so importing this needs just gymnasium, not a
running Kit app. The ids keep their original names (stored in recorded
datasets and robomimic checkpoints); they are the `agile_legs` base. The
`FixedBase` id is the same demo task on the `fixed` base.
"""

import gymnasium as gym

gym.register(
    id="HumanoidArc-G1-PickPlace-Demo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1PickPlaceDemoEnvCfg",
        # IsaacLab's G1 locomanipulation BC-RNN config: same observation terms as this env.
        "robomimic_bc_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomanipulation.pick_place:agents/robomimic/bc_rnn_low_dim.json"
        ),
        # The same plus +-1.5cm training-time observation noise (--algo bc_obs_noise). Without it
        # the policy copies the current wrist pose (actions sit ~5mm from it) and drifts in rollouts.
        "robomimic_bc_obs_noise_cfg_entry_point": f"{__name__}:agents/robomimic/bc_rnn_low_dim_obs_noise.json",
        # The same with a 50-step (1s) RNN context instead of 10 (--algo bc_obs_noise_h50).
        "robomimic_bc_obs_noise_h50_cfg_entry_point": f"{__name__}:agents/robomimic/bc_rnn_low_dim_obs_noise_h50.json",
    },
    disable_env_checker=True,
)

gym.register(
    id="HumanoidArc-G1-PickPlace-Mimic-v0",
    entry_point=f"{__name__}.mimic_env:G1PickPlaceMimicEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.mimic_env_cfg:G1PickPlaceMimicEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="HumanoidArc-G1-PickPlace-FixedBase-Demo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:G1PickPlaceFixedBaseDemoEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="HumanoidArc-G1-PickPlace-Randomized-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1PickPlaceRandomizedEnvCfg",
        "robomimic_bc_obs_noise_h50_cfg_entry_point": f"{__name__}:agents/robomimic/bc_rnn_low_dim_obs_noise_h50.json",
    },
    disable_env_checker=True,
)

gym.register(
    id="HumanoidArc-G1-PickPlace-RandomizedRobot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1PickPlaceRandomizedRobotEnvCfg",
        "robomimic_bc_obs_noise_h50_cfg_entry_point": f"{__name__}:agents/robomimic/bc_rnn_low_dim_obs_noise_h50.json",
    },
    disable_env_checker=True,
)
