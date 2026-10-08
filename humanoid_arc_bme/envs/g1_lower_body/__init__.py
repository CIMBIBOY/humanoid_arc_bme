"""Gym ids for the lower-body training env (String entry points: no Kit needed to import)."""

import gymnasium as gym

gym.register(
    id="HumanoidArc-G1-LowerBody-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1LowerBodyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1LowerBodyPPORunnerCfg",
    },
    disable_env_checker=True,
)
gym.register(
    id="HumanoidArc-G1-LowerBody-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:G1LowerBodyEnvCfg_PLAY"},
    disable_env_checker=True,
)
gym.register(
    id="HumanoidArc-G1-LowerBody-Easy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1LowerBodyEasyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1LowerBodyPPORunnerCfg",
    },
    disable_env_checker=True,
)
