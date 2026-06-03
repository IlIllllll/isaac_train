import gymnasium as gym

from . import agents, flat_env_cfg

gym.register(
    id="Tracking-Flat-T800-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Exp-Current10s-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatSingleMotionCurrent10sEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Exp-Current3p2s-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatSingleMotionCurrent3p2sEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Exp-RelaxedDone3p2s-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatSingleMotionRelaxedDone3p2sEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Exp-RelaxedDoneLowVel3p2s-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatSingleMotionRelaxedDoneLowVel3p2sEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-HighDyn-Warmup3p2s-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatHighDynWarmup3p2sEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800HighDynPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-HighDyn-Stable3p2s-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatHighDynStable3p2sEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800HighDynPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-HighDyn-Final3p2s-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatHighDynFinal3p2sEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800HighDynPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-HighDyn-SpeedLead3p2s-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatHighDynSpeedLead3p2sEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800HighDynPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-T800-HighDyn-PhaseLocalKick3p2s-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatHighDynPhaseLocalKick3p2sEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800HighDynPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-HighDyn-PhaseLocalRiot3p2s-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatHighDynPhaseLocalRiot3p2sEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800HighDynPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatLowFreqPPORunnerCfg",
    },
)
