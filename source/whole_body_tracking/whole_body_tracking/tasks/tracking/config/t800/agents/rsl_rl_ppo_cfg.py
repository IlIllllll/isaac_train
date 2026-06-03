from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class T800FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "t800_flat"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class T800HighDynPPORunnerCfg(T800FlatPPORunnerCfg):
    """PPO settings for high-dynamic 3.2s single-motion training."""

    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = 48
        self.max_iterations = 30000
        self.save_interval = 500
        self.experiment_name = "highdyn_single"
        self.algorithm.learning_rate = 5.0e-4
        self.algorithm.desired_kl = 0.008
        self.algorithm.entropy_coef = 0.003
        self.algorithm.num_mini_batches = 8


LOW_FREQ_SCALE = 0.5


@configclass
class T800FlatLowFreqPPORunnerCfg(T800FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)
