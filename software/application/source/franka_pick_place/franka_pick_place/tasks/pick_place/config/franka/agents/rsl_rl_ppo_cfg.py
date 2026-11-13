from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class PickPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    clip_actions = None
    experiment_name = "franka_pick_place"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PickPlace360PPORunnerCfg(PickPlacePPORunnerCfg):

    max_iterations = 6000

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.algorithm.gamma = 0.99
