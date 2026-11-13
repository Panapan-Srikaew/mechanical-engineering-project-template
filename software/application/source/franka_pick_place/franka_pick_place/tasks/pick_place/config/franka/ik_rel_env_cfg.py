from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils import configclass

from franka_pick_place.tasks.pick_place.mdp.actions import ClippedDifferentialIKActionCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from . import joint_pos_env_cfg


@configclass
class FrankaPickPlaceIKRelEnvCfg(joint_pos_env_cfg.FrankaPickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.arm_action = ClippedDifferentialIKActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.1,
            action_clip=1.0,
            body_offset=ClippedDifferentialIKActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )


@configclass
class FrankaPickPlaceIKRelEnvCfg_PLAY(FrankaPickPlaceIKRelEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
