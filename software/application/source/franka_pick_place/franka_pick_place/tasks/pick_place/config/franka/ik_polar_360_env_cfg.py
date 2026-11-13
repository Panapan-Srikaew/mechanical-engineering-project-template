from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from ... import mdp
from . import ik_rel_360_env_cfg

CMD = "object_pose"


@configclass
class FrankaPickPlaceIKPolar360EnvCfg(ik_rel_360_env_cfg.FrankaPickPlaceIKRel360EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action.polar_action = True

        obs = self.observations.policy
        obs.object_position.func = mdp.object_position_hand_frame
        obs.ee_position.func = mdp.ee_position_hand_frame
        obs.target_pose.func = mdp.target_pose_hand_frame
        obs.object_to_target.func = mdp.object_to_target_hand_frame
        obs.arm_state.func = mdp.arm_controller_state_hand_frame

        self.curriculum.sector_weights = CurrTerm(func=mdp.adapt_sector_weights)


@configclass
class FrankaPickPlaceIKPolar360EnvCfg_PLAY(FrankaPickPlaceIKPolar360EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.curriculum.spawn_angle = None
        self.curriculum.place_tol = None
        self.curriculum.sector_weights = None
