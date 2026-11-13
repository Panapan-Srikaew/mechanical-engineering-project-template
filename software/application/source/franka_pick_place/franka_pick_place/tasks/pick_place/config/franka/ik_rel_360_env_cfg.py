import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ... import mdp
from ...mdp.reward_math import PickPlaceParams
from . import ik_rel_env_cfg

RADIUS = (0.40, 0.70)
HALF_ANGLE_START = math.radians(30)
HALF_ANGLE_END = math.radians(160)
CURRICULUM_SUCCESS = 0.6
ANGLE_RATE = math.radians(130) / (300 * 24)
PLACE_TOL_START, PLACE_TOL_END = 0.03, 0.015
PLACE_TOL_RATE = (PLACE_TOL_START - PLACE_TOL_END) / (300 * 24)
MAX_DELTA_ANGLE = math.radians(90)
HAND_STEP = 0.005
WRIST_STEP = 0.02
SMOOTHING = 0.3
EPISODE_S = 15.0
PARAMS_360 = PickPlaceParams(
    place_tol=PLACE_TOL_END, fine_std=0.05, lift_hold_after_pick=True,
)
TABLE_RADIUS = 1.0


@configclass
class FrankaPickPlaceIKRel360EnvCfg(ik_rel_env_cfg.FrankaPickPlaceIKRelEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        rest_z = PARAMS_360.object_rest_z

        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -0.5]),
            spawn=sim_utils.CylinderCfg(
                radius=TABLE_RADIUS,
                height=1.0,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.28)),
            ),
        )
        self.scene.env_spacing = 2.5

        self.events.reset_object_position = EventTerm(
            func=mdp.reset_object_polar,
            mode="reset",
            params={
                "radius_range": RADIUS, "half_angle": HALF_ANGLE_END, "rest_z": rest_z,
                "drop_height_range": (0.0, 0.05), "asset_cfg": SceneEntityCfg("object"),
            },
        )
        old = self.commands.object_pose
        self.commands.object_pose = mdp.PolarTargetCommandCfg(
            asset_name=old.asset_name,
            body_name=old.body_name,
            resampling_time_range=(EPISODE_S + 1.0, EPISODE_S + 1.0),
            debug_vis=True,
            ranges=old.ranges,
            radius_range=RADIUS,
            half_angle=HALF_ANGLE_END,
            max_delta_angle=MAX_DELTA_ANGLE,
        )
        self.curriculum.spawn_angle = CurrTerm(
            func=mdp.adapt_spawn_angle,
            params={
                "start_angle": HALF_ANGLE_START, "end_angle": HALF_ANGLE_END,
                "rate": ANGLE_RATE, "success_threshold": CURRICULUM_SUCCESS,
            },
        )
        self.curriculum.place_tol = CurrTerm(
            func=mdp.adapt_place_tol,
            params={
                "start_tol": PLACE_TOL_START, "end_tol": PLACE_TOL_END, "rate": PLACE_TOL_RATE,
                "success_threshold": CURRICULUM_SUCCESS, "angle_done": HALF_ANGLE_END,
            },
        )

        arm = self.actions.arm_action
        arm.scale = (HAND_STEP,) * 3 + (WRIST_STEP,) * 3
        arm.smoothing = SMOOTHING
        arm.integrate_target = True
        arm.max_lead_pos = 0.05
        self.episode_length_s = EPISODE_S

        self.rewards.action_l2.weight = -5e-3
        self.rewards.transport_fine.weight = 15.0
        self.curriculum.action_l2.params["start_weight"] = -5e-3

        self.terminations.invalid_state = DoneTerm(func=mdp.invalid_state)

        for group in (self.rewards, self.terminations):
            for term in group.__dict__.values():
                if hasattr(term, "params") and "p" in term.params:
                    term.params["p"] = PARAMS_360

        self.events.object_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("object"), "static_friction_range": (0.5, 1.2),
                "dynamic_friction_range": (0.4, 1.0), "restitution_range": (0.0, 0.05),
                "num_buckets": 64, "make_consistent": True,
            },
        )
        self.events.finger_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="panda_.*finger"), "static_friction_range": (0.7, 1.3),
                "dynamic_friction_range": (0.6, 1.1), "restitution_range": (0.0, 0.0),
                "num_buckets": 64, "make_consistent": True,
            },
        )
        self.events.object_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("object"), "mass_distribution_params": (0.7, 1.5), "operation": "scale"},
        )
        obs = self.observations.policy
        obs.enable_corruption = True
        obs.joint_pos.noise = Unoise(n_min=-0.005, n_max=0.005)
        obs.joint_vel.noise = Unoise(n_min=-0.02, n_max=0.02)
        obs.object_position.noise = Unoise(n_min=-0.005, n_max=0.005)
        obs.object_to_target.noise = Unoise(n_min=-0.005, n_max=0.005)
        obs.arm_state = ObsTerm(func=mdp.arm_controller_state)

        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = (1.6, -1.6, 1.3)
        self.viewer.lookat = (0.0, 0.0, 0.1)


@configclass
class FrankaPickPlaceIKRel360EnvCfg_PLAY(FrankaPickPlaceIKRel360EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.curriculum.spawn_angle = None
        self.curriculum.place_tol = None
