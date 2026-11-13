from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_physx.physics import PhysxCfg

from . import mdp
from .mdp.reward_math import DEFAULT_WEIGHTS, PickPlaceParams

OBJECT_SPAWN_RANGE = {"x": (-0.10, 0.10), "y": (-0.25, 0.25), "z": (0.0, 0.10), "yaw": (-3.14, 3.14)}
TARGET_X = (0.40, 0.60)
TARGET_Y = (-0.25, 0.25)
TARGET_Z = (0.0206, 0.0206)

PARAMS = PickPlaceParams()
FINGERS = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])
CMD = "object_pose"


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    object: RigidObjectCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0, 0, 0.707, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:

    object_pose = mdp.PickPlaceTargetCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.PickPlaceTargetCommandCfg.Ranges(
            pos_x=TARGET_X, pos_y=TARGET_Y, pos_z=TARGET_Z, roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        target_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": CMD})
        object_to_target = ObsTerm(func=mdp.object_to_target_in_robot_root_frame, params={"command_name": CMD})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": OBJECT_SPAWN_RANGE,
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


_W = DEFAULT_WEIGHTS


def _stage(signal: str, weight: float) -> RewTerm:
    return RewTerm(
        func=mdp.stage_reward,
        weight=weight,
        params={"signal": signal, "command_name": CMD, "p": PARAMS, "fingers_cfg": FINGERS},
    )


@configclass
class RewardsCfg:
    reach = _stage("reach", _W["reach"])
    grasp = _stage("grasp", _W["grasp"])
    lift = _stage("lift", _W["lift"])
    transport_coarse = _stage("transport_coarse", _W["transport_coarse"])
    transport_fine = _stage("transport_fine", _W["transport_fine"])
    place = _stage("place", _W["place"])
    success_bonus = RewTerm(
        func=mdp.success_bonus,
        weight=_W["success_bonus"],
        params={"command_name": CMD, "p": PARAMS, "fingers_cfg": FINGERS},
    )

    object_dropped = RewTerm(func=mdp.is_terminated_term, weight=_W["object_dropped"], params={"term_keys": "object_dropped"})

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=_W["action_rate"])
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=_W["joint_vel"],
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    rot_action = RewTerm(func=mdp.arm_rot_action_l2, weight=_W["rot_action"])
    action_l2 = RewTerm(func=mdp.action_l2, weight=_W["action_l2"])


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )
    success = DoneTerm(
        func=mdp.task_success,
        params={"command_name": CMD, "p": PARAMS, "fingers_cfg": FINGERS},
    )


def _ramp(term: str, end_weight: float) -> CurrTerm:
    return CurrTerm(
        func=mdp.ramp_reward_weight,
        params={
            "term_name": term, "start_weight": _W[term], "end_weight": end_weight,
            "start_step": 800 * 24, "end_step": 2000 * 24,
        },
    )


@configclass
class CurriculumCfg:
    action_rate = _ramp("action_rate", -0.3)
    joint_vel = _ramp("joint_vel", -1e-2)
    rot_action = _ramp("rot_action", -0.2)
    action_l2 = _ramp("action_l2", -5e-2)


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 8.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            friction_correlation_distance=0.00625,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
        )
