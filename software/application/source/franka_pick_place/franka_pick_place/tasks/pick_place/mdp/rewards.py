from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

from ._compat import as_torch
from .reward_math import PickPlaceParams, stage_signals

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def compute_signals(
    env: "ManagerBasedRLEnv",
    command_name: str,
    p: PickPlaceParams,
    fingers_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> dict:
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_pos_w = as_torch(ee.data.target_pos_w)[:, 0, :]
    obj_pos_w = as_torch(obj.data.root_pos_w)[:, :3]

    cmd = env.command_manager.get_command(command_name)
    goal_pos_w, _ = combine_frame_transforms(
        as_torch(robot.data.root_pos_w), as_torch(robot.data.root_quat_w), cmd[:, :3]
    )

    d_ee = torch.norm(ee_pos_w - obj_pos_w, dim=1)
    d_goal = torch.norm(goal_pos_w - obj_pos_w, dim=1)
    d_goal_xy = torch.norm(goal_pos_w[:, :2] - obj_pos_w[:, :2], dim=1)
    opening = as_torch(robot.data.joint_pos)[:, fingers_cfg.joint_ids].sum(dim=1)
    obj_speed = torch.norm(as_torch(obj.data.root_lin_vel_w), dim=1)

    was_lifted = getattr(env, "_pick_place_was_lifted", None)
    if was_lifted is None or was_lifted.shape[0] != env.num_envs:
        was_lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    was_lifted = was_lifted & (env.episode_length_buf > 1)

    close_cmd = env.action_manager.get_term("gripper_action").raw_actions[:, 0] < 0

    s = stage_signals(
        d_ee, d_goal, obj_pos_w[:, 2], goal_pos_w[:, 2], opening, obj_speed, p,
        was_lifted=was_lifted, close_cmd=close_cmd, d_goal_xy=d_goal_xy,
    )
    env._pick_place_was_lifted = was_lifted | (s["lifted_latch"] > 0.5)
    return s


def stage_reward(
    env: "ManagerBasedRLEnv",
    signal: str,
    command_name: str,
    p: PickPlaceParams,
    fingers_cfg: SceneEntityCfg,
) -> torch.Tensor:
    return compute_signals(env, command_name, p, fingers_cfg)[signal]


def success_bonus(
    env: "ManagerBasedRLEnv",
    command_name: str,
    p: PickPlaceParams,
    fingers_cfg: SceneEntityCfg,
    extra_steps: int = 50,
    constant: float | None = None,
) -> torch.Tensor:
    s = compute_signals(env, command_name, p, fingers_cfg)["success"]
    if constant is not None:
        return s * constant
    remaining = (env.max_episode_length - env.episode_length_buf).float()
    return s * (remaining + extra_steps)


def task_success(
    env: "ManagerBasedRLEnv",
    command_name: str,
    p: PickPlaceParams,
    fingers_cfg: SceneEntityCfg,
) -> torch.Tensor:
    return compute_signals(env, command_name, p, fingers_cfg)["success"] > 0.5


def arm_rot_action_l2(env: "ManagerBasedRLEnv", action_name: str = "arm_action") -> torch.Tensor:
    a = env.action_manager.get_term(action_name).raw_actions
    if a.shape[1] != 6:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.sum(torch.square(a[:, 3:6]), dim=1)


def invalid_state(
    env: "ManagerBasedRLEnv",
    max_joint_vel: float = 100.0,
    max_object_dist: float = 3.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    q = as_torch(robot.data.joint_pos)
    qd = as_torch(robot.data.joint_vel)
    pos = as_torch(obj.data.root_pos_w) - env.scene.env_origins
    vel = as_torch(obj.data.root_lin_vel_w)
    finite = (
        torch.isfinite(q).all(1) & torch.isfinite(qd).all(1) & torch.isfinite(pos).all(1) & torch.isfinite(vel).all(1)
    )
    sane = (qd.abs().amax(1) < max_joint_vel) & (torch.linalg.vector_norm(pos, dim=1) < max_object_dist)
    return ~(finite & sane)
