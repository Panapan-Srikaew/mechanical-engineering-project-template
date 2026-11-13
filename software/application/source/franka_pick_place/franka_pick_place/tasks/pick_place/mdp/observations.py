from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from ._compat import as_torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    pos_b, _ = subtract_frame_transforms(
        as_torch(robot.data.root_pos_w), as_torch(robot.data.root_quat_w), as_torch(obj.data.root_pos_w)[:, :3]
    )
    return pos_b


def ee_position_in_robot_root_frame(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    ee = env.scene[ee_frame_cfg.name]
    pos_b, _ = subtract_frame_transforms(
        as_torch(robot.data.root_pos_w), as_torch(robot.data.root_quat_w), as_torch(ee.data.target_pos_w)[:, 0, :]
    )
    return pos_b


def object_to_target_in_robot_root_frame(
    env: "ManagerBasedRLEnv",
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    target_b = env.command_manager.get_command(command_name)[:, :3]
    return target_b - object_position_in_robot_root_frame(env, robot_cfg, object_cfg)


def gripper_opening(env: "ManagerBasedRLEnv", fingers_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[fingers_cfg.name]
    return as_torch(robot.data.joint_pos)[:, fingers_cfg.joint_ids].sum(dim=1, keepdim=True)


def arm_controller_state(env: "ManagerBasedRLEnv", action_name: str = "arm_action") -> torch.Tensor:
    term = env.action_manager.get_term(action_name)
    if not hasattr(term, "_target_pos"):
        return torch.zeros(env.num_envs, 9, device=env.device)
    ee_pos, _ = term._compute_frame_pose()
    lead = (term._target_pos - ee_pos) / term.cfg.max_lead_pos
    lead = torch.where(term._target_init.unsqueeze(-1), torch.zeros_like(lead), lead)
    return torch.cat([lead, term._smoothed], dim=1)


def _hand_azimuth(env: "ManagerBasedRLEnv") -> torch.Tensor:
    p = ee_position_in_robot_root_frame(env)
    return torch.atan2(p[:, 1], p[:, 0])


def _rot_z(v: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(ang), torch.sin(ang)
    return torch.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]], dim=1)


def object_position_hand_frame(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _rot_z(object_position_in_robot_root_frame(env), -_hand_azimuth(env))


def ee_position_hand_frame(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _rot_z(ee_position_in_robot_root_frame(env), -_hand_azimuth(env))


def target_pose_hand_frame(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    phi = _hand_azimuth(env)
    z_axis = torch.zeros_like(cmd[:, :3])
    z_axis[:, 2] = 1.0
    quat = math_utils.quat_mul(math_utils.quat_from_angle_axis(-phi, z_axis), cmd[:, 3:7])
    return torch.cat([_rot_z(cmd[:, :3], -phi), quat], dim=1)


def object_to_target_hand_frame(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    return _rot_z(object_to_target_in_robot_root_frame(env, command_name), -_hand_azimuth(env))


def arm_controller_state_hand_frame(env: "ManagerBasedRLEnv", action_name: str = "arm_action") -> torch.Tensor:
    s = arm_controller_state(env, action_name)
    return torch.cat([_rot_z(s[:, :3], -_hand_azimuth(env)), s[:, 3:]], dim=1)
