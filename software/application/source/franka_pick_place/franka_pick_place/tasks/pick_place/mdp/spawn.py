from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def spawn_half_angle(env: "ManagerBasedEnv", default: float) -> float:
    return getattr(env, "_pp_half_angle", default)


def sample_polar(
    n: int, radius_range: tuple[float, float], half_angle: float, device, sector_weights: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    r0, r1 = radius_range
    r = torch.sqrt(torch.empty(n, device=device).uniform_(r0 * r0, r1 * r1))
    if sector_weights is None:
        a = torch.empty(n, device=device).uniform_(-half_angle, half_angle)
    else:
        k = sector_weights.numel()
        idx = torch.multinomial(sector_weights.to(device).float(), n, replacement=True)
        a = -half_angle + (idx + torch.rand(n, device=device)) * (2.0 * half_angle / k)
    return r, a


def reset_object_polar(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    radius_range: tuple[float, float],
    half_angle: float,
    rest_z: float,
    drop_height_range: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
    dev = asset.device
    weights = getattr(env, "_pp_sector_weights", None)
    r, a = sample_polar(n, radius_range, spawn_half_angle(env, half_angle), dev, weights)
    if not hasattr(env, "_pp_spawn_angle"):
        env._pp_spawn_angle = torch.zeros(env.num_envs, device=dev)
    env._pp_spawn_angle[env_ids] = a
    z = rest_z + torch.empty(n, device=dev).uniform_(*drop_height_range)
    pos = torch.stack([r * torch.cos(a), r * torch.sin(a), z], dim=1) + env.scene.env_origins[env_ids]
    yaw = torch.empty(n, device=dev).uniform_(-math.pi, math.pi)
    zero = torch.zeros(n, device=dev)
    quat = math_utils.quat_from_euler_xyz(zero, zero, yaw)
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=env_ids)


def ramp_spawn_angle(
    env: "ManagerBasedRLEnv",
    env_ids,
    start_angle: float,
    end_angle: float,
    start_step: int,
    end_step: int,
) -> float:
    t = (env.common_step_counter - start_step) / max(1, end_step - start_step)
    t = min(max(t, 0.0), 1.0)
    env._pp_half_angle = start_angle + t * (end_angle - start_angle)
    return math.degrees(env._pp_half_angle)
