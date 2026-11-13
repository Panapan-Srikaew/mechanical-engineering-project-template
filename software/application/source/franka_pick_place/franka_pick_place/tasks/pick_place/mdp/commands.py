from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from ._compat import as_torch
from .spawn import spawn_half_angle


class PickPlaceTargetCommand(UniformPoseCommand):
    cfg: "PickPlaceTargetCommandCfg"

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if isinstance(env_ids, slice) or env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        obj = self._env.scene[self.cfg.object_name]
        obj_pos_b, _ = subtract_frame_transforms(
            as_torch(self.robot.data.root_pos_w)[env_ids],
            as_torch(self.robot.data.root_quat_w)[env_ids],
            as_torch(obj.data.root_pos_w)[env_ids, :3],
        )
        r = self.cfg.ranges
        for _ in range(self.cfg.max_tries):
            d_xy = torch.norm(self.pose_command_b[env_ids, :2] - obj_pos_b[:, :2], dim=1)
            bad = d_xy < self.cfg.min_distance
            if not bad.any():
                break
            ids = env_ids[bad]
            n = ids.numel()
            self.pose_command_b[ids, 0] = torch.empty(n, device=self.device).uniform_(*r.pos_x)
            self.pose_command_b[ids, 1] = torch.empty(n, device=self.device).uniform_(*r.pos_y)


@configclass
class PickPlaceTargetCommandCfg(UniformPoseCommandCfg):
    class_type: type = PickPlaceTargetCommand
    object_name: str = "object"
    min_distance: float = 0.10
    max_tries: int = 20


class PolarTargetCommand(PickPlaceTargetCommand):

    cfg: "PolarTargetCommandCfg"

    def _resample_command(self, env_ids: Sequence[int]):
        UniformPoseCommand._resample_command(self, env_ids)
        if isinstance(env_ids, slice) or env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        obj = self._env.scene[self.cfg.object_name]
        obj_pos_b, _ = subtract_frame_transforms(
            as_torch(self.robot.data.root_pos_w)[env_ids],
            as_torch(self.robot.data.root_quat_w)[env_ids],
            as_torch(obj.data.root_pos_w)[env_ids, :3],
        )
        cube_angle = torch.atan2(obj_pos_b[:, 1], obj_pos_b[:, 0])
        half = spawn_half_angle(self._env, self.cfg.half_angle)
        todo = torch.ones(env_ids.numel(), dtype=torch.bool, device=self.device)
        for _ in range(self.cfg.max_tries):
            n = int(todo.sum())
            r0, r1 = self.cfg.radius_range
            r = torch.sqrt(torch.empty(n, device=self.device).uniform_(r0 * r0, r1 * r1))
            d = torch.empty(n, device=self.device).uniform_(-self.cfg.max_delta_angle, self.cfg.max_delta_angle)
            a = torch.clamp(cube_angle[todo] + d, -half, half)
            ids = env_ids[todo]
            self.pose_command_b[ids, 0] = r * torch.cos(a)
            self.pose_command_b[ids, 1] = r * torch.sin(a)
            d_xy = torch.norm(self.pose_command_b[env_ids, :2] - obj_pos_b[:, :2], dim=1)
            todo = d_xy < self.cfg.min_distance
            if not todo.any():
                break


@configclass
class PolarTargetCommandCfg(PickPlaceTargetCommandCfg):
    class_type: type = PolarTargetCommand
    radius_range: tuple[float, float] = (0.40, 0.70)
    half_angle: float = 2.79
    max_delta_angle: float = 1.57
