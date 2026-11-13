from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass


class ClippedDifferentialIKAction(DifferentialInverseKinematicsAction):
    cfg: "ClippedDifferentialIKActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._smoothed = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_quat[:, 3] = 1.0
        self._target_init = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._q_home = None

    def process_actions(self, actions: torch.Tensor):
        c = self.cfg.action_clip
        a = torch.clamp(actions, -c, c)
        if self.cfg.smoothing < 1.0:
            self._smoothed = self.cfg.smoothing * a + (1.0 - self.cfg.smoothing) * self._smoothed
            a = self._smoothed
        if self.cfg.integrate_target:
            a = self._integrate(a)
        super().process_actions(a)
        self._raw_actions[:] = actions

    def _integrate(self, a: torch.Tensor) -> torch.Tensor:
        ee_pos, ee_quat = self._compute_frame_pose()
        init = self._target_init
        if init.any():
            self._target_pos[init] = ee_pos[init]
            self._target_quat[init] = ee_quat[init]
            self._target_init[:] = False
        step = a * self._scale
        if self.cfg.polar_action:
            tgt, tq_base, rotvec = self._polar_step(step)
        else:
            tgt, tq_base, rotvec = self._target_pos + step[:, :3], self._target_quat, step[:, 3:6]
        lead = tgt - ee_pos
        n = torch.linalg.vector_norm(lead, dim=1, keepdim=True).clamp(min=1e-9)
        lead = lead * torch.clamp(self.cfg.max_lead_pos / n, max=1.0)
        self._target_pos = ee_pos + lead
        ang = torch.linalg.vector_norm(rotvec, dim=1)
        axis = rotvec / ang.clamp(min=1e-9).unsqueeze(-1)
        dq = math_utils.quat_from_angle_axis(ang, axis)
        tq = math_utils.quat_mul(dq, tq_base)
        rot_lead = math_utils.axis_angle_from_quat(math_utils.quat_mul(tq, math_utils.quat_inv(ee_quat)))
        rn = torch.linalg.vector_norm(rot_lead, dim=1, keepdim=True).clamp(min=1e-9)
        rot_lead = rot_lead * torch.clamp(self.cfg.max_lead_rot / rn, max=1.0)
        rlen = torch.linalg.vector_norm(rot_lead, dim=1)
        self._target_quat = math_utils.quat_mul(
            math_utils.quat_from_angle_axis(rlen, rot_lead / rlen.clamp(min=1e-9).unsqueeze(-1)), ee_quat
        )
        return torch.cat([lead, rot_lead], dim=1) / self._scale

    def _polar_step(self, step: torch.Tensor):
        tp = self._target_pos
        r = torch.linalg.vector_norm(tp[:, :2], dim=1)
        phi = torch.atan2(tp[:, 1], tp[:, 0]) + step[:, 1] / r.clamp(min=0.15)
        dphi = phi - torch.atan2(tp[:, 1], tp[:, 0])
        r_new = (r + step[:, 0]).clamp(min=0.15)
        tgt = torch.stack([r_new * torch.cos(phi), r_new * torch.sin(phi), tp[:, 2] + step[:, 2]], dim=1)
        z_axis = torch.zeros_like(tp)
        z_axis[:, 2] = 1.0
        tq_base = math_utils.quat_mul(math_utils.quat_from_angle_axis(dphi, z_axis), self._target_quat)
        c, s = torch.cos(phi), torch.sin(phi)
        w = step[:, 3:6]
        rotvec = torch.stack([c * w[:, 0] - s * w[:, 1], s * w[:, 0] + c * w[:, 1], w[:, 2]], dim=1)
        return tgt, tq_base, rotvec

    def apply_actions(self):
        if self.cfg.nullspace_gain <= 0.0:
            return super().apply_actions()
        ee_pos, ee_quat = self._compute_frame_pose()
        q = self._asset.data.joint_pos.torch[:, self._joint_ids]
        jac = self._compute_frame_jacobian()
        q_des = self._ik_controller.compute(ee_pos, ee_quat, jac, q)
        q_des = q_des + self._nullspace_step(jac, q)
        self._asset.set_joint_position_target_index(target=q_des, joint_ids=self._joint_ids)

    def _nullspace_step(self, jac: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        if self._q_home is None:
            self._q_home = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()
            lim = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids[0]]
            self._j1_range = (lim[:, 0] + 0.1, lim[:, 1] - 0.1)
        tgt = self._ik_controller.ee_pos_des
        q_pref = self._q_home.clone()
        q_pref[:, 0] = torch.maximum(torch.minimum(torch.atan2(tgt[:, 1], tgt[:, 0]), self._j1_range[1]), self._j1_range[0])
        jt = jac.transpose(1, 2)
        eye6 = torch.eye(jac.shape[1], device=jac.device)
        j_pinv = jt @ torch.linalg.inv(jac @ jt + (0.05**2) * eye6)
        null = torch.eye(jac.shape[2], device=jac.device) - j_pinv @ jac
        dq = (null @ (self.cfg.nullspace_gain * (q_pref - q)).unsqueeze(-1)).squeeze(-1)
        return dq.clamp(-self.cfg.nullspace_max_step, self.cfg.nullspace_max_step)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._smoothed[ids] = 0.0
        self._target_init[ids] = True


@configclass
class ClippedDifferentialIKActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type = ClippedDifferentialIKAction
    action_clip: float = 1.0
    smoothing: float = 1.0
    integrate_target: bool = False
    max_lead_pos: float = 0.03
    max_lead_rot: float = 0.3
    polar_action: bool = False
    nullspace_gain: float = 0.0
    nullspace_max_step: float = 0.02
