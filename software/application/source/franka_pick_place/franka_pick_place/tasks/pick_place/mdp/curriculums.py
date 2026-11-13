from __future__ import annotations

import math

import torch

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ramp_reward_weight(ManagerTermBase):

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._term_cfg = env.reward_manager.get_term_cfg(cfg.params["term_name"])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        start_weight: float,
        end_weight: float,
        start_step: int,
        end_step: int,
    ) -> float:
        a = (env.common_step_counter - start_step) / max(end_step - start_step, 1)
        a = min(max(a, 0.0), 1.0)
        self._term_cfg.weight = start_weight + a * (end_weight - start_weight)
        env.reward_manager.set_term_cfg(term_name, self._term_cfg)
        return self._term_cfg.weight


def success_rate_ema(env, env_ids, alpha: float = 0.005) -> float:
    step = env.common_step_counter
    if getattr(env, "_pp_succ_step", None) != step:
        env._pp_succ_step = step
        if env_ids is not None and len(env_ids) > 0:
            s = env.termination_manager.get_term("success")[env_ids].float().mean().item()
            env._pp_succ_ema = (1.0 - alpha) * getattr(env, "_pp_succ_ema", 0.0) + alpha * s
    return getattr(env, "_pp_succ_ema", 0.0)


def adapt_spawn_angle(env, env_ids, start_angle: float, end_angle: float, rate: float, success_threshold: float):
    cur = getattr(env, "_pp_half_angle", None)
    if cur is None:
        cur = start_angle
    if success_rate_ema(env, env_ids) > success_threshold:
        cur = min(end_angle, cur + rate)
    env._pp_half_angle = cur
    return math.degrees(cur)


def adapt_place_tol(env, env_ids, start_tol: float, end_tol: float, rate: float, success_threshold: float,
                    angle_done: float | None = None):
    cur = getattr(env, "_pp_place_tol", None)
    if cur is None:
        cur = start_tol
    ready = angle_done is None or getattr(env, "_pp_half_angle", 0.0) >= angle_done - 1e-6
    if ready and success_rate_ema(env, env_ids) > success_threshold:
        cur = max(end_tol, cur - rate)
    if cur != getattr(env, "_pp_place_tol", None):
        env._pp_place_tol = cur
        for manager in (env.reward_manager, env.termination_manager):
            for name in manager.active_terms:
                p = manager.get_term_cfg(name).params.get("p")
                if p is not None:
                    p.place_tol = cur
    return cur * 100.0


def adapt_sector_weights(env, env_ids, num_sectors: int = 8, alpha: float = 0.005, floor: float = 0.25,
                         half_angle: float = math.radians(160.0)):
    dev = env.device
    if not hasattr(env, "_pp_sector_succ"):
        env._pp_sector_succ = torch.zeros(num_sectors, device=dev)
    angle = getattr(env, "_pp_spawn_angle", None)
    if angle is not None and env_ids is not None and len(env_ids) > 0 and env.common_step_counter > 0:
        ids = torch.as_tensor(env_ids, device=dev, dtype=torch.long)
        sec = ((angle[ids] + half_angle) / (2.0 * half_angle) * num_sectors).long().clamp(0, num_sectors - 1)
        succ = env.termination_manager.get_term("success")[ids].float()
        cnt = torch.bincount(sec, minlength=num_sectors).float()
        tot = torch.bincount(sec, weights=succ, minlength=num_sectors)
        seen = cnt > 0
        mean = torch.where(seen, tot / cnt.clamp(min=1.0), env._pp_sector_succ)
        env._pp_sector_succ = torch.where(seen, (1.0 - alpha) * env._pp_sector_succ + alpha * mean, env._pp_sector_succ)
    full = getattr(env, "_pp_half_angle", half_angle) >= half_angle - 1e-6
    env._pp_sector_weights = floor + (1.0 - env._pp_sector_succ) if full else None
    width = 2.0 * math.degrees(half_angle) / num_sectors
    lo = -math.degrees(half_angle)
    return {f"succ_{lo + i * width:+.0f}": env._pp_sector_succ[i].item() for i in range(num_sectors)}
