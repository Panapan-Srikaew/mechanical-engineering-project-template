from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except Exception:
    torch = None


@dataclass
class PickPlaceParams:
    object_rest_z: float = 0.0206
    max_opening: float = 0.08
    hold_open_range: tuple = (0.015, 0.068)
    open_thresh: float = 0.07
    near_thresh: float = 0.04
    lift_height: float = 0.03
    place_tol: float = 0.03
    place_speed: float = 0.05
    retreat_dist: float = 0.06
    elevated_thresh: float = 0.02
    goal_zone_scale: float = 2.0
    reach_std: float = 0.1
    reach_std_coarse: float = 0.4
    coarse_std: float = 0.3
    fine_std: float = 0.05
    lift_hold_after_pick: bool = False
    coarse_xy: bool = False


DEFAULT_WEIGHTS = dict(
    reach=1.0,
    grasp=2.0,
    lift=15.0,
    transport_coarse=16.0,
    transport_fine=5.0,
    place=10.0,
    success_bonus=50.0,
    object_dropped=-200.0,
    action_rate=-1e-4,
    joint_vel=-1e-4,
    rot_action=-1e-3,
    action_l2=-1e-3,
)

SIGNAL_NAMES = ("reach", "grasp", "lift", "transport_coarse", "transport_fine", "place", "success")


def _xp(x):
    if isinstance(x, np.ndarray):
        return np
    return torch


def _kernel(xp, d, std):
    return 1.0 - xp.tanh(d / std)


def _f(xp, mask):
    return xp.where(mask, 1.0, 0.0)


def stage_signals(
    d_ee, d_goal, obj_z, goal_z, opening, obj_speed, p: PickPlaceParams = PickPlaceParams(), was_lifted=None,
    close_cmd=None, d_goal_xy=None,
):
    xp = _xp(d_ee)
    lo, hi = p.hold_open_range

    near = d_ee < p.near_thresh
    holding_width = (opening > lo) & (opening < hi)
    grasped = near & holding_width
    if close_cmd is not None:
        grasped = grasped & close_cmd
    lifted = (obj_z - p.object_rest_z) > p.lift_height
    lifted_now = grasped & lifted
    picked = (xp.ones_like(lifted_now) > 0) if was_lifted is None else (was_lifted | lifted_now)

    at_goal = d_goal < p.place_tol
    goal_zone = (d_goal < p.place_tol * p.goal_zone_scale) & picked
    settled = at_goal & (obj_speed < p.place_speed) & picked
    on_table = (goal_z - p.object_rest_z) < p.elevated_thresh

    finishing = goal_zone & on_table

    open_frac = xp.clip(opening / p.max_opening, 0.0, 1.0)
    retreat_frac = xp.clip(d_ee / p.retreat_dist, 0.0, 1.0)
    released = (opening > p.open_thresh) & (d_ee > p.retreat_dist)

    reach_k = 0.5 * _kernel(xp, d_ee, p.reach_std) + 0.5 * _kernel(xp, d_ee, p.reach_std_coarse)
    reach = reach_k * _f(xp, ~finishing)
    grasp = _f(xp, grasped & ~finishing)
    lift = _f(xp, grasped) * xp.clip((obj_z - p.object_rest_z) / p.lift_height, 0.0, 1.0)
    if p.lift_hold_after_pick and was_lifted is not None:
        lift = xp.where(was_lifted & (grasped | goal_zone), 1.0, lift)

    transport_gate = _f(xp, lifted_now | goal_zone)
    d_coarse = d_goal_xy if (p.coarse_xy and d_goal_xy is not None) else d_goal
    transport_coarse = transport_gate * _kernel(xp, d_coarse, p.coarse_std)
    transport_fine = transport_gate * _kernel(xp, d_goal, p.fine_std)

    place_table = _f(xp, settled) * (0.3 + 0.35 * open_frac + 0.35 * retreat_frac)
    place_air = _f(xp, settled & grasped)
    place = xp.where(on_table, place_table, place_air)

    success = _f(xp, xp.where(on_table, settled & released, settled & grasped))

    return dict(
        reach=reach,
        grasp=grasp,
        lift=lift,
        transport_coarse=transport_coarse,
        transport_fine=transport_fine,
        place=place,
        success=success,
        lifted_latch=_f(xp, picked),
    )


def weighted_step_reward(signals: dict, weights: dict = DEFAULT_WEIGHTS):
    return (
        weights["reach"] * signals["reach"]
        + weights["grasp"] * signals["grasp"]
        + weights["lift"] * signals["lift"]
        + weights["transport_coarse"] * signals["transport_coarse"]
        + weights["transport_fine"] * signals["transport_fine"]
        + weights["place"] * signals["place"]
    )
