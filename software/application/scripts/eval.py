import argparse
import os

from isaaclab.app import AppLauncher
from rsl_rl_utils import add_legacy_args

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Franka-IK-Polar-360-Play-v0")
parser.add_argument("--num_envs", type=int, default=1000)
parser.add_argument("--checkpoint", type=str, default=None, help="model_*.pt (default: newest under logs/)")
parser.add_argument("--log_root", type=str, default=os.path.join(PROJECT_DIR, "logs", "rsl_rl", "franka_pick_place"))
parser.add_argument("--obs_clamp", type=float, default=None,
                    help="clamp observations to +-value before the policy, as train.py does (100); off by default")
parser.add_argument("--plot_name", type=str, default="eval_by_angle.png",
                    help="angle plot written next to the checkpoint ('none' = no plot; use another name for variations)")
add_legacy_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import franka_pick_place
from rsl_rl_utils import adapt_agent_cfg, apply_legacy_overrides, newest_checkpoint


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    apply_legacy_overrides(env_cfg, args_cli)
    env_cfg.commands.object_pose.debug_vis = False
    agent_cfg = adapt_agent_cfg(load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point"))
    if args_cli.clip_actions is not None:
        agent_cfg.clip_actions = args_cli.clip_actions
    ckpt = args_cli.checkpoint or newest_checkpoint(args_cli.log_root)
    print(f"[INFO] evaluating {ckpt}")

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    u = env.unwrapped
    n = u.num_envs

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    cube0 = u.scene["object"].data.root_pos_w.torch[:, :2] - u.scene.env_origins[:, :2]
    spawn_deg = torch.rad2deg(torch.atan2(cube0[:, 1], cube0[:, 0])).cpu()
    outcome = torch.full((n,), -1, dtype=torch.long, device=u.device)
    steps_to_success = torch.zeros(n, device=u.device)
    dt = u.step_dt
    acc = {"abs": 0.0, "sat": 0.0, "n": 0, "flip": 0.0, "nflip": 0, "jerk": 0.0, "njerk": 0}
    prev_ee = prev_arm = prev_v = None
    prev_valid = prev_m = None
    for t in range(int(u.max_episode_length) + 5):
        run_mask = outcome < 0
        with torch.inference_mode():
            if args_cli.obs_clamp is not None:
                if torch.is_tensor(obs):
                    obs = obs.clamp(-args_cli.obs_clamp, args_cli.obs_clamp)
                else:
                    for k in list(obs.keys()):
                        obs[k] = obs[k].clamp(-args_cli.obs_clamp, args_cli.obs_clamp)
            obs_in = (obs if torch.is_tensor(obs) else obs["policy"]).clone()
            a = policy(obs)
            obs = env.step(a)[0]
        arm = a[:, :-1].clone()
        ee = (u.scene["ee_frame"].data.target_pos_w.torch[:, 0] - u.scene.env_origins).clone()
        tm = u.termination_manager
        running = outcome < 0
        succ = tm.get_term("success") & running
        drop = tm.get_term("object_dropped") & running & ~succ
        tout = tm.get_term("time_out") & running & ~succ & ~drop
        steps_to_success[succ] = float(t + 1)
        outcome[succ], outcome[drop], outcome[tout] = 1, 2, 0

        valid = run_mask & (outcome < 0)
        ob = obs if torch.is_tensor(obs) else obs["policy"]
        acc["obs_nonfinite"] = acc.get("obs_nonfinite", 0) + int((run_mask & ~torch.isfinite(ob).all(dim=1)).sum())
        a_max = arm.abs().nan_to_num(nan=float("inf")).max(dim=1).values
        bad = run_mask & (a_max > 100)
        acc["outlier"] = acc.get("outlier", 0) + int(bad.sum())
        if bad.any():
            acc["outlier_obs_max"] = max(acc.get("outlier_obs_max", 0.0), obs_in[bad].abs().max().item())
            dims = (obs_in[bad].abs() > 100).nonzero()[:, 1].tolist()
            acc.setdefault("outlier_dims", {})
            for d_ in dims:
                acc["outlier_dims"][d_] = acc["outlier_dims"].get(d_, 0) + 1
            acc.setdefault("outlier_envs", set()).update(bad.nonzero()[:, 0].tolist())
        acc["obs_p_max"] = max(acc.get("obs_p_max", 0.0), ob[run_mask].abs().max().item() if run_mask.any() else 0.0)
        finite = a_max <= 100
        acc["nonfinite"] = acc.get("nonfinite", 0) + int((run_mask & ~torch.isfinite(arm).all(dim=1)).sum())
        m_ok = run_mask & finite
        acc["abs"] += arm[m_ok].abs().sum().item()
        acc["sat"] += (arm[m_ok].abs() >= 1.0).sum().item()
        acc["n"] += int(m_ok.sum()) * arm.shape[1]
        if prev_ee is not None:
            m = valid & prev_valid
            acc["flip"] += ((arm * prev_arm) < 0)[m].sum().item()
            acc["nflip"] += int(m.sum()) * arm.shape[1]
            v = (ee - prev_ee) / dt
            if prev_v is not None:
                m2 = m & prev_m
                acc["jerk"] += ((v - prev_v).norm(dim=1) / dt)[m2].sum().item()
                acc["njerk"] += int(m2.sum())
            prev_v, prev_m = v, m
        prev_ee, prev_arm, prev_valid = ee, arm, valid
        if (outcome >= 0).all():
            break

    s = (outcome == 1).float().mean().item()
    d = (outcome == 2).float().mean().item()
    t = (outcome == 0).float().mean().item()
    print("\n================ EVALUATION ================")
    print(f"checkpoint     : {ckpt}")
    print(f"episodes       : {n}")
    print(f"success rate   : {s * 100:.1f} %")
    print(f"dropped        : {d * 100:.1f} %")
    print(f"time-out       : {t * 100:.1f} %")
    if s > 0:
        print(f"time to success: {steps_to_success[outcome == 1].mean().item() * dt:.2f} s (mean)")
    print("---- smoothness (lower = smoother) ----")
    print(f"mean |arm action|  : {acc['abs'] / max(1, acc['n']):.2f}   (raw policy output; ~1 or less is smooth)")
    print(f"non-finite actions : {acc.get('nonfinite', 0)} env-steps (NaN/inf policy output; excluded from the averages); "
          f"non-finite observations: {acc.get('obs_nonfinite', 0)}")
    print(f"action outliers    : {acc.get('outlier', 0)} env-steps with |a| > 100 (excluded); largest |obs| that produced one: "
          f"{acc.get('outlier_obs_max', 0.0):.3g}; largest |obs| overall: {acc.get('obs_p_max', 0.0):.3g}")
    if acc.get("outlier_envs"):
        print(f"  outliers in {len(acc['outlier_envs'])} envs; obs dims > 100 at those steps (dim: count): "
              f"{dict(sorted(acc.get('outlier_dims', {}).items()))}")
    print(f"actions |a| >= 1   : {acc['sat'] / max(1, acc['n']) * 100:.1f} %   (at or beyond the IK clip)")
    print(f"sign flips / step  : {acc['flip'] / max(1, acc['nflip']) * 100:.1f} %")
    print(f"hand acceleration  : {acc['jerk'] / max(1, acc['njerk']):.1f} m/s^2 (mean)")
    print("---- success by cube spawn angle (0 deg = in front, +-180 = behind) ----")
    edges = list(range(-180, 181, 45))
    ok = (outcome == 1).cpu()
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (spawn_deg >= lo) & (spawn_deg < hi)
        k = int(m.sum())
        rows.append((lo, hi, k, ok[m].float().mean().item() * 100 if k else float("nan")))
        print(f"  {lo:+4d} .. {hi:+4d} deg : " + (f"{rows[-1][3]:5.1f} %  ({k} episodes)" if k else "  -   (no episodes)"))
    print("============================================")
    print(f"SUMMARY success={s * 100:.2f} dropped={d * 100:.2f} timeout={t * 100:.2f} "
          f"time={(steps_to_success[outcome == 1].mean().item() * dt) if s > 0 else float('nan'):.3f} "
          f"accel={acc['jerk'] / max(1, acc['njerk']):.3f}")
    if args_cli.plot_name.lower() != "none":
        save_angle_plot(rows, os.path.join(os.path.dirname(ckpt), args_cli.plot_name))
    env.close()


def save_angle_plot(rows, path):
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="polar")
    for lo, hi, k, rate in rows:
        if k == 0:
            continue
        ax.bar(math.radians((lo + hi) / 2), rate, width=math.radians(hi - lo) * 0.95,
               color=plt.cm.RdYlGn(rate / 100), edgecolor="k", alpha=0.85)
        ax.text(math.radians((lo + hi) / 2), max(rate, 15) + 12, f"{rate:.0f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_yticklabels([])
    ax.set_title("success by cube spawn angle\n(0 deg = in front of the robot)", fontsize=10)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
