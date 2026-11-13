import argparse
import os
import sys

from isaaclab.app import AppLauncher
from rsl_rl_utils import add_legacy_args

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Franka-IK-Polar-360-Play-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--log_root", type=str, default=os.path.join(PROJECT_DIR, "logs", "rsl_rl", "franka_pick_place"))
add_legacy_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import franka_pick_place
from rsl_rl_utils import adapt_agent_cfg, apply_legacy_overrides, newest_checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import urdf_analysis as ua

OUT = ua.OUT
ARM = [f"panda_joint{i}" for i in range(1, 8)]


def quat_xyzw_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main():
    os.makedirs(OUT, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    apply_legacy_overrides(env_cfg, args_cli)
    env_cfg.commands.object_pose.debug_vis = False
    agent_cfg = adapt_agent_cfg(load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point"))
    ckpt = args_cli.checkpoint or newest_checkpoint(args_cli.log_root)
    print(f"[INFO] recording joints with {ckpt}")
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    u = env.unwrapped
    robot = u.scene["robot"]
    jids, jnames = robot.find_joints(ARM, preserve_order=True)
    fids, _ = robot.find_joints(["panda_finger_joint1", "panda_finger_joint2"], preserve_order=True)
    hid = robot.find_bodies("panda_hand")[0][0]
    n, dt = u.num_envs, u.step_dt

    links, joints = ua.parse_urdf(ua.DEFAULT_URDF)
    jd = {j["name"]: j for j in joints}
    lim = robot.data.joint_pos_limits.torch[0].cpu().numpy()
    vlim = robot.data.joint_vel_limits.torch[0].cpu().numpy()
    elim = robot.data.joint_effort_limits.torch[0].cpu().numpy()
    kp = robot.data.joint_stiffness.torch[0].cpu().numpy()
    kd = robot.data.joint_damping.torch[0].cpu().numpy()
    mass = robot.data.default_mass.torch[0].cpu().numpy()
    rows = ["| Joint | URDF range (rad) | Sim range (rad) | URDF vel (rad/s) | Sim vel (rad/s) | URDF effort | Sim effort | Sim stiffness Kp | Sim damping Kd |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name in ARM + ["panda_finger_joint1", "panda_finger_joint2"]:
        k = robot.joint_names.index(name)
        j = jd[name]
        rows.append(f"| `{name}` | {j['lower']:.4f} … {j['upper']:.4f} | {lim[k, 0]:.4f} … {lim[k, 1]:.4f} | {j['velocity']:g} | "
                    f"{vlim[k]:.3f} | {j['effort']:g} | {elim[k]:g} | {kp[k]:g} | {kd[k]:g} |")
    rows += ["", "| Link | URDF mass (kg) | Sim mass (kg) |", "|---|---|---|"]
    for b, name in enumerate(robot.body_names):
        um = links.get(name, {}).get("mass")
        rows.append(f"| `{name}` | {'—' if um is None else f'{um:.3f}'} | {mass[b]:.3f} |")
    open(os.path.join(OUT, "sim_vs_urdf.md"), "w", encoding="utf-8").write("\n".join(rows) + "\n")
    print("\n".join(rows))

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    cube0 = u.scene["object"].data.root_pos_w.torch[:, :2] - u.scene.env_origins[:, :2]
    spawn_deg = torch.rad2deg(torch.atan2(cube0[:, 1], cube0[:, 0])).cpu().numpy()
    T = int(u.max_episode_length) + 5
    Q = np.full((T, n, 7), np.nan, np.float32)
    DQ = np.full((T, n, 7), np.nan, np.float32)
    HP = np.full((T, n, 3), np.nan, np.float32)
    HR = np.full((T, n, 4), np.nan, np.float32)
    FING = np.full((T, n), np.nan, np.float32)
    outcome = np.full(n, -1)
    steps = np.zeros(n)
    for t in range(T):
        running = outcome < 0
        with torch.inference_mode():
            obs = env.step(policy(obs))[0]
        tm = u.termination_manager
        succ = (tm.get_term("success")).cpu().numpy() & running
        drop = (tm.get_term("object_dropped")).cpu().numpy() & running & ~succ
        tout = (tm.get_term("time_out")).cpu().numpy() & running & ~succ & ~drop
        steps[succ] = t + 1
        outcome[succ], outcome[drop], outcome[tout] = 1, 2, 0
        keep = running & (outcome < 0)
        q = robot.data.joint_pos.torch[:, jids].cpu().numpy()
        dq = robot.data.joint_vel.torch[:, jids].cpu().numpy()
        rp = robot.data.root_pos_w.torch.cpu().numpy()
        rq = robot.data.root_quat_w.torch.cpu().numpy()
        bp = robot.data.body_pos_w.torch[:, hid].cpu().numpy()
        bq = robot.data.body_quat_w.torch[:, hid].cpu().numpy()
        f = robot.data.joint_pos.torch[:, fids].sum(dim=1).cpu().numpy()
        Q[t, keep], DQ[t, keep], FING[t, keep] = q[keep], dq[keep], f[keep]
        for e in np.nonzero(keep)[0]:
            R0 = quat_xyzw_to_R(rq[e])
            HP[t, e] = R0.T @ (bp[e] - rp[e])
            HR[t, e] = bq[e]
        HR_root = rq
        if (outcome >= 0).all():
            break
    env.close()
    np.savez_compressed(os.path.join(OUT, "joint_record.npz"), q=Q, dq=DQ, hand_pos=HP, hand_quat=HR, root_quat=HR_root,
                        finger=FING, outcome=outcome, steps=steps, spawn_deg=spawn_deg, dt=dt)
    print(f"success {np.mean(outcome == 1) * 100:.1f}% of {n}")
    analyse(Q, DQ, HP, HR, HR_root, outcome, steps, spawn_deg, dt, joints)


def analyse(Q, DQ, HP, HR, root_quat, outcome, steps, spawn_deg, dt, joints):
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rev = [j for j in joints if j["type"] == "revolute"]
    lo = np.degrees([j["lower"] for j in rev])
    hi = np.degrees([j["upper"] for j in rev])
    vmax = np.degrees([j["velocity"] for j in rev])
    T, n, _ = Q.shape

    idx = np.argwhere(~np.isnan(Q[:, :, 0]))
    rng = np.random.default_rng(0)
    pick = idx[rng.choice(len(idx), size=min(5000, len(idx)), replace=False)]
    ep, er = [], []
    for t, e in pick:
        P = ua.fk_urdf(joints, Q[t, e].astype(float))[0]["panda_hand"]
        ep.append(np.linalg.norm(P[:3, 3] - HP[t, e]))
        Rs = quat_xyzw_to_R(root_quat[e]).T @ quat_xyzw_to_R(HR[t, e])
        er.append(math.degrees(math.acos(np.clip((np.trace(P[:3, :3].T @ Rs) - 1) / 2, -1, 1))))
    ep, er = np.array(ep), np.array(er)
    txt = [
        "FK validation: URDF forward kinematics (scripts/urdf_analysis.py) fed with the joint angles of the simulation,",
        "compared with the panda_hand pose that PhysX reports (robot base frame).",
        f"samples (time steps x envs)  : {len(pick)}",
        f"position error  mean / max   : {ep.mean() * 1000:.3f} / {ep.max() * 1000:.3f} mm",
        f"rotation error  mean / max   : {er.mean():.3f} / {er.max():.3f} deg",
    ]
    open(os.path.join(OUT, "fk_validation.txt"), "w", encoding="utf-8").write("\n".join(txt) + "\n")
    print("\n".join(txt))

    ok = outcome == 1
    examples = []
    for target in (0, 90, 155):
        cand = np.nonzero(ok)[0]
        if len(cand) == 0:
            break
        e = cand[np.argmin(np.abs(np.abs(spawn_deg[cand]) - target))]
        examples.append(e)
    fig, axs = plt.subplots(7, 1, figsize=(10, 13), sharex=True)
    for k in range(7):
        ax = axs[k]
        ax.axhspan(lo[k], hi[k], color="#e8f1fa")
        ax.axhline(lo[k], color="C3", lw=1.2)
        ax.axhline(hi[k], color="C3", lw=1.2)
        for c, e in enumerate(examples):
            tt = np.arange(T) * dt
            ax.plot(tt, np.degrees(Q[:, e, k]), color=f"C{c}", lw=1.6,
                    label=f"cube at {spawn_deg[e]:+.0f}° (success {steps[e] * dt:.1f} s)" if k == 0 else None)
        ax.set_ylabel(f"J{k + 1} (°)")
        span = hi[k] - lo[k]
        ax.set_ylim(lo[k] - 0.08 * span, hi[k] + 0.08 * span)
    axs[0].legend(loc="upper right", fontsize=8, ncol=3)
    axs[-1].set_xlabel("time (s)")
    fig.suptitle("Joint angles of 3 successful episodes (red lines = URDF <limit lower/upper>)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "joint_trajectories.png"), dpi=110)
    plt.close(fig)

    qd = np.degrees(Q.reshape(-1, 7))
    qd = qd[~np.isnan(qd[:, 0])]
    norm = (qd - lo) / (hi - lo)
    near = (np.minimum(qd - lo, hi - qd) < 5.0).mean(axis=0) * 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    a1.violinplot([norm[:, k] for k in range(7)], showextrema=True)
    a1.axhline(0, color="C3")
    a1.axhline(1, color="C3")
    a1.set_xticks(range(1, 8))
    a1.set_xticklabels([f"J{k + 1}" for k in range(7)])
    a1.set_ylabel("position in URDF range (0 = lower, 1 = upper)")
    a1.set_title("Joint angle distribution over all recorded steps")
    a2.bar([f"J{k + 1}" for k in range(7)], near, color="C1")
    for k, v in enumerate(near):
        a2.text(k, v + 0.2, f"{v:.1f}", ha="center", fontsize=8)
    a2.set_ylabel("% of time within 5° of a limit")
    a2.set_title("Time spent near a URDF limit")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "joint_usage.png"), dpi=110)
    plt.close(fig)

    q1max = np.nanmax(np.abs(np.degrees(Q[:, :, 0])), axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for o, c, lab in ((1, "C2", "success"), (0, "C3", "time-out"), (2, "k", "dropped")):
        m = outcome == o
        if m.any():
            ax.scatter(spawn_deg[m], q1max[m], s=12, c=c, label=f"{lab} ({m.sum()})")
    ax.axhline(math.degrees(rev[0]["upper"]), color="C3", ls="--", label="J1 limit 166° (URDF)")
    ax.set_xlabel("cube spawn angle around the robot (°, 0 = front)")
    ax.set_ylabel("max |q1| during the episode (°)")
    ax.set_title("Base joint J1 needed to reach the cube")
    ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "joint1_vs_spawn.png"), dpi=110)
    plt.close(fig)

    dqd = np.abs(np.degrees(DQ.reshape(-1, 7)))
    dqd = dqd[~np.isnan(dqd[:, 0])]
    p99 = np.percentile(dqd, 99, axis=0)
    mx = dqd.max(axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(7)
    ax.bar(x - 0.27, vmax, 0.27, color="0.8", label="URDF velocity limit")
    ax.bar(x, mx, 0.27, color="C1", label="max in simulation")
    ax.bar(x + 0.27, p99, 0.27, color="C0", label="99th percentile")
    ax.set_xticks(x)
    ax.set_xticklabels([f"J{k + 1}" for k in range(7)])
    ax.set_ylabel("joint speed (°/s)")
    ax.set_title("Joint speeds of the trained policy vs URDF limits")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "joint_velocity.png"), dpi=110)
    plt.close(fig)
    with open(os.path.join(OUT, "joint_stats.md"), "w", encoding="utf-8") as f:
        f.write("| Joint | URDF range (°) | used min…max (°) | % time within 5° of limit | URDF vel limit (°/s) | max speed (°/s) | 99th pct (°/s) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for k in range(7):
            f.write(f"| J{k + 1} | {lo[k]:.0f} … {hi[k]:.0f} | {qd[:, k].min():.0f} … {qd[:, k].max():.0f} | {near[k]:.1f} | "
                    f"{vmax[k]:.0f} | {mx[k]:.0f} | {p99[k]:.0f} |\n")
    print(open(os.path.join(OUT, "joint_stats.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
    simulation_app.close()
