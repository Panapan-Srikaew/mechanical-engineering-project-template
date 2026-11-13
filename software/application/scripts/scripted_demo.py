import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Franka-IK-Rel-360-Play-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--trace", action="store_true", help="print the state of the first 3 envs every 25 steps")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import franka_pick_place

APPROACH, DESCEND, CLOSE, LIFT, CARRY, LOWER, OPEN, RETREAT = range(8)
NAMES = ["approach", "descend", "close", "lift", "carry", "lower", "open", "retreat"]


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.terminations.time_out = None
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    n, dev = u.num_envs, u.device
    scale = torch.as_tensor(env_cfg.actions.arm_action.scale, device=dev, dtype=torch.float32).flatten()
    scale = scale[:3] if scale.numel() >= 3 else scale.repeat(3)
    max_cmd = torch.clamp(0.05 / scale, max=1.0)
    env.reset()
    spawn_xy = (u.scene["object"].data.root_pos_w.torch[:, :2] - u.scene.env_origins[:, :2]).clone()

    phase = torch.zeros(n, dtype=torch.long, device=dev)
    timer = torch.zeros(n, device=dev)
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    outcome = torch.full((n,), -1, device=dev)
    ret = torch.zeros(n, device=dev)
    grip_pos = torch.zeros(n, 3, device=dev)

    for t in range(int(u.max_episode_length)):
        obj = u.scene["object"].data.root_pos_w.torch[:, :3] - u.scene.env_origins
        ee = u.scene["ee_frame"].data.target_pos_w.torch[:, 0] - u.scene.env_origins
        goal = u.command_manager.get_command("object_pose")[:, :3]

        up = torch.tensor([0.0, 0.0, 1.0], device=dev)
        tgt = ee.clone()
        close = torch.zeros(n, dtype=torch.bool, device=dev)

        m = phase == APPROACH
        tgt[m] = obj[m] + 0.10 * up
        m = phase == DESCEND
        tgt[m] = obj[m] + 0.005 * up
        m = phase == CLOSE
        tgt[m] = grip_pos[m]
        close[m] = True
        m = phase == LIFT
        tgt[m] = grip_pos[m] + 0.12 * up
        close[m] = True
        m = phase == CARRY
        tgt[m] = goal[m] + 0.10 * up
        close[m] = True
        m = phase == LOWER
        tgt[m] = goal[m] + 0.005 * up
        close[m] = True
        m = phase == OPEN
        tgt[m] = ee[m]
        m = phase == RETREAT
        tgt[m] = goal[m] + 0.15 * up

        arm = torch.maximum(torch.minimum((tgt - ee) / scale, max_cmd), -max_cmd)
        rot = torch.zeros(n, 3, device=dev)
        grip = torch.where(close, -1.0, 1.0).unsqueeze(1)
        actions = torch.cat([arm, rot, grip], dim=1)
        _, rew, terminated, truncated, _ = env.step(actions)
        ret += rew * (~done)

        d = torch.norm(tgt - ee, dim=1)
        timer += 1
        nxt = torch.zeros(n, dtype=torch.bool, device=dev)
        nxt |= (phase == APPROACH) & (d < 0.02)
        nxt |= (phase == DESCEND) & (d < 0.01)
        nxt |= (phase == CLOSE) & (timer > 15)
        nxt |= (phase == LIFT) & (d < 0.02)
        nxt |= (phase == CARRY) & (d < 0.01)
        nxt |= (phase == LOWER) & (d < 0.008)
        nxt |= (phase == OPEN) & (timer > 15)
        grip_pos[(phase == DESCEND) & nxt] = ee[(phase == DESCEND) & nxt]
        phase = torch.where(nxt & (phase < RETREAT), phase + 1, phase)
        timer[nxt] = 0

        if args_cli.trace and t % 25 == 0:
            for i in range(min(3, n)):
                print(f"t={t:3d} env{i} {NAMES[phase[i]]:8s} ee={ee[i].tolist()} tgt={tgt[i].tolist()} "
                      f"obj={obj[i].tolist()} d={d[i]:.3f}")
            terms = u.reward_manager.get_active_iterable_terms(0)
            print("        env0 reward terms: " + " ".join(f"{k}={v[0]:+.3f}" for k, v in terms if v[0] != 0))

        tm = u.termination_manager
        newly = (terminated | truncated) & ~done
        outcome[newly & tm.get_term("success")] = 1
        outcome[newly & tm.get_term("object_dropped")] = 2
        done |= newly
        if done.all():
            break

    print("\n=========== SCRIPTED DEMO ===========")
    print(f"episodes      : {n}")
    print(f"success rate  : {(outcome == 1).float().mean() * 100:.1f} %")
    print(f"dropped       : {(outcome == 2).float().mean() * 100:.1f} %")
    print(f"mean return   : {ret.mean():.1f}   (success episodes end early, return incl. bonus)")
    ang = torch.rad2deg(torch.atan2(spawn_xy[:, 1], spawn_xy[:, 0]))
    for lo in range(-180, 180, 45):
        m = (ang >= lo) & (ang < lo + 45)
        if m.any():
            print(f"  cube at {lo:+4d}..{lo + 45:+4d} deg: success {(outcome[m] == 1).float().mean() * 100:5.1f} %  ({int(m.sum())})")
    hist = torch.bincount(phase[outcome != 1], minlength=8).tolist()
    print("unfinished envs stuck in phase:", {NAMES[i]: c for i, c in enumerate(hist) if c})
    print("=====================================")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
