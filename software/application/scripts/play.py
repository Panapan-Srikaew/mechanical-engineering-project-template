import argparse
import os

from isaaclab.app import AppLauncher
from rsl_rl_utils import add_legacy_args

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Franka-IK-Polar-360-Play-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None, help="model_*.pt (default: newest under logs/)")
parser.add_argument("--log_root", type=str, default=os.path.join(PROJECT_DIR, "logs", "rsl_rl", "franka_pick_place"))
parser.add_argument("--video", action="store_true", help="record an mp4 into <run>/videos/play and exit")
parser.add_argument("--video_length", type=int, default=800, help="video length in control steps (50 per second)")
parser.add_argument("--export", action="store_true", help="write <run>/exported/policy.onnx + policy.pt (TorchScript) and exit")
add_legacy_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
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
    agent_cfg = adapt_agent_cfg(load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point"))
    if args_cli.clip_actions is not None:
        agent_cfg.clip_actions = args_cli.clip_actions
    ckpt = args_cli.checkpoint or newest_checkpoint(args_cli.log_root)
    print(f"[INFO] loading {ckpt}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_dir = os.path.join(os.path.dirname(ckpt), "videos", "play")
        env = gym.wrappers.RecordVideo(
            env, video_folder=video_dir, step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length, disable_logger=True,
        )
        print(f"[INFO] recording {args_cli.video_length} steps to {video_dir}")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    if args_cli.export:
        out = os.path.join(os.path.dirname(ckpt), "exported")
        runner.export_policy_to_jit(out, "policy.pt")
        runner.export_policy_to_onnx(out, "policy.onnx")
        print(f"[INFO] exported policy.onnx / policy.pt to {out}")
        env.close()
        return

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            out = env.step(actions)
            obs = out[0]
        step += 1
        if args_cli.video and step >= args_cli.video_length:
            break
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
