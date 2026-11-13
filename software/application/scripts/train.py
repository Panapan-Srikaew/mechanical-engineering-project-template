import argparse
import math
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Franka pick-and-place with rsl_rl PPO.")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Franka-IK-Polar-360-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--resume", type=str, default=None, help="path to a model_*.pt checkpoint to resume from")
parser.add_argument(
    "--curriculum_state", type=str, default=None,
    help="resume the gated curricula of the 360 task: 'ANGLE_DEG,TOL_CM' (last Curriculum/* values in the log)",
)
parser.add_argument("--entropy_coef", type=float, default=None, help="override PPO entropy_coef (default 0.006)")
parser.add_argument(
    "--penalty_end", type=str, default=None,
    help="override the end weight of ramped penalties, e.g. 'action_l2=-0.1,action_rate=-0.3' (run 19)",
)
parser.add_argument(
    "--robot_urdf", type=str, default=None,
    help="train on a robot converted from this URDF (with the robot cfg's physx settings) instead of franka.usd (run 20)",
)
parser.add_argument("--urdf_finger_vel", type=float, default=None, help="finger speed limit (m/s) for --robot_urdf (URDF: 0.2)")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parser.add_argument("--log_root", type=str, default=os.path.join(PROJECT_DIR, "logs", "rsl_rl"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import franka_pick_place
from rsl_rl_utils import adapt_agent_cfg

class FiniteVecEnvWrapper(RslRlVecEnvWrapper):

    OBS_LIMIT, REW_MIN, REW_MAX = 100.0, -10.0, 1000.0

    def step(self, actions):
        obs, rew, dones, extras = super().step(torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0))
        for k in obs.keys():
            obs[k] = torch.nan_to_num(obs[k], nan=0.0, posinf=0.0, neginf=0.0).clamp_(-self.OBS_LIMIT, self.OBS_LIMIT)
        rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0).clamp_(self.REW_MIN, self.REW_MAX)
        return obs, rew, dones, extras


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    if args_cli.curriculum_state:
        angle_deg, tol_cm = (float(x) for x in args_cli.curriculum_state.split(","))
        env_cfg.curriculum.spawn_angle.params["start_angle"] = math.radians(angle_deg)
        env_cfg.curriculum.place_tol.params["start_tol"] = tol_cm / 100.0
    if args_cli.robot_urdf:
        from rsl_rl_utils import apply_robot_urdf

        apply_robot_urdf(env_cfg, args_cli.robot_urdf)
        if args_cli.urdf_finger_vel is not None:
            env_cfg.scene.robot.actuators["panda_hand"].velocity_limit_sim = args_cli.urdf_finger_vel
            print(f"[INFO] finger velocity limit set to {args_cli.urdf_finger_vel} m/s")
    if args_cli.penalty_end:
        for item in args_cli.penalty_end.split(","):
            term, value = item.split("=")
            getattr(env_cfg.curriculum, term.strip()).params["end_weight"] = float(value)
            print(f"[INFO] penalty {term.strip()}: end weight {float(value)}")
    agent_cfg = adapt_agent_cfg(load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point"))
    if args_cli.entropy_coef is not None:
        agent_cfg.algorithm.entropy_coef = args_cli.entropy_coef
        print(f"[INFO] entropy_coef: {args_cli.entropy_coef}")
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    log_dir = os.path.abspath(
        os.path.join(args_cli.log_root, agent_cfg.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    )
    print(f"[INFO] logging to: {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = FiniteVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if args_cli.resume:
        print(f"[INFO] resuming from {args_cli.resume}")
        runner.load(args_cli.resume)
        env.unwrapped.common_step_counter = runner.current_learning_iteration * agent_cfg.num_steps_per_env

    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
