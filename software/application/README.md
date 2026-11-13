# Application software

All project code. Runs on a PC (Windows 11 + NVIDIA GPU) in Isaac Sim / Isaac Lab.

| | |
|---|---|
| OS | Windows 11 (Linux works too: use `./isaaclab.sh` instead of `isaaclab.bat`) |
| Runtime | Python 3.12 with Isaac Sim 6.0.1 + Isaac Lab 3.0 |
| Libraries | `isaaclab`, `isaaclab_rl`, `rsl-rl-lib==5.0.1`, `torch`, `matplotlib` |
| GPU | NVIDIA, ≥ 8 GB (4,096 training robots use ~10 GB) |

## Layout
```
software/application/
├── source/franka_pick_place/          Python package (pip install -e), registers the tasks
│   └── franka_pick_place/tasks/pick_place/
│       ├── pick_place_env_cfg.py        scene, observations, rewards, terminations
│       ├── config/franka/
│       │   ├── ik_polar_360_env_cfg.py  main task: hand commanded in polar coordinates
│       │   ├── ik_rel_360_env_cfg.py    360° task with x/y/z hand commands (run 13–17)
│       │   ├── ik_rel_env_cfg.py        small task in front of the robot (run 8–12)
│       │   ├── joint_pos_env_cfg.py     joint angle commands (run 1–7)
│       │   └── agents/rsl_rl_ppo_cfg.py PPO settings
│       └── mdp/                         actions, observations, rewards, curriculum, goal, cube spawn
├── scripts/
│   ├── train.py           train with PPO
│   ├── play.py            watch the robot, record a video, export ONNX
│   ├── eval.py            test on 1,000 episodes, success by angle
│   ├── rsl_rl_utils.py    shared helpers used by train / play / eval
│   ├── urdf_analysis.py   URDF → joint/link tables, DH, FK, workspace (no simulator needed)
│   ├── record_joints.py   record the joints while the trained robot works
│   ├── scripted_demo.py   hand-written controller (comparison)
│   └── plot_training.py   training curves
├── tests/test_reward_math.py   reward unit test
└── watch_robot.bat             double-click to watch the robot in Isaac Sim
```

## Install and run
See [docs/](../../docs/README.md).
