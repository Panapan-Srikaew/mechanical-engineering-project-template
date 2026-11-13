# Machine skills: pick & place learned with RL

The "skill" of this project is the trained policy (a neural network). It reads the state of the robot, cube and goal, and commands the hand every 0.02 s.

| File in [`pick_place_policy/`](pick_place_policy) | Use |
|---|---|
| `model_7998.pt` | rsl_rl checkpoint (run 19) for `play.py` / `eval.py` / further training |
| `policy.pt` | TorchScript (actor only) |
| `policy.onnx` + `policy.onnx.data` | ONNX for use outside Isaac Sim (e.g. ROS / a real robot) · input `obs` [1, 50] → output [1, 7] |
| `agent.yaml` | PPO settings used for training |

## Skill specification
| | |
|---|---|
| Supported robot | Franka Emika Panda + Franka Hand (URDF in `hardware/mechanical/urdf`) |
| Software | Isaac Sim 6.0.1 / Isaac Lab 3.0 / rsl-rl-lib 5.0.1 (or ONNX Runtime) |
| Control rate | 50 Hz |
| Workspace | Cube and goal on the table, 0.40–0.70 m from the base, ±160° (goal ≤ 90° around the base from the cube) |
| Object | ≈ 4.1 cm cube, 0.7–1.5× the trained mass, friction 0.5–1.2 |
| Expected result | Cube placed ≤ 1.5 cm from the goal in ~5 s (98.6 % success in simulation) |
| Safety limits | Hand ≤ 0.25 m/s, wrist ≤ 1 rad/s (enforced in the action layer) |

### Input (50 observations, in this order)
`joint_pos(9) · joint_vel(9) · object_position(3) · ee_position(3) · target_pose(7) · object_to_target(3) · last_action(7) · arm_state(9)`
All positions are in the **frame turned to face the hand around the base** (`mdp/observations.py: *_hand_frame`), not the base frame.

### Output (7 actions, clipped to ±1)
`[radial, around the base, up/down, roll, pitch, yaw, hand]`. Convert to a hand target with `_polar_step` (`mdp/actions.py`), smooth with the EMA, then feed differential IK.
Last value ≥ 0 = open hand, < 0 = close hand (Isaac Lab `BinaryJointPositionAction`).

## How to use
```bat
cd software\application
REM watch the robot in the Isaac Sim window
isaaclab.bat -p scripts\play.py --num_envs 1 --viz kit --checkpoint ..\..\machine-skills\pick_place_policy\model_7998.pt
REM or double-click watch_robot.bat
```

## Before using it on a real robot
- Not tested on a real robot. Start at low speed, with someone watching and an emergency stop.
- The policy keeps joint 6 near its upper limit (215°) 67 % of the time and joint 4 near its lower limit 20 % of the time ([hardware/mechanical §6](../hardware/mechanical/README.md)). A real robot may stop on a joint-limit reflex.
- The policy was trained on `franka.usd`, whose link inertias are 10⁻⁴ of the URDF values ([hardware/mechanical §5](../hardware/mechanical/README.md)). Check its behaviour on a robot model with the real inertias first.
- `arm_state` (internal state of the velocity controller) must be computed exactly as in the simulation.
- The cube position must come from a perception system (e.g. a camera) with ≤ ~5 mm error, as in training.
