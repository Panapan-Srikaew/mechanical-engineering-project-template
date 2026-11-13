# Franka Panda 360° Pick & Place with Reinforcement Learning in Isaac Sim

A 7-DoF Franka Emika Panda arm learns, with PPO in NVIDIA Isaac Sim / Isaac Lab, to pick up a cube placed anywhere around it and place it on a random goal. The robot's mechanical model (URDF) is analysed in detail and linked to the learning results.


## Project overview

- **Project name:** Franka Panda 360° Pick & Place with PPO in Isaac Sim / Isaac Lab
- **Problem / purpose:** teach a robot arm to pick up a cube placed at a random spot all around it (±160°, 0.40–0.70 m from the base), carry it to a random goal and place it within 1.5 cm, moving slowly and smoothly (hand ≤ 0.25 m/s)
- **Team:**
  - Panapan Srikaew (6730281821) – Isaac Sim setup and operation, running the simulation system
  - Parithat Muenma (6730296221) – URDF analysis and robot model study
  - Panyawat Wipassuchanan (6730303021) – system design, model training and reward tuning
- **Advisor:** Assoc. Prof. Dr. Ratchatin Chancharoen
- **Status:** Complete (simulation)
- **Last updated:** 2026-11-13

## Repository structure

| Path | Purpose |
|---|---|
| [`hardware/mechanical/`](hardware/mechanical/README.md) | Robot model (URDF): links, joints, DH table, forward kinematics, workspace, URDF vs. simulator |
| [`hardware/electronics/`](hardware/electronics/README.md) | Not used (simulation only) |
| [`hardware/bom.csv`](hardware/bom.csv) | Computing resources and software used |
| [`software/firmware/`](software/firmware/README.md) | Not used (no microcontroller) |
| [`software/application/`](software/application/README.md) | Isaac Lab task and the training / evaluation / analysis scripts |
| [`machine-skills/`](machine-skills/README.md) | Trained pick & place policy (`.pt` / `.onnx`), inputs, outputs and limits |
| [`simulation/`](simulation/README.md) | Scene, physics, observations, actions, reward, PPO settings |
| [`docs/`](docs/README.md) | Installation and usage, report and slides |
| [`media/`](media/README.md) | Images, GIFs and videos |
| [`experiments/`](experiments/README.md) | Training runs, results and robustness tests |
| [`publications/`](publications/README.md) | References |

## Collaboration and file practices

Software versions: Isaac Sim 6.0.1, Isaac Lab 3.0, rsl-rl-lib 5.0.1, PyTorch 2.10 (see [`docs/`](docs/README.md)). Training logs and intermediate checkpoints stay on the training PC; the repository keeps the final model, the evaluation results and the plots.

## Safety and validation

Everything is simulated; nothing has been run on a real robot. The robot model was checked against the simulator (forward kinematics, joint limits, masses) in [`hardware/mechanical/`](hardware/mechanical/README.md). All results come from 1,000 test episodes (see [`experiments/`](experiments/README.md)).
