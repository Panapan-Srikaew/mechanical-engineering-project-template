# URDF figures

Made with `scripts/urdf_analysis.py` (from the URDF only) and `scripts/record_joints.py` (trained policy, 256 episodes). Used in [`hardware/mechanical/`](../../hardware/mechanical/README.md).

| File | Description |
|---|---|
| `kinematic_tree.png` | Link–joint tree from the URDF |
| `robot_frames_3d.png` | Frames and rotation axes of joints 1–7 at the start pose |
| `joint_limits.png` | Range of each joint from the URDF |
| `workspace.png` | Positions the hand can reach vs. the cube area |
| `joint_usage.png` | Joint angles used by the trained robot and time spent near a limit |
