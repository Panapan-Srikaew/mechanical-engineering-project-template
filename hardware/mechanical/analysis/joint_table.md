| Joint | Type | Parent → Child | Origin xyz (m) | Origin rpy (°) | Axis | Range | Max velocity | Max effort |
|---|---|---|---|---|---|---|---|---|
| `panda_joint1` | revolute | panda_link0 → panda_link1 | 0 0 0.333 | 0 0 0 | 0 0 1 | -166° … 166° | 2.175 rad/s (125°/s) | 87 Nm |
| `panda_joint2` | revolute | panda_link1 → panda_link2 | 0 0 0 | -90 0 0 | 0 0 1 | -101° … 101° | 2.175 rad/s (125°/s) | 87 Nm |
| `panda_joint3` | revolute | panda_link2 → panda_link3 | 0 -0.316 0 | 90 0 0 | 0 0 1 | -166° … 166° | 2.175 rad/s (125°/s) | 87 Nm |
| `panda_joint4` | revolute | panda_link3 → panda_link4 | 0.0825 0 0 | 90 0 0 | 0 0 1 | -176° … -4° | 2.175 rad/s (125°/s) | 87 Nm |
| `panda_joint5` | revolute | panda_link4 → panda_link5 | -0.0825 0.384 0 | -90 0 0 | 0 0 1 | -166° … 166° | 2.610 rad/s (150°/s) | 12 Nm |
| `panda_joint6` | revolute | panda_link5 → panda_link6 | 0 0 0 | 90 0 0 | 0 0 1 | -1° … 215° | 2.610 rad/s (150°/s) | 12 Nm |
| `panda_joint7` | revolute | panda_link6 → panda_link7 | 0.088 0 0 | 90 0 0 | 0 0 1 | -166° … 166° | 2.610 rad/s (150°/s) | 12 Nm |
| `panda_joint8` | fixed | panda_link7 → panda_link8 | 0 0 0.107 | 0 0 0 | — | — | — | — |
| `panda_hand_joint` | fixed | panda_link8 → panda_hand | 0 0 0 | 0 0 -45 | — | — | — | — |
| `panda_finger_joint1` | prismatic | panda_hand → panda_leftfinger | 0 0 0.0584 | 0 0 0 | 0 1 0 | 0 … 4 cm | 0.2 m/s | 20 N |
| `panda_finger_joint2` | prismatic (mimic panda_finger_joint1) | panda_hand → panda_rightfinger | 0 0 0.0584 | 0 0 0 | 0 -1 0 | 0 … 4 cm | 0.2 m/s | 20 N |
