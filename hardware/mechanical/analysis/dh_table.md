Modified DH (Craig) parameters of the Panda arm, derived from the URDF joint origins.
T_i = RotX(alpha_{i-1}) · TransX(a_{i-1}) · RotZ(theta_i) · TransZ(d_i)

| i | a_{i-1} (m) | d_i (m) | alpha_{i-1} (°) | theta_i |
|---|---|---|---|---|
| 1 | 0 | 0.333 | 0 | q1 |
| 2 | 0 | 0 | -90 | q2 |
| 3 | 0 | 0.316 | 90 | q3 |
| 4 | 0.0825 | 0 | 90 | q4 |
| 5 | -0.0825 | 0.384 | -90 | q5 |
| 6 | 0 | 0 | 90 | q6 |
| 7 | 0.088 | 0 | 90 | q7 |
| flange (`panda_joint8`, fixed) | 0 | 0.107 | 0 | 0 |
| hand (`panda_hand_joint`, fixed) | 0 | 0 | 0 | −45° (yaw) |
| TCP (between fingertips) | 0 | 0.1034 | 0 | 0 |
