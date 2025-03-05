#!/usr/bin/env python3

import numpy as np
from tools.quat2rot import quaternion_to_rotation_matrix

# 计算第一帧位姿

################################# for quad
# ImMesh
# matrix_values = [
#     5.304626993818075675e-01, -8.474622417882305969e-01, 2.042261633276983360e-02, -8.377865843928848644e-02,
#     8.463450710843981595e-01, 5.308216667107832354e-01, 4.391332211528171242e-02, 3.370663104058911230e+00,
#     -4.805564502133059107e-02, -6.009799083416286596e-03, 9.988265833638158009e-01, 7.037440120229881968e-01
# ]
# T_lidar = np.vstack([np.array(matrix_values).reshape(3, 4), [0, 0, 0, 1]])

# T_lidar_imu = np.array([[-1.0, 0, 0, -0.006253],
#                         [0, -1.0, 0, 0.011775],
#                         [0, 0, 1.0, -0.028535],
#                         [0, 0, 0, 1]])
#
# T = T_lidar @ T_lidar_imu

################################# for math easy
T_lidar = np.eye(4)

# SLAMesh
T_lidar[:3, :3] = quaternion_to_rotation_matrix(-0.00987445, 0.00774057, 0.842868, 0.537974)
T_lidar[:3, 3] = np.array([-23.7176, -31.2646, 1.03258])

# ImMesh
# T_lidar[:3, :3] = quaternion_to_rotation_matrix(-0.00248205, 0.00444627, 0.842838, 0.538143)
# T_lidar[:3, 3] = np.array([-23.7202, -31.2861, 1.04326])

T_lidar_imu = np.array([
    [1.0, 0, 0, 0.014],
    [0, 1.0, 0, -0.012],
    [0, 0, 1.0, -0.015],
    [0, 0, 0, 1.0]])

T = T_lidar @ T_lidar_imu

################################ print
print(np.array2string(T[:, :], separator=', ', precision=8, floatmode='fixed'))