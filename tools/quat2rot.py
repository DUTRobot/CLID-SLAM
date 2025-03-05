#!/usr/bin/env python3
import numpy as np

# 四元数转旋转矩阵

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """
    将四元数转换为旋转矩阵。

    输入:
    - qx, qy, qz, qw: 四元数的分量

    返回:
    - rotation_matrix: 3x3的旋转矩阵
    """
    # 四元数的归一化
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    # 计算旋转矩阵
    rotation_matrix = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]
    ])

    return rotation_matrix