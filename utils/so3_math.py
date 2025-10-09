#!/usr/bin/env python3
# @file      so3_math.py
# @author    Junlong Jiang     [jiangjunlong@mail.dlut.edu.cn]
# Copyright (c) 2025 Junlong Jiang, all rights reserved
import torch


# [v]_x = [[0,   -v3,   v2],
#          [v3,    0,  -v1],
#          [-v2,  v1,    0]]
# and v = [v1, v2, v3]^T
# v X w = [v]_x w
def vec2skew(v: torch.Tensor):
    """返回向量v的对应反对称矩阵[v]_x"""
    zero = torch.zeros_like(v[0])
    return torch.tensor(
        [[zero, -v[2], v[1]], [v[2], zero, -v[0]], [-v[1], v[0], zero]],
        device=v.device,
        dtype=v.dtype,
    )


def batch_vec2skew(v: torch.Tensor):
    """将一个批次的向量转换为一个批次的斜对称矩阵"""
    skew_sym = torch.zeros((v.shape[0], 3, 3), dtype=v.dtype, device=v.device)
    skew_sym[:, 0, 1] = -v[:, 2]
    skew_sym[:, 0, 2] = v[:, 1]
    skew_sym[:, 1, 0] = v[:, 2]
    skew_sym[:, 1, 2] = -v[:, 0]
    skew_sym[:, 2, 0] = -v[:, 1]
    skew_sym[:, 2, 1] = v[:, 0]

    return skew_sym


# For normalized axis a and angle theta theta
# a_x * a_x = aa^T - I
# exp([a*theta]_x) = I + sin(theta)*[a]_x + (1-cos(theta))*[a]_x*[a]_x
# = I + sin(theta)*[a]_x + (1-cos(theta))*(aa^T - I)
# = cos(theta)*I + (1-cos(theta))*aa^T + sin(theta
def so3Exp(so3: torch.Tensor):
    """将 so3 向量转换为 SO3 旋转矩阵"""
    theta = torch.norm(so3)
    I = torch.eye(3, device=so3.device, dtype=so3.dtype)
    if theta <= 1e-8:
        return I
    axis = so3 / theta
    skew = vec2skew(axis)
    SO3 = I + torch.sin(theta) * skew + (1 - torch.cos(theta)) * skew @ skew
    return SO3


def SO3Log(R: torch.Tensor) -> torch.Tensor:
    """将旋转矩阵 SO3 转换为旋转向量 so3"""
    trace = R.trace()
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)

    w = torch.tensor(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        device=R.device,
        dtype=R.dtype,
    )

    sin_theta = torch.sin(theta)

    # 小角度数值稳定处理
    if theta.abs() < 1e-4:
        scale = 0.5 + theta**2 / 12.0
    else:
        scale = 0.5 * theta / (sin_theta + 1e-8)

    return scale * w


def A_T(v: torch.Tensor):
    """根据给定的三维向量v，计算相应的旋转矩阵"""
    squared_norm = torch.dot(v, v)
    norm = torch.sqrt(squared_norm)
    I = torch.eye(3, device=v.device, dtype=v.dtype)

    if norm < 1e-11:
        return I
    else:
        S = vec2skew(v)
        term1 = (1 - torch.cos(norm)) / squared_norm
        term2 = (1 - torch.sin(norm) / norm) / squared_norm
        return I + term1 * S + term2 * torch.matmul(S, S)
