#!/usr/bin/env python3
# @file      so3_math.py
# @author    Junlong Jiang     [jiangjunlong@mail.dlut.edu.cn]
# Copyright (c) 2025 Junlong Jiang, all rights reserved

import torch
import numpy as np

def vec2skew(v: torch.Tensor):
    """创建一个3x3的斜对称矩阵，对应于向量v的叉积操作"""
    zero = torch.zeros_like(v[0])
    return torch.tensor([
        [zero, -v[2], v[1]],
        [v[2], zero, -v[0]],
        [-v[1], v[0], zero]
    ], device=v.device, dtype=v.dtype)


def so3Exp(so3: torch.Tensor):
    """将 so3 向量转换为 SO3 旋转矩阵。

    参数:
    so3 (torch.Tensor): 形状为(3,)的 so3 向量。

    返回:
    torch.Tensor: 形状为(3, 3)的 SO3 旋转矩阵。
    """
    so3_norm = torch.norm(so3)
    if so3_norm <= 1e-7:
        return torch.eye(3, device=so3.device, dtype=so3.dtype)

    so3_skew_sym = vec2skew(so3)
    I = torch.eye(3, device=so3.device, dtype=so3.dtype)

    SO3 = I + (so3_skew_sym / so3_norm) * torch.sin(so3_norm) + \
          (so3_skew_sym @ so3_skew_sym / (so3_norm * so3_norm)) * (1 - torch.cos(so3_norm))
    return SO3


def SO3Log(SO3: torch.Tensor):
    '''李群转换为李代数

    参数:
    SO3 (torch.Tensor): 形状为(3, 3)的 SO3 旋转矩阵。

    返回:
    torch.Tensor: 形状为(3,)的 so3 向量。
    '''
    # 计算旋转角度 theta
    trace = SO3.trace()
    theta = torch.acos((trace - 1) / 2) if trace <= 3 - 1e-6 else 0.0

    # 计算so3向量
    so3 = torch.tensor([
        SO3[2, 1] - SO3[1, 2],
        SO3[0, 2] - SO3[2, 0],
        SO3[1, 0] - SO3[0, 1]
    ], device=SO3.device)

    # 调整so3向量的尺度
    if abs(theta) < 0.001:
        so3 = 0.5 * so3
    else:
        so3 = 0.5 * theta / torch.sin(theta) * so3

    return so3


def A_T(v: torch.Tensor):
    """根据给定的三维向量v，计算相应的旋转矩阵"""
    squared_norm = torch.dot(v, v)  # 计算向量的模的平方
    norm = torch.sqrt(squared_norm)  # 向量的模
    identity = torch.eye(3, device=v.device, dtype=v.dtype)  # 单位矩阵

    if norm < 1e-11:
        return identity
    else:
        S = vec2skew(v)  # 计算斜对称矩阵
        term1 = (1 - torch.cos(norm)) / squared_norm
        term2 = (1 - torch.sin(norm) / norm) / squared_norm
        return identity + term1 * S + term2 * torch.matmul(S, S)