#!/usr/bin/env python3
# @file      error_state_iekf.py
# @author    Junlong Jiang     [jiangjunlong@mail.dlut.edu.cn]
# Copyright (c) 2025 Junlong Jiang, all rights reserved

import math
import torch
import numpy as np
from model.decoder import Decoder
from model.neural_points import NeuralPoints
from utils.config import Config
from utils.so3_math import vec2skew, so3Exp, SO3Log, batch_vec2skew
from utils.tools import get_gradient, transform_torch

G_m_s2 = 9.81  # 定义全局重力加速度


class StateIkfom:
    """18维的状态量x定义: 对应顺序为旋转(3), 位置(3), 速度(3), 角速度偏置(3), 加速度偏置(3), 重力向量(3)"""

    def __init__(
        self, dtype, pos=None, rot=None, vel=None, bg=None, ba=None, grav=None
    ):
        self.dtype = dtype
        self.rot = torch.eye(3, dtype=self.dtype) if rot is None else rot
        self.pos = torch.zeros(3, dtype=self.dtype) if pos is None else pos
        self.vel = torch.zeros(3, dtype=self.dtype) if vel is None else vel
        self.bg = torch.zeros(3, dtype=self.dtype) if bg is None else bg
        self.ba = torch.zeros(3, dtype=self.dtype) if ba is None else ba
        self.grav = (
            torch.tensor([0.0, 0.0, -G_m_s2], dtype=self.dtype)
            if grav is None
            else grav
        )

    def cpu(self):
        """将所有张量转移到CPU"""
        self.rot = self.rot.cpu()
        self.pos = self.pos.cpu()
        self.vel = self.vel.cpu()
        self.bg = self.bg.cpu()
        self.ba = self.ba.cpu()
        self.grav = self.grav.cpu()

    def cuda(self):
        """将所有张量转移到GPU"""
        self.rot = self.rot.cuda()
        self.pos = self.pos.cuda()
        self.vel = self.vel.cuda()
        self.bg = self.bg.cuda()
        self.ba = self.ba.cuda()
        self.grav = self.grav.cuda()


class InputIkfom:
    """输入向量类定义，用于表示陀螺仪和加速度计的测量值。"""

    def __init__(self, dtype, acc: np.array, gyro: np.array):
        self.dtype = dtype
        self.acc = torch.tensor(acc, dtype=self.dtype)
        self.gyro = torch.tensor(gyro, dtype=self.dtype)


def boxplus(state: StateIkfom, delta: torch.tensor):
    """广义加法操作"""
    new_state = StateIkfom(state.dtype)
    new_state.rot = state.rot @ so3Exp(delta[0:3])
    new_state.pos = state.pos + delta[3:6]
    new_state.vel = state.vel + delta[6:9]
    new_state.bg = state.bg + delta[9:12]
    new_state.ba = state.ba + delta[12:15]
    new_state.grav = state.grav + delta[15:18]
    return new_state


def boxminus(x1: StateIkfom, x2: StateIkfom):
    """广义减法操作，计算两个状态之间的差"""
    delta_rot = SO3Log(x2.rot.T @ x1.rot)
    delta_pos = x1.pos - x2.pos
    delta_vel = x1.vel - x2.vel
    delta_bg = x1.bg - x2.bg
    delta_ba = x1.ba - x2.ba
    delta_grav = x1.grav - x2.grav
    delta = torch.concatenate(
        [delta_rot, delta_pos, delta_vel, delta_bg, delta_ba, delta_grav]
    )
    return delta


class IEKFOM:
    """迭代扩展卡尔曼滤波器类"""

    def __init__(
        self,
        config: Config,
        neural_points: NeuralPoints,
        geo_decoder: Decoder,
    ):
        self.config = config
        self.silence = config.silence
        self.neural_points = neural_points
        self.geo_decoder = geo_decoder
        self.device = self.config.device
        self.dtype = config.dtype
        self.tran_dtype = config.tran_dtype

        self.x = StateIkfom(self.tran_dtype)  # 初始化状态
        self.P = torch.eye(18, dtype=self.tran_dtype)  # 初始化状态协方差矩阵
        self.P[9:12, 9:12] = self.P[9:12, 9:12] * 1e-4  # 初始陀螺仪偏置协方差
        self.P[12:15, 12:15] = self.P[12:15, 12:15] * 1e-3  # 初始加速度计协方差
        self.P[15:18, 15:18] = self.P[15:18, 15:18] * 1e-4  # 初始重力协方差
        self.Q = self.process_noise_covariance()  # 前向传播白噪声协方差
        self.R_inv = None  # 测量噪声协方差
        self.eps = 0.001  # 收敛阈值
        self.max_iteration = self.config.reg_iter_n  # 最大迭代轮数

    def process_noise_covariance(self):
        """噪声协方差Q的初始化"""
        Q = torch.zeros((12, 12), dtype=self.config.tran_dtype)
        Q[:3, :3] = self.config.measurement_noise_covariance * torch.eye(3)
        Q[3:6, 3:6] = self.config.measurement_noise_covariance * torch.eye(3)
        Q[6:9, 6:9] = self.config.bias_noise_covariance * torch.eye(3)
        Q[9:12, 9:12] = self.config.bias_noise_covariance * torch.eye(3)
        return Q

    def df_dx(self, s: StateIkfom, in_: InputIkfom, dt: float):
        """计算状态转移函数的雅可比矩阵"""
        # omega_ = in_.gyro - s.bg
        acc_ = in_.acc - s.ba
        df_dx = torch.eye(18, dtype=self.tran_dtype)
        I_dt = torch.eye(3, dtype=self.tran_dtype) * dt
        # df_dx[0:3, 0:3] = so3Exp(-omega_ * dt)
        # so3Exp(-omega_ * dt) 可以近似为I
        df_dx[0:3, 0:3] = torch.eye(3, dtype=self.tran_dtype)
        df_dx[0:3, 9:12] = -I_dt
        df_dx[3:6, 6:9] = I_dt
        df_dx[6:9, 0:3] = -s.rot @ vec2skew(acc_) * dt
        df_dx[6:9, 12:15] = -s.rot * dt
        df_dx[6:9, 15:18] = I_dt

        return df_dx

    def df_dw(self, s: StateIkfom, in_: InputIkfom, dt: float):
        """计算过程噪声的雅可比矩阵"""
        # omega_ = in_.gyro - s.bg
        I = torch.eye(3, dtype=self.tran_dtype)
        cov = torch.zeros((18, 12), dtype=self.tran_dtype)
        # cov[0:3, 0:3] = -A_T(omega_ * dt)
        # -A(w dt)可以简化为-I
        cov[0:3, 0:3] = -I
        cov[6:9, 3:6] = -s.rot  # -R
        cov[9:12, 6:9] = I
        cov[12:15, 9:12] = I
        cov = cov * dt

        return cov

    def predict(self, i_in: InputIkfom, dt: float):
        """前向传播，在cpu上执行前向传播效率高得多"""
        f = self.f_model(self.x, i_in)
        df_dx = self.df_dx(self.x, i_in, dt)
        df_dw = self.df_dw(self.x, i_in, dt)

        self.x = boxplus(self.x, f * dt)
        self.P = df_dx @ self.P @ df_dx.T + df_dw @ self.Q @ df_dw.T

    def f_model(self, s: StateIkfom, in_: InputIkfom):
        """获取运动方程，用于描述状态如何随时间演变"""
        res = torch.zeros(18, dtype=self.tran_dtype)
        a_inertial = s.rot @ (in_.acc - s.ba) + s.grav
        res[:3] = in_.gyro - s.bg
        res[3:6] = s.vel
        res[6:9] = a_inertial
        return res

    def h_model(self, pc_imu: torch.tensor):
        bs = self.config.infer_bs
        mask_min_nn_count = self.config.track_mask_query_nn_k
        min_grad_norm = self.config.reg_min_grad_norm
        max_grad_norm = self.config.reg_max_grad_norm

        T = torch.eye(4)
        T[:3, :3] = self.x.rot
        T[:3, 3] = self.x.pos

        pc_map = transform_torch(pc_imu, T)
        sample_count = pc_map.shape[0]
        iter_n = math.ceil(sample_count / bs)

        sdf_pred = torch.zeros(sample_count, device=pc_map.device)
        sdf_std = torch.zeros(sample_count, device=pc_map.device)
        mc_mask = torch.zeros(sample_count, device=pc_map.device, dtype=torch.bool)
        sdf_grad = torch.zeros((sample_count, 3), device=pc_map.device)
        certainty = torch.zeros(sample_count, device=pc_map.device)

        # 分批处理，计算点云的SDF预测值
        for n in range(iter_n):
            head = n * bs
            tail = min((n + 1) * bs, sample_count)
            batch_coord = pc_map[head:tail, :]
            batch_coord.requires_grad_(True)

            (
                batch_geo_feature,
                _,
                weight_knn,
                nn_count,
                batch_certainty,
            ) = self.neural_points.query_feature(
                batch_coord,
                training_mode=False,
                query_locally=True,
                query_color_feature=False,
            )  # inference mode

            batch_sdf = self.geo_decoder.sdf(batch_geo_feature)
            if not self.config.weighted_first:
                batch_sdf_mean = torch.sum(batch_sdf * weight_knn, dim=1)
                batch_sdf_var = torch.sum(
                    (weight_knn * (batch_sdf - batch_sdf_mean.unsqueeze(-1)) ** 2),
                    dim=1,
                )
                batch_sdf_std = torch.sqrt(batch_sdf_var).squeeze(1)
                batch_sdf = batch_sdf_mean.squeeze(1)
                sdf_std[head:tail] = batch_sdf_std.detach()

            batch_sdf_grad = get_gradient(batch_coord, batch_sdf)
            sdf_grad[head:tail, :] = batch_sdf_grad.detach()
            sdf_pred[head:tail] = batch_sdf.detach()
            mc_mask[head:tail] = nn_count >= mask_min_nn_count
            certainty[head:tail] = batch_certainty.detach()

        # 剔除异常观测
        grad_norm = sdf_grad.norm(dim=-1, keepdim=True).squeeze()
        max_sdf_std = self.config.surface_sample_range_m * self.config.max_sdf_std_ratio
        valid_idx = (
            mc_mask
            & (grad_norm < max_grad_norm)
            & (grad_norm > min_grad_norm)
            & (sdf_std < max_sdf_std)
        )
        valid_points = pc_map[valid_idx]
        N = valid_points.shape[0]
        pc_imu = pc_imu[valid_idx]
        grad_norm = grad_norm[valid_idx]
        sdf_pred = sdf_pred[valid_idx]
        sdf_grad = sdf_grad[valid_idx]

        # 计算雅可比矩阵
        H = torch.zeros((N, 18), device=self.device, dtype=self.tran_dtype)
        pc_imu_hat = batch_vec2skew(pc_imu)
        rotation = self.x.rot.to(dtype=self.dtype).unsqueeze(0)
        A = torch.bmm(rotation.repeat(N, 1, 1), pc_imu_hat)
        H[:, 0:3] = -torch.bmm(sdf_grad.unsqueeze(1), A).squeeze(1)
        H[:, 3:6] = sdf_grad

        # 计算不确定度（对精度有一个轻微的提升）
        sdf_residual = sdf_pred.to(dtype=self.tran_dtype)
        grad_anomaly = (grad_norm - 1.0).to(dtype=self.tran_dtype)
        w_grad = 1 / (1 + grad_anomaly**2)
        w_res = 0.4 / (0.4 + sdf_residual**2)
        self.R_inv = w_grad * w_res * 1000

        return sdf_residual, H, valid_points

    def update_iterated(self, pc_imu: torch.tensor):
        """
        使用迭代方法更新状态估计。

        Args:
        source_points (np.array): 测量点云，假定为 Nx3 矩阵。
        maximum_iter (int): 最大迭代次数。
        """
        # 将状态量和协方差矩阵转移到GPU
        self.x.cuda()
        self.P = self.P.cuda()
        valid_flag = True
        converged = False

        x_propagated = self.x
        P_inv = torch.linalg.inv(self.P)
        I = torch.eye(18, device=self.device, dtype=self.tran_dtype)
        term_thre_deg = self.config.reg_term_thre_deg
        term_thre_m = self.config.reg_term_thre_m

        for i in range(self.max_iteration):
            dx_new = boxminus(self.x, x_propagated)
            z, H, valid_points = self.h_model(pc_imu)
            N = valid_points.shape[0]
            source_point_count = pc_imu.shape[0]

            if N / source_point_count < 0.2 and i == self.max_iteration - 1:
                if not self.config.silence:
                    print(
                        "[bold yellow](Warning) registration failed: not enough valid points[/bold yellow]"
                    )
                valid_flag = False

            H_T_R_inv = H.T * self.R_inv
            S = H_T_R_inv @ H

            K_front = torch.linalg.inv(S + P_inv)
            K = K_front @ H_T_R_inv

            dx_ = -K @ z + (K @ H - I) @ dx_new
            self.x = boxplus(self.x, dx_)
            tran_m = dx_[3:6].norm()
            rot_angle_deg = dx_[0:3].norm() * 180.0 / np.pi

            # 第一种迭代终止判定方式（有物理含义）
            if (
                rot_angle_deg < term_thre_deg
                and tran_m < term_thre_m
                and torch.all(torch.abs(dx_[6:]) < self.eps)
            ):
                if not self.config.silence:
                    print("Converged after", i, "iterations")
                converged = True

            # 第二种迭代终止判定方式
            # if torch.all(torch.abs(dx_) < self.eps):
            #     if not self.config.silence:
            #         print("Converged after", i, "iterations")
            #     converged = True

            if not valid_flag or converged:
                break

        self.P = (I - K @ H) @ self.P
        updated_pose = torch.eye(4, dtype=self.dtype, device=self.device)
        updated_pose[:3, :3] = self.x.rot.to(self.dtype)
        updated_pose[:3, 3] = self.x.pos.to(self.dtype)

        # 将状态量和协方差矩阵转移到CPU
        self.x.cpu()
        self.P = self.P.cpu()
        return updated_pose, valid_flag
