#!/usr/bin/env python3
# @file      esekfom.py
# @author    Junlong Jiang     [jiangjunlong@mail.dlut.edu.cn]
# Copyright (c) 2025 Junlong Jiang, all rights reserved

import torch

from utils.config import Config
from utils.tools import transform_torch


class DataSampler:
    def __init__(self, config: Config):
        self.config = config
        self.dev = config.device

    def sample(self, points_torch, neural_points, cur_pose_torch):
        dev = self.dev
        surface_sample_range = self.config.surface_sample_range_m
        surface_sample_n = self.config.surface_sample_n
        freespace_front_sample_n = self.config.free_front_n
        free_front_min_ratio = self.config.free_sample_begin_ratio

        # 1 as the exact measurement
        all_sample_n = surface_sample_n + freespace_front_sample_n + 1

        # get sample points
        point_num = points_torch.shape[0]
        distances = torch.linalg.norm(points_torch, dim=1, keepdim=True)  # ray distances (scaled)

        # Part 0. the exact measured point
        measured_sample_sdf = torch.zeros(point_num, device=points_torch.device)
        measured_sample_dist_ratio = torch.ones_like(distances)

        # Part 1. close-to-surface uniform sampling
        # uniform sample in the close-to-surface range (+- range)
        surface_sample_displacement = torch.randn(point_num * surface_sample_n, 1, device=dev) * surface_sample_range

        repeated_dist = distances.repeat(surface_sample_n, 1)
        surface_sample_dist_ratio = surface_sample_displacement / repeated_dist + 1.0  # 1.0 means on the surface

        # Part 2. free space (in front of surface) uniform sampling
        # 如果你想要更好地重建细小的物体（例如电线杆、树枝），则需要更多的自由空间样本以产生空间雕刻效果

        sigma_ratio = 2.0
        repeated_dist = distances.repeat(freespace_front_sample_n, 1)
        free_max_ratio = 1.0 - sigma_ratio * surface_sample_range / repeated_dist
        free_diff_ratio = free_max_ratio - free_front_min_ratio
        free_sample_front_dist_ratio = (
                torch.rand(point_num * freespace_front_sample_n, 1, device=dev)
                * free_diff_ratio
                + free_front_min_ratio
        )
        all_sample_dist_ratio = torch.cat(
            (
                measured_sample_dist_ratio,
                surface_sample_dist_ratio,
                free_sample_front_dist_ratio,
            ),
            0,
        )

        # 根据表面采样平移量计算符号
        sdf_sign = torch.where(surface_sample_displacement.squeeze(1) < 0, 1, -1)
        repeated_points = points_torch.repeat(all_sample_n, 1)
        all_sample_points = repeated_points * all_sample_dist_ratio
        mask = torch.ones(point_num * all_sample_n, dtype=torch.bool, device=self.config.device)

        # 表面采样点的全部坐标
        surface_sample_count = point_num * surface_sample_n
        sample_points = all_sample_points[point_num:]
        sample_points_global = transform_torch(sample_points, cur_pose_torch)
        dist, valid_mask = neural_points.region_specific_sdf_estimations(sample_points_global)
        mask[point_num:] = valid_mask
        surface_sample_sdf = sdf_sign * dist[:surface_sample_count]
        sdf_label = torch.cat((measured_sample_sdf, surface_sample_sdf, dist[surface_sample_count:]), 0)

        repeated_dist = distances.repeat(all_sample_n, 1)
        if self.config.dist_weight_on:  # far away surface samples would have lower weight
            weight_tensor = (
                    1
                    + self.config.dist_weight_scale * 0.5
                    - (repeated_dist / self.config.max_range)
                    * self.config.dist_weight_scale
            )  # [0.6, 1.4]
        else:

            depths_tensor = repeated_dist * all_sample_dist_ratio
            weight_tensor = torch.ones_like(depths_tensor)

        # TODO: also add lower weight for surface samples with large incidence angle
        weight_tensor = weight_tensor.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1)

        return all_sample_points[mask], sdf_label[mask], weight_tensor[mask]
