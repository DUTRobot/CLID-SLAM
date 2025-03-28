#!/usr/bin/env python3
# @file      data_sampler.py
# @author    Junlong Jiang     [jiangjunlong@mail.dlut.edu.cn]
# Copyright (c) 2025 Junlong Jiang, all rights reserved

import torch

from model.local_point_cloud_map import LocalPointCloudMap
from utils.config import Config
from utils.tools import transform_torch


class DataSampler:
    def __init__(self, config: Config):
        self.config = config
        self.dev = config.device

    def sample(self, points_torch, local_point_cloud_map: LocalPointCloudMap, cur_pose_torch):
        dev = self.dev
        surface_sample_range = self.config.surface_sample_range_m
        surface_sample_n = self.config.surface_sample_n
        freespace_behind_sample_n = self.config.free_behind_n
        freespace_front_sample_n = self.config.free_front_n
        all_sample_n = (
            surface_sample_n + freespace_behind_sample_n + freespace_front_sample_n + 1
        )  # 1 as the exact measurement
        free_front_min_ratio = self.config.free_sample_begin_ratio
        free_sample_end_dist = self.config.free_sample_end_dist_m

        # get sample points
        point_num = points_torch.shape[0]
        distances = torch.linalg.norm(
            points_torch, dim=1, keepdim=True
        )  # ray distances (scaled)

        # Part 0. the exact measured point
        measured_sample_displacement = torch.zeros_like(distances)
        measured_sample_dist_ratio = torch.ones_like(distances)

        # Part 1. close-to-surface uniform sampling
        # uniform sample in the close-to-surface range (+- range)
        surface_sample_displacement = (
            torch.randn(point_num * surface_sample_n, 1, device=dev)
            * surface_sample_range
        )

        repeated_dist = distances.repeat(surface_sample_n, 1)
        surface_sample_dist_ratio = (
            surface_sample_displacement / repeated_dist + 1.0
        )  # 1.0 means on the surface


        # Part 2. free space (in front of surface) uniform sampling
        # if you want to reconstruct the thin objects (like poles, tree branches) well, you need more freespace samples to have
        # a space carving effect

        sigma_ratio = 2.0
        repeated_dist = distances.repeat(freespace_front_sample_n, 1)
        free_max_ratio = 1.0 - sigma_ratio * surface_sample_range / repeated_dist
        free_diff_ratio = free_max_ratio - free_front_min_ratio
        free_sample_front_dist_ratio = (
            torch.rand(point_num * freespace_front_sample_n, 1, device=dev)
            * free_diff_ratio
            + free_front_min_ratio
        )
        free_sample_front_displacement = (
            free_sample_front_dist_ratio - 1.0
        ) * repeated_dist

        # Part 3. free space (behind surface) uniform sampling
        repeated_dist = distances.repeat(freespace_behind_sample_n, 1)
        free_max_ratio = free_sample_end_dist / repeated_dist + 1.0
        free_behind_min_ratio = 1.0 + sigma_ratio * surface_sample_range / repeated_dist
        free_diff_ratio = free_max_ratio - free_behind_min_ratio

        free_sample_behind_dist_ratio = (
            torch.rand(point_num * freespace_behind_sample_n, 1, device=dev)
            * free_diff_ratio
            + free_behind_min_ratio
        )

        free_sample_behind_displacement = (
            free_sample_behind_dist_ratio - 1.0
        ) * repeated_dist


        # all together
        all_sample_displacement = torch.cat(
            (
                measured_sample_displacement,
                surface_sample_displacement,
                free_sample_front_displacement,
                free_sample_behind_displacement,
            ),
            0,
        )
        all_sample_dist_ratio = torch.cat(
            (
                measured_sample_dist_ratio,
                surface_sample_dist_ratio,
                free_sample_front_dist_ratio,
                free_sample_behind_dist_ratio,
            ),
            0,
        )

        repeated_points = points_torch.repeat(all_sample_n, 1)
        repeated_dist = distances.repeat(all_sample_n, 1)
        all_sample_points = repeated_points * all_sample_dist_ratio
        ####################################### Added By Jiang Junlong #################################################
        # 根据表面采样平移量计算符号
        sdf_sign = torch.where(surface_sample_displacement.squeeze(1) < 0, 1, -1)
        mask = torch.ones(point_num * all_sample_n, dtype=torch.bool, device=self.config.device)

        # 表面采样点的全部坐标
        surface_sample_count = point_num * surface_sample_n
        surface_sample_points = all_sample_points[point_num: point_num + surface_sample_count]
        surface_sample_points_G = transform_torch(surface_sample_points, cur_pose_torch)
        dist, valid_mask = local_point_cloud_map.region_specific_sdf_estimation(surface_sample_points_G)
        mask[point_num:point_num + surface_sample_count] = valid_mask
        surface_sample_sdf = sdf_sign * dist[:surface_sample_count]
        sdf_label_tensor = all_sample_displacement.squeeze(1)
        sdf_label_tensor *= -1  # convert to the same sign as
        # print("surface_sample_sdf", surface_sample_sdf.shape)
        # print("sdf_label_tensor", sdf_label_tensor.shape)

        sdf_label = torch.cat((sdf_label_tensor[:point_num], surface_sample_sdf, sdf_label_tensor[point_num + surface_sample_count:]), 0)
        # sdf_label = torch.clamp(sdf_label, -0.4, 0.4)

        # depth tensor of all the samples
        depths_tensor = repeated_dist * all_sample_dist_ratio
        # get the weight vector as the inverse of sigma
        weight_tensor = torch.ones_like(depths_tensor)
        if self.config.dist_weight_on:  # far away surface samples would have lower weight
            weight_tensor[:point_num + surface_sample_count] = (
                1
                + self.config.dist_weight_scale * 0.5
                - (repeated_dist[:point_num + surface_sample_count] / self.config.max_range)
                * self.config.dist_weight_scale
            )  # [0.6, 1.4]

        weight_tensor[point_num + surface_sample_count:] *= -1.0

        # Convert from the all ray surface + all ray free order to the ray-wise (surface + free) order

        all_sample_points = (
            all_sample_points.reshape(all_sample_n, -1, 3)
            .transpose(0, 1)
            .reshape(-1, 3)
        )
        sdf_label = (
            sdf_label.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1)
        )

        weight_tensor = (
            weight_tensor.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1)
        )

        mask = (
            mask.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1)
        )
        return all_sample_points[mask], sdf_label[mask], weight_tensor[mask]
