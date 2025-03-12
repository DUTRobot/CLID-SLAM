#!/usr/bin/env python3
# @file      neural_points.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved
# Modifications by:
# Junlong Jiang [jiangjunlong@mail.dlut.edu.cn]
# Copyright (c) 2025 Junlong Jiang, all rights reserved.

import math
import sys

import matplotlib.cm as cm
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print

from utils.config import Config
from utils.tools import (
    apply_quaternion_rotation,
    quat_multiply,
    rotmat_to_quat,
    transform_batch_torch,
    voxel_down_sample_min_value_torch,
    voxel_down_sample_torch,
    estimate_plane
)


# 定义神经点云的类
class NeuralPoints(nn.Module):
    def __init__(self, config: Config) -> None:

        super().__init__()

        self.config = config
        self.silence = config.silence

        # self.point_level_num = config.point_level_num

        self.geo_feature_dim = config.feature_dim  # 几何特征的维度
        self.geo_feature_std = config.feature_std  # 几何特征的标准差

        self.color_feature_dim = config.feature_dim  # 颜色特征维度
        self.color_feature_std = config.feature_std  # 颜色特征的标准差

        # 是否使用高斯位置编码
        if config.use_gaussian_pe:
            self.position_encoder_geo = GaussianFourierFeatures(config)
            self.position_encoder_color = GaussianFourierFeatures(config)
        else:
            self.position_encoder_geo = PositionalEncoder(config)
            self.position_encoder_color = PositionalEncoder(config)

        self.mean_grid_sampling = False # NOTE: sample the gravity center of the points inside the voxel or keep the point that is closest to the voxel center

        self.device = config.device
        self.dtype = config.dtype
        self.idx_dtype = torch.int64  # torch.int64/32 does not have much speed difference

        self.resolution = config.voxel_size_m

        self.buffer_size = config.buffer_size  # 缓冲区大小

        self.temporal_local_map_on = True
        self.local_map_radius = self.config.local_map_radius
        self.diff_travel_dist_local = (self.config.local_map_radius * self.config.local_map_travel_dist_ratio)

        self.diff_ts_local = self.config.diff_ts_local  # not used now, switch to travel distance

        self.local_orientation = torch.eye(3, device=self.device)

        self.cur_ts = 0  # current frame No. or the current timestamp
        self.max_ts = 0

        self.travel_dist = None  # for determine the local map, update from the dataset class for each frame
        self.est_poses = None
        self.after_pgo = False

        # for hashing (large prime numbers)
        self.primes = torch.tensor([73856093, 19349669, 83492791], dtype=self.idx_dtype, device=self.device)

        # 初始化
        # 全局地图
        self.buffer_pt_index = torch.full((self.buffer_size,), -1, dtype=self.idx_dtype, device=self.device)

        self.neural_points = torch.empty((0, 3), dtype=self.dtype, device=self.device)
        self.point_orientations = torch.empty((0, 4), dtype=self.dtype, device=self.device)  # as quaternion
        self.geo_features = torch.empty((1, self.geo_feature_dim), dtype=self.dtype, device=self.device)
        if self.config.color_on:
            self.color_features = torch.empty((1, self.color_feature_dim), dtype=self.dtype, device=self.device)
        else:
            self.color_features = None
        # here, the ts represent the actually processed frame id (not neccessarily the frame id of the dataset)
        self.point_ts_create = torch.empty((0,), device=self.device, dtype=torch.int)  # 创建的时间戳
        self.point_ts_update = torch.empty((0,), device=self.device, dtype=torch.int)  # 最近更新的时间戳
        self.point_certainties = torch.empty((0,), dtype=self.dtype, device=self.device)

        # 局部地图
        # 局部地图-神经点的属性：位置、旋转（四元数）、几何特征、颜色特征、不确定度、更新时间
        self.local_neural_points = torch.empty((0, 3), dtype=self.dtype, device=self.device)
        self.local_point_orientations = torch.empty((0, 4), dtype=self.dtype, device=self.device)  # as quaternion
        self.local_geo_features = nn.Parameter()
        self.local_color_features = nn.Parameter()
        self.local_point_certainties = torch.empty((0,), dtype=self.dtype, device=self.device)
        self.local_point_ts_update = torch.empty((0,), device=self.device, dtype=torch.int)
        self.local_mask = None
        self.global2local = None

        # 设置近邻搜索的区域
        self.set_search_neighborhood(num_nei_cells=config.num_nei_cells, search_alpha=config.search_alpha)

        self.memory_footprint = []

        self.to(self.device)

    # 检查神经点云是否为空
    def is_empty(self):
        return self.neural_points.shape[0] == 0

    # 返回全部地图的神经点云的数量
    def count(self):
        return self.neural_points.shape[0]

    # 返回局部地图的神经点云的数量
    def local_count(self):
        return self.local_neural_points.shape[0]

    # 打印内存使用的情况
    def print_memory(self):
        if not self.silence:
            print("# Global neural point: %d" % (self.count()))
            print("# Local  neural point: %d" % (self.local_count()))
        neural_point_count = self.count()
        point_dim = self.config.feature_dim + 3 + 4  # feature plus neural point position and orientation
        if self.color_features is not None:
            point_dim += self.config.feature_dim  # also include the color feature
        cur_memory = neural_point_count * point_dim * 4 / 1024 / 1024  # as float32
        if not self.silence:
            print("Memory consumption: %f (MB)" % cur_memory)
        self.memory_footprint.append(cur_memory)

    # 神经点的可视化
    def get_neural_points_o3d(
            self,
            query_global: bool = True,
            color_mode: int = -1,
            random_down_ratio: int = 1,
    ):

        ratio_vis = 1.5
        # TODO: visualize orientation as normal

        if query_global:
            neural_points_np = self.neural_points[::random_down_ratio].cpu().detach().numpy().astype(np.float64)
        # points_orientation_np = self.point_orientations[::random_down_ratio].cpu().detach().numpy().astype(np.float64)
        else:
            neural_points_np = self.local_neural_points[::random_down_ratio].cpu().detach().numpy().astype(np.float64)

        neural_pc_o3d = o3d.geometry.PointCloud()
        neural_pc_o3d.points = o3d.utility.Vector3dVector(neural_points_np)

        if color_mode == 0:  # "geo_feature"
            if query_global:
                neural_features_vis = self.geo_features[:-1:random_down_ratio].detach()
            else:
                neural_features_vis = self.local_geo_features[:-1:random_down_ratio].detach()
            neural_features_vis = F.normalize(neural_features_vis, p=2, dim=1)
            neural_features_np = neural_features_vis.cpu().numpy().astype(np.float64)
            neural_pc_o3d.colors = o3d.utility.Vector3dVector(neural_features_np[:, 0:3] * ratio_vis)

            # print("run TSNE")
            # print(np.shape(neural_features_np)[0])
            # tsne_result = tsne.fit_transform(neural_features_np)
            # print("TSNE done")
            # neural_pc_o3d.colors = o3d.utility.Vector3dVector(tsne_result)

        elif color_mode == 1:  # "color_feature"
            if self.color_features is None:
                return neural_pc_o3d
            if query_global:
                neural_features_vis = self.color_features[
                                      :-1:random_down_ratio
                                      ].detach()
            else:
                neural_features_vis = self.local_color_features[
                                      :-1:random_down_ratio
                                      ].detach()
            neural_features_vis = F.normalize(neural_features_vis, p=2, dim=1)
            neural_features_np = neural_features_vis.cpu().numpy().astype(np.float64)
            neural_pc_o3d.colors = o3d.utility.Vector3dVector(
                neural_features_np[:, 0:3] * ratio_vis
            )

        elif color_mode == 2:  # "ts": # frame number (ts) as the color
            if query_global:
                if self.config.use_mid_ts:
                    show_ts = ((self.point_ts_create + self.point_ts_update) / 2).int()
                else:
                    show_ts = self.point_ts_create
                ts_np = (
                    show_ts[::random_down_ratio]
                    .cpu()
                    .detach()
                    .numpy()
                    .astype(np.float64)
                )
            else:
                ts_np = (
                    self.local_point_ts_update[::random_down_ratio]
                    .cpu()
                    .detach()
                    .numpy()
                    .astype(np.float64)
                )
            ts_np = np.clip(ts_np / self.max_ts, 0.0, 1.0)
            color_map = cm.get_cmap("jet")
            ts_color = color_map(ts_np)[:, :3].astype(np.float64)
            neural_pc_o3d.colors = o3d.utility.Vector3dVector(ts_color)

        elif color_mode == 3:  # "certainty" # certainty as color
            if query_global:
                certainty_np = (
                        1.0
                        - self.point_certainties[::random_down_ratio]
                        .cpu()
                        .detach()
                        .numpy()
                        .astype(np.float64)
                        / 1000.0
                )
            else:
                certainty_np = (
                        1.0
                        - self.local_point_certainties[::random_down_ratio]
                        .cpu()
                        .detach()
                        .numpy()
                        .astype(np.float64)
                        / 1000.0
                )
            # print(self.local_point_certainties)
            certainty_color = np.repeat(certainty_np.reshape(-1, 1), 3, axis=1)
            neural_pc_o3d.colors = o3d.utility.Vector3dVector(certainty_color)

        elif color_mode == 4:  # "random" # random color
            random_color = np.random.rand(neural_points_np.shape[0], 3).astype(
                np.float64
            )
            neural_pc_o3d.colors = o3d.utility.Vector3dVector(random_color)

        return neural_pc_o3d

    def update(
            self,
            points: torch.Tensor,
            sensor_position: torch.Tensor,
            sensor_orientation: torch.Tensor,
            cur_ts,
    ):
        """
        使用新观测的点云更新神经点地图

        参数:
        - points: torch.Tensor，点云张量，形状为[N, 3]，N是点的数量。
        - sensor_position: torch.Tensor，传感器的当前位置，形状为[3]。
        - sensor_orientation: torch.Tensor，传感器的当前姿态，形状为[4]，表示为四元数。
        - cur_ts: 当前时间戳或帧编号，用于跟踪观测点的时序信息。
        """
        # update the neural point map using new observations

        cur_resolution = self.resolution
        # if self.mean_grid_sampling:
        #     sample_points = meanGridSampling(points, resolution=cur_resolution)
        # take the point that is the closest to the voxel center (now used)
        sample_idx = voxel_down_sample_torch(points, cur_resolution)
        sample_points = points[sample_idx]

        grid_coords = (sample_points / cur_resolution).floor().to(self.primes)
        buffer_size = int(self.buffer_size)
        hash = torch.fmod((grid_coords * self.primes).sum(-1), buffer_size)

        hash_idx = self.buffer_pt_index[hash]

        # not occupied before or is occupied but already far away (then it would be a hash collision)
        if not self.is_empty():
            vec_points = self.neural_points[hash_idx] - sample_points
            dist2 = torch.sum(vec_points**2, dim=-1)

            # the voxel is not occupied before or the case when hash collision happens
            # delta_t = (cur_ts - self.point_ts_create[hash_idx]) # use time diff
            # use travel dist diff
            delta_travel_dist = self.travel_dist[cur_ts] - self.travel_dist[self.point_ts_update[hash_idx]]

            # the last time mask is necessary (but better change to the accumulated distance or the pose uncertainty), done
            # 1. 哈希表该索引未占用
            # 2. 距离大于三个根号3 * cur_resolution
            # 3. 点云数据超过一定时间未更新，可以对旧的数据进行覆盖更新
            update_mask = ((hash_idx == -1) | (dist2 > 3 * cur_resolution ** 2)
                           | (delta_travel_dist > self.diff_travel_dist_local))
        else:
            update_mask = torch.ones(hash_idx.shape, dtype=torch.bool, device=self.device)

        # 满足条件的新插入点
        added_pt = sample_points[update_mask]
        # 新插入点的数量
        new_point_count = added_pt.shape[0]

        cur_pt_idx = self.buffer_pt_index[hash]
        # 分配新的神经点
        cur_pt_count = self.neural_points.shape[0]
        cur_pt_idx[update_mask] = torch.arange(new_point_count, dtype=self.idx_dtype, device=self.device) + cur_pt_count

        # torch.cat could be slow for large map
        self.buffer_pt_index[hash] = cur_pt_idx
        # 这里添加神经点
        self.neural_points = torch.cat((self.neural_points, added_pt), 0)

        # 初始化插入新点的旋转
        added_orientations = [[1, 0, 0, 0]] * new_point_count
        added_orientations = torch.tensor(added_orientations, dtype=self.dtype, device=self.device)
        self.point_orientations = torch.cat((self.point_orientations, added_orientations), 0)

        # 初始化插入新点的时间戳
        new_points_ts = (torch.ones(new_point_count, device=self.device, dtype=torch.int) * cur_ts)
        self.point_ts_create = torch.cat((self.point_ts_create, new_points_ts), 0)
        self.point_ts_update = torch.cat((self.point_ts_update, new_points_ts), 0)

        # with padding in the end
        new_fts = self.geo_feature_std * torch.randn(
            new_point_count + 1,
            self.geo_feature_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.geo_features = torch.cat((self.geo_features[:-1], new_fts), 0)

        # with padding in the end
        if self.color_features is not None:
            new_fts = self.color_feature_std * torch.randn(
                new_point_count + 1,
                self.color_feature_dim,
                device=self.device,
                dtype=self.dtype,
            )
            self.color_features = torch.cat((self.color_features[:-1], new_fts), 0)

        # 初始化插入新点的不确定度
        new_certainty = torch.zeros(new_point_count, device=self.device, dtype=self.dtype, requires_grad=False)
        self.point_certainties = torch.cat((self.point_certainties, new_certainty), 0)

        self.reset_local_map(sensor_position, sensor_orientation, cur_ts)  # no need to recreate hash

    def delete_points(self, points: torch.Tensor, point_cloud, frame_id):
        # T0 = get_time()
        cur_resolution = self.resolution
        sample_idx = voxel_down_sample_torch(points, cur_resolution)
        sample_points = points[sample_idx]

        grid_coords = (sample_points / cur_resolution).floor().to(self.primes)
        buffer_size = int(self.buffer_size)
        hash = torch.fmod((grid_coords * self.primes).sum(-1), buffer_size)
        delete_indices = self.buffer_pt_index[hash]
        delete_pts = self.neural_points[delete_indices]
        neighb_pts_sub = delete_pts - sample_points  # [N,K,3]
        dist = torch.sqrt(torch.sum(neighb_pts_sub ** 2, dim=-1))
        delete_indices = delete_indices[dist < 0.1]
        grid_coords_not = (point_cloud / cur_resolution).floor().to(self.primes)
        hash_reserve = torch.fmod((grid_coords_not * self.primes).sum(-1), buffer_size)
        retain_indices = self.buffer_pt_index[hash_reserve]

        delete_mask = torch.zeros(self.neural_points.size(0), dtype=torch.bool)
        delete_mask[delete_indices] = True
        delete_mask[retain_indices] = False
        self.neural_points = self.neural_points[~delete_mask]
        self.point_orientations = self.point_orientations[~delete_mask]
        self.point_ts_create = self.point_ts_create[~delete_mask]
        self.point_ts_update = self.point_ts_update[~delete_mask]
        self.point_certainties = self.point_certainties[~delete_mask]

        # with padding
        delete_mask = torch.cat((delete_mask, torch.tensor([False]).to(delete_mask)), dim=0)
        self.geo_features = self.geo_features[~delete_mask]
        if delete_mask.sum().item() > 0:
            self.recreate_hash(None, None, False, False)
        # T1 = get_time()
        # print(T1 - T0)

    def reset_local_map(
            self,
            sensor_position: torch.Tensor,
            sensor_orientation: torch.Tensor,
            cur_ts: int,
            use_travel_dist: bool = True,
            diff_ts_local: int = 50,
    ):
        # TODO: not very efficient, optimize the code

        self.cur_ts = cur_ts
        self.max_ts = max(self.max_ts, cur_ts)

        if self.config.use_mid_ts:
            point_ts_used = ((self.point_ts_create + self.point_ts_update) / 2).int()
        else:
            point_ts_used = self.point_ts_create

        if use_travel_dist:  # self.travel_dist as torch tensor
            delta_travel_dist = torch.abs(self.travel_dist[cur_ts] - self.travel_dist[point_ts_used])
            time_mask = (delta_travel_dist < self.diff_travel_dist_local)
        else:  # use delta_t
            delta_t = torch.abs(cur_ts - point_ts_used)
            time_mask = (delta_t < diff_ts_local)

        # speed up by calulating distance only with the t filtered points
        masked_vec2sensor = self.neural_points[time_mask] - sensor_position
        masked_dist2sensor = torch.sum(masked_vec2sensor ** 2, dim=-1)  # dist square

        dist_mask = (masked_dist2sensor < self.local_map_radius ** 2)
        time_mask_idx = torch.nonzero(time_mask).squeeze()  # True index
        local_mask_idx = time_mask_idx[dist_mask]  # True index

        local_mask = torch.full((time_mask.shape), False, dtype=torch.bool, device=self.device)

        local_mask[local_mask_idx] = True

        self.local_neural_points = self.neural_points[local_mask]
        self.local_point_orientations = self.point_orientations[local_mask]
        self.local_point_certainties = self.point_certainties[local_mask]
        self.local_point_ts_update = self.point_ts_update[local_mask]

        # padding with one element in the end
        local_mask = torch.cat((local_mask, torch.tensor([True], device=self.device)))
        self.local_mask = local_mask

        # if Flase (not in the local map), the mapping get an idx as -1
        global2local = torch.full_like(local_mask, -1).long()

        local_indices = torch.nonzero(local_mask).flatten()
        local_point_count = local_indices.size(0)
        global2local[local_indices] = torch.arange(local_point_count, device=self.device)
        global2local[-1] = -1  # invalid idx is still invalid after mapping

        self.global2local = global2local

        self.local_geo_features = nn.Parameter(self.geo_features[local_mask])
        if self.color_features is not None:
            self.local_color_features = nn.Parameter(self.color_features[local_mask])

        self.local_orientation = sensor_orientation  # not used

    def assign_local_to_global(self):

        local_mask = self.local_mask
        # self.neural_points[local_mask[:-1]] = self.local_neural_points
        # self.point_orientations[local_mask[:-1]] = self.local_point_orientations
        self.geo_features[local_mask] = self.local_geo_features.data
        if self.color_features is not None:
            self.color_features[local_mask] = self.local_color_features.data
        self.point_certainties[local_mask[:-1]] = self.local_point_certainties
        self.point_ts_update[local_mask[:-1]] = self.local_point_ts_update

        # print("mean certainty for the neural points:", torch.mean(self.point_certainties))

    def query_feature(
            self,
            query_points: torch.Tensor,
            query_ts: torch.Tensor = None,
            training_mode: bool = True,
            query_locally: bool = True,
            query_geo_feature: bool = True,
            query_color_feature: bool = False,
    ):

        if not query_geo_feature and not query_color_feature:
            sys.exit("you need to at least query one kind of feature")

        batch_size = query_points.shape[0]

        geo_features_vector = None
        color_features_vector = None

        nn_k = self.config.query_nn_k

        # T0 = get_time()

        # the slow part
        dists2, idx = self.radius_neighborhood_search(
            query_points, time_filtering=self.temporal_local_map_on and query_locally
        )

        # [N, K], [N, K]
        # if query globally, we do not have the time filtering

        # T10 = get_time()

        # print("K=", idx.shape[-1]) # K
        if query_locally:
            idx = self.global2local[idx]  # [N, K] # get the local idx using the global2local mapping

        nn_counts = (idx >= 0).sum(dim=-1)  # then it could be larger than nn_k because this is before the sorting

        # T1 = get_time()

        dists2[idx == -1] = 9e3  # invalid, set to large distance
        sorted_dist2, sorted_neigh_idx = torch.sort(dists2, dim=1)  # sort according to distance
        sorted_idx = idx.gather(1, sorted_neigh_idx)
        dists2 = sorted_dist2[:, :nn_k]  # only take the knn
        idx = sorted_idx[:, :nn_k]  # sorted local idx, only take the knn

        # dist2, idx are all with the shape [N, K]

        # T2 = get_time()

        valid_mask = idx >= 0  # [N, K]

        if query_geo_feature:
            geo_features = torch.zeros(
                batch_size,
                nn_k,
                self.geo_feature_dim,
                device=self.device,
                dtype=self.dtype,
            )  # [N, K, F]
            if query_locally:
                geo_features[valid_mask] = self.local_geo_features[idx[valid_mask]]
            else:
                geo_features[valid_mask] = self.geo_features[idx[valid_mask]]
            if self.config.layer_norm_on:
                geo_features = F.layer_norm(geo_features, [self.geo_feature_dim])
        if query_color_feature and self.color_features is not None:
            # [N, K, F]
            color_features = torch.zeros(batch_size, nn_k, self.color_feature_dim, device=self.device, dtype=self.dtype)
            if query_locally:
                color_features[valid_mask] = self.local_color_features[idx[valid_mask]]
            else:
                color_features[valid_mask] = self.color_features[idx[valid_mask]]
            if self.config.layer_norm_on:
                color_features = F.layer_norm(color_features, [self.color_feature_dim])

        N, K = valid_mask.shape  # K = nn_k here

        if query_locally:
            certainty = self.local_point_certainties[idx]  # [N, K]
            neighb_vector = (query_points.view(-1, 1, 3) - self.local_neural_points[idx])  # [N, K, 3]
            quat = self.local_point_orientations[idx]  # [N, K, 4]
        else:
            certainty = self.point_certainties[idx]  # [N, K]
            neighb_vector = (query_points.view(-1, 1, 3) - self.neural_points[idx])  # [N, K, 3]
            quat = self.point_orientations[idx]  # [N, K, 4]

        # quat[...,1:] *= -1. # inverse (not needed)
        # This has been doubly checked
        if self.after_pgo:
            # [N, K, 3] # passive rotation (axis rotation w.r.t point)
            neighb_vector = apply_quaternion_rotation(quat, neighb_vector)
        neighb_vector[~valid_mask] = torch.zeros(1, 3, device=self.device, dtype=self.dtype)

        if self.config.pos_encoding_band > 0:
            neighb_vector = self.position_encoder_geo(neighb_vector)  # [N, K, P]

        if query_geo_feature:
            geo_features_vector = torch.cat((geo_features, neighb_vector), dim=2)  # [N, K, F+P]
        if query_color_feature and self.color_features is not None:
            color_features_vector = torch.cat((color_features, neighb_vector), dim=2)  # [N, K, F+P]

        eps = 1e-15  # avoid nan (dividing by 0)

        weight_vector = 1.0 / (dists2 + eps)  # [N, K] # Inverse distance weighting (IDW), distance square

        weight_vector[~valid_mask] = 0.0  # pad for invalid voxels
        weight_vector[nn_counts == 0] = eps  # all 0 would cause NaN during normalization

        # apply the normalization of weight
        weight_row_sums = torch.sum(weight_vector, dim=1).unsqueeze(1)
        weight_vector = torch.div(weight_vector, weight_row_sums)  # [N, K] # normalize the weight, to make the sum as 1

        # print(weight_vector)
        weight_vector[~valid_mask] = 0.0  # invalid has zero weight

        with torch.no_grad():
            # Certainty accumulation for each neural point according to the weight
            # Use scatter_add_ to accumulate the values for each index
            if training_mode:  # only do it during the training mode
                idx[~valid_mask] = 0  # scatter_add don't accept -1 index
                if query_locally:
                    self.local_point_certainties.scatter_add_(
                        dim=0, index=idx.flatten(), src=weight_vector.flatten()
                    )
                    if query_ts is not None:  # update the last update ts for each neural point
                        idx_ts = query_ts.view(-1, 1).repeat(1, K)
                        idx_ts[~valid_mask] = 0
                        self.local_point_ts_update.scatter_reduce_(
                            dim=0,
                            index=idx.flatten(),
                            src=idx_ts.flatten(),
                            reduce="amax",
                            include_self=True,
                        )
                        # print(self.local_point_ts_update)
                else:
                    self.point_certainties.scatter_add_(dim=0, index=idx.flatten(), src=weight_vector.flatten())
                # queried_certainty = None

                certainty[~valid_mask] = 0.0
                queried_certainty = torch.sum(certainty * weight_vector, dim=1)

            else:  # inference mode
                certainty[~valid_mask] = 0.0
                queried_certainty = torch.sum(certainty * weight_vector, dim=1)

        weight_vector = weight_vector.unsqueeze(-1)  # [N, K, 1]

        if self.config.weighted_first:
            if query_geo_feature:
                geo_features_vector = torch.sum(geo_features_vector * weight_vector, dim=1)  # [N, F+P]

            if query_color_feature and self.color_features is not None:
                color_features_vector = torch.sum(color_features_vector * weight_vector, dim=1)  # [N, F+P]

        # T3 = get_time()

        # in msbuffe
        # print("time for nn     :", (T1-T0) * 1e3) # ////
        # print("time for sorting:", (T2-T1) * 1e3) # //
        # print("time for feature:", (T3-T2) * 1e3) # ///

        return geo_features_vector, color_features_vector, weight_vector, nn_counts, queried_certainty

    # prune inactive uncertain neural points
    # 修剪不活跃和不确定性比较大的神经点
    def prune_map(self, prune_certainty_thre, min_prune_count=500):
        diff_travel_dist = torch.abs(self.travel_dist[self.cur_ts] - self.travel_dist[self.point_ts_update])
        inactive_mask = diff_travel_dist > self.diff_travel_dist_local
        prune_mask = inactive_mask & (self.point_certainties < prune_certainty_thre)  # True for prune
        # prune_mask = self.point_certainties < prune_certainty_thre # True for prune

        prune_count = torch.sum(prune_mask).item()
        if prune_count > min_prune_count:
            if not self.silence:
                print("# Prune neural points: ", prune_count)

            self.neural_points = self.neural_points[~prune_mask]
            self.point_orientations = self.point_orientations[~prune_mask]
            self.point_ts_create = self.point_ts_create[~prune_mask]
            self.point_ts_update = self.point_ts_update[~prune_mask]
            self.point_certainties = self.point_certainties[~prune_mask]

            # with padding
            prune_mask = torch.cat((prune_mask, torch.tensor([False]).to(prune_mask)), dim=0)
            self.geo_features = self.geo_features[~prune_mask]
            if self.config.color_on:
                self.color_features = self.color_features[~prune_mask]
            # recreate hash and local map then
            return True
        return False

    def adjust_map(self, pose_diff_torch):
        # for each neural point, use its ts to find the diff between old and new pose, transform the position and rotate the orientation
        # we use the mid_ts for each neural point

        self.after_pgo = True

        if self.config.use_mid_ts:  # 使用中值时间戳
            used_ts = ((self.point_ts_create + self.point_ts_update) / 2).int()
        else:  # 使用创建时间戳
            used_ts = self.point_ts_create

        self.neural_points = transform_batch_torch(self.neural_points, pose_diff_torch[used_ts])

        diff_quat_torch = rotmat_to_quat(pose_diff_torch[:, :3, :3])  # rotation part

        self.point_orientations = (quat_multiply(diff_quat_torch[used_ts], self.point_orientations)
                                   .to(self.point_orientations))

    def recreate_hash(
            self,
            sensor_position: torch.Tensor,
            sensor_orientation: torch.Tensor,
            kept_points: bool = True,
            with_ts: bool = True,
            cur_ts=0,
    ):

        cur_resolution = self.resolution

        self.buffer_pt_index = torch.full((self.buffer_size,), -1, dtype=self.idx_dtype, device=self.device)  # reset

        # take the point that is closer to the current timestamp (now used)
        # also update the timestep of neural points during merging
        if with_ts:
            if self.config.use_mid_ts:
                ts_used = ((self.point_ts_create + self.point_ts_update) / 2).int()
            else:
                ts_used = self.point_ts_create
            ts_diff = torch.abs(ts_used - cur_ts).float()
            sample_idx = voxel_down_sample_min_value_torch(
                self.neural_points, cur_resolution, ts_diff
            )
        else:
            # take the point that has a larger certainity
            sample_idx = voxel_down_sample_min_value_torch(
                self.neural_points,
                cur_resolution,
                self.point_certainties.max() - self.point_certainties,
            )

        if kept_points:
            # don't filter the neural points (keep them, only merge when neccessary, figure out the better merging method later)
            sample_points = self.neural_points[sample_idx]
            grid_coords = (sample_points / cur_resolution).floor().to(self.primes)
            hash = torch.fmod(
                (grid_coords * self.primes).sum(-1), int(self.buffer_size)
            )
            self.buffer_pt_index[hash] = sample_idx

        else:
            if not self.silence:
                print("Filter duplicated neural points")

            # only kept those filtered
            self.neural_points = self.neural_points[sample_idx]
            self.point_orientations = self.point_orientations[
                sample_idx
            ]  # as quaternion
            self.point_ts_create = self.point_ts_create[sample_idx]
            self.point_ts_update = self.point_ts_update[sample_idx]
            self.point_certainties = self.point_certainties[sample_idx]

            sample_idx_pad = torch.cat((sample_idx, torch.tensor([-1]).to(sample_idx)))
            self.geo_features = self.geo_features[
                sample_idx_pad
            ]  # with padding in the end
            if self.color_features is not None:
                self.color_features = self.color_features[
                    sample_idx_pad
                ]  # with padding in the end

            new_point_count = self.neural_points.shape[0]

            grid_coords = (self.neural_points / cur_resolution).floor().to(self.primes)
            hash = torch.fmod(
                (grid_coords * self.primes).sum(-1), int(self.buffer_size)
            )
            self.buffer_pt_index[hash] = torch.arange(
                new_point_count, dtype=self.idx_dtype, device=self.device
            )

        if sensor_position is not None:
            self.reset_local_map(sensor_position, sensor_orientation, cur_ts)

        if not kept_points:  # merged
            self.print_memory()  # show the updated memory after merging

    def set_search_neighborhood(self, num_nei_cells: int = 1, search_alpha: float = 1.0):
        """
        设置搜索邻域的大小和形状，用于确定点云中点的近邻

        参数:
            num_nei_cells (int): 定义搜索邻域立方体边长的一半，以体素为单位
            search_alpha (float): 用于扩展或收缩邻域球体的半径，影响最终包含的邻居数量

        功能描述:
            - 通过生成一个立方体的网格，然后筛选出位于指定球形邻域内的点来确定搜索邻域
            - 球形邻域由 num_nei_cells 和 search_alpha 共同定义
            - 该方法用于优化近邻搜索，通过减少计算距离的点的数量来加快训练和查询速度

        说明:
            - 该方法是初始化搜索结构的关键步骤，它直接影响邻近搜索的效率和精确度。
            - 参数 num_nei_cells 和 search_alpha 可以根据具体应用调整以平衡搜索精度和性能。
        """
        dx = torch.arange(
            -num_nei_cells,
            num_nei_cells + 1,
            device=self.primes.device,
            dtype=self.primes.dtype,
        )

        coords = torch.meshgrid(dx, dx, dx, indexing="ij")
        dx = torch.stack(coords, dim=-1).reshape(-1, 3)  # [K,3]

        dx2 = torch.sum(dx**2, dim=-1)
        self.neighbor_dx = dx[dx2 < (num_nei_cells + search_alpha) ** 2]
        self.max_valid_dist = 1.732 * ((num_nei_cells + 1) * self.resolution)

        dx = torch.arange(-1, 2, device=self.primes.device, dtype=self.primes.dtype)
        coords = torch.meshgrid(dx, dx, dx, indexing="ij")
        dx = torch.stack(coords, dim=-1).reshape(-1, 3)  # [K,3]
        dx2 = torch.sum(dx ** 2, dim=-1)

        self.neighbor_idx = dx[dx2 <= 2]
        self.max_valid_range = 2 * self.resolution
    def radius_neighborhood_search(self, points: torch.Tensor, time_filtering: bool = False):
        # T0 = get_time()
        cur_resolution = self.resolution
        cur_buffer_size = int(self.buffer_size)

        with torch.no_grad():
            grid_coords = (points / cur_resolution).floor().to(self.primes)  # [N,3]
            neighbord_cells = (grid_coords[..., None, :] + self.neighbor_dx)  # [N,K,3] # int64
            hash = torch.fmod((neighbord_cells * self.primes).sum(-1), cur_buffer_size)

        # [N,K] # with negative number (but actually the same)

        # T12 = get_time()

        neighb_idx = self.buffer_pt_index[hash]

        # T2 = get_time()

        if time_filtering:  # now is actually travel distance filtering
            diff_travel_dist = torch.abs(
                self.travel_dist[self.cur_ts]
                - self.travel_dist[self.point_ts_create[neighb_idx]]
            )
            local_t_window_mask = diff_travel_dist < self.diff_travel_dist_local

            # diff_ts = torch.abs(self.cur_ts - self.point_ts_create[neighb_idx]).squeeze(-1)
            # local_t_window_mask = diff_ts < self.diff_ts_local

            neighb_idx[~local_t_window_mask] = -1

        # T3 = get_time()

        neighb_pts = self.neural_points[neighb_idx]
        neighb_pts_sub = neighb_pts - points.view(-1, 1, 3)  # [N,K,3]

        dist2 = torch.sqrt(torch.sum(neighb_pts_sub ** 2, dim=-1)).clone()
        dist2[neighb_idx == -1] = self.max_valid_dist

        # if the dist is too large (indicating a hash collision), also mask the index as invalid
        neighb_idx[dist2 > self.max_valid_dist] = -1

        # T4 = get_time()

        # print("time for get neighbor idx:", (T1-T0) * 1e3)  # |
        # # print("time for hashing func    :", (T12-T1) * 1e3)
        # print("time for hashing         :", (T2-T1) * 1e3)  # ||||
        # print("time for time filtering  :", (T3-T2) * 1e3)  # |
        # print("time for distance        :", (T4-T3) * 1e3)  # |||
        # torch.autograd.set_detect_anomaly(True)

        return dist2, neighb_idx

    def region_specific_sdf_estimations(self, points: torch.Tensor):
        point_num = points.shape[0]
        sdf_abs = torch.zeros(point_num, device=points.device)
        surface_mask = torch.ones(point_num, dtype=torch.bool, device=self.config.device)

        bs = 262144  # 256 × 1024，接近 26 万，是 64 的倍数
        iter_n = math.ceil(point_num / bs)
        # 为了避免爆显存，采用分批处理的办法
        for n in range(iter_n):
            head = n * bs
            tail = min((n + 1) * bs, point_num)
            batch_points = points[head:tail, :]
            batch_coords = (batch_points / self.resolution).floor().to(self.primes)
            batch_neighbord_cells = (batch_coords[..., None, :] + self.neighbor_idx)
            batch_hash = torch.fmod((batch_neighbord_cells * self.primes).sum(-1), self.buffer_size)
            batch_neighb_idx = self.buffer_pt_index[batch_hash]
            batch_neighb_pts = self.neural_points[batch_neighb_idx]
            batch_dist = torch.norm(batch_neighb_pts - batch_points.view(-1, 1, 3), dim=-1)
            batch_dist = torch.where(batch_neighb_idx == -1, self.max_valid_range, batch_dist)

            # k nearst neighbors neural points
            batch_sdf_abs, batch_min_idx = torch.topk(batch_dist, 4, largest=False, dim=1)

            # For Ablation, comment below
            # Point-to-plane distance
            batch_min_idx_expanded = batch_min_idx.unsqueeze(-1).expand(-1, -1, 3)
            batch_knn_points = torch.gather(batch_neighb_pts, 1, batch_min_idx_expanded)
            valid_fit_mask = batch_sdf_abs[:, 3] < self.max_valid_range
            valid_batch_knn_points = batch_knn_points[valid_fit_mask]
            unit_normal_vector = torch.zeros_like(batch_points)
            plane_constant = torch.zeros(batch_points.size(0), device=batch_points.device)
            fit_success = torch.zeros(batch_points.size(0), dtype=torch.bool, device=batch_points.device)

            valid_unit_normal_vector, valid_plane_constant, valid_fit_success = estimate_plane(valid_batch_knn_points)
            unit_normal_vector[valid_fit_mask] = valid_unit_normal_vector
            plane_constant[valid_fit_mask] = valid_plane_constant
            fit_success[valid_fit_mask] = valid_fit_success

            fit_success &= batch_sdf_abs[:, 3] < self.max_valid_range  # 平面拟合失败
            surface_mask[head:tail] &= (batch_sdf_abs[:, 0] < self.max_valid_range)
            distance = torch.abs(torch.sum(unit_normal_vector * batch_points, dim=1) + plane_constant)
            sdf_abs[head:tail][fit_success] = distance[fit_success]
            sdf_abs[head:tail][~fit_success] = batch_sdf_abs[:, 0][~fit_success]

            # For Ablation
            # only using point-to-nearst point distance
            # sdf_abs[head:tail] = batch_sdf_abs[:, 0]
            # surface_mask[head:tail] = (batch_sdf_abs[:, 0] <= self.max_valid_range)

        if not self.config.silence:
            print(surface_mask.sum().item() / surface_mask.numel())
        return sdf_abs, surface_mask

    # a faster way to get the certainty at a batch of query points
    def query_certainty(self, query_points: torch.Tensor):

        _, idx = self.radius_neighborhood_search(query_points)  # only the self voxel

        # idx = self.global2local[0][idx] # [N, K] # get the local idx using the global2local mapping
        # certainty = self.local_hier_certainty[0][idx] # [N, K] # directly global search

        certainty = self.point_certainties[idx]
        certainty[idx < 0] = 0.0

        query_points_certainty = torch.max(certainty, dim=-1)[0]

        # print(query_points_certainty)

        return query_points_certainty

    # clear the temp data that is not needed
    def clear_temp(self, clean_more: bool = False):
        self.buffer_pt_index = None
        self.local_neural_points = None
        self.local_point_orientations = None
        self.local_geo_features = nn.Parameter()
        self.local_color_features = nn.Parameter()
        self.local_point_certainties = None
        self.local_point_ts_update = None
        self.local_mask = None
        self.global2local = None

        # Also only used for debugging, can be removed
        if clean_more:
            self.point_ts_create = None
            self.point_ts_update = None
            self.point_certainties = None

    def get_map_o3d_bbx(self):
        map_min, _ = torch.min(self.neural_points, dim=0)
        map_max, _ = torch.max(self.neural_points, dim=0)

        # print(map_min)

        o3d_bbx = o3d.geometry.AxisAlignedBoundingBox(
            map_min.cpu().detach().numpy(), map_max.cpu().detach().numpy()
        )

        return o3d_bbx

    # def feature_tsne(self):
    #     tsne = TSNE(n_components=3, perplexity=30, n_iter=300)
    #     tsne_result = tsne.fit_transform(self.geo_features[:-1].cpu().detach().numpy())


# the positional encoding is actually not used
# Borrow from Louis's LocNDF
# https://github.com/PRBonn/LocNDF
class PositionalEncoder(nn.Module):
    # out_dim = in_dimnesionality * (2 * bands + 1)
    def __init__(self, config: Config):
        super().__init__()

        self.freq = torch.tensor(config.pos_encoding_freq)
        self.num_bands = config.pos_encoding_band
        self.dimensionality = config.pos_input_dim
        self.base = torch.tensor(config.pos_encoding_base)

        self.out_dim = self.dimensionality * (2 * self.num_bands + 1)

        # self.num_bands = floor(feature_size/dimensionality/2)

    def forward(self, x):
        # print(x)
        x = x[..., : self.dimensionality, None]
        device, dtype, orig_x = x.device, x.dtype, x

        scales = torch.logspace(
            0.0,
            torch.log(self.freq / 2) / torch.log(self.base),
            self.num_bands,
            base=self.base,
            device=device,
            dtype=dtype,
        )
        # Fancy reshaping
        scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

        x = x * scales * torch.pi
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = torch.cat((x, orig_x), dim=-1)
        x = x.flatten(-2, -1)
        # print(x.shape)

        # print(x)

        return x

    def featureSize(self):
        return self.out_dim


# Borrow from Louis's Loc_NDF
# https://github.com/PRBonn/LocNDF
class GaussianFourierFeatures(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.freq = torch.tensor(config.pos_encoding_freq)
        self.num_bands = config.pos_encoding_band
        self.dimensionality = config.pos_input_dim

        self.register_buffer(
            "B", torch.randn([self.dimensionality, self.num_bands]) * self.freq
        )

        self.out_dim = self.num_bands * 2 + self.dimensionality

    def forward(self, x):
        x_proj = (2.0 * torch.pi * x) @ self.B
        return torch.cat([x, torch.sin(x_proj), torch.cos(x_proj)], axis=-1)

    def featureSize(self):
        return self.out_dim
