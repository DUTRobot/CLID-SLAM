#!/usr/bin/env python3
# @file      local_point_cloud_map.py
# @author    Junlong Jiang     [jiangjunlong@mail.dlut.edu.cn]
# Copyright (c) 2025 Junlong Jiang, all rights reserved
import math
import torch
from utils.config import Config
from utils.tools import voxel_down_sample_torch


class LocalPointCloudMap:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.idx_dtype = torch.int64
        self.dtype = config.dtype
        self.device = config.device
        self.resolution = config.local_voxel_size_m
        self.buffer_size = config.local_buffer_size

        self.buffer_pt_index = -torch.ones(
            self.buffer_size, dtype=self.idx_dtype, device=self.device
        )
        self.local_point_cloud_map = torch.empty(
            (0, 3), dtype=torch.float32, device=self.device
        )

        self.primes = torch.tensor(
            [73856093, 19349663, 83492791], dtype=self.idx_dtype, device=self.device
        )
        self.neighbor_idx = None
        self.max_valid_range = None
        self.set_search_neighborhood()
        self.map_size = config.local_map_size

    def voxel_hash(self, points):
        grid_coords = (points / self.resolution).floor().to(self.primes)
        hash_values = torch.fmod((grid_coords * self.primes).sum(-1), self.buffer_size)
        return hash_values

    def insert_points(self, points):
        sample_idx = voxel_down_sample_torch(points, self.resolution)
        sample_points = points[sample_idx]
        hash_values = self.voxel_hash(sample_points)
        hash_idx = self.buffer_pt_index[hash_values]

        update_mask = hash_idx == -1
        new_points = sample_points[update_mask]

        cur_pt_count = self.local_point_cloud_map.shape[0]
        self.buffer_pt_index[hash_values[update_mask]] = (
            torch.arange(new_points.shape[0], device=self.device) + cur_pt_count
        )

        self.local_point_cloud_map = torch.cat(
            (self.local_point_cloud_map, new_points), 0
        )

    def update_map(self, sensor_position, points):
        self.insert_points(points)
        distances = torch.norm(self.local_point_cloud_map - sensor_position, dim=-1)
        keep_mask = distances < self.map_size
        self.local_point_cloud_map = self.local_point_cloud_map[keep_mask]

        new_buffer_pt_index = -torch.ones(
            self.buffer_size, dtype=self.idx_dtype, device=self.device
        )
        new_hash_values = self.voxel_hash(self.local_point_cloud_map)
        new_buffer_pt_index[new_hash_values] = torch.arange(
            self.local_point_cloud_map.shape[0], device=self.device
        )

        self.buffer_pt_index = new_buffer_pt_index

    def set_search_neighborhood(
        self, num_nei_cells: int = 1, search_alpha: float = 0.2
    ):
        dx = torch.arange(
            -num_nei_cells,
            num_nei_cells + 1,
            device=self.primes.device,
            dtype=self.primes.dtype,
        )

        coords = torch.meshgrid(dx, dx, dx, indexing="ij")
        dx = torch.stack(coords, dim=-1).reshape(-1, 3)  # [K,3]

        dx2 = torch.sum(dx**2, dim=-1)
        self.neighbor_idx = dx[dx2 < (num_nei_cells + search_alpha) ** 2]
        self.max_valid_range = 1.732 * (num_nei_cells + 1) * self.resolution
        # in the sphere --> smaller K --> faster training
        # when num_cells = 2              when num_cells = 3
        # alpha 0.2, K = 33               alpha 0.2, K = 147
        # alpha 0.3, K = 57               alpha 0.5, K = 179
        # alpha 0.5, K = 81               alpha 1.0, K = 251
        # alpha 1.0, K = 93
        # alpha 2.0, K = 125

    def region_specific_sdf_estimation(self, points: torch.Tensor):
        point_num = points.shape[0]
        sdf_abs = torch.ones(point_num, device=points.device) * self.max_valid_range
        surface_mask = torch.ones(
            point_num, dtype=torch.bool, device=self.config.device
        )

        bs = 262144  # 256 × 1024
        iter_n = math.ceil(point_num / bs)
        # 为了避免爆显存，采用分批处理的办法
        for n in range(iter_n):
            head, tail = n * bs, min((n + 1) * bs, point_num)
            batch_pts = points[head:tail, :]
            batch_coords = (batch_pts / self.resolution).floor().to(self.primes)
            batch_neighbord_cells = batch_coords[..., None, :] + self.neighbor_idx
            batch_hash = torch.fmod(
                (batch_neighbord_cells * self.primes).sum(-1), self.buffer_size
            )
            batch_neighb_idx = self.buffer_pt_index[batch_hash]
            batch_neighb_pts = self.local_point_cloud_map[batch_neighb_idx]
            batch_dist = torch.norm(batch_neighb_pts - batch_pts.view(-1, 1, 3), dim=-1)
            batch_dist = torch.where(
                batch_neighb_idx == -1, self.max_valid_range, batch_dist
            )

            # k nearst neighbors neural points
            batch_sdf_abs, batch_min_idx = torch.topk(
                batch_dist, 4, largest=False, dim=1
            )
            batch_min_idx_expanded = batch_min_idx.unsqueeze(-1).expand(-1, -1, 3)
            batch_knn_points = torch.gather(batch_neighb_pts, 1, batch_min_idx_expanded)
            valid_fit_mask = batch_sdf_abs[:, 3] < self.max_valid_range
            valid_batch_knn_points = batch_knn_points[valid_fit_mask]
            normal = torch.zeros_like(batch_pts)
            plane_constant = torch.zeros(batch_pts.size(0), device=batch_pts.device)
            fit_success = torch.zeros(
                batch_pts.size(0), dtype=torch.bool, device=batch_pts.device
            )

            valid_normal, valid_plane_constant, valid_fit_success = estimate_plane(
                valid_batch_knn_points
            )
            normal[valid_fit_mask] = valid_normal
            plane_constant[valid_fit_mask] = valid_plane_constant
            fit_success[valid_fit_mask] = valid_fit_success

            fit_success &= batch_sdf_abs[:, 3] < self.max_valid_range  # 平面拟合失败
            surface_mask[head:tail] &= batch_sdf_abs[:, 0] < self.max_valid_range
            distance = torch.abs(torch.sum(normal * batch_pts, dim=1) + plane_constant)
            sdf_abs[head:tail][fit_success] = distance[fit_success]
            sdf_abs[head:tail][~fit_success] = batch_sdf_abs[:, 0][~fit_success]

        if not self.config.silence:
            print(surface_mask.sum().item() / surface_mask.numel())
        return sdf_abs, surface_mask


def estimate_plane(
    points: torch.Tensor, eta_threshold: float = 0.2, threshold: float = 0.1
):
    """Estimates planes from a given set of 3D points using Singular Value Decomposition (SVD)"""

    def fit_planes(points: torch.Tensor):
        """Fits multiple planes using SVD"""
        centroid = points.mean(dim=1, keepdim=True)
        centered_points = points - centroid
        U, S, Vh = torch.linalg.svd(
            centered_points, full_matrices=False
        )  # Perform SVD.

        # The normal vector of the plane is the last row of Vh (since Vh is the transpose of V).
        normals = Vh[:, -1, :]
        return normals, centroid.squeeze(1), S

    def is_valid_planes(singular_values: torch.Tensor, eta_threshold: float):
        """Determines whether the fitted planes are valid based on the η value."""
        lambda_min = singular_values[:, -1]  # The smallest singular value.
        lambda_mid = singular_values[:, 1]  # The middle singular value.

        eta = lambda_min / (lambda_mid + 1e-6)
        return eta <= eta_threshold

    m, num_points, _ = points.shape
    normal_vector, centroids, singular_values = fit_planes(
        points
    )  # Fit planes to the input points.

    # Initialize normal vectors with zeros.
    normal = torch.zeros((m, 3), dtype=points.dtype, device=points.device)

    valid_mask = is_valid_planes(singular_values, eta_threshold)
    normal_vector = normal_vector[valid_mask]
    normal[valid_mask] = normal_vector
    plane_constant = -1.0 * torch.sum(normal * centroids, dim=1)

    # Compute the distance of each point to its respective plane.
    # distances = torch.abs((points @ normal.unsqueeze(-1)).squeeze() + plane_constant.unsqueeze(-1))
    distances = torch.abs(
        torch.bmm(points, normal.unsqueeze(-1)).squeeze() + plane_constant.unsqueeze(-1)
    )
    fit_success = torch.max(distances, dim=1).values <= threshold
    mask = fit_success & valid_mask

    return normal, plane_constant, mask
