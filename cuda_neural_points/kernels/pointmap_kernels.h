#pragma once
#include <cuda_runtime.h>
#include <ATen/core/ScalarType.h>

// 体素哈希
void launch_voxel_hash_kernel(const float* points, const int* primes, float resolution, int buffer_size, int* hash_values, int n_points, cudaStream_t stream);

// 点插入
void launch_insert_points_kernel(const float* points, const int* hash_values, int* buffer_pt_index, int n_points, cudaStream_t stream);

// 体素哈希邻域搜索（高效版本）
void launch_voxel_hash_neighbor_search_kernel(
    const float* query_points,
    const int* primes,
    const int* buffer_pt_index,
    const float* neural_points,
    const int* neighbor_dx,
    int neighbor_K,
    float resolution,
    int buffer_size,
    float max_valid_dist2,
    int* neighbor_indices,
    float* neighbor_dist2,
    int n_query,
    cudaStream_t stream
);

// 特征插值（支持多通道/多类型）
void launch_feature_interpolate_kernel(const float* query_points, const float* map_points, const void* map_features, int* neighbor_indices, int n_query, int k, int feat_dim, void* out_features, at::ScalarType dtype, cudaStream_t stream);

// 置信度累积
void launch_accumulate_certainty_kernel(const int* neighbor_indices, float* certainties, int n_query, int k, cudaStream_t stream);

// 点云更新/融合（update）
void launch_update_points_kernel(
    const float* points,
    int* buffer_pt_index,
    float* neural_points,
    float* point_orientations,
    float* geo_features,
    float* color_features,
    int* point_ts_create,
    int* point_ts_update,
    float* point_certainties,
    int* cur_count,
    int* hash_values,
    int buffer_size,
    int geo_dim,
    int color_dim,
    float geo_std,
    float color_std,
    int cur_ts,
    unsigned long long seed,
    int N,
    cudaStream_t stream
); 