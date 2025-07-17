// pointmap_kernels.cu
// 神经点云核心CUDA kernel实现，支持体素哈希、点插入、邻域搜索、特征插值、置信度累积、点云更新等
// 便于C++/pybind11高效调用

#include "pointmap_kernels.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <curand_kernel.h>
#include <algorithm>
#include <stdexcept>
#include <stdint.h>
#include <cstdio>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")

// =========================================================
// 1. 体素哈希 kernel, 计算点云在体素哈希表中的索引
// =========================================================
__global__ void voxel_hash_kernel(const float* points, const int* primes, float resolution, int buffer_size, int* hash_values, int n_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;
    // 使用向量化加载，减少内存访问次数
    float3 point = *reinterpret_cast<const float3*>(&points[idx * 3]);
    int grid_x = static_cast<int>(floorf(point.x / resolution));
    int grid_y = static_cast<int>(floorf(point.y / resolution));
    int grid_z = static_cast<int>(floorf(point.z / resolution));
    // 使用向量化计算哈希，更简洁高效
    int3 grid_coord = make_int3(grid_x, grid_y, grid_z);
    int hash = (grid_coord.x * primes[0] + grid_coord.y * primes[1] + grid_coord.z * primes[2]) % buffer_size;
    hash = hash < 0 ? hash + buffer_size : hash;
    hash_values[idx] = hash;
}

void launch_voxel_hash_kernel(const float* points, const int* primes, float resolution, int buffer_size, int* hash_values, int n_points, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n_points + block_size - 1) / block_size;
    voxel_hash_kernel<<<grid_size, block_size, 0, stream>>>(points, primes, resolution, buffer_size, hash_values, n_points);
}

// =========================================================
// 2. 神经点插入 kernel（原子操作防冲突）
// =========================================================
__global__ void insert_points_kernel(const float* points, const int* hash_values, int* buffer_pt_index, int n_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;
    int hash = hash_values[idx];
    // CUDA哈希表/并发插入的标准写法, 使用原子操作防止冲突
    // 如果buffer_pt_index[hash]当前是-1（没人占），就把它设为idx（当前线程“抢占”这个槽位）。
    // 如果不是-1（已经被别的线程抢了），什么都不做。
    atomicCAS(&buffer_pt_index[hash], -1, idx);
}

void launch_insert_points_kernel(const float* points, const int* hash_values, int* buffer_pt_index, int n_points, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n_points + block_size - 1) / block_size;
    insert_points_kernel<<<grid_size, block_size, 0, stream>>>(points, hash_values, buffer_pt_index, n_points);
}

// =========================================================
// 3. 体素哈希邻域搜索 kernel（高效版本）
// =========================================================
__global__ void voxel_hash_neighbor_search_kernel(
    const float* query_points,      // [N, 3] 查询点坐标
    const int* primes,              // [3] 哈希用素数
    const int* buffer_pt_index,     // [buffer_size] 哈希表，存储点索引
    const float* neural_points,     // [M, 3] 神经点坐标
    const int* neighbor_dx,         // [K, 3] 邻域体素的相对偏移
    int neighbor_K,                 // 邻域体素数量
    float resolution,               // 体素分辨率
    int buffer_size,                // 哈希表大小
    float max_valid_dist2,          // 最大有效距离的平方
    int* neighbor_indices,          // [N, K] 输出：每个查询点的邻居点索引
    float* neighbor_dist2,          // [N, K] 输出：每个邻居的距离平方
    int n_query                     // 查询点数量
) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= n_query) return;
    
    // 读取当前查询点坐标
    float3 query_pt = *reinterpret_cast<const float3*>(&query_points[query_idx * 3]);
    
    // 计算查询点所在体素坐标
    int3 query_grid = make_int3(
        __float2int_rd(query_pt.x / resolution),
        __float2int_rd(query_pt.y / resolution),
        __float2int_rd(query_pt.z / resolution)
    );
    
    // 预加载邻域偏移到共享内存（如果neighbor_K不太大）
    #if __CUDA_ARCH__ >= 600 // 支持共享内存的架构
    __shared__ int shared_neighbor_dx[256]; // 假设最大256个邻域
    if (threadIdx.x < neighbor_K * 3) {
        shared_neighbor_dx[threadIdx.x] = neighbor_dx[threadIdx.x];
    }
    __syncthreads();
    #endif
    
    // 遍历所有邻域体素
    #pragma unroll
    for (int nei_idx = 0; nei_idx < neighbor_K; ++nei_idx) {
        // 计算邻域体素坐标
        #if __CUDA_ARCH__ >= 600
        int3 nei_grid = make_int3(
            query_grid.x + shared_neighbor_dx[nei_idx * 3 + 0],
            query_grid.y + shared_neighbor_dx[nei_idx * 3 + 1],
            query_grid.z + shared_neighbor_dx[nei_idx * 3 + 2]
        );
        #else
        int3 nei_grid = make_int3(
            query_grid.x + neighbor_dx[nei_idx * 3 + 0],
            query_grid.y + neighbor_dx[nei_idx * 3 + 1],
            query_grid.z + neighbor_dx[nei_idx * 3 + 2]
        );
        #endif
        
        // 计算该体素的哈希值
        int hash = (nei_grid.x * primes[0] + nei_grid.y * primes[1] + nei_grid.z * primes[2]) % buffer_size;
        hash = hash < 0 ? hash + buffer_size : hash;
        
        // 查找该体素内的点索引
        int pt_idx = buffer_pt_index[hash];
        
        // 如果该体素没有点，直接标记为无效邻居
        if (pt_idx < 0) {
            neighbor_indices[query_idx * neighbor_K + nei_idx] = -1;
            neighbor_dist2[query_idx * neighbor_K + nei_idx] = max_valid_dist2;
            continue;
        }
        
        // 计算距离
        float3 neural_pt = *reinterpret_cast<const float3*>(&neural_points[pt_idx * 3]);
        // 使用向量化距离计算
        float3 diff = make_float3(query_pt.x - neural_pt.x, query_pt.y - neural_pt.y, query_pt.z - neural_pt.z);
        float dist2 = __fmaf_rn(diff.x, diff.x, __fmaf_rn(diff.y, diff.y, diff.z * diff.z)); // FMA优化
        
        // 检查是否在有效距离内
        if (dist2 < max_valid_dist2) {
            neighbor_indices[query_idx * neighbor_K + nei_idx] = pt_idx;
            neighbor_dist2[query_idx * neighbor_K + nei_idx] = dist2;
        } else {
            neighbor_indices[query_idx * neighbor_K + nei_idx] = -1;
            neighbor_dist2[query_idx * neighbor_K + nei_idx] = max_valid_dist2;
        }
    }
}

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
) {
    int block_size = 128;
    int grid_size = (n_query + block_size - 1) / block_size;
    voxel_hash_neighbor_search_kernel<<<grid_size, block_size, 0, stream>>>(
        query_points, primes, buffer_pt_index, neural_points, neighbor_dx,
        neighbor_K, resolution, buffer_size, max_valid_dist2,
        neighbor_indices, neighbor_dist2, n_query
    );
}

// =========================================================
// 4. 特征插值 kernel（支持多通道/多类型）
// =========================================================
template <typename scalar_t>
__global__ void feature_interpolate_kernel(const float* query_points, const float* map_points, const scalar_t* map_features, int* neighbor_indices, int n_query, int k, int feat_dim, scalar_t* out_features) {
    int qidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (qidx >= n_query) return;
    for (int d = 0; d < feat_dim; ++d) {
        scalar_t val = 0;
        int valid = 0;
        for (int i = 0; i < k; ++i) {
            int nidx = neighbor_indices[qidx * k + i];
            if (nidx >= 0) {
                val += map_features[nidx * feat_dim + d];
                valid++;
            }
        }
        out_features[qidx * feat_dim + d] = valid > 0 ? val / valid : 0;
    }
}

void launch_feature_interpolate_kernel(const float* query_points, const float* map_points, const void* map_features, int* neighbor_indices, int n_query, int k, int feat_dim, void* out_features, at::ScalarType dtype, cudaStream_t stream) {
    int block_size = 128;
    int grid_size = (n_query + block_size - 1) / block_size;
    if (dtype == at::kFloat) {
        feature_interpolate_kernel<float><<<grid_size, block_size, 0, stream>>>(query_points, map_points, (const float*)map_features, neighbor_indices, n_query, k, feat_dim, (float*)out_features);
    } else if (dtype == at::kHalf) {
        feature_interpolate_kernel<at::Half><<<grid_size, block_size, 0, stream>>>(query_points, map_points, (const at::Half*)map_features, neighbor_indices, n_query, k, feat_dim, (at::Half*)out_features);
    } else if (dtype == at::kDouble) {
        feature_interpolate_kernel<double><<<grid_size, block_size, 0, stream>>>(query_points, map_points, (const double*)map_features, neighbor_indices, n_query, k, feat_dim, (double*)out_features);
    } else {
        throw std::runtime_error("Unsupported dtype in feature_interpolate_kernel");
    }
}

// =========================================================
// 5. 置信度累积 kernel
// =========================================================
__global__ void accumulate_certainty_kernel(const int* neighbor_indices, float* certainties, int n_query, int k) {
    int qidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (qidx >= n_query) return;
    for (int i = 0; i < k; ++i) {
        int nidx = neighbor_indices[qidx * k + i];
        if (nidx >= 0) {
            atomicAdd(&certainties[nidx], 1.0f);
        }
    }
}

void launch_accumulate_certainty_kernel(const int* neighbor_indices, float* certainties, int n_query, int k, cudaStream_t stream) {
    int block_size = 128;
    int grid_size = (n_query + block_size - 1) / block_size;
    accumulate_certainty_kernel<<<grid_size, block_size, 0, stream>>>(neighbor_indices, certainties, n_query, k);
}

// =========================================================
// 6. 点云更新/融合 kernel（update）
// =========================================================
__global__ void update_points_kernel(
    const float* points,           // [N, 3]
    int* buffer_pt_index,          // [buffer_size]
    float* neural_points,          // [M, 3]
    float* point_orientations,     // [M, 4]
    float* geo_features,           // [M, geo_dim]
    float* color_features,         // [M, color_dim] (可为nullptr)
    int* point_ts_create,          // [M]
    int* point_ts_update,          // [M]
    float* point_certainties,      // [M]
    int* cur_count,                // [1] 全局点数计数器
    int* hash_values,              // [N]
    int buffer_size,
    int geo_dim,
    int color_dim,
    float geo_std,
    float color_std,
    int cur_ts,
    unsigned long long seed,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int hash = hash_values[idx];
    // 原子插入，防止冲突
    int insert_idx = atomicCAS(&buffer_pt_index[hash], -1, -2); // -2: 正在写入
    if (insert_idx == -1) {
        // 分配新点索引
        int pt_idx = atomicAdd(cur_count, 1);
        // 向量化写入点坐标
        *reinterpret_cast<float3*>(&neural_points[pt_idx * 3]) = *reinterpret_cast<const float3*>(&points[idx * 3]);
        // orientation: 单位四元数 (1,0,0,0)
        float4 orientation = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
        *reinterpret_cast<float4*>(&point_orientations[pt_idx * 4]) = orientation;
        // geo_features: 高斯随机
        curandState state;
        curand_init(seed + idx, 0, 0, &state);
        for (int d = 0; d < geo_dim; ++d) {
            geo_features[pt_idx * geo_dim + d] = geo_std * curand_normal(&state);
        }
        // color_features: 高斯随机
        if (color_features != nullptr && color_dim > 0) {
            for (int d = 0; d < color_dim; ++d) {
                color_features[pt_idx * color_dim + d] = color_std * curand_normal(&state);
            }
        }
        // ts
        point_ts_create[pt_idx] = cur_ts;
        point_ts_update[pt_idx] = cur_ts;
        // certainty
        point_certainties[pt_idx] = 0.0f;
        // buffer_pt_index 指向新点
        buffer_pt_index[hash] = pt_idx;
    } else if (insert_idx == -2) {
        // 有其他线程正在写入，什么都不做
    } else {
        // 已有点，什么都不做（可扩展为更新）
    }
}

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
) {
    int block_size = 128;
    int grid_size = (N + block_size - 1) / block_size;
    update_points_kernel<<<grid_size, block_size, 0, stream>>>(
        points, buffer_pt_index, neural_points, point_orientations, geo_features, color_features,
        point_ts_create, point_ts_update, point_certainties, cur_count, hash_values, buffer_size,
        geo_dim, color_dim, geo_std, color_std, cur_ts, seed, N
    );
} 