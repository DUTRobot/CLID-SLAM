#pragma once
#include <torch/extension.h>
#include <vector>
#include <memory>

class NeuralPointsCUDA {
public:
    NeuralPointsCUDA(int buffer_size, float resolution, int geo_feature_dim, int color_feature_dim, int max_points = 10000);
    ~NeuralPointsCUDA();

    // 基础信息
    int count() const;
    bool is_empty() const;

    // 点云更新/融合
    void update(const torch::Tensor& points, const torch::Tensor& sensor_position, const torch::Tensor& sensor_orientation, int cur_ts);

    // 局部地图重置
    void reset_local_map(const torch::Tensor& sensor_position, const torch::Tensor& sensor_orientation, int cur_ts, bool use_travel_dist = true, int diff_ts_local = 50, bool reboot_map = false);

    // 特征查询/插值
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    query_feature(const torch::Tensor& query_points, const torch::Tensor& query_ts, bool training_mode = true, bool query_locally = true, bool query_geo_feature = true, bool query_color_feature = false);

    // 局部到全局同步
    void assign_local_to_global();

    // 稀疏化/裁剪
    void prune_map(float prune_certainty_thre, int min_prune_count = 500, bool global_prune = false);

    // 全局优化后调整
    void adjust_map(const torch::Tensor& pose_diff_torch);

    // 哈希重建
    void recreate_hash(const torch::Tensor& sensor_position, const torch::Tensor& sensor_orientation, bool kept_points = true, bool with_ts = true, int cur_ts = 0);

    // 邻域搜索
    std::pair<torch::Tensor, torch::Tensor> radius_neighborhood_search(const torch::Tensor& query_points, bool time_filtering = false);

    // 置信度查询
    torch::Tensor query_certainty(const torch::Tensor& query_points);

    // 临时数据清理
    void clear_temp(bool clean_more = false);

    // 其他辅助接口
    void set_search_neighborhood(int num_nei_cells = 1, float search_alpha = 1.0);

    // ... 其他接口可根据需要补充

private:
    // 内部成员变量声明（略）
}; 