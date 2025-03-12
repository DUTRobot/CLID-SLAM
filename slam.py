#!/usr/bin/env python3
# @file      pin_slam.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved
# Modifications by:
# Junlong Jiang [jiangjunlong@mail.dlut.edu.cn]
# Copyright (c) 2025 Junlong Jiang, all rights reserved.

import os
import sys

import numpy as np
import open3d as o3d
import torch
from rich import print
from tqdm import tqdm

from utils.dataset_indexing import set_dataset_path
from utils.slam_dataset import SLAMDataset
from model.decoder import Decoder
from model.neural_points import NeuralPoints
from utils.mapper import Mapper
from utils.config import Config
from utils.esekfom import IESEKF
from utils.mesher import Mesher
from utils.tools import freeze_model, get_time, save_implicit_map, setup_experiment, split_chunks
from utils.visualizer import MapVisualizer

def run_slam(config_path=None, dataset_name=None, sequence_name=None, seed=None):
    torch.set_num_threads(16)  # 设置为16个线程，限制使用的线程数，使用太多的线程会导致电脑卡死
    config = Config()
    if config_path is not None:
        config.load(config_path)
        set_dataset_path(config, dataset_name, sequence_name)
        if seed is not None:
            config.seed = seed
        argv = ['slam.py', config_path, dataset_name, sequence_name, str(seed)]
        run_path = setup_experiment(config, argv)
    else:
        if len(sys.argv) > 1:
            config.load(sys.argv[1])
        else:
            sys.exit("Please provide the path to the config file.\nTry: \
                    python3 slam.py path_to_config.yaml [dataset_name] [sequence_name] [random_seed]")
            # specific dataset [optional]
        if len(sys.argv) == 3:
            set_dataset_path(config, sys.argv[2])
        if len(sys.argv) > 3:
            set_dataset_path(config, sys.argv[2], sys.argv[3])
        if len(sys.argv) > 4:  # random seed [optional]
            config.seed = int(sys.argv[4])
        run_path = setup_experiment(config, sys.argv)
        print("[bold green]CLID-SLAM starts[/bold green]", "📍")

    # non-blocking visualizer
    if config.o3d_vis_on:
        o3d_vis = MapVisualizer(config)

    # 初始化MLP解码器
    geo_mlp = Decoder(config, config.geo_mlp_hidden_dim, config.geo_mlp_level, 1)

    # 初始化神经点云地图
    neural_points = NeuralPoints(config)

    # 初始化数据集
    dataset = SLAMDataset(config)

    # 里程计跟踪模块
    kf_state = IESEKF(config, neural_points, geo_mlp)
    dataset.kf_state = kf_state

    # 建图模块
    mapper = Mapper(config, dataset, neural_points, geo_mlp)

    # 网格重建
    mesher = Mesher(config, neural_points, geo_mlp, None, None)

    last_frame = dataset.total_pc_count - 1

    for frame_id in tqdm(range(dataset.total_pc_count)):

        # I. 加载数据和预处理
        T0 = get_time()
        dataset.read_frame(frame_id)

        T1 = get_time()
        valid_frame = dataset.preprocess_frame()
        if not valid_frame:
            dataset.processed_frame += 1
            continue

        T2 = get_time()

        # II. 里程计定位
        if frame_id > 0:
            if config.track_on:
                cur_pose_torch, valid_flag = kf_state.update_iterated(dataset.cur_source_points)
                dataset.lose_track = not valid_flag
                dataset.update_odom_pose(cur_pose_torch)  # update dataset.cur_pose_torch

                if not valid_flag and config.o3d_vis_on and o3d_vis.debug_mode > 0:
                    o3d_vis.stop()

        travel_dist = dataset.travel_dist[:frame_id + 1]
        neural_points.travel_dist = torch.tensor(travel_dist, device=config.device, dtype=config.dtype)

        T3 = get_time()
        # III: 建图和光束平差优化
        # if lose track, we will not update the map and data pool (don't let the wrong pose to corrupt the map)
        # if the robot stop, also don't process this frame, since there's no new oberservations
        if not dataset.lose_track and not dataset.stop_status:
            mapper.process_frame(dataset.cur_point_cloud_torch, dataset.cur_sem_labels_torch,
                                 dataset.cur_pose_torch, frame_id, (config.dynamic_filter_on and frame_id > 0))
        else:
            mapper.determine_used_pose()
            neural_points.reset_local_map(dataset.cur_pose_torch[:3, 3], None, frame_id)  # not efficient for large map

        T4 = get_time()

        # for the first frame, we need more iterations to do the initialization (warm-up)
        # 计算当前帧建图的迭代轮数
        cur_iter_num = config.iters * config.init_iter_ratio if frame_id == 0 else config.iters
        if dataset.stop_status:
            cur_iter_num = max(1, cur_iter_num - 10)
        #  在某一帧后固定解码器的参数
        if frame_id == config.freeze_after_frame:  # freeze the decoder after certain frame
            freeze_model(geo_mlp)

        # mapping with fixed poses (every frame)
        if frame_id % config.mapping_freq_frame == 0:
            mapper.mapping(cur_iter_num)

        T5 = get_time()

        # regular saving logs
        if config.log_freq_frame > 0 and (frame_id + 1) % config.log_freq_frame == 0:
            dataset.write_results_log()

        # IV: 网格重建和可视化
        if config.o3d_vis_on:  # if visualizer is off, there's no need to reconstruct the mesh

            o3d_vis.cur_frame_id = frame_id  # frame id in the data folder
            dataset.update_o3d_map()

            T6 = get_time()

            if frame_id == last_frame:
                o3d_vis.vis_global = True
                o3d_vis.ego_view = False
                mapper.free_pool()

            neural_pcd = None
            if o3d_vis.render_neural_points or (frame_id == last_frame):  # last frame also vis
                neural_pcd = neural_points.get_neural_points_o3d(query_global=o3d_vis.vis_global,
                                                                 color_mode=o3d_vis.neural_points_vis_mode,
                                                                 random_down_ratio=1)  # select from geo_feature, ts and certainty

            # reconstruction by marching cubes
            cur_mesh = None
            if config.mesh_freq_frame > 0:
                if o3d_vis.render_mesh and (frame_id == 0 or frame_id == last_frame or (
                        frame_id + 1) % config.mesh_freq_frame == 0):
                    # update map bbx
                    global_neural_pcd_down = neural_points.get_neural_points_o3d(query_global=True,
                                                                                 random_down_ratio=23)  # prime number
                    dataset.map_bbx = global_neural_pcd_down.get_axis_aligned_bounding_box()

                    mesh_path = None  # no need to save the mesh
                    if frame_id == last_frame and config.save_mesh:  # save the mesh at the last frame
                        mc_cm_str = str(round(o3d_vis.mc_res_m * 1e2))
                        mesh_path = os.path.join(run_path, "mesh",
                                                 'mesh_frame_' + str(frame_id) + "_" + mc_cm_str + "cm.ply")

                    # figure out how to do it efficiently
                    if not o3d_vis.vis_global:  # only build the local mesh
                        chunks_aabb = split_chunks(global_neural_pcd_down, dataset.cur_bbx,
                                                   o3d_vis.mc_res_m * 100)  # reconstruct in chunks
                        cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, o3d_vis.mc_res_m, mesh_path, True,
                                                                      config.semantic_on, config.color_on,
                                                                      filter_isolated_mesh=True,
                                                                      mesh_min_nn=o3d_vis.mesh_min_nn)
                    else:
                        aabb = global_neural_pcd_down.get_axis_aligned_bounding_box()
                        chunks_aabb = split_chunks(global_neural_pcd_down, aabb,
                                                   o3d_vis.mc_res_m * 300)  # reconstruct in chunks
                        cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, o3d_vis.mc_res_m, mesh_path, False,
                                                                      config.semantic_on, config.color_on,
                                                                      filter_isolated_mesh=True,
                                                                      mesh_min_nn=o3d_vis.mesh_min_nn)
            cur_sdf_slice = None
            if config.sdfslice_freq_frame > 0:
                if o3d_vis.render_sdf and (
                        frame_id == 0 or frame_id == last_frame or (frame_id + 1) % config.sdfslice_freq_frame == 0):
                    slice_res_m = config.voxel_size_m * 0.2
                    sdf_bound = config.surface_sample_range_m * 4.0
                    query_sdf_locally = True
                    if o3d_vis.vis_global:
                        cur_sdf_slice_h = mesher.generate_bbx_sdf_hor_slice(dataset.map_bbx, dataset.cur_pose_ref[
                            2, 3] + o3d_vis.sdf_slice_height, slice_res_m, False, -sdf_bound,
                                                                            sdf_bound)  # horizontal slice
                    else:
                        cur_sdf_slice_h = mesher.generate_bbx_sdf_hor_slice(dataset.cur_bbx, dataset.cur_pose_ref[
                            2, 3] + o3d_vis.sdf_slice_height, slice_res_m, query_sdf_locally, -sdf_bound,
                                                                            sdf_bound)  # horizontal slice (local)
                    if config.vis_sdf_slice_v:
                        cur_sdf_slice_v = mesher.generate_bbx_sdf_ver_slice(dataset.cur_bbx, dataset.cur_pose_ref[0, 3],
                                                                            slice_res_m, query_sdf_locally, -sdf_bound,
                                                                            sdf_bound)  # vertical slice (local)
                        cur_sdf_slice = cur_sdf_slice_h + cur_sdf_slice_v
                    else:
                        cur_sdf_slice = cur_sdf_slice_h

            pool_pcd = mapper.get_data_pool_o3d(down_rate=17,
                                                only_cur_data=o3d_vis.vis_only_cur_samples) if o3d_vis.render_data_pool else None  # down rate should be a prime number
            odom_poses, gt_poses, pgo_poses = dataset.get_poses_np_for_vis()
            o3d_vis.update_traj(dataset.cur_pose_ref, odom_poses, gt_poses, pgo_poses)
            o3d_vis.update(dataset.cur_frame_o3d, dataset.cur_pose_ref, cur_sdf_slice, cur_mesh, neural_pcd, pool_pcd)

        # loop & pgo in the end, visualization and I/O time excluded
        # 我们这个系统把回环删了
        cur_frame_process_time = np.array([T2 - T1, T3 - T2, T4 - T3, T5 - T4, 0])
        dataset.time_table.append(cur_frame_process_time)  # in s
        dataset.processed_frame += 1

    # V. 保存结果
    if config.track_on:
        pose_eval_results = dataset.write_results()

    neural_points.recreate_hash(None, None, False, False)  # merge the final neural point map
    neural_points.prune_map(config.max_prune_certainty, 0)  # prune uncertain points for the final output
    if config.save_map:
        neural_pcd = neural_points.get_neural_points_o3d(query_global=True, color_mode=0)
        # write the neural point cloud
        o3d.io.write_point_cloud(os.path.join(run_path, "map", "neural_points.ply"), neural_pcd)
    neural_points.clear_temp()  # clear temp data for output

    if config.save_map:
        save_implicit_map(run_path, neural_points, geo_mlp, None, None)
    if config.save_merged_pc:
        dataset.write_merged_point_cloud()  # replay: save merged point cloud map

    if config.o3d_vis_on:
        while True:
            o3d_vis.ego_view = False
            o3d_vis.update(dataset.cur_frame_o3d, dataset.cur_pose_ref, cur_sdf_slice, cur_mesh, neural_pcd, pool_pcd)
            odom_poses, gt_poses, pgo_poses = dataset.get_poses_np_for_vis()
            o3d_vis.update_traj(dataset.cur_pose_ref, odom_poses, gt_poses, pgo_poses)

    return pose_eval_results


if __name__ == "__main__":
    run_slam()
