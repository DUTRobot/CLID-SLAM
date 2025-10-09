#!/usr/bin/env python3
# @file      pin_slam.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved
# Modifications by:
# Junlong Jiang [jiangjunlong@mail.dlut.edu.cn]
# Copyright (c) 2025 Junlong Jiang, all rights reserved.

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import time
import numpy as np
import open3d as o3d
import torch
import torch.multiprocessing as mp
import wandb
from rich import print
from tqdm import tqdm
from gui import slam_gui
from gui.gui_utils import ParamsGUI, VisPacket, ControlPacket, get_latest_queue
from model.decoder import Decoder
from model.local_point_cloud_map import LocalPointCloudMap
from model.neural_points import NeuralPoints
from utils.config import Config
from utils.dataset_indexing import set_dataset_path
from utils.error_state_iekf import IEKFOM
from utils.mapper import Mapper
from utils.mesher import Mesher
from utils.slam_dataset import SLAMDataset
from utils.tools import (
    freeze_model,
    get_time,
    save_implicit_map,
    setup_experiment,
    split_chunks,
    remove_gpu_cache,
    create_bbx_o3d,
)


def run_slam(config_path=None, dataset_name=None, sequence_name=None, seed=None):
    torch.set_num_threads(
        16
    )  # 设置为16个线程，限制使用的线程数，使用太多的线程会导致电脑卡死
    config = Config()
    if config_path is not None:
        config.load(config_path)
        set_dataset_path(config, dataset_name, sequence_name)
        if seed is not None:
            config.seed = seed
        argv = ["slam.py", config_path, dataset_name, sequence_name, str(seed)]
        run_path = setup_experiment(config, argv)
    else:
        if len(sys.argv) > 1:
            config.load(sys.argv[1])
        else:
            sys.exit(
                "Please provide the path to the config file.\nTry: \
                    python3 slam.py path_to_config.yaml [dataset_name] [sequence_name] [random_seed]"
            )
            # specific dataset [optional]
        if len(sys.argv) == 3:
            set_dataset_path(config, sys.argv[2])
        if len(sys.argv) > 3:
            set_dataset_path(config, sys.argv[2], sys.argv[3])
        if len(sys.argv) > 4:  # random seed [optional]
            config.seed = int(sys.argv[4])
        run_path = setup_experiment(config, sys.argv)
        print("⚔️", "[bold green]CLID-SLAM starts[/bold green]")

    if config.o3d_vis_on:
        mp.set_start_method("spawn")  # don't forget this

    # 初始化MLP解码器
    geo_mlp = Decoder(config, config.geo_mlp_hidden_dim, config.geo_mlp_level, 1)
    mlp_dict = {"sdf": geo_mlp, "semantic": None, "color": None}

    # 初始化神经点云地图
    neural_points = NeuralPoints(config)
    local_point_cloud_map = LocalPointCloudMap(config)

    # 初始化数据集
    dataset = SLAMDataset(config)

    # 里程计跟踪模块
    iekfom = IEKFOM(config, neural_points, geo_mlp)
    dataset.tracker = iekfom

    # 建图模块
    mapper = Mapper(config, dataset, neural_points, local_point_cloud_map, geo_mlp)

    # 网格重建
    mesher = Mesher(config, neural_points, mlp_dict)

    last_frame = dataset.total_pc_count - 1

    # 可视化
    q_main2vis = q_vis2main = None
    if config.o3d_vis_on:
        # communicator between the processes
        q_main2vis = mp.Queue()
        q_vis2main = mp.Queue()

        params_gui = ParamsGUI(
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
            config=config,
            local_map_default_on=config.local_map_default_on,
            mesh_default_on=config.mesh_default_on,
            sdf_default_on=config.sdf_default_on,
            neural_point_map_default_on=config.neural_point_map_default_on,
        )
        gui_process = mp.Process(target=slam_gui.run, args=(params_gui,))
        gui_process.start()
        time.sleep(3)  # second

        # visualizer configs
        vis_visualize_on = True
        vis_source_pc_weight = False
        vis_global_on = not config.local_map_default_on
        vis_mesh_on = config.mesh_default_on
        vis_mesh_freq_frame = config.mesh_freq_frame
        vis_mesh_mc_res_m = config.mc_res_m
        vis_mesh_min_nn = config.mesh_min_nn
        vis_sdf_on = config.sdf_default_on
        vis_sdf_freq_frame = config.sdfslice_freq_frame
        vis_sdf_slice_height = config.sdf_slice_height
        vis_sdf_res_m = config.vis_sdf_res_m

    cur_mesh = None
    cur_sdf_slice = None

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
                cur_pose_torch, valid_flag = iekfom.update_iterated(
                    dataset.cur_source_points
                )
                dataset.lose_track = not valid_flag
                dataset.update_odom_pose(
                    cur_pose_torch
                )  # update dataset.cur_pose_torch

        travel_dist = dataset.travel_dist[: frame_id + 1]
        neural_points.travel_dist = torch.tensor(
            travel_dist, device=config.device, dtype=config.dtype
        )  # always update this
        valid_mapping_flag = (not dataset.lose_track) and (not dataset.stop_status)

        T3 = get_time()
        # III: 建图和光束平差优化
        # if lose track, we will not update the map and data pool (don't let the wrong pose to corrupt the map)
        # if the robot stop, also don't process this frame, since there's no new oberservations
        if not dataset.lose_track and valid_mapping_flag:
            mapper.process_frame(
                dataset.cur_point_cloud_torch,
                dataset.cur_sem_labels_torch,
                dataset.cur_pose_torch,
                frame_id,
                (config.dynamic_filter_on and frame_id > 0),
            )
        else:
            mapper.determine_used_pose()
            neural_points.reset_local_map(
                dataset.cur_pose_torch[:3, 3], None, frame_id
            )  # not efficient for large map

        T4 = get_time()

        # for the first frame, we need more iterations to do the initialization (warm-up)
        # 计算当前帧建图的迭代轮数
        cur_iter_num = (
            config.iters * config.init_iter_ratio if frame_id == 0 else config.iters
        )
        if dataset.stop_status:
            cur_iter_num = max(1, cur_iter_num - 10)
        #  在某一帧后固定解码器的参数
        if (
            frame_id == config.freeze_after_frame
        ):  # freeze the decoder after certain frame
            freeze_model(geo_mlp)

        # mapping with fixed poses (every frame)
        if frame_id % config.mapping_freq_frame == 0:
            mapper.mapping(cur_iter_num)

        T5 = get_time()

        # regular saving logs
        if config.log_freq_frame > 0 and (frame_id + 1) % config.log_freq_frame == 0:
            dataset.write_results_log()

        remove_gpu_cache()

        # IV: 网格重建和可视化
        if config.o3d_vis_on:
            if not q_vis2main.empty():
                control_packet: ControlPacket = get_latest_queue(q_vis2main)

                vis_visualize_on = control_packet.flag_vis
                vis_global_on = control_packet.flag_global
                vis_mesh_on = control_packet.flag_mesh
                vis_sdf_on = control_packet.flag_sdf
                vis_source_pc_weight = control_packet.flag_source
                vis_mesh_mc_res_m = control_packet.mc_res_m
                vis_mesh_min_nn = control_packet.mesh_min_nn
                vis_mesh_freq_frame = control_packet.mesh_freq_frame
                vis_sdf_slice_height = control_packet.sdf_slice_height
                vis_sdf_freq_frame = control_packet.sdf_freq_frame
                vis_sdf_res_m = control_packet.sdf_res_m

                while control_packet.flag_pause:
                    time.sleep(0.1)
                    if not q_vis2main.empty():
                        control_packet = get_latest_queue(q_vis2main)
                        if not control_packet.flag_pause:
                            break

            if vis_visualize_on:
                dataset.update_o3d_map()
                # Only PIN-SLAM has
                # if config.track_on and frame_id > 0 and vis_source_pc_weight and (weight_pc_o3d is not None):
                #     dataset.cur_frame_o3d = weight_pc_o3d

                # T7 = get_time()
                T6 = get_time()

                # reconstruction by marching cubes
                # Only PIN-SLAM has
                # if vis_mesh_on and (frame_id == 0 or frame_id == last_frame or (
                #         frame_id + 1) % vis_mesh_freq_frame == 0 or pgm.last_loop_idx == frame_id):
                if vis_mesh_on and (
                    frame_id == 0
                    or frame_id == last_frame
                    or (frame_id + 1) % vis_mesh_freq_frame == 0
                ):
                    # update map bbx
                    global_neural_pcd_down = neural_points.get_neural_points_o3d(
                        query_global=True, random_down_ratio=37
                    )  # prime number
                    dataset.map_bbx = (
                        global_neural_pcd_down.get_axis_aligned_bounding_box()
                    )

                    if not vis_global_on:  # only build the local mesh
                        chunks_aabb = split_chunks(
                            global_neural_pcd_down,
                            dataset.cur_bbx,
                            vis_mesh_mc_res_m * 100,
                        )  # reconstruct in chunks
                        cur_mesh = mesher.recon_aabb_collections_mesh(
                            chunks_aabb,
                            vis_mesh_mc_res_m,
                            None,
                            True,
                            config.semantic_on,
                            config.color_on,
                            filter_isolated_mesh=True,
                            mesh_min_nn=vis_mesh_min_nn,
                        )
                    else:
                        aabb = global_neural_pcd_down.get_axis_aligned_bounding_box()
                        chunks_aabb = split_chunks(
                            global_neural_pcd_down, aabb, vis_mesh_mc_res_m * 300
                        )  # reconstruct in chunks
                        cur_mesh = mesher.recon_aabb_collections_mesh(
                            chunks_aabb,
                            vis_mesh_mc_res_m,
                            None,
                            False,
                            config.semantic_on,
                            config.color_on,
                            filter_isolated_mesh=True,
                            mesh_min_nn=vis_mesh_min_nn,
                        )
                        # cur_sdf_slice = None

                if vis_sdf_on and (
                    frame_id == 0
                    or frame_id == last_frame
                    or (frame_id + 1) % vis_sdf_freq_frame == 0
                ):
                    sdf_bound = config.surface_sample_range_m * 4.0
                    vis_sdf_bbx = create_bbx_o3d(
                        dataset.cur_pose_ref[:3, 3], config.max_range / 2
                    )
                    cur_sdf_slice_h = mesher.generate_bbx_sdf_hor_slice(
                        vis_sdf_bbx,
                        dataset.cur_pose_ref[2, 3] + vis_sdf_slice_height,
                        vis_sdf_res_m,
                        True,
                        -sdf_bound,
                        sdf_bound,
                    )  # horizontal slice (local)
                    if config.vis_sdf_slice_v:
                        cur_sdf_slice_v = mesher.generate_bbx_sdf_ver_slice(
                            dataset.cur_bbx,
                            dataset.cur_pose_ref[0, 3],
                            vis_sdf_res_m,
                            True,
                            -sdf_bound,
                            sdf_bound,
                        )  # vertical slice (local)
                        cur_sdf_slice = cur_sdf_slice_h + cur_sdf_slice_v
                    else:
                        cur_sdf_slice = cur_sdf_slice_h

                pool_pcd = mapper.get_data_pool_o3d(down_rate=37)
                odom_poses, gt_poses, pgo_poses = dataset.get_poses_np_for_vis()
                loop_edges = None
                # Only PIN-SLAM has
                # loop_edges = pgm.loop_edges_vis if config.pgo_on else None

                packet_to_vis: VisPacket = VisPacket(
                    frame_id=frame_id, travel_dist=travel_dist[-1]
                )

                if not neural_points.is_empty():
                    packet_to_vis.add_neural_points_data(
                        neural_points,
                        only_local_map=(not vis_global_on),
                        pca_color_on=config.decoder_freezed,
                    )

                if dataset.cur_frame_o3d is not None:
                    packet_to_vis.add_scan(
                        np.array(dataset.cur_frame_o3d.points, dtype=np.float64),
                        np.array(dataset.cur_frame_o3d.colors, dtype=np.float64),
                    )

                if cur_mesh is not None:
                    packet_to_vis.add_mesh(
                        np.array(cur_mesh.vertices, dtype=np.float64),
                        np.array(cur_mesh.triangles),
                        np.array(cur_mesh.vertex_colors, dtype=np.float64),
                    )

                if cur_sdf_slice is not None:
                    packet_to_vis.add_sdf_slice(
                        np.array(cur_sdf_slice.points, dtype=np.float64),
                        np.array(cur_sdf_slice.colors, dtype=np.float64),
                    )

                if pool_pcd is not None:
                    packet_to_vis.add_sdf_training_pool(
                        np.array(pool_pcd.points, dtype=np.float64),
                        np.array(pool_pcd.colors, dtype=np.float64),
                    )

                packet_to_vis.add_traj(odom_poses, gt_poses, pgo_poses, loop_edges)

                q_main2vis.put(packet_to_vis)

                T8 = get_time()

                # if not config.silence:
                #     print("time for o3d update             (ms): {:.2f}".format((T7 - T6) * 1e3))
                #     print("time for visualization          (ms): {:.2f}".format((T8 - T7) * 1e3))

        cur_frame_process_time = np.array([T2 - T1, T3 - T2, T4 - T3, T5 - T4, 0])
        # cur_frame_process_time = np.array([T2 - T1, T3 - T2, T5 - T4, T6 - T5, T4 - T3])  # loop & pgo in the end, visualization and I/O time excluded
        dataset.time_table.append(cur_frame_process_time)  # in s

        if config.wandb_vis_on:
            wandb_log_content = {
                "frame": frame_id,
                "timing(s)/preprocess": T2 - T1,
                "timing(s)/tracking": T3 - T2,
                "timing(s)/pgo": T4 - T3,
                "timing(s)/mapping": T6 - T4,
            }
            wandb.log(wandb_log_content)

        dataset.processed_frame += 1

    # V. 保存结果
    mapper.free_pool()
    pose_eval_results = dataset.write_results()

    neural_points.prune_map(
        config.max_prune_certainty, 0, True
    )  # prune uncertain points for the final output
    neural_points.recreate_hash(
        None, None, False, False
    )  # merge the final neural point map
    neural_pcd = neural_points.get_neural_points_o3d(query_global=True, color_mode=0)
    if config.save_map:
        o3d.io.write_point_cloud(
            os.path.join(run_path, "map", "neural_points.ply"), neural_pcd
        )  # write the neural point cloud
    neural_points.clear_temp()  # clear temp data for output

    output_mc_res_m = config.mc_res_m * 0.6
    mc_cm_str = str(round(output_mc_res_m * 1e2))
    if config.save_mesh:
        chunks_aabb = split_chunks(
            neural_pcd,
            neural_pcd.get_axis_aligned_bounding_box(),
            output_mc_res_m * 300,
        )  # reconstruct in chunks
        mesh_path = os.path.join(run_path, "mesh", "mesh_" + mc_cm_str + "cm.ply")
        print("Reconstructing the global mesh with resolution {} cm".format(mc_cm_str))
        cur_mesh = mesher.recon_aabb_collections_mesh(
            chunks_aabb,
            output_mc_res_m,
            mesh_path,
            False,
            config.semantic_on,
            config.color_on,
            filter_isolated_mesh=True,
            mesh_min_nn=config.mesh_min_nn,
        )
        print("Reconstructing the global mesh done")
    neural_points.clear_temp()  # clear temp data for output
    if config.save_map:
        save_implicit_map(run_path, neural_points, mlp_dict)
        # lcd_npmc.save_context_dict(mapper.used_poses, run_path)
        print(
            "Use 'python vis_pin_map.py {} -m {} -o mesh_out_{}cm.ply' to inspect the map offline.".format(
                run_path, output_mc_res_m, mc_cm_str
            )
        )

    if config.save_merged_pc:
        dataset.write_merged_point_cloud()  # replay: save merged point cloud map

    remove_gpu_cache()

    if config.o3d_vis_on:
        while True:
            if not q_vis2main.empty():
                q_vis2main.get()

            packet_to_vis: VisPacket = VisPacket(
                frame_id=frame_id, travel_dist=travel_dist[-1], slam_finished=True
            )

            if not neural_points.is_empty():
                packet_to_vis.add_neural_points_data(
                    neural_points,
                    only_local_map=False,
                    pca_color_on=config.decoder_freezed,
                )

            if cur_mesh is not None:
                packet_to_vis.add_mesh(
                    np.array(cur_mesh.vertices, dtype=np.float64),
                    np.array(cur_mesh.triangles),
                    np.array(cur_mesh.vertex_colors, dtype=np.float64),
                )
                cur_mesh = None

            packet_to_vis.add_traj(odom_poses, gt_poses, pgo_poses, loop_edges)

            q_main2vis.put(packet_to_vis)
            time.sleep(1.0)

    return pose_eval_results


if __name__ == "__main__":
    run_slam()
