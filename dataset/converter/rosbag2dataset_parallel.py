#!/usr/bin/env python3
# @file      rosbag2dataset_parallel.py
# @author    Junlong Jiang     [jiangjunlong@mail.dlut.edu.cn]
# Copyright (c) 2025 Junlong Jiang, all rights reserved
import csv
import os
import yaml
from multiprocessing import Process, Queue
from typing import List, Tuple

import numpy as np
import rosbag
import sensor_msgs.point_cloud2 as pc2
from plyfile import PlyData, PlyElement

G_M_S2 = 9.81  # Gravitational constant in m/s^2


def load_config(path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(path, "r") as file:
        return yaml.safe_load(file)


def write_ply(filename: str, data: tuple) -> bool:
    """Writes point cloud data along with timestamps to a PLY file."""
    # Ensure timestamp data is a 2D array with one column
    points, timestamps = data
    combined_data = np.hstack([points, timestamps.reshape(-1, 1)])
    structured_array = np.core.records.fromarrays(
        combined_data.transpose(), names=["x", "y", "z", "intensity", "timestamp"]
    )
    PlyData([PlyElement.describe(structured_array, "vertex")], text=False).write(
        filename
    )
    return True


def write_csv(
    filename: str,
    imu_data_pool: List[Tuple[float, float, float, float, float, float, float]],
) -> None:
    """Write IMU data to a CSV file."""
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        )
        for imu_data in imu_data_pool:
            writer.writerow(imu_data)


def extract_lidar_data(msg) -> Tuple[np.ndarray, np.ndarray]:
    """Extract point cloud data and timestamps from a LiDAR message."""
    pc_data = list(pc2.read_points(msg, skip_nans=True))
    pc_array = np.array(pc_data)
    timestamps = pc_array[:, 4] * 1e-9  # Convert to seconds
    return pc_array[:, :4], timestamps


def process_lidar_data(
    batch_data: List[Tuple[str, Tuple[np.ndarray, np.ndarray]]],
) -> None:
    """Process a batch of LiDAR data and save as PLY files."""
    for i, (ply_file_path, data) in enumerate(batch_data):
        if write_ply(ply_file_path, data):
            print(f"Exported LiDAR point cloud PLY file: {ply_file_path}")


def sync_and_save(config: dict) -> None:
    """Synchronize and save LiDAR and IMU data from a ROS bag file."""
    os.makedirs(config["output_folder"], exist_ok=True)
    os.makedirs(os.path.join(config["output_folder"], "lidar"), exist_ok=True)
    os.makedirs(os.path.join(config["output_folder"], "imu"), exist_ok=True)

    in_bag = rosbag.Bag(config["input_bag"])

    frame_index = 0
    start_flag = False
    imu_last_timestamp = None
    imu_data_pool = []
    lidar_timestamp_queue = Queue()

    processes = []
    batch_size = config["batch_size"]  # Number of messages per batch
    batch_lidar_data = []

    for topic, msg, t in in_bag.read_messages(
        topics=[config["imu_topic"], config["lidar_topic"]]
    ):
        current_timestamp = t.to_sec()

        if topic == config["lidar_topic"]:
            if not start_flag:
                start_flag = True
            else:
                csv_file_path = os.path.join(
                    config["output_folder"], "imu", f"{frame_index}.csv"
                )
                write_csv(csv_file_path, imu_data_pool)
                imu_data_pool = []
                print(f"Exported IMU measurement CSV file: {csv_file_path}")

            if len(batch_lidar_data) >= batch_size:
                p = Process(target=process_lidar_data, args=(batch_lidar_data,))
                p.start()
                processes.append(p)
                batch_lidar_data = []

            lidar_frame_timestamp = msg.header.stamp.to_sec()
            lidar_timestamp_queue.put(lidar_frame_timestamp)

            ply_file_path = os.path.join(
                config["output_folder"], "lidar", f"{frame_index}.ply"
            )
            point_cloud_data = extract_lidar_data(msg)
            batch_lidar_data.append((ply_file_path, point_cloud_data))

            imu_last_timestamp = current_timestamp
            frame_index += 1

            if 0 < config["end_frame"] <= frame_index:
                break

        elif topic == config["imu_topic"]:
            if start_flag:
                time_delta = current_timestamp - imu_last_timestamp
                imu_last_timestamp = current_timestamp
                imu_data = (
                    time_delta,
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z,
                )
                imu_data_pool.append(imu_data)

    if batch_lidar_data:
        p = Process(target=process_lidar_data, args=(batch_lidar_data,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    with open(
        os.path.join(config["output_folder"], "pose_timestamps.txt"), "w", newline=""
    ) as file:
        print("Writing pose timestamps...")
        writer = csv.writer(file)
        writer.writerow(["timestamp"])
        while not lidar_timestamp_queue.empty():
            lidar_timestamp = lidar_timestamp_queue.get()
            writer.writerow([lidar_timestamp])
        print("Pose timestamps written successfully.")


if __name__ == "__main__":
    config = load_config("./dataset/converter/config/rosbag2dataset.yaml")
    sync_and_save(config)
