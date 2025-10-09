#!/usr/bin/env python3
# @file      rosbag2dataset_parallel.py
# @author    Junlong Jiang     [jiangjunlong@mail.dlut.edu.cn]
# Copyright (c) 2025 Junlong Jiang, all rights reserved
import sys
import csv
import os
import cv2
import yaml
import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from typing import List, Tuple
from plyfile import PlyData, PlyElement
from multiprocessing import Process, Queue

pc2.sys = sys


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


def compressed_image_to_numpy(msg) -> np.ndarray:
    """Convert sensor_msgs/CompressedImage to OpenCV BGR image (np.ndarray)."""
    # msg.format 可能是 "jpeg", "png", "rgb8; jpeg compressed", etc.
    np_arr = np.frombuffer(msg.data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode CompressedImage")
    return img


def sync_and_save(config: dict) -> None:
    """Synchronize and save LiDAR and IMU data from a ROS bag file."""
    os.makedirs(config["output_folder"], exist_ok=True)
    os.makedirs(os.path.join(config["output_folder"], "lidar"), exist_ok=True)
    os.makedirs(os.path.join(config["output_folder"], "imu"), exist_ok=True)
    os.makedirs(os.path.join(config["output_folder"], "image"), exist_ok=True)

    in_bag = rosbag.Bag(config["input_bag"])
    bridge = CvBridge()

    frame_index = 0
    image_index = 0
    start_flag = False
    imu_last_timestamp = None
    imu_data_pool = []
    lidar_timestamp_queue = Queue()

    processes = []
    batch_size = config["batch_size"]  # Number of messages per batch
    batch_lidar_data = []

    for topic, msg, t in in_bag.read_messages(
        topics=[config["imu_topic"], config["lidar_topic"], config["image_topic"]]
    ):
        current_timestamp = t.to_sec()

        if topic == config["image_topic"]:
            if not start_flag:
                # start_flag = True
                continue
            try:
                if msg._type == "sensor_msgs/Image":
                    img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                elif msg._type == "sensor_msgs/CompressedImage":
                    img = compressed_image_to_numpy(msg)
                else:
                    raise TypeError("Unsupported image type")
                img_path = os.path.join(
                    config["output_folder"], "image", f"{frame_index}.png"
                )
                cv2.imwrite(img_path, img)
                print(f"Exported image: {img_path}")
                image_index += 1
            except Exception as e:
                print(f"[Warning] Failed to extract image: {e}")

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

            lidar_timestamp_queue.put(msg.header.stamp.to_sec())

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
        os.path.join(config["output_folder"], "pose_ts.txt"), "w", newline=""
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
