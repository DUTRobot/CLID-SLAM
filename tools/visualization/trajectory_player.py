#!/usr/bin/env python

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# 用于可视化播放轨迹

def read_tum_file(filename):
    poses = []
    with open(filename, 'r') as file:
        for line in file:
            data = line.strip().split()
            if len(data) >= 8:
                timestamp = float(data[0])
                tx = float(data[1])
                ty = float(data[2])
                tz = float(data[3])
                qx = float(data[4])
                qy = float(data[5])
                qz = float(data[6])
                qw = float(data[7])
                poses.append((timestamp, tx, ty, tz, qx, qy, qz, qw))
    return poses


def publish_trajectories(true_poses, predict_poses):
    true_pub = rospy.Publisher('true_trajectory_marker', Marker, queue_size=10)
    predict_pub = rospy.Publisher('predict_trajectory_marker', Marker, queue_size=10)
    rospy.init_node('vis_node', anonymous=True)
    rate = rospy.Rate(200)  # 10 Hz

    true_marker = Marker()
    true_marker.header.frame_id = "map"
    true_marker.type = Marker.LINE_STRIP
    true_marker.action = Marker.ADD
    true_marker.scale.x = 0.2  # Line width
    true_marker.color.a = 1.0  # Alpha channel
    true_marker.color.r = 0.8  # Red color

    predict_marker = Marker()
    predict_marker.header.frame_id = "map"
    predict_marker.type = Marker.LINE_STRIP
    predict_marker.action = Marker.ADD
    predict_marker.scale.x = 0.2  # Line width
    predict_marker.color.a = 1.0  # Alpha channel
    predict_marker.color.g = 0.8  # Blue color

    for true_pose, predict_pose in zip(true_poses, predict_poses):
        # True trajectory
        true_timestamp, tx, ty, tz, qx, qy, qz, qw = true_pose
        true_point = Point()
        true_point.x = tx
        true_point.y = ty
        true_point.z = tz
        true_marker.points.append(true_point)

        # Publish the markers
        true_pub.publish(true_marker)
        # Predicted trajectory (predict)
        predict_timestamp, otx, oty, otz, oqx, oqy, oqz, oqw = predict_pose
        predict_point = Point()
        predict_point.x = otx
        predict_point.y = oty
        predict_point.z = otz
        predict_marker.points.append(predict_point)
        predict_pub.publish(predict_marker)

        rate.sleep()

if __name__ == '__main__':
    true_poses = (
        read_tum_file('./gt_poses_tum_vis.txt'))
    predict_poses = (
        read_tum_file('/./odom_poses_tum_vis.txt'))
    publish_trajectories(true_poses, predict_poses)

