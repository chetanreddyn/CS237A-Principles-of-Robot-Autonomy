#!/usr/bin/env python3

import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformListener

from icp_utils import icp, open3d_icp


def get_pcd_array_from_point_cloud(pcd_msg: PointCloud2):
    pcd = point_cloud2.read_points_list(pcd_msg, field_names=["x", "y", "z"], skip_nans=True)
    return np.array(pcd)


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    rotation_matrix = np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)]
    ])
    return rotation_matrix


class ICPNode(Node):
    def __init__(self):
        super().__init__('icp_node')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.2, self.get_tfs)

        self.prev_pcd = None
        self.pose = None
        self.lidar_pose = None
        self.transformation = np.eye(4)
        self.use_open3d = True
        
        self.pcd_sub = self.create_subscription(PointCloud2, "/velodyne_points", self.pcd_callback, 10)
        self.marker_pub = self.create_publisher(Marker, "/marker", 10)
        
        self.marker = Marker()
        self.marker.header.frame_id = "odom"
        self.marker.ns = "icp"
        self.marker.type = 2
        self.marker.pose.orientation.w = 1.0
        self.marker.scale.x = 0.1
        self.marker.scale.y = 0.1
        self.marker.scale.z = 0.1
        self.marker.color.r = 1.0
        self.marker.color.g = 0.0
        self.marker.color.b = 0.0
        self.marker.color.a = 1.0

    def get_tfs(self):
        try:
            init_transform = self.tf_buffer.lookup_transform(target_frame='odom', source_frame='base_footprint', time=Time())
            
            translation = init_transform.transform.translation
            tx, ty, tz = translation.x, translation.y, translation.z
            
            rotation = init_transform.transform.rotation
            qx, qy, qz, qw = rotation.x, rotation.y, rotation.z, rotation.w
            rotation_matrix = quaternion_to_rotation_matrix(qx, qy, qz, qw)

            self.pose = np.eye(4)
            self.pose[:3, :3] = rotation_matrix
            self.pose[:3, 3] = [tx, ty, tz]
            self.get_logger().info('Set initial pose.')
            
            lidar_transform = self.tf_buffer.lookup_transform(target_frame='base_footprint', source_frame='velodyne', time=Time())
            
            lidar_translation = lidar_transform.transform.translation
            lx, ly, lz = lidar_translation.x, lidar_translation.y, lidar_translation.z

            lidar_rotation = lidar_transform.transform.rotation
            l_qx, l_qy, l_qz, l_qw = lidar_rotation.x, lidar_rotation.y, lidar_rotation.z, lidar_rotation.w
            lidar_rotation_matrix = quaternion_to_rotation_matrix(l_qx, l_qy, l_qz, l_qw)

            self.lidar_pose = np.eye(4)
            self.lidar_pose[:3, :3] = lidar_rotation_matrix
            self.lidar_pose[:3, 3] = [lx, ly, lz]
            self.get_logger().info('Set lidar pose.')
            
            self.timer.cancel()
            
        except Exception as e:
            self.pose = None
            self.lidar_pose = None

    def pcd_callback(self, msg: PointCloud2):
        if self.pose is None or self.lidar_pose is None:
            return
        
        if self.prev_pcd is None:
            pts = get_pcd_array_from_point_cloud(msg)
            self.prev_pcd = o3d.geometry.PointCloud()
            self.prev_pcd.points = o3d.utility.Vector3dVector(pts)
            self.prev_pcd = self.prev_pcd.uniform_down_sample(10)
            self.get_logger().info('Initial point cloud received.')
            return

        ### TODO: Task 3.5 ###
        pts = get_pcd_array_from_point_cloud(msg)
        pts_tf = self.lidar_pose @ np.block([[pts.T], [np.ones(len(pts))]])
        pts_tf = (pts_tf[:3,:]).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_tf)
        pcd = pcd.uniform_down_sample(10)
        ### Task 3.5 ###

        ### TODO: Task 3.6 ###
        if self.use_open3d:
            self.transformation = open3d_icp(pcd, self.prev_pcd, self.transformation)
        else:
            self.transformation = icp(np.asarray(pcd.points), np.asarray(self.prev_pcd.points), self.transformation)
        ### Task 3.6 ###
        
        ### TODO: Task 3.7 ###
        self.pose = self.pose @ self.transformation
        ### Task 3.7 ###

        self.marker.header.stamp = self.get_clock().now().to_msg()
        ### TODO: Task 3.8 ###
        self.marker.pose.position.x = self.pose[0, 3]
        self.marker.pose.position.y = self.pose[1, 3]
        self.marker_pub.publish(self.marker)
        self.prev_pcd = pcd
        ### Task 3.8 ###


def main(args=None):
    rclpy.init(args=args)
    node = ICPNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
