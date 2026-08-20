#!/usr/bin/env python3

import math

import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2


class CvToPointCloud(Node):
    """Convert high-cost cells from the combined metric BEV to PointCloud2."""

    def __init__(self):
        super().__init__('cv_2_ptcld')
        self.bridge = CvBridge()

        self.declare_parameter(
            'input_topic',
            '/limo/nav_map_package/online/metric_bev/cost_grid_combined',
        )
        self.declare_parameter(
            'output_topic',
            '/limo/nav_map_package/online/cv_2_ptcld/points',
        )
        self.declare_parameter(
            'debug_topic',
            '/limo/nav_map_package/online/cv_2_ptcld/debug',
        )
        self.declare_parameter('cost_threshold', 40.0)
        self.declare_parameter('point_z', 0.0)
        self.declare_parameter('publish_debug', True)

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        debug_topic = self.get_parameter('debug_topic').value

        input_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        cloud_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        debug_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='cost', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        self.points_pub = self.create_publisher(
            PointCloud2,
            output_topic,
            cloud_qos,
        )
        self.debug_pub = self.create_publisher(
            Image,
            debug_topic,
            debug_qos,
        )
        self.grid_sub = self.create_subscription(
            OccupancyGrid,
            input_topic,
            self.grid_callback,
            input_qos,
        )

        self.get_logger().info(
            f'Converting cells from {input_topic} to {output_topic}'
        )

    def grid_callback(self, msg: OccupancyGrid):
        width = msg.info.width
        height = msg.info.height
        expected_size = width * height
        if width == 0 or height == 0 or len(msg.data) != expected_size:
            self.get_logger().warning(
                'Ignoring malformed occupancy grid: '
                f'{width}x{height}, {len(msg.data)} cells'
            )
            return

        threshold = float(self.get_parameter('cost_threshold').value)
        point_z = float(self.get_parameter('point_z').value)
        costs = np.asarray(msg.data, dtype=np.int16).reshape(height, width)
        rows, cols = np.nonzero(costs > threshold)

        if (
            self.get_parameter('publish_debug').value
            and self.debug_pub.get_subscription_count() > 0
        ):
            debug_image = np.zeros((height, width), dtype=np.uint8)
            debug_image[rows, cols] = 255
            debug_image = np.rot90(np.flipud(debug_image), k=1)
            debug_msg = self.bridge.cv2_to_imgmsg(
                np.ascontiguousarray(debug_image),
                encoding='mono8',
            )
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)

        resolution = msg.info.resolution
        local_x = (cols.astype(np.float64) + 0.5) * resolution
        local_y = (rows.astype(np.float64) + 0.5) * resolution

        origin = msg.info.origin
        quaternion = origin.orientation
        sin_yaw = 2.0 * (
            quaternion.w * quaternion.z + quaternion.x * quaternion.y
        )
        cos_yaw = 1.0 - 2.0 * (
            quaternion.y * quaternion.y + quaternion.z * quaternion.z
        )
        yaw = math.atan2(sin_yaw, cos_yaw)

        cos_angle = math.cos(yaw)
        sin_angle = math.sin(yaw)
        x = origin.position.x + cos_angle * local_x - sin_angle * local_y
        y = origin.position.y + sin_angle * local_x + cos_angle * local_y

        selected_costs = costs[rows, cols].astype(np.float32)
        points = np.column_stack((
            x.astype(np.float32),
            y.astype(np.float32),
            np.full(rows.size, point_z, dtype=np.float32),
            selected_costs,
        ))

        cloud = point_cloud2.create_cloud(
            msg.header,
            self.fields,
            points.tolist(),
        )
        self.points_pub.publish(cloud)


def main(args=None):
    rclpy.init(args=args)
    node = CvToPointCloud()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
