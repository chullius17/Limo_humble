#!/usr/bin/env python3

import math
import time
from functools import partial

import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener


class CvToPointCloud(Node):
    """Convert obstacle and street metric BEV grids to separate clouds."""

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
        self.declare_parameter(
            'street_input_topic',
            '/limo/nav_map_package/online/metric_bev/cost_grid_blue',
        )
        self.declare_parameter(
            'street_output_topic',
            '/limo/nav_map_package/online/cv_2_ptcld/street_points',
        )
        self.declare_parameter(
            'street_debug_topic',
            '/limo/nav_map_package/online/cv_2_ptcld/street_debug',
        )
        self.declare_parameter('cost_threshold', 40.0)
        self.declare_parameter('point_z', 0.0)
        self.declare_parameter('publish_debug', True)
        self.declare_parameter('output_frame', 'base_link')
        self.declare_parameter('transform_timeout_sec', 0.05)

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        debug_topic = self.get_parameter('debug_topic').value
        street_input_topic = self.get_parameter('street_input_topic').value
        street_output_topic = self.get_parameter('street_output_topic').value
        street_debug_topic = self.get_parameter('street_debug_topic').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.last_tf_warning = 0.0

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
        self.street_points_pub = self.create_publisher(
            PointCloud2,
            street_output_topic,
            cloud_qos,
        )
        self.street_debug_pub = self.create_publisher(
            Image,
            street_debug_topic,
            debug_qos,
        )
        self.grid_sub = self.create_subscription(
            OccupancyGrid,
            input_topic,
            partial(
                self.grid_callback,
                points_pub=self.points_pub,
                debug_pub=self.debug_pub,
                source_name='obstacle',
            ),
            input_qos,
        )
        self.street_grid_sub = self.create_subscription(
            OccupancyGrid,
            street_input_topic,
            partial(
                self.grid_callback,
                points_pub=self.street_points_pub,
                debug_pub=self.street_debug_pub,
                source_name='street',
            ),
            input_qos,
        )

        self.get_logger().info(
            f'Converting cells from {input_topic} to {output_topic} in frame '
            f'{self.get_parameter("output_frame").value}'
        )
        self.get_logger().info(
            f'Converting street cells from {street_input_topic} to '
            f'{street_output_topic} in frame '
            f'{self.get_parameter("output_frame").value}'
        )

    def _lookup_transform(self, target_frame, source_frame, stamp):
        if not source_frame:
            raise TransformException(
                'Input occupancy grid has an empty frame_id'
            )
        if source_frame == target_frame:
            return None

        timeout = Duration(
            seconds=float(
                self.get_parameter('transform_timeout_sec').value
            )
        )
        try:
            return self.tf_buffer.lookup_transform(
                target_frame, source_frame, Time.from_msg(stamp), timeout
            )
        except TransformException:
            # Sensor messages can be a few milliseconds ahead of the latest TF.
            return self.tf_buffer.lookup_transform(
                target_frame, source_frame, Time(), timeout
            )

    @staticmethod
    def _transform_xyz(xyz, transform):
        if transform is None or xyz.size == 0:
            return xyz

        translation = transform.transform.translation
        quaternion = transform.transform.rotation
        norm = math.sqrt(
            quaternion.x * quaternion.x
            + quaternion.y * quaternion.y
            + quaternion.z * quaternion.z
            + quaternion.w * quaternion.w
        )
        if norm == 0.0:
            raise ValueError('TF contains a zero-norm quaternion')

        qx = quaternion.x / norm
        qy = quaternion.y / norm
        qz = quaternion.z / norm
        qw = quaternion.w / norm
        rotation = np.array([
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ], dtype=np.float64)
        offset = np.array(
            [translation.x, translation.y, translation.z],
            dtype=np.float64,
        )
        return xyz @ rotation.T + offset

    def _warn_transform(self, message):
        now = time.monotonic()
        if now - self.last_tf_warning >= 2.0:
            self.get_logger().warning(message)
            self.last_tf_warning = now

    def grid_callback(
        self,
        msg: OccupancyGrid,
        points_pub,
        debug_pub,
        source_name: str,
    ):
        """Convert one labeled cost grid into its corresponding point cloud."""
        width = msg.info.width
        height = msg.info.height
        expected_size = width * height
        if width == 0 or height == 0 or len(msg.data) != expected_size:
            self.get_logger().warning(
                f'Ignoring malformed {source_name} occupancy grid: '
                f'{width}x{height}, {len(msg.data)} cells'
            )
            return

        threshold = float(self.get_parameter('cost_threshold').value)
        point_z = float(self.get_parameter('point_z').value)
        costs = np.asarray(msg.data, dtype=np.int16).reshape(height, width)
        rows, cols = np.nonzero(costs > threshold)

        if (
            self.get_parameter('publish_debug').value
            and debug_pub.get_subscription_count() > 0
        ):
            debug_image = np.zeros((height, width), dtype=np.uint8)
            debug_image[rows, cols] = 255
            debug_image = np.rot90(np.flipud(debug_image), k=1)
            debug_msg = self.bridge.cv2_to_imgmsg(
                np.ascontiguousarray(debug_image),
                encoding='mono8',
            )
            debug_msg.header = msg.header
            debug_pub.publish(debug_msg)

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

        output_frame = str(self.get_parameter('output_frame').value)
        if not output_frame:
            self._warn_transform('Cannot publish cloud: output_frame is empty')
            return
        try:
            transform = self._lookup_transform(
                output_frame, msg.header.frame_id, msg.header.stamp
            )
            xyz = self._transform_xyz(
                np.column_stack((
                    x,
                    y,
                    np.full(rows.size, point_z, dtype=np.float64),
                )),
                transform,
            )
        except (TransformException, ValueError) as error:
            self._warn_transform(
                f'Cannot transform {msg.header.frame_id} to '
                f'{output_frame}: {error}'
            )
            return

        selected_costs = costs[rows, cols].astype(np.float32)
        points = np.column_stack((
            xyz.astype(np.float32),
            selected_costs,
        ))

        cloud_header = Header()
        cloud_header.stamp = msg.header.stamp
        cloud_header.frame_id = output_frame

        cloud = point_cloud2.create_cloud(
            cloud_header,
            self.fields,
            points.tolist(),
        )
        points_pub.publish(cloud)


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
