#!/usr/bin/env python3

import math
import time

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener


class LaserCvFusion(Node):
    """Project LaserScan points and merge them with the latest CV cloud."""

    def __init__(self):
        super().__init__('laser_cv_fusion')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter(
            'cv_cloud_topic',
            '/limo/nav_map_package/online/cv_2_ptcld/points',
        )
        self.declare_parameter(
            'output_topic',
            '/limo/nav_map_package/online/laser_cv_fusion/points',
        )
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('max_cv_age_sec', 0.5)
        self.declare_parameter('laser_cost', 100.0)
        self.declare_parameter('transform_timeout_sec', 0.05)

        scan_topic = self.get_parameter('scan_topic').value
        cv_topic = self.get_parameter('cv_cloud_topic').value
        output_topic = self.get_parameter('output_topic').value

        cloud_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(
                name='cost', offset=12, datatype=PointField.FLOAT32, count=1
            ),
            PointField(
                name='source', offset=16, datatype=PointField.FLOAT32, count=1
            ),
        ]

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.latest_cv_points = None
        self.latest_cv_stamp = None
        self.last_tf_warning = 0.0

        self.cloud_pub = self.create_publisher(
            PointCloud2, output_topic, cloud_qos
        )
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, qos_profile_sensor_data
        )
        self.cv_sub = self.create_subscription(
            PointCloud2, cv_topic, self.cv_callback, cloud_qos
        )

        self.get_logger().info(
            f'Fusing {scan_topic} and {cv_topic} into {output_topic} '
            f'in frame {self.get_parameter("target_frame").value}'
        )

    @staticmethod
    def _stamp_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _warn_tf(self, message):
        now = time.monotonic()
        if now - self.last_tf_warning >= 2.0:
            self.get_logger().warning(message)
            self.last_tf_warning = now

    def _lookup_transform(self, source_frame, stamp):
        target_frame = self.get_parameter('target_frame').value
        if not source_frame or source_frame == target_frame:
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
            # Gazebo can publish sensors faster than /clock; latest TF avoids
            # rejecting otherwise valid samples a few milliseconds in future.
            try:
                return self.tf_buffer.lookup_transform(
                    target_frame, source_frame, Time(), timeout
                )
            except TransformException as error:
                self._warn_tf(
                    f'Cannot transform {source_frame} to {target_frame}: '
                    f'{error}'
                )
                raise

    @staticmethod
    def _transform_points(points, transform):
        if transform is None or points.size == 0:
            return points

        translation = transform.transform.translation
        q = transform.transform.rotation
        norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        if norm == 0.0:
            return points
        x, y, z, w = q.x / norm, q.y / norm, q.z / norm, q.w / norm
        rotation = np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
             2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
             2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
             1.0 - 2.0 * (x * x + y * y)],
        ], dtype=np.float64)
        offset = np.array(
            [translation.x, translation.y, translation.z], dtype=np.float64
        )
        return points @ rotation.T + offset

    def cv_callback(self, msg):
        field_names = {field.name for field in msg.fields}
        if not {'x', 'y', 'z'}.issubset(field_names):
            self.get_logger().warning('CV cloud has no x, y, z fields')
            return

        has_cost = 'cost' in field_names
        requested_fields = ('x', 'y', 'z', 'cost') if has_cost else ('x', 'y', 'z')
        values = list(point_cloud2.read_points(
            msg, field_names=requested_fields, skip_nans=True
        ))
        if values:
            data = np.asarray(values, dtype=np.float64)
            xyz = data[:, :3]
            costs = data[:, 3] if has_cost else np.zeros(data.shape[0])
        else:
            xyz = np.empty((0, 3), dtype=np.float64)
            costs = np.empty(0, dtype=np.float64)

        try:
            transform = self._lookup_transform(msg.header.frame_id, msg.header.stamp)
        except TransformException:
            return
        xyz = self._transform_points(xyz, transform)
        self.latest_cv_points = np.column_stack((
            xyz,
            costs,
            np.ones(xyz.shape[0], dtype=np.float64),
        )).astype(np.float32)
        self.latest_cv_stamp = self._stamp_seconds(msg.header.stamp)

    def scan_callback(self, msg):
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        valid = (
            np.isfinite(ranges)
            & (ranges >= float(msg.range_min))
            & (ranges <= float(msg.range_max))
        )
        indices = np.flatnonzero(valid)
        angles = float(msg.angle_min) + indices * float(msg.angle_increment)
        distances = ranges[indices]
        xyz = np.column_stack((
            distances * np.cos(angles),
            distances * np.sin(angles),
            np.zeros(indices.size, dtype=np.float64),
        ))

        try:
            transform = self._lookup_transform(msg.header.frame_id, msg.header.stamp)
        except TransformException:
            return
        xyz = self._transform_points(xyz, transform)
        laser_points = np.column_stack((
            xyz,
            np.full(
                xyz.shape[0],
                float(self.get_parameter('laser_cost').value),
                dtype=np.float64,
            ),
            np.zeros(xyz.shape[0], dtype=np.float64),
        )).astype(np.float32)

        point_sets = [laser_points]
        if self.latest_cv_points is not None and self.latest_cv_stamp is not None:
            age = abs(
                self._stamp_seconds(msg.header.stamp) - self.latest_cv_stamp
            )
            max_age = float(self.get_parameter('max_cv_age_sec').value)
            if max_age < 0.0 or age <= max_age:
                point_sets.append(self.latest_cv_points)

        fused_points = np.concatenate(point_sets, axis=0)
        header = msg.header
        header.frame_id = self.get_parameter('target_frame').value
        cloud = point_cloud2.create_cloud(
            header, self.fields, fused_points.tolist()
        )
        self.cloud_pub.publish(cloud)


def main(args=None):
    rclpy.init(args=args)
    node = LaserCvFusion()
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
