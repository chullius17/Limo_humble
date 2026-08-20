#!/usr/bin/env python3

import math
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
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
from sensor_msgs.msg import Image, LaserScan, PointCloud2
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
            '/limo/nav_map_package/online/laser_cv_fusion/scan',
        )
        self.declare_parameter(
            'debug_topic',
            '/limo/nav_map_package/online/laser_cv_fusion/debug',
        )
        self.declare_parameter('max_cv_age_sec', 0.5)
        self.declare_parameter('transform_timeout_sec', 0.05)
        self.declare_parameter('publish_debug', True)
        self.declare_parameter('debug_image_size', 600)
        self.declare_parameter('debug_resolution', 0.01)
        self.declare_parameter('debug_point_radius', 1)

        scan_topic = self.get_parameter('scan_topic').value
        cv_topic = self.get_parameter('cv_cloud_topic').value
        output_topic = self.get_parameter('output_topic').value
        debug_topic = self.get_parameter('debug_topic').value

        cloud_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()
        self.latest_cv_points = None
        self.latest_cv_stamp = None
        self.latest_cv_frame = None
        self.last_tf_warning = 0.0

        self.scan_pub = self.create_publisher(
            LaserScan, output_topic, cloud_qos
        )
        self.debug_pub = self.create_publisher(Image, debug_topic, cloud_qos)
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, qos_profile_sensor_data
        )
        self.cv_sub = self.create_subscription(
            PointCloud2, cv_topic, self.cv_callback, cloud_qos
        )

        self.get_logger().info(
            f'Fusing {scan_topic} and {cv_topic} into {output_topic} '
            'using the input LaserScan frame'
        )

    @staticmethod
    def _stamp_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _warn_tf(self, message):
        now = time.monotonic()
        if now - self.last_tf_warning >= 2.0:
            self.get_logger().warning(message)
            self.last_tf_warning = now

    def _lookup_transform(self, target_frame, source_frame, stamp):
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

        requested_fields = ('x', 'y', 'z')
        values = point_cloud2.read_points(
            msg, field_names=requested_fields, skip_nans=True
        )
        if isinstance(values, np.ndarray) and values.dtype.names:
            xyz = np.column_stack((
                np.asarray(values['x']).reshape(-1),
                np.asarray(values['y']).reshape(-1),
                np.asarray(values['z']).reshape(-1),
            )).astype(np.float64, copy=False)
        else:
            values = list(values)
            if values:
                xyz = np.asarray(values, dtype=np.float64)[:, :3]
            else:
                xyz = np.empty((0, 3), dtype=np.float64)

        self.latest_cv_points = xyz
        self.latest_cv_stamp = self._stamp_seconds(msg.header.stamp)
        self.latest_cv_frame = msg.header.frame_id

    def scan_callback(self, msg):
        fused_ranges = np.asarray(msg.ranges, dtype=np.float64).copy()
        invalid = ~np.isfinite(fused_ranges)
        fused_ranges[invalid] = math.inf

        cv_points = None
        if (
            self.latest_cv_points is not None
            and self.latest_cv_stamp is not None
            and self.latest_cv_frame
        ):
            age = abs(
                self._stamp_seconds(msg.header.stamp) - self.latest_cv_stamp
            )
            max_age = float(self.get_parameter('max_cv_age_sec').value)
            if max_age < 0.0 or age <= max_age:
                cv_points = self.latest_cv_points

        if cv_points is not None and cv_points.size:
            try:
                transform = self._lookup_transform(
                    msg.header.frame_id,
                    self.latest_cv_frame,
                    msg.header.stamp,
                )
            except TransformException:
                transform = None
                cv_points = None

            if cv_points is not None:
                cv_points = self._transform_points(cv_points, transform)
                cv_ranges = np.hypot(cv_points[:, 0], cv_points[:, 1])
                cv_angles = np.arctan2(cv_points[:, 1], cv_points[:, 0])
                bin_indices = np.rint(
                    (cv_angles - float(msg.angle_min))
                    / float(msg.angle_increment)
                ).astype(np.int64)
                valid = (
                    (bin_indices >= 0)
                    & (bin_indices < fused_ranges.size)
                    & np.isfinite(cv_ranges)
                    & (cv_ranges >= float(msg.range_min))
                    & (cv_ranges <= float(msg.range_max))
                )
                np.minimum.at(
                    fused_ranges, bin_indices[valid], cv_ranges[valid]
                )

        output = LaserScan()
        output.header = msg.header
        output.angle_min = msg.angle_min
        output.angle_max = msg.angle_max
        output.angle_increment = msg.angle_increment
        output.time_increment = msg.time_increment
        output.scan_time = msg.scan_time
        output.range_min = msg.range_min
        output.range_max = msg.range_max
        output.ranges = fused_ranges.tolist()
        output.intensities = list(msg.intensities)
        self.scan_pub.publish(output)

        if self.get_parameter('publish_debug').value:
            self._publish_debug_image(msg, fused_ranges)

    def _publish_debug_image(self, scan, ranges):
        image_size = max(
            1, int(self.get_parameter('debug_image_size').value)
        )
        resolution = float(self.get_parameter('debug_resolution').value)
        if resolution <= 0.0:
            self.get_logger().warning('debug_resolution must be positive')
            return

        indices = np.arange(ranges.size, dtype=np.float64)
        angles = float(scan.angle_min) + indices * float(scan.angle_increment)
        valid = (
            np.isfinite(ranges)
            & (ranges >= float(scan.range_min))
            & (ranges <= float(scan.range_max))
        )
        x = ranges[valid] * np.cos(angles[valid])
        y = ranges[valid] * np.sin(angles[valid])

        center = image_size // 2
        rows = np.rint(center - x / resolution).astype(np.int64)
        cols = np.rint(center - y / resolution).astype(np.int64)
        visible = (
            (rows >= 0)
            & (rows < image_size)
            & (cols >= 0)
            & (cols < image_size)
        )
        rows = rows[visible]
        cols = cols[visible]

        image = np.zeros((image_size, image_size), dtype=np.uint8)
        radius = max(
            0, int(self.get_parameter('debug_point_radius').value)
        )
        for row_offset in range(-radius, radius + 1):
            for col_offset in range(-radius, radius + 1):
                point_rows = rows + row_offset
                point_cols = cols + col_offset
                inside = (
                    (point_rows >= 0)
                    & (point_rows < image_size)
                    & (point_cols >= 0)
                    & (point_cols < image_size)
                )
                image[point_rows[inside], point_cols[inside]] = 255

        debug_msg = self.bridge.cv2_to_imgmsg(image, encoding='mono8')
        debug_msg.header = scan.header
        self.debug_pub.publish(debug_msg)


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
