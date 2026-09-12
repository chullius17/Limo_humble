#!/usr/bin/env python3
"""Transform a metric boundary cloud and project it onto a planar frame."""

from collections import deque
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformException, TransformListener


class SimpleBev(Node):
    """Express a PointCloud2 in the BEV frame and flatten it onto its plane."""

    def __init__(self):
        super().__init__('simple_bev')

        self.declare_parameter(
            'pointcloud_topic', 'limo/cv_package/boundaries/points')
        self.declare_parameter(
            'bev_pointcloud_topic', 'limo/cv_package/bev/points')
        self.declare_parameter('plane_frame', 'base_link')
        self.declare_parameter('plane_z', 0.0)
        self.declare_parameter('enable_telemetry', True)
        self.declare_parameter('telemetry_window_size', 60)
        self.declare_parameter('telemetry_log_interval_frames', 30)

        input_topic = self.get_parameter('pointcloud_topic').value
        output_topic = self.get_parameter('bev_pointcloud_topic').value
        self.plane_frame = self.get_parameter('plane_frame').value
        self.plane_z = float(self.get_parameter('plane_z').value)
        self.telemetry_enabled = bool(
            self.get_parameter('enable_telemetry').value)
        self.telemetry_window_size = int(
            self.get_parameter('telemetry_window_size').value)
        self.telemetry_interval = int(
            self.get_parameter('telemetry_log_interval_frames').value)

        if not self.plane_frame:
            raise ValueError('plane_frame cannot be empty')
        if not np.isfinite(self.plane_z):
            raise ValueError('plane_z must be finite')
        if self.telemetry_window_size <= 0 or self.telemetry_interval <= 0:
            raise ValueError('Telemetry window and interval must be positive')

        cloud_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.publisher = self.create_publisher(
            PointCloud2, output_topic, cloud_qos)
        self.subscription = self.create_subscription(
            PointCloud2, input_topic, self.pointcloud_callback, cloud_qos)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.received_count = 0
        self.published_count = 0
        self.tf_drop_count = 0
        self.point_counts = deque(maxlen=self.telemetry_window_size)
        self.processing_times = deque(maxlen=self.telemetry_window_size)

        self.get_logger().info(
            f'Simple BEV ready: {input_topic} -> {output_topic}, '
            f'plane {self.plane_frame} at z={self.plane_z:.3f} m.')

    @staticmethod
    def quaternion_to_rotation(quaternion):
        """Return the rotation matrix represented by a ROS quaternion."""
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w
        norm = x * x + y * y + z * z + w * w
        if norm < 1e-12:
            raise ValueError('TF contains an invalid zero quaternion')

        scale = 2.0 / norm
        return np.array([
            [1.0 - scale * (y * y + z * z),
             scale * (x * y - z * w),
             scale * (x * z + y * w)],
            [scale * (x * y + z * w),
             1.0 - scale * (x * x + z * z),
             scale * (y * z - x * w)],
            [scale * (x * z - y * w),
             scale * (y * z + x * w),
             1.0 - scale * (x * x + y * y)],
        ], dtype=np.float32)

    @staticmethod
    def xyz_fields(msg):
        """Validate and return the three float32 coordinate fields."""
        fields = {field.name: field for field in msg.fields}
        missing = [name for name in ('x', 'y', 'z') if name not in fields]
        if missing:
            raise ValueError(
                f'PointCloud2 is missing fields: {", ".join(missing)}')

        xyz = tuple(fields[name] for name in ('x', 'y', 'z'))
        for field in xyz:
            if (field.datatype != PointField.FLOAT32 or field.count != 1 or
                    field.offset + 4 > msg.point_step):
                raise ValueError(
                    f'Field {field.name} must be one FLOAT32 value')
        return xyz

    @staticmethod
    def field_view(data, msg, field):
        """Create a strided NumPy view without unpacking the entire cloud."""
        byte_order = '>' if msg.is_bigendian else '<'
        return np.ndarray(
            shape=(msg.height, msg.width),
            dtype=np.dtype(f'{byte_order}f4'),
            buffer=data,
            offset=field.offset,
            strides=(msg.row_step, msg.point_step),
        )

    def pointcloud_callback(self, msg):
        started_at = time.perf_counter()
        self.received_count += 1

        if not msg.header.frame_id:
            self.get_logger().warning(
                'Ignoring PointCloud2 with empty frame_id',
                throttle_duration_sec=2.0,
            )
            return

        try:
            xyz_fields = self.xyz_fields(msg)
            if msg.width * msg.height:
                transform = self.tf_buffer.lookup_transform(
                    self.plane_frame,
                    msg.header.frame_id,
                    Time.from_msg(msg.header.stamp),
                )
                output_data = self.transform_and_project(
                    msg, xyz_fields, transform)
            else:
                output_data = bytes(msg.data)
        except TransformException as error:
            self.tf_drop_count += 1
            self.get_logger().warning(
                f'Waiting for TF {msg.header.frame_id} -> '
                f'{self.plane_frame}: {error}',
                throttle_duration_sec=2.0,
            )
            return
        except (TypeError, ValueError) as error:
            self.get_logger().error(
                f'Cannot transform PointCloud2: {error}',
                throttle_duration_sec=2.0,
            )
            return

        output = PointCloud2()
        output.header.stamp = msg.header.stamp
        output.header.frame_id = self.plane_frame
        output.height = msg.height
        output.width = msg.width
        output.fields = msg.fields
        output.is_bigendian = msg.is_bigendian
        output.point_step = msg.point_step
        output.row_step = msg.row_step
        output.data = output_data
        output.is_dense = msg.is_dense
        self.publisher.publish(output)

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        self.published_count += 1
        self.point_counts.append(msg.width * msg.height)
        self.processing_times.append(elapsed_ms)
        self.log_telemetry()

    def transform_and_project(self, msg, xyz_fields, transform):
        """Apply the TF, then orthogonally project points onto z=plane_z."""
        output_data = bytearray(msg.data)
        input_views = [
            self.field_view(msg.data, msg, field) for field in xyz_fields
        ]
        output_views = [
            self.field_view(output_data, msg, field) for field in xyz_fields
        ]

        points = np.column_stack([
            coordinate.reshape(-1) for coordinate in input_views
        ]).astype(np.float32, copy=False)
        rotation = self.quaternion_to_rotation(
            transform.transform.rotation)
        translation = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ], dtype=np.float32)
        transformed = points @ rotation.T
        transformed += translation
        transformed[:, 2] = self.plane_z

        for index, coordinate in enumerate(output_views):
            coordinate[...] = transformed[:, index].reshape(
                msg.height, msg.width)
        return bytes(output_data)

    def log_telemetry(self):
        if (not self.telemetry_enabled or
                self.received_count % self.telemetry_interval):
            return

        average_points = float(np.mean(self.point_counts))
        average_ms = float(np.mean(self.processing_times))
        self.get_logger().info(
            f'Simple BEV: received={self.received_count}, '
            f'published={self.published_count}, '
            f'tf_dropped={self.tf_drop_count}, '
            f'points_avg={average_points:.0f}, time_avg={average_ms:.2f} ms')


def main(args=None):
    rclpy.init(args=args)
    node = SimpleBev()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
