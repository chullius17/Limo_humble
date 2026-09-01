"""Maintain a motion-compensated local cloud from CV point observations."""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Twist
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker


POINT_DTYPE = np.dtype([
    ('x', '<f4'),
    ('y', '<f4'),
    ('z', '<f4'),
    ('cost', '<f4'),
    ('confidence', '<f4'),
])
INPUT_FIELDS = ('x', 'y', 'cost', 'confidence')


class LocalMap(Node):
    """Reproject, decay and replace observations on a local canvas."""

    def __init__(self):
        super().__init__('local_map')

        self.declare_parameter(
            'input_topic',
            '/limo/nav_map_package/online/cv_cloud',
        )
        self.declare_parameter(
            'output_topic',
            '/limo/nav_map_package/online/persistent_cloud',
        )
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('odometry_frame', 'odom')
        self.declare_parameter(
            'bounding_box_topic',
            '/limo/nav_map_package/online/cloud_bounding_box',
        )
        self.declare_parameter(
            'roi_marker_topic',
            '/limo/nav_map_package/online/cloud_roi',
        )
        self.declare_parameter(
            'roi_frame',
            'online_metric_bev_origin_combined',
        )
        self.declare_parameter('roi_trapezoid_height', 1.85)
        self.declare_parameter('roi_near_base_width', 0.60)
        self.declare_parameter('roi_far_base_width', 2.65)
        self.declare_parameter('bounding_box_length', 2.46)
        self.declare_parameter('bounding_box_width', 2.66)
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('maximum_decay_per_cycle', 0.02)
        self.declare_parameter('linear_speed_at_max_decay', 0.50)
        self.declare_parameter('angular_speed_at_max_decay', 1.00)
        self.declare_parameter('linear_stationary_threshold', 0.01)
        self.declare_parameter('angular_stationary_threshold', 0.02)
        self.declare_parameter('cmd_vel_timeout_sec', 0.50)
        self.declare_parameter('minimum_confidence', 0.30)
        self.declare_parameter('maximum_points', 2000)
        self.declare_parameter('point_statistics_window_cycles', 30)

        self.input_topic = str(self.get_parameter('input_topic').value)
        self.output_topic = str(self.get_parameter('output_topic').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.odometry_frame = str(
            self.get_parameter('odometry_frame').value
        )
        self.bounding_box_topic = str(
            self.get_parameter('bounding_box_topic').value
        )
        self.roi_marker_topic = str(
            self.get_parameter('roi_marker_topic').value
        )
        self.roi_frame = str(self.get_parameter('roi_frame').value)
        self.roi_trapezoid_height = float(
            self.get_parameter('roi_trapezoid_height').value
        )
        self.roi_near_base_width = float(
            self.get_parameter('roi_near_base_width').value
        )
        self.roi_far_base_width = float(
            self.get_parameter('roi_far_base_width').value
        )
        self.bounding_box_length = float(
            self.get_parameter('bounding_box_length').value
        )
        self.bounding_box_width = float(
            self.get_parameter('bounding_box_width').value
        )
        self.cmd_vel_topic = str(
            self.get_parameter('cmd_vel_topic').value
        )
        self.maximum_decay_per_cycle = float(
            self.get_parameter('maximum_decay_per_cycle').value
        )
        self.linear_speed_at_max_decay = float(
            self.get_parameter('linear_speed_at_max_decay').value
        )
        self.angular_speed_at_max_decay = float(
            self.get_parameter('angular_speed_at_max_decay').value
        )
        self.linear_stationary_threshold = float(
            self.get_parameter('linear_stationary_threshold').value
        )
        self.angular_stationary_threshold = float(
            self.get_parameter('angular_stationary_threshold').value
        )
        self.cmd_vel_timeout_sec = float(
            self.get_parameter('cmd_vel_timeout_sec').value
        )
        self.minimum_confidence = float(
            self.get_parameter('minimum_confidence').value
        )
        self.maximum_points = int(
            self.get_parameter('maximum_points').value
        )
        self.point_statistics_window_cycles = int(
            self.get_parameter('point_statistics_window_cycles').value
        )

        for name, frame in (
            ('base_frame', self.base_frame),
            ('odometry_frame', self.odometry_frame),
            ('roi_frame', self.roi_frame),
        ):
            if not frame:
                raise ValueError(f'{name} must not be empty')
        for name, value in (
            ('roi_trapezoid_height', self.roi_trapezoid_height),
            ('roi_near_base_width', self.roi_near_base_width),
            ('roi_far_base_width', self.roi_far_base_width),
            ('bounding_box_length', self.bounding_box_length),
            ('bounding_box_width', self.bounding_box_width),
        ):
            if value <= 0.0:
                raise ValueError(f'{name} must be greater than zero')
        if not self.cmd_vel_topic:
            raise ValueError('cmd_vel_topic must not be empty')
        if not 0.0 <= self.maximum_decay_per_cycle <= 1.0:
            raise ValueError(
                'maximum_decay_per_cycle must be between 0 and 1'
            )
        if self.linear_stationary_threshold < 0.0:
            raise ValueError(
                'linear_stationary_threshold must be zero or greater'
            )
        if self.angular_stationary_threshold < 0.0:
            raise ValueError(
                'angular_stationary_threshold must be zero or greater'
            )
        if (
            self.linear_speed_at_max_decay
            <= self.linear_stationary_threshold
        ):
            raise ValueError(
                'linear_speed_at_max_decay must exceed the stationary '
                'threshold'
            )
        if (
            self.angular_speed_at_max_decay
            <= self.angular_stationary_threshold
        ):
            raise ValueError(
                'angular_speed_at_max_decay must exceed the stationary '
                'threshold'
            )
        if self.cmd_vel_timeout_sec < 0.0:
            raise ValueError('cmd_vel_timeout_sec must be zero or greater')
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError('minimum_confidence must be between 0 and 1')
        if self.maximum_points <= 0:
            raise ValueError('maximum_points must be greater than zero')
        if self.point_statistics_window_cycles <= 0:
            raise ValueError(
                'point_statistics_window_cycles must be greater than zero'
            )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.points_xy = np.empty((0, 2), dtype=np.float64)
        self.costs = np.empty(0, dtype=np.float32)
        self.confidences = np.empty(0, dtype=np.float32)
        self.previous_odom_from_base = None
        self.roi_from_base_pose = None
        self.commanded_linear_speed = 0.0
        self.commanded_angular_speed = 0.0
        self.last_cmd_vel_time_ns = None
        self.point_statistics_count = 0
        self.point_statistics_sum = 0
        self.point_statistics_min = None
        self.point_statistics_max = None
        self.decay_statistics_sum = 0.0
        self.decay_statistics_min = None
        self.decay_statistics_max = None

        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.cloud_callback,
            input_qos,
        )
        self.create_subscription(
            Twist,
            self.cmd_vel_topic,
            self.cmd_vel_callback,
            10,
        )
        self.cloud_pub = self.create_publisher(
            PointCloud2,
            self.output_topic,
            output_qos,
        )
        self.bounding_box_pub = self.create_publisher(
            Marker,
            self.bounding_box_topic,
            output_qos,
        )
        self.roi_marker_pub = self.create_publisher(
            Marker,
            self.roi_marker_topic,
            output_qos,
        )

        self.get_logger().info(
            f'Persistent cloud: {self.input_topic} -> {self.output_topic}, '
            f'frame={self.base_frame}, motion reference={self.odometry_frame}, '
            f'decay=0..{self.maximum_decay_per_cycle:.2f}/cycle from '
            f'{self.cmd_vel_topic}, '
            f'threshold={self.minimum_confidence:.2f}'
        )

    @staticmethod
    def _yaw(quaternion):
        return math.atan2(
            2.0 * (
                quaternion.w * quaternion.z
                + quaternion.x * quaternion.y
            ),
            1.0 - 2.0 * (
                quaternion.y * quaternion.y
                + quaternion.z * quaternion.z
            ),
        )

    @staticmethod
    def _pose_from_transform(transform):
        translation = transform.transform.translation
        return (
            translation.x,
            translation.y,
            LocalMap._yaw(transform.transform.rotation),
        )

    @staticmethod
    def _transform_xy(points_xy, pose):
        """Apply a planar target-from-source pose to an ``Nx2`` array."""
        if points_xy.size == 0:
            return points_xy.copy()
        translation_x, translation_y, yaw = pose
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        output = np.empty_like(points_xy, dtype=np.float64)
        output[:, 0] = (
            translation_x
            + cos_yaw * points_xy[:, 0]
            - sin_yaw * points_xy[:, 1]
        )
        output[:, 1] = (
            translation_y
            + sin_yaw * points_xy[:, 0]
            + cos_yaw * points_xy[:, 1]
        )
        return output

    @staticmethod
    def _inverse_transform_xy(points_xy, pose):
        """Apply the inverse of a planar target-from-source pose."""
        if points_xy.size == 0:
            return points_xy.copy()
        translation_x, translation_y, yaw = pose
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        delta_x = points_xy[:, 0] - translation_x
        delta_y = points_xy[:, 1] - translation_y
        output = np.empty_like(points_xy, dtype=np.float64)
        output[:, 0] = cos_yaw * delta_x + sin_yaw * delta_y
        output[:, 1] = -sin_yaw * delta_x + cos_yaw * delta_y
        return output

    def _lookup_transform(
        self,
        target_frame,
        source_frame,
        stamp,
        allow_latest=True,
    ):
        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time.from_msg(stamp),
            )
        except TransformException:
            if not allow_latest:
                raise
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
            )

    def _read_input_cloud(self, msg):
        """Read the fixed CV-cloud interface without per-point Python loops."""
        if msg.header.frame_id != self.base_frame:
            raise ValueError(
                f'Input cloud frame must be {self.base_frame!r}, got '
                f'{msg.header.frame_id!r}'
            )
        point_count = int(msg.width) * int(msg.height)
        if point_count == 0:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )
        if msg.point_step <= 0 or msg.row_step < msg.point_step * msg.width:
            raise ValueError('Input cloud has invalid point or row stride')
        if len(msg.data) < msg.row_step * msg.height:
            raise ValueError('Input cloud data is shorter than its geometry')

        fields = {field.name: field for field in msg.fields}
        for name in INPUT_FIELDS:
            field = fields.get(name)
            if field is None:
                raise ValueError(f'Input cloud is missing field {name!r}')
            if field.datatype != PointField.FLOAT32 or field.count != 1:
                raise ValueError(f'Input field {name!r} must be FLOAT32[1]')
            if field.offset + 4 > msg.point_step:
                raise ValueError(f'Input field {name!r} exceeds point_step')

        float_format = '>f4' if msg.is_bigendian else '<f4'
        dtype = np.dtype({
            'names': list(INPUT_FIELDS),
            'formats': [float_format] * len(INPUT_FIELDS),
            'offsets': [fields[name].offset for name in INPUT_FIELDS],
            'itemsize': msg.point_step,
        })
        organized = np.ndarray(
            shape=(msg.height, msg.width),
            dtype=dtype,
            buffer=msg.data,
            strides=(msg.row_step, msg.point_step),
        )
        flat = organized.reshape(-1)
        points_xy = np.column_stack((flat['x'], flat['y'])).astype(
            np.float64,
            copy=False,
        )
        costs = flat['cost'].astype(np.float32, copy=False)
        confidences = flat['confidence'].astype(np.float32, copy=False)
        valid = (
            np.all(np.isfinite(points_xy), axis=1)
            & np.isfinite(costs)
            & np.isfinite(confidences)
        )
        return (
            points_xy[valid],
            costs[valid],
            np.clip(confidences[valid], 0.0, 1.0),
        )

    def cmd_vel_callback(self, msg):
        """Cache commanded planar speeds for confidence decay."""
        self.commanded_linear_speed = abs(float(msg.linear.x))
        self.commanded_angular_speed = abs(float(msg.angular.z))
        self.last_cmd_vel_time_ns = self.get_clock().now().nanoseconds

    def _decay_factor(self):
        """Return the fractional confidence loss for the current cycle."""
        if self.last_cmd_vel_time_ns is None:
            return 0.0
        if self.cmd_vel_timeout_sec > 0.0:
            age_sec = (
                self.get_clock().now().nanoseconds
                - self.last_cmd_vel_time_ns
            ) * 1.0e-9
            if age_sec > self.cmd_vel_timeout_sec:
                return 0.0

        linear_ratio = np.clip(
            (
                self.commanded_linear_speed
                - self.linear_stationary_threshold
            ) / (
                self.linear_speed_at_max_decay
                - self.linear_stationary_threshold
            ),
            0.0,
            1.0,
        )
        angular_ratio = np.clip(
            (
                self.commanded_angular_speed
                - self.angular_stationary_threshold
            ) / (
                self.angular_speed_at_max_decay
                - self.angular_stationary_threshold
            ),
            0.0,
            1.0,
        )
        motion_ratio = max(float(linear_ratio), float(angular_ratio))
        return self.maximum_decay_per_cycle * motion_ratio

    def _reproject_existing(self, current_odom_from_base, decay_factor):
        """Move the previous base-frame cloud into the current base frame."""
        if self.previous_odom_from_base is None or self.points_xy.size == 0:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )

        odom_xy = self._transform_xy(
            self.points_xy,
            self.previous_odom_from_base,
        )
        current_base_xy = self._inverse_transform_xy(
            odom_xy,
            current_odom_from_base,
        )
        confidences = self.confidences * (1.0 - decay_factor)
        half_width = 0.5 * self.bounding_box_width
        keep = (
            (current_base_xy[:, 0] >= 0.0)
            & (current_base_xy[:, 0] <= self.bounding_box_length)
            & (np.abs(current_base_xy[:, 1]) <= half_width)
            & (confidences >= self.minimum_confidence)
        )
        return (
            current_base_xy[keep],
            self.costs[keep],
            confidences[keep],
        )

    def _points_in_roi_frame(self, base_points, stamp):
        if base_points.size == 0:
            return base_points.copy()
        if self.roi_frame == self.base_frame:
            return base_points
        if self.roi_from_base_pose is None:
            transform = self._lookup_transform(
                self.roi_frame,
                self.base_frame,
                stamp,
            )
            self.roi_from_base_pose = self._pose_from_transform(transform)
        return self._transform_xy(base_points, self.roi_from_base_pose)

    def _inside_trapezoid(self, points_in_roi_frame):
        if points_in_roi_frame.size == 0:
            return np.empty(0, dtype=bool)
        x = points_in_roi_frame[:, 0]
        interpolation = np.clip(
            x / self.roi_trapezoid_height,
            0.0,
            1.0,
        )
        width = (
            self.roi_near_base_width
            + interpolation
            * (self.roi_far_base_width - self.roi_near_base_width)
        )
        return (
            (x >= 0.0)
            & (x <= self.roi_trapezoid_height)
            & (np.abs(points_in_roi_frame[:, 1]) <= 0.5 * width)
        )

    def _replace_observation_region(
        self,
        old_points,
        old_costs,
        old_confidences,
        input_points,
        input_costs,
        input_confidences,
        stamp,
    ):
        old_inside = self._inside_trapezoid(
            self._points_in_roi_frame(old_points, stamp)
        )
        input_inside = self._inside_trapezoid(
            self._points_in_roi_frame(input_points, stamp)
        )
        self.points_xy = np.concatenate((
            old_points[~old_inside],
            input_points[input_inside],
        ))
        self.costs = np.concatenate((
            old_costs[~old_inside],
            input_costs[input_inside],
        )).astype(np.float32, copy=False)
        self.confidences = np.concatenate((
            old_confidences[~old_inside],
            input_confidences[input_inside],
        )).astype(np.float32, copy=False)

        if self.costs.size > self.maximum_points:
            ordered = np.lexsort((self.costs, self.confidences))
            keep = ordered[-self.maximum_points:]
            self.points_xy = self.points_xy[keep]
            self.costs = self.costs[keep]
            self.confidences = self.confidences[keep]

    def _make_cloud(self, stamp):
        cloud_data = np.empty(self.costs.size, dtype=POINT_DTYPE)
        cloud_data['x'] = self.points_xy[:, 0]
        cloud_data['y'] = self.points_xy[:, 1]
        cloud_data['z'] = 0.0
        cloud_data['cost'] = self.costs
        cloud_data['confidence'] = self.confidences

        cloud = PointCloud2()
        cloud.header.stamp = stamp
        cloud.header.frame_id = self.base_frame
        cloud.height = 1
        cloud.width = len(cloud_data)
        cloud.fields = [
            PointField(
                name=name,
                offset=offset,
                datatype=PointField.FLOAT32,
                count=1,
            )
            for name, offset in (
                ('x', 0),
                ('y', 4),
                ('z', 8),
                ('cost', 12),
                ('confidence', 16),
            )
        ]
        cloud.is_bigendian = False
        cloud.point_step = POINT_DTYPE.itemsize
        cloud.row_step = cloud.point_step * cloud.width
        cloud.data = cloud_data.tobytes()
        cloud.is_dense = True
        return cloud

    def _update_cycle_statistics(self, decay_factor):
        """Log point-count and decay statistics over a cycle window."""
        point_count = int(self.costs.size)
        self.point_statistics_count += 1
        self.point_statistics_sum += point_count
        self.decay_statistics_sum += decay_factor
        if self.point_statistics_min is None:
            self.point_statistics_min = point_count
            self.point_statistics_max = point_count
            self.decay_statistics_min = decay_factor
            self.decay_statistics_max = decay_factor
        else:
            self.point_statistics_min = min(
                self.point_statistics_min,
                point_count,
            )
            self.point_statistics_max = max(
                self.point_statistics_max,
                point_count,
            )
            self.decay_statistics_min = min(
                self.decay_statistics_min,
                decay_factor,
            )
            self.decay_statistics_max = max(
                self.decay_statistics_max,
                decay_factor,
            )

        if (
            self.point_statistics_count
            < self.point_statistics_window_cycles
        ):
            return

        average = (
            self.point_statistics_sum / self.point_statistics_count
        )
        decay_average = (
            self.decay_statistics_sum / self.point_statistics_count
        )
        self.get_logger().info(
            f'Persistent cloud points over '
            f'{self.point_statistics_count} cycles: '
            f'average={average:.1f}, '
            f'min={self.point_statistics_min}, '
            f'max={self.point_statistics_max}; '
            f'decay average={decay_average:.4f}, '
            f'min={self.decay_statistics_min:.4f}, '
            f'max={self.decay_statistics_max:.4f}'
        )
        self.point_statistics_count = 0
        self.point_statistics_sum = 0
        self.point_statistics_min = None
        self.point_statistics_max = None
        self.decay_statistics_sum = 0.0
        self.decay_statistics_min = None
        self.decay_statistics_max = None

    def _make_bounding_box_marker(self, stamp):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.base_frame
        marker.ns = 'online_cloud_bounds'
        marker.id = 0
        marker.pose.orientation.w = 1.0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.025
        marker.color.r = 1.0
        marker.color.a = 1.0
        length = self.bounding_box_length
        half_width = 0.5 * self.bounding_box_width
        marker.points = [
            Point(x=0.0, y=-half_width, z=0.03),
            Point(x=length, y=-half_width, z=0.03),
            Point(x=length, y=half_width, z=0.03),
            Point(x=0.0, y=half_width, z=0.03),
            Point(x=0.0, y=-half_width, z=0.03),
        ]
        return marker

    def _make_roi_marker(self, stamp):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.roi_frame
        marker.ns = 'online_cloud_roi'
        marker.id = 0
        marker.pose.orientation.w = 1.0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.025
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.a = 1.0
        height = self.roi_trapezoid_height
        near_half_width = 0.5 * self.roi_near_base_width
        far_half_width = 0.5 * self.roi_far_base_width
        marker.points = [
            Point(x=0.0, y=-near_half_width, z=0.04),
            Point(x=height, y=-far_half_width, z=0.04),
            Point(x=height, y=far_half_width, z=0.04),
            Point(x=0.0, y=near_half_width, z=0.04),
            Point(x=0.0, y=-near_half_width, z=0.04),
        ]
        return marker

    def cloud_callback(self, msg):
        try:
            current_transform = self._lookup_transform(
                self.odometry_frame,
                self.base_frame,
                msg.header.stamp,
                allow_latest=False,
            )
            current_odom_from_base = self._pose_from_transform(
                current_transform
            )
            decay_factor = self._decay_factor()
            old_points, old_costs, old_confidences = (
                self._reproject_existing(
                    current_odom_from_base,
                    decay_factor,
                )
            )
            input_points, input_costs, input_confidences = (
                self._read_input_cloud(msg)
            )
            self._replace_observation_region(
                old_points,
                old_costs,
                old_confidences,
                input_points,
                input_costs,
                input_confidences,
                msg.header.stamp,
            )
        except (TransformException, ValueError) as exc:
            self.get_logger().warn(
                f'Cannot update persistent cloud: {exc}',
                throttle_duration_sec=2.0,
            )
            return

        self.previous_odom_from_base = current_odom_from_base
        self._update_cycle_statistics(decay_factor)
        self.cloud_pub.publish(self._make_cloud(msg.header.stamp))
        self.bounding_box_pub.publish(
            self._make_bounding_box_marker(msg.header.stamp)
        )
        self.roi_marker_pub.publish(self._make_roi_marker(msg.header.stamp))


def main(args=None):
    rclpy.init(args=args)
    node = LocalMap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
