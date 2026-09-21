# Copyright 2026 Giulio Cataldo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fuse live and reprojected semantic points into a local OccupancyGrid."""

import array
import math

from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
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
from visualization_msgs.msg import Marker, MarkerArray

from online_map_package.semantic_memory import SemanticMemory, transform_xy


OUTPUT_CLOUD_DTYPE = np.dtype({
    'names': ['x', 'y', 'z', 'class_id'],
    'formats': ['<f4', '<f4', '<f4', 'u1'],
    'offsets': [0, 4, 8, 12],
    'itemsize': 16,
})


class LocalMapFinal(Node):
    """Fuse live and reprojected boardwalk/yellow-line evidence into a grid."""

    def __init__(self):
        super().__init__('local_map_final')

        self.declare_parameter(
            'marker_topic',
            '/limo/map_package/online/local_map_final/markers',
        )
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('rectangle_length', 2.50)
        self.declare_parameter('rectangle_width', 2.66)
        self.declare_parameter('trapezoid_height', 1.95)
        self.declare_parameter('trapezoid_near_width', 0.60)
        self.declare_parameter('inner_trapezoid_inset', 0.15)
        self.declare_parameter('line_width', 0.025)
        self.declare_parameter('publish_rate', 2.0)

        self.marker_topic = str(self.get_parameter('marker_topic').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.rectangle_length = float(
            self.get_parameter('rectangle_length').value
        )
        self.rectangle_width = float(
            self.get_parameter('rectangle_width').value
        )
        self.trapezoid_height = float(
            self.get_parameter('trapezoid_height').value
        )
        self.trapezoid_near_width = float(
            self.get_parameter('trapezoid_near_width').value
        )
        self.inner_trapezoid_inset = float(
            self.get_parameter('inner_trapezoid_inset').value
        )
        self.line_width = float(self.get_parameter('line_width').value)
        self.publish_rate = float(
            self.get_parameter('publish_rate').value
        )

        if not self.marker_topic:
            raise ValueError('marker_topic must not be empty')
        if not self.base_frame:
            raise ValueError('base_frame must not be empty')
        for name, value in (
            ('rectangle_length', self.rectangle_length),
            ('rectangle_width', self.rectangle_width),
            ('trapezoid_height', self.trapezoid_height),
            ('trapezoid_near_width', self.trapezoid_near_width),
            ('inner_trapezoid_inset', self.inner_trapezoid_inset),
            ('line_width', self.line_width),
            ('publish_rate', self.publish_rate),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and greater than zero')
        if self.trapezoid_height > self.rectangle_length:
            raise ValueError(
                'trapezoid_height cannot exceed rectangle_length'
            )
        # Geometry is built only at startup, before any sensor callbacks run.
        self.yellow_vertices = self._trapezoid_points()
        self.inner_vertices = self._trapezoid_points(self.inner_trapezoid_inset)
        self.region_markers = MarkerArray()
        self.region_markers.markers = [
            self._rectangle_marker(),
            self._trapezoid_marker(),
            self._inner_trapezoid_marker(),
        ]

        defaults = {
            'input_topic': '/limo/cv_package/visual_ptcld/points',
            'output_topic': '/limo/map_package/online/local_costmap',
            'output_cloud_topic': (
                '/limo/map_package/online/local_map_final/points'),
            'odometry_frame': 'odom',
            'maximum_points': 300,
            'minimum_confidence': 0.30,
            'confidence_decay_per_sec': 0.10,
            'yellow_decay_multiplier': 3.0,
            'cmd_vel_topic': '/cmd_vel',
            'linear_speed_at_max_decay': 0.50,
            'angular_speed_at_max_decay': 1.00,
            'linear_stationary_threshold': 0.01,
            'angular_stationary_threshold': 0.02,
            'cmd_vel_timeout_sec': 0.0,
            'voxel_size': 0.08,
            'grid_resolution': 0.02,
            'inflation_radius': 0.10,
            'grid_publish_rate': 10.0,
            'live_cloud_timeout_sec': 0.50,
            'tf_wait_timeout_sec': 0.20,
            'max_pending_clouds': 10,
            'yellow_line_cost': 60,
            'soft_obstacle_cost': 30,
            'boardwalk_cost': 90,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
            setattr(self, name, self.get_parameter(name).value)
        if (not self.input_topic or not self.output_topic
                or not self.output_cloud_topic
                or not self.odometry_frame or not self.cmd_vel_topic):
            raise ValueError(
                'input_topic, output_topic, output_cloud_topic, '
                'cmd_vel_topic and odometry_frame must not be empty')
        for name in (
                'maximum_points', 'voxel_size', 'grid_resolution',
                'grid_publish_rate', 'live_cloud_timeout_sec',
                'tf_wait_timeout_sec'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and greater than zero')
        if (not math.isfinite(self.inflation_radius)
                or self.inflation_radius < 0.0):
            raise ValueError('inflation_radius must be finite and >= 0')
        if (not isinstance(self.max_pending_clouds, int)
                or isinstance(self.max_pending_clouds, bool)
                or self.max_pending_clouds <= 0):
            raise ValueError('max_pending_clouds must be a positive integer')
        for name in ('yellow_line_cost', 'soft_obstacle_cost', 'boardwalk_cost'):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1 or value > 100:
                raise ValueError(f'{name} must be an integer in [1, 100]')
        if not 0.0 < self.minimum_confidence <= 1.0:
            raise ValueError('minimum_confidence must be in (0, 1]')
        if (not math.isfinite(self.confidence_decay_per_sec)
                or self.confidence_decay_per_sec < 0.0):
            raise ValueError('confidence_decay_per_sec must be finite and >= 0')
        if (not math.isfinite(self.yellow_decay_multiplier)
                or self.yellow_decay_multiplier <= 1.0):
            raise ValueError('yellow_decay_multiplier must be greater than 1')
        for name in (
                'linear_stationary_threshold',
                'angular_stationary_threshold', 'cmd_vel_timeout_sec'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f'{name} must be finite and >= 0')
        for speed, threshold in (
                ('linear_speed_at_max_decay',
                 'linear_stationary_threshold'),
                ('angular_speed_at_max_decay',
                 'angular_stationary_threshold')):
            value = getattr(self, speed)
            if (not math.isfinite(value)
                    or value <= getattr(self, threshold)):
                raise ValueError(
                    f'{speed} must be finite and greater than {threshold}')
        self.memory = SemanticMemory(
            self.rectangle_length, self.rectangle_width,
            self.trapezoid_height, self.trapezoid_near_width,
            self.inner_trapezoid_inset,
            self.maximum_points, self.minimum_confidence,
            self.confidence_decay_per_sec,
            self.yellow_decay_multiplier, self.voxel_size)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.last_clock_ns = None
        self.pending_clouds = []
        self.cells_x = int(round(self.rectangle_length / self.grid_resolution))
        self.cells_y = int(round(self.rectangle_width / self.grid_resolution))
        if (not math.isclose(
                self.cells_x * self.grid_resolution,
                self.rectangle_length, abs_tol=1e-9)
                or not math.isclose(
                    self.cells_y * self.grid_resolution,
                    self.rectangle_width, abs_tol=1e-9)):
            raise ValueError(
                'grid_resolution must divide rectangle dimensions exactly')
        self.grid_origin_y = -0.5 * self.rectangle_width
        radius_cells = int(math.ceil(
            self.inflation_radius / self.grid_resolution))
        self.inflation_offsets = [
            (dy, dx)
            for dy in range(-radius_cells, radius_cells + 1)
            for dx in range(-radius_cells, radius_cells + 1)
            if math.hypot(dx, dy) * self.grid_resolution
            <= self.inflation_radius + 1e-12
        ]
        self.live_points_odom = np.empty((0, 2), dtype=np.float64)
        self.live_classes = np.empty(0, dtype=np.uint8)
        self.live_stamp_sec = None
        self.commanded_linear_speed = 0.0
        self.commanded_angular_speed = 0.0
        self.last_cmd_vel_time_ns = None

        marker_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.marker_pub = self.create_publisher(
            MarkerArray,
            self.marker_topic,
            marker_qos,
        )
        self.grid_pub = self.create_publisher(
            OccupancyGrid, self.output_topic, marker_qos)
        self.cloud_pub = self.create_publisher(
            PointCloud2, self.output_cloud_topic, marker_qos)
        self.cloud_sub = self.create_subscription(
            PointCloud2, self.input_topic, self.cloud_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.cmd_vel_sub = self.create_subscription(
            Twist, self.cmd_vel_topic, self.cmd_vel_callback, 10)
        self.grid_timer = self.create_timer(
            1.0 / self.grid_publish_rate, self.publish_grid)
        # Retry without blocking the executor that also receives TF and /clock.
        self.tf_retry_timer = self.create_timer(0.02, self.retry_pending_clouds)
        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.publish_markers,
        )

        self.get_logger().info(
            f'Local-map regions: topic={self.marker_topic}, '
            f'frame={self.base_frame}, rectangle='
            f'{self.rectangle_length:.2f}x{self.rectangle_width:.2f} m, '
            f'trapezoid={self.trapezoid_height:.2f} m '
            f'({self.trapezoid_near_width:.2f}->'
            f'{self.rectangle_width:.2f} m)'
        )
        self.get_logger().info(
            f'Local semantic grid: {self.input_topic} -> {self.output_topic} '
            f'and {self.output_cloud_topic}, '
            f'live/grid_classes=1..6, persistent_classes=2/4, '
            f'maximum_points={self.maximum_points}; '
            f'tf_wait={self.tf_wait_timeout_sec:g}s, '
            f'pending_clouds<={self.max_pending_clouds}; '
            'admission=inside yellow and outside red trapezoid; '
            f'yellow_decay={self.yellow_decay_multiplier:g}x; '
            f'motion_decay={self.cmd_vel_topic} '
            f'(linear={self.linear_speed_at_max_decay:g}m/s, '
            f'angular={self.angular_speed_at_max_decay:g}rad/s at maximum); '
            f'geometry={self.cells_x}x{self.cells_y} at '
            f'{self.grid_resolution:.3f}m, '
            f'inflation={self.inflation_radius:g}m; costs=1/5:0,'
            f'2:{self.yellow_line_cost},3:{self.soft_obstacle_cost},'
            f'4/6:{self.boardwalk_cost}')

    def _check_clock(self):
        now = self.get_clock().now()
        if self.last_clock_ns is not None and now.nanoseconds < self.last_clock_ns:
            self.memory.reset()
            self.pending_clouds.clear()
            self.live_points_odom = np.empty((0, 2), dtype=np.float64)
            self.live_classes = np.empty(0, dtype=np.uint8)
            self.live_stamp_sec = None
            self.commanded_linear_speed = 0.0
            self.commanded_angular_speed = 0.0
            self.last_cmd_vel_time_ns = None
        self.last_clock_ns = now.nanoseconds
        return now

    def _odom_pose(self, stamp):
        transform = self.tf_buffer.lookup_transform(
            self.odometry_frame, self.base_frame, stamp)
        q = transform.transform.rotation
        translation = transform.transform.translation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (translation.x, translation.y, yaw), transform.header.stamp

    def _read_cloud(self, msg):
        if msg.header.frame_id != self.base_frame:
            raise ValueError(f'Expected cloud in {self.base_frame}')
        if not msg.width or not msg.height:
            return np.empty((0, 2)), np.empty(0, dtype=np.uint8)
        if (msg.point_step <= 0 or msg.row_step < msg.width * msg.point_step
                or len(msg.data) < msg.row_step * msg.height):
            raise ValueError('Invalid pointcloud layout')
        fields = {field.name: field for field in msg.fields}
        formats, offsets = [], []
        endian = '>' if msg.is_bigendian else '<'
        for name, datatype, size, fmt in (
                ('x', PointField.FLOAT32, 4, endian + 'f4'),
                ('y', PointField.FLOAT32, 4, endian + 'f4'),
                ('z', PointField.FLOAT32, 4, endian + 'f4'),
                ('class_id', PointField.UINT8, 1, 'u1')):
            field = fields.get(name)
            if (field is None or field.datatype != datatype or field.count != 1
                    or field.offset + size > msg.point_step):
                raise ValueError(f'Invalid or missing field {name}')
            formats.append(fmt)
            offsets.append(field.offset)
        dtype = np.dtype({
            'names': ['x', 'y', 'z', 'class_id'], 'formats': formats,
            'offsets': offsets, 'itemsize': msg.point_step,
        })
        points = np.ndarray(
            (msg.height, msg.width), dtype=dtype, buffer=msg.data,
            strides=(msg.row_step, msg.point_step)).reshape(-1)
        valid = (np.isfinite(points['x']) & np.isfinite(points['y'])
                 & np.isfinite(points['z']))
        return (np.column_stack((points['x'][valid], points['y'][valid])),
                points['class_id'][valid])

    def cloud_callback(self, msg):
        """Queue observations until their exact timestamp TF is available."""
        now = self._check_clock()
        stamp = Time.from_msg(msg.header.stamp)
        seconds = stamp.nanoseconds * 1e-9
        try:
            xy, classes = self._read_cloud(msg)
        except ValueError as error:
            self.get_logger().warn(
                f'Invalid semantic cloud: {error}',
                throttle_duration_sec=2.0)
            return
        self._process_pending_clouds(now)
        if (self.memory.last_observation_stamp is not None
                and seconds <= self.memory.last_observation_stamp):
            return
        if any(entry[0].nanoseconds == stamp.nanoseconds
               for entry in self.pending_clouds):
            return
        self.pending_clouds.append((stamp, xy, classes, now.nanoseconds))
        self.pending_clouds.sort(key=lambda entry: entry[0].nanoseconds)
        if len(self.pending_clouds) > self.max_pending_clouds:
            self.pending_clouds.pop(0)
            self.get_logger().warn(
                'Semantic TF queue full: dropped oldest cloud',
                throttle_duration_sec=2.0)
        self._process_pending_clouds(now)

    def retry_pending_clouds(self):
        """Recover delayed TF even if no more sensor frames arrive."""
        self._process_pending_clouds(self._check_clock())

    def _process_pending_clouds(self, now):
        """Drain in timestamp order, with a bounded ROS-time wait on arrival."""
        while self.pending_clouds:
            stamp, xy, classes, received_ns = self.pending_clouds[0]
            if ((now.nanoseconds - received_ns) * 1e-9
                    >= self.tf_wait_timeout_sec):
                self.pending_clouds.pop(0)
                self.get_logger().warn(
                    'Semantic cloud expired while waiting for timestamp TF',
                    throttle_duration_sec=2.0)
                continue
            try:
                pose, _ = self._odom_pose(stamp)
            except TransformException:
                # Do not let newer observations overtake a pending frame.
                return
            self.pending_clouds.pop(0)
            seconds = stamp.nanoseconds * 1e-9
            self.memory.observe(xy, classes, pose, seconds)
            self.live_points_odom = transform_xy(xy, pose)
            self.live_classes = classes.copy()
            # Preserve sensor time: waiting must not extend the live lifetime.
            self.live_stamp_sec = seconds

    def cmd_vel_callback(self, msg):
        """Cache commanded planar speeds for confidence decay."""
        self.commanded_linear_speed = abs(float(msg.linear.x))
        self.commanded_angular_speed = abs(float(msg.angular.z))
        self.last_cmd_vel_time_ns = self.get_clock().now().nanoseconds

    def _motion_ratio(self, now_ns=None):
        """Scale decay from zero at rest to one at configured speeds."""
        if self.last_cmd_vel_time_ns is None:
            return 0.0
        if now_ns is None:
            now_ns = self.get_clock().now().nanoseconds
        if self.cmd_vel_timeout_sec > 0.0:
            age_sec = (now_ns - self.last_cmd_vel_time_ns) * 1e-9
            if age_sec > self.cmd_vel_timeout_sec:
                return 0.0
        linear_ratio = np.clip(
            (self.commanded_linear_speed - self.linear_stationary_threshold)
            / (self.linear_speed_at_max_decay
               - self.linear_stationary_threshold),
            0.0, 1.0)
        angular_ratio = np.clip(
            (self.commanded_angular_speed - self.angular_stationary_threshold)
            / (self.angular_speed_at_max_decay
               - self.angular_stationary_threshold),
            0.0, 1.0)
        return max(float(linear_ratio), float(angular_ratio))

    def _rasterize(self, points, classes):
        """Rasterize fixed semantic costs, retaining the maximum per cell."""
        grid = np.zeros((self.cells_y, self.cells_x), dtype=np.int16)
        if not len(points):
            return grid
        valid = (
            np.isfinite(points).all(axis=1)
            & np.isin(classes, [1, 2, 3, 4, 5, 6])
            & (points[:, 0] >= 0.0)
            & (points[:, 0] < self.rectangle_length)
            & (points[:, 1] >= self.grid_origin_y)
            & (points[:, 1] < -self.grid_origin_y)
        )
        if not np.any(valid):
            return grid
        selected = points[valid]
        selected_classes = classes[valid]
        cell_x = np.floor(
            selected[:, 0] / self.grid_resolution).astype(np.int64)
        cell_y = np.floor(
            (selected[:, 1] - self.grid_origin_y) /
            self.grid_resolution).astype(np.int64)
        # Match the offline semantic map: both road classes are free,
        # and interior boardwalk (6) has the same cost as boundary (4).
        class_costs = np.array([
            0, 0, self.yellow_line_cost, self.soft_obstacle_cost,
            self.boardwalk_cost, 0, self.boardwalk_cost,
        ], dtype=np.int16)
        costs = class_costs[selected_classes]
        flat_indices = cell_y * self.cells_x + cell_x
        np.maximum.at(grid.ravel(), flat_indices, costs)
        return self._inflate(grid)

    def _inflate(self, grid):
        """Dilate nonzero semantic costs within the configured radius."""
        if self.inflation_radius <= 0.0 or not np.any(grid):
            return grid
        inflated = grid.copy()
        height, width = grid.shape
        for dy, dx in self.inflation_offsets:
            if dx == 0 and dy == 0:
                continue
            source_y0, source_y1 = max(0, -dy), min(height, height - dy)
            source_x0, source_x1 = max(0, -dx), min(width, width - dx)
            target_y0, target_y1 = source_y0 + dy, source_y1 + dy
            target_x0, target_x1 = source_x0 + dx, source_x1 + dx
            np.maximum(
                inflated[target_y0:target_y1, target_x0:target_x1],
                grid[source_y0:source_y1, source_x0:source_x1],
                out=inflated[target_y0:target_y1, target_x0:target_x1],
            )
        return inflated

    def _make_grid(self, costs, stamp):
        grid = OccupancyGrid()
        grid.header.stamp = stamp
        grid.header.frame_id = self.base_frame
        grid.info.map_load_time = stamp
        grid.info.resolution = self.grid_resolution
        grid.info.width = self.cells_x
        grid.info.height = self.cells_y
        grid.info.origin.position.x = 0.0
        grid.info.origin.position.y = self.grid_origin_y
        grid.info.origin.orientation.w = 1.0
        grid.data = array.array(
            'b', costs.astype(np.int8, copy=False).ravel().tobytes())
        return grid

    def _make_cloud(self, points, classes, stamp):
        """Serialize every finite live or reprojected semantic point."""
        valid = np.isfinite(points).all(axis=1)
        selected = points[valid]
        selected_classes = classes[valid]
        cloud_data = np.empty(len(selected), dtype=OUTPUT_CLOUD_DTYPE)
        cloud_data['x'] = selected[:, 0]
        cloud_data['y'] = selected[:, 1]
        cloud_data['z'] = 0.0
        cloud_data['class_id'] = selected_classes

        cloud = PointCloud2()
        cloud.header.stamp = stamp
        cloud.header.frame_id = self.base_frame
        cloud.height = 1
        cloud.width = len(cloud_data)
        cloud.fields = [
            PointField(
                name=name, offset=offset, datatype=datatype, count=1)
            for name, offset, datatype in (
                ('x', 0, PointField.FLOAT32),
                ('y', 4, PointField.FLOAT32),
                ('z', 8, PointField.FLOAT32),
                ('class_id', 12, PointField.UINT8),
            )
        ]
        cloud.is_bigendian = False
        cloud.point_step = OUTPUT_CLOUD_DTYPE.itemsize
        cloud.row_step = cloud.point_step * cloud.width
        cloud.data = array.array('B', cloud_data.tobytes())
        cloud.is_dense = True
        return cloud

    def publish_grid(self):
        """Fuse the latest live cloud with the continuously reprojected memory."""
        now = self._check_clock()
        now_sec = now.nanoseconds * 1e-9
        has_live = (
            self.live_stamp_sec is not None
            and 0.0 <= now_sec - self.live_stamp_sec <=
            self.live_cloud_timeout_sec
            and len(self.live_points_odom) > 0)
        if not has_live and not len(self.memory.points):
            points = np.empty((0, 2))
            classes = np.empty(0, dtype=np.uint8)
            stamp = now.to_msg()
            self.grid_pub.publish(self._make_grid(
                self._rasterize(points, classes), stamp))
            self.cloud_pub.publish(self._make_cloud(points, classes, stamp))
            return
        try:
            pose, stamp = self._odom_pose(Time())
        except TransformException as error:
            self.get_logger().warn(
                f'Cannot publish local semantic grid: {error}',
                throttle_duration_sec=2.0)
            return
        memory_xy, memory_classes, _ = self.memory.prune(
            pose, now_sec, self._motion_ratio(now.nanoseconds))
        if has_live:
            live_xy = transform_xy(self.live_points_odom, pose, inverse=True)
            points = np.concatenate((live_xy, memory_xy))
            classes = np.concatenate((self.live_classes, memory_classes))
        else:
            points, classes = memory_xy, memory_classes
        self.grid_pub.publish(self._make_grid(
            self._rasterize(points, classes), stamp))
        self.cloud_pub.publish(self._make_cloud(points, classes, stamp))

    def _line_marker(self, marker_id, namespace, points, red, green, z):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.base_frame
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.line_width
        marker.color.r = red
        marker.color.g = green
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.frame_locked = True
        marker.points = [Point(x=x, y=y, z=z) for x, y in points]
        return marker

    def _rectangle_marker(self):
        half_width = 0.5 * self.rectangle_width
        points = (
            (0.0, -half_width),
            (self.rectangle_length, -half_width),
            (self.rectangle_length, half_width),
            (0.0, half_width),
            (0.0, -half_width),
        )
        return self._line_marker(
            0, 'local_map_rectangle', points, 0.0, 1.0, 0.03
        )

    def _trapezoid_points(self, inset=0.0):
        """Build the ROI; the inset variant keeps the front edge aligned."""
        near_half_width = 0.5 * self.trapezoid_near_width
        far_half_width = 0.5 * self.rectangle_width
        slope = (far_half_width - near_half_width) / self.trapezoid_height
        lateral_shift = inset * math.hypot(1.0, slope)
        near_half_width += slope * inset - lateral_shift
        far_half_width -= lateral_shift
        far_x = self.rectangle_length
        near_x = self.rectangle_length - self.trapezoid_height + inset
        if near_x >= far_x or min(near_half_width, far_half_width) <= 0.0:
            raise ValueError('inner_trapezoid_inset collapses the trapezoid')
        return (
            (near_x, -near_half_width),
            (far_x, -far_half_width),
            (far_x, far_half_width),
            (near_x, near_half_width),
            (near_x, -near_half_width),
        )

    def _trapezoid_marker(self):
        return self._line_marker(
            1, 'local_map_trapezoid', self.yellow_vertices, 1.0, 1.0, 0.04
        )

    def _inner_trapezoid_marker(self):
        return self._line_marker(
            2, 'local_map_inner_trapezoid',
            self.inner_vertices,
            1.0, 0.0, 0.04
        )

    def publish_markers(self):
        """Publish all three outlines in one RViz update."""
        stamp = self.get_clock().now().to_msg()
        for marker in self.region_markers.markers:
            marker.header.stamp = stamp
        self.marker_pub.publish(self.region_markers)


def main(args=None):
    rclpy.init(args=args)
    node = LocalMapFinal()
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
