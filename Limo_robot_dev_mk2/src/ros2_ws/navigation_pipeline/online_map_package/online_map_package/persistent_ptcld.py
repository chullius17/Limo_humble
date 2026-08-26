"""Convert the local metric BEV cost grid into a voxelized point cloud."""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
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


class PersistentPointCloud(Node):
    """Publish the current local cost grid as a voxelized global XY cloud."""

    def __init__(self):
        super().__init__('persistent_cloud')

        self.declare_parameter(
            'input_topic',
            '/limo/nav_map_package/online/metric_bev/cost_grid_combined',
        )
        self.declare_parameter(
            'output_topic',
            '/limo/nav_map_package/online/persistent_cloud',
        )
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter(
            'bounding_box_topic',
            '/limo/nav_map_package/online/cloud_bounding_box',
        )
        self.declare_parameter(
            'roi_marker_topic',
            '/limo/nav_map_package/online/cloud_roi',
        )
        self.declare_parameter('bounding_box_length', 2.46)
        self.declare_parameter('bounding_box_width', 2.66)
        # Aggregate the 9.2 mm BEV cells into 3 cm voxels to keep point count,
        # memory use and publication cost suitable for the LIMO computer.
        # Setting this to zero still enables native-grid resolution if needed.
        self.declare_parameter('voxel_size', 0.03)
        self.declare_parameter('source_block_size', 5)
        self.declare_parameter('minimum_cells_per_voxel', 5)
        self.declare_parameter('minimum_cost', 30.0)
        # 2k points are about 40 kB per PointCloud2 message (five float32
        # fields), a practical ceiling while the full pipeline runs on Nano.
        self.declare_parameter('maximum_points', 2000)

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.global_frame = self.get_parameter('global_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.bounding_box_topic = self.get_parameter(
            'bounding_box_topic'
        ).value
        self.roi_marker_topic = self.get_parameter('roi_marker_topic').value
        self.bounding_box_length = float(
            self.get_parameter('bounding_box_length').value
        )
        self.bounding_box_width = float(
            self.get_parameter('bounding_box_width').value
        )
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.point_resolution = None
        self.source_block_size = int(
            self.get_parameter('source_block_size').value
        )
        self.minimum_cells_per_voxel = int(
            self.get_parameter('minimum_cells_per_voxel').value
        )
        self.minimum_cost = float(self.get_parameter('minimum_cost').value)
        self.maximum_points = int(self.get_parameter('maximum_points').value)

        if self.voxel_size < 0.0:
            raise ValueError('voxel_size must be zero or greater')
        if self.bounding_box_length <= 0.0:
            raise ValueError('bounding_box_length must be greater than zero')
        if self.bounding_box_width <= 0.0:
            raise ValueError('bounding_box_width must be greater than zero')
        if self.source_block_size <= 0:
            raise ValueError('source_block_size must be greater than zero')
        if self.minimum_cells_per_voxel <= 0:
            raise ValueError('minimum_cells_per_voxel must be greater than zero')
        if not 0.0 <= self.minimum_cost <= 100.0:
            raise ValueError('minimum_cost must be between 0 and 100')
        if self.maximum_points <= 0:
            raise ValueError('maximum_points must be greater than zero')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.voxel_indices = np.empty((0, 2), dtype=np.int64)
        self.voxel_costs = np.empty(0, dtype=np.float32)

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
            OccupancyGrid,
            self.input_topic,
            self.grid_callback,
            input_qos,
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
            f'Voxel cloud: {self.input_topic} -> {self.output_topic} '
            f'in {self.global_frame} '
            f'(resolution={self.voxel_size or "native grid"})'
        )

    @staticmethod
    def _yaw(quaternion) -> float:
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
    def _apply_planar_transform(x, y, transform):
        yaw = PersistentPointCloud._yaw(transform.transform.rotation)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return (
            transform.transform.translation.x + cos_yaw * x - sin_yaw * y,
            transform.transform.translation.y + sin_yaw * x + cos_yaw * y,
        )

    def _lookup_transform(self, source_frame, stamp):
        if not source_frame:
            raise TransformException('Cost grid has an empty frame_id')
        try:
            return self.tf_buffer.lookup_transform(
                self.global_frame,
                source_frame,
                Time.from_msg(stamp),
            )
        except TransformException:
            return self.tf_buffer.lookup_transform(
                self.global_frame,
                source_frame,
                Time(),
            )

    def _grid_observations(self, msg):
        """Return maximum cost for voxels supported by positive grid cells."""
        costs = np.asarray(msg.data, dtype=np.float32)
        if costs.size != msg.info.width * msg.info.height:
            raise ValueError('OccupancyGrid data size does not match its geometry')

        cost_grid = costs.reshape(msg.info.height, msg.info.width)
        block_size = self.source_block_size
        padded_height = (
            (msg.info.height + block_size - 1) // block_size * block_size
        )
        padded_width = (
            (msg.info.width + block_size - 1) // block_size * block_size
        )
        padded = np.full(
            (padded_height, padded_width),
            -1.0,
            dtype=np.float32,
        )
        padded[:msg.info.height, :msg.info.width] = cost_grid
        blocks = padded.reshape(
            padded_height // block_size,
            block_size,
            padded_width // block_size,
            block_size,
        )
        eligible = blocks >= self.minimum_cost
        block_counts = np.count_nonzero(eligible, axis=(1, 3))
        block_costs = np.max(
            np.where(eligible, blocks, -1.0),
            axis=(1, 3),
        )
        block_rows, block_cols = np.nonzero(
            block_counts >= self.minimum_cells_per_voxel
        )
        if block_rows.size == 0:
            return (
                np.empty((0, 2), dtype=np.int64),
                np.empty(0, dtype=np.float32),
            )
        selected_costs = block_costs[block_rows, block_cols]

        # Apply the output ceiling before coordinate generation and TF. The
        # final cap remains as a safeguard after global voxel collisions.
        if selected_costs.size > self.maximum_points:
            strongest = np.argpartition(
                selected_costs,
                -self.maximum_points,
            )[-self.maximum_points:]
            block_rows = block_rows[strongest]
            block_cols = block_cols[strongest]
            selected_costs = selected_costs[strongest]

        row_starts = block_rows * block_size
        col_starts = block_cols * block_size
        row_sizes = np.minimum(block_size, msg.info.height - row_starts)
        col_sizes = np.minimum(block_size, msg.info.width - col_starts)
        local_x = (
            col_starts.astype(np.float32) + 0.5 * col_sizes
        ) * msg.info.resolution
        local_y = (
            row_starts.astype(np.float32) + 0.5 * row_sizes
        ) * msg.info.resolution

        origin_yaw = self._yaw(msg.info.origin.orientation)
        cos_yaw = math.cos(origin_yaw)
        sin_yaw = math.sin(origin_yaw)
        source_x = (
            msg.info.origin.position.x
            + cos_yaw * local_x
            - sin_yaw * local_y
        )
        source_y = (
            msg.info.origin.position.y
            + sin_yaw * local_x
            + cos_yaw * local_y
        )

        if msg.header.frame_id == self.global_frame:
            world_x, world_y = source_x, source_y
        else:
            transform = self._lookup_transform(
                msg.header.frame_id,
                msg.header.stamp,
            )
            world_x, world_y = self._apply_planar_transform(
                source_x,
                source_y,
                transform,
            )

        voxel_indices = np.column_stack((
            np.floor(world_x / self.point_resolution).astype(np.int64),
            np.floor(world_y / self.point_resolution).astype(np.int64),
        ))
        unique_voxels, inverse = np.unique(
            voxel_indices,
            axis=0,
            return_inverse=True,
        )
        maximum_costs = np.zeros(len(unique_voxels), dtype=np.float32)
        np.maximum.at(maximum_costs, inverse, selected_costs)
        return unique_voxels, maximum_costs

    def _replace_points(self, voxel_indices, costs):
        """Build the cloud exclusively from voxels in the current grid."""
        valid = np.isfinite(costs) & (costs >= self.minimum_cost)
        self.voxel_indices = voxel_indices[valid]
        self.voxel_costs = costs[valid].astype(np.float32, copy=False)

        if self.voxel_costs.size > self.maximum_points:
            strongest = np.argpartition(
                self.voxel_costs,
                -self.maximum_points,
            )[-self.maximum_points:]
            self.voxel_indices = self.voxel_indices[strongest]
            self.voxel_costs = self.voxel_costs[strongest]

    def _make_cloud(self, stamp):
        point_dtype = np.dtype([
            ('x', '<f4'),
            ('y', '<f4'),
            ('z', '<f4'),
            ('cost', '<f4'),
            ('confidence', '<f4'),
        ])
        cloud_data = np.empty(self.voxel_costs.size, dtype=point_dtype)
        cloud_data['x'] = (
            self.voxel_indices[:, 0] + 0.5
        ) * self.point_resolution
        cloud_data['y'] = (
            self.voxel_indices[:, 1] + 0.5
        ) * self.point_resolution
        cloud_data['z'] = 0.0
        cloud_data['cost'] = self.voxel_costs
        cloud_data['confidence'] = 1.0

        cloud = PointCloud2()
        cloud.header.stamp = stamp
        cloud.header.frame_id = self.global_frame
        cloud.height = 1
        cloud.width = len(cloud_data)
        cloud.fields = [
            PointField(name=name, offset=offset, datatype=PointField.FLOAT32, count=1)
            for name, offset in (
                ('x', 0),
                ('y', 4),
                ('z', 8),
                ('cost', 12),
                ('confidence', 16),
            )
        ]
        cloud.is_bigendian = False
        cloud.point_step = point_dtype.itemsize
        cloud.row_step = cloud.point_step * cloud.width
        cloud.data = cloud_data.tobytes()
        cloud.is_dense = True
        return cloud

    def _make_bounding_box_marker(self, stamp):
        """Create robot-centric bounds matching the filtering canvas."""
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.base_frame
        marker.ns = 'online_cloud_bounds'
        marker.id = 0
        marker.pose.orientation.w = 1.0

        # online_filtering rounds its requested 2.45 x 2.65 m canvas up to
        # 2.46 x 2.66 m at 2 cm resolution. base_link is centered on its rear
        # edge, so x starts at zero and y is symmetric around the centerline.
        length = self.bounding_box_length
        half_width = 0.5 * self.bounding_box_width

        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.025
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.points = [
            Point(x=0.0, y=-half_width, z=0.03),
            Point(x=length, y=-half_width, z=0.03),
            Point(x=length, y=half_width, z=0.03),
            Point(x=0.0, y=half_width, z=0.03),
            Point(x=0.0, y=-half_width, z=0.03),
        ]
        return marker

    def _make_roi_marker(self, stamp):
        """Create a yellow dynamic trapezoid enclosing the current cloud."""
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.base_frame
        marker.ns = 'online_cloud_roi'
        marker.id = 0
        marker.pose.orientation.w = 1.0

        if self.voxel_costs.size == 0:
            marker.action = Marker.DELETE
            return marker

        base_transform = self._lookup_transform(self.base_frame, stamp)
        base_x = base_transform.transform.translation.x
        base_y = base_transform.transform.translation.y
        base_yaw = self._yaw(base_transform.transform.rotation)
        cos_yaw = math.cos(base_yaw)
        sin_yaw = math.sin(base_yaw)
        world_x = (
            self.voxel_indices[:, 0] + 0.5
        ) * self.point_resolution
        world_y = (
            self.voxel_indices[:, 1] + 0.5
        ) * self.point_resolution
        delta_x = world_x - base_x
        delta_y = world_y - base_y
        local_x = cos_yaw * delta_x + sin_yaw * delta_y
        local_y = -sin_yaw * delta_x + cos_yaw * delta_y

        rectangle_length = self.bounding_box_length
        rectangle_half_width = 0.5 * self.bounding_box_width
        inside = (local_x >= 0.0) & (local_x <= rectangle_length)
        if not np.any(inside):
            marker.action = Marker.DELETE
            return marker
        x = local_x[inside]
        abs_y = np.abs(local_y[inside])

        half_cell = 0.5 * self.point_resolution
        rear_x = max(0.0, float(np.min(x)) - half_cell)
        front_x = min(
            rectangle_length,
            float(np.max(x)) + half_cell,
        )
        trapezoid_length = max(front_x - rear_x, self.point_resolution)

        # The rear third determines the near width. Starting from the 0.60 m
        # filtering near width, solve the linear envelope required to contain
        # every remaining point at its normalized longitudinal coordinate.
        normalized_x = np.clip(
            (x - rear_x) / trapezoid_length,
            0.0,
            1.0,
        )
        near_zone = normalized_x <= 1.0 / 3.0
        near_half_width = 0.30
        if np.any(near_zone):
            near_half_width = max(
                near_half_width,
                float(np.max(abs_y[near_zone])),
            )
        away_from_rear = normalized_x > 1.0e-6
        required_far_width = (
            abs_y[away_from_rear]
            - near_half_width * (1.0 - normalized_x[away_from_rear])
        ) / normalized_x[away_from_rear]
        far_half_width = near_half_width
        if required_far_width.size > 0:
            far_half_width = max(
                far_half_width,
                float(np.max(required_far_width)),
            )
        near_half_width = min(near_half_width, rectangle_half_width)
        far_half_width = min(far_half_width, rectangle_half_width)

        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.025
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.points = [
            Point(x=rear_x, y=-near_half_width, z=0.04),
            Point(x=front_x, y=-far_half_width, z=0.04),
            Point(x=front_x, y=far_half_width, z=0.04),
            Point(x=rear_x, y=near_half_width, z=0.04),
            Point(x=rear_x, y=-near_half_width, z=0.04),
        ]
        return marker

    def grid_callback(self, msg):
        if not msg.data:
            return
        requested_resolution = self.voxel_size or float(msg.info.resolution)
        if requested_resolution <= 0.0:
            self.get_logger().warn(
                'Cannot integrate a grid with non-positive resolution',
                throttle_duration_sec=2.0,
            )
            return
        if (
            self.point_resolution is not None
            and not math.isclose(self.point_resolution, requested_resolution)
        ):
            self.get_logger().warn(
                'Input grid resolution changed'
            )
        self.point_resolution = requested_resolution
        try:
            voxel_indices, costs = self._grid_observations(msg)
        except (TransformException, ValueError) as exc:
            self.get_logger().warn(
                f'Cannot integrate cost grid: {exc}',
                throttle_duration_sec=2.0,
            )
            return

        self._replace_points(voxel_indices, costs)
        self.cloud_pub.publish(self._make_cloud(msg.header.stamp))
        marker = self._make_bounding_box_marker(msg.header.stamp)
        self.bounding_box_pub.publish(marker)
        try:
            roi_marker = self._make_roi_marker(msg.header.stamp)
        except TransformException as exc:
            self.get_logger().warn(
                f'Cannot draw cloud ROI: {exc}',
                throttle_duration_sec=2.0,
            )
        else:
            self.roi_marker_pub.publish(roi_marker)


def main(args=None):
    rclpy.init(args=args)
    node = PersistentPointCloud()
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
