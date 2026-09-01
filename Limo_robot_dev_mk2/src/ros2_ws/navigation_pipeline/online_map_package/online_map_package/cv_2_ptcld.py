"""Convert the metric BEV cost grid into a downsampled point cloud."""

import math
import time

import numpy as np
import rclpy
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


POINT_DTYPE = np.dtype([
    ('x', '<f4'),
    ('y', '<f4'),
    ('z', '<f4'),
    ('cost', '<f4'),
    ('confidence', '<f4'),
])


class CvToPointCloud(Node):
    """Transform a CV cost grid and apply its two reduction stages."""

    def __init__(self):
        super().__init__('cv_2_ptcld')

        self.declare_parameter(
            'input_topic',
            '/limo/nav_map_package/online/metric_bev/cost_grid_combined',
        )
        self.declare_parameter(
            'output_topic',
            '/limo/nav_map_package/online/cv_cloud',
        )
        self.declare_parameter('output_frame', 'base_link')
        self.declare_parameter('voxel_size', 0.03)
        self.declare_parameter('source_block_size', 5)
        self.declare_parameter('minimum_cells_per_block', 5)
        self.declare_parameter('minimum_cost', 30.0)
        self.declare_parameter('high_cost_threshold', 95.0)
        self.declare_parameter('high_cost_downsampling_factor', 2)
        self.declare_parameter('maximum_points', 2000)
        self.declare_parameter('statistics_window_cycles', 30)

        self.input_topic = str(self.get_parameter('input_topic').value)
        self.output_topic = str(self.get_parameter('output_topic').value)
        self.output_frame = str(self.get_parameter('output_frame').value)
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.source_block_size = int(
            self.get_parameter('source_block_size').value
        )
        self.minimum_cells_per_block = int(
            self.get_parameter('minimum_cells_per_block').value
        )
        self.minimum_cost = float(
            self.get_parameter('minimum_cost').value
        )
        self.high_cost_threshold = float(
            self.get_parameter('high_cost_threshold').value
        )
        self.high_cost_downsampling_factor = int(
            self.get_parameter('high_cost_downsampling_factor').value
        )
        self.maximum_points = int(
            self.get_parameter('maximum_points').value
        )
        self.statistics_window_cycles = int(
            self.get_parameter('statistics_window_cycles').value
        )

        if not self.output_frame:
            raise ValueError('output_frame must not be empty')
        if self.voxel_size < 0.0:
            raise ValueError('voxel_size must be zero or greater')
        if self.source_block_size <= 0:
            raise ValueError('source_block_size must be greater than zero')
        if self.minimum_cells_per_block <= 0:
            raise ValueError(
                'minimum_cells_per_block must be greater than zero'
            )
        maximum_cells = self.source_block_size ** 2
        if self.minimum_cells_per_block > maximum_cells:
            raise ValueError(
                'minimum_cells_per_block cannot exceed block area'
            )
        if not 0.0 <= self.minimum_cost <= 100.0:
            raise ValueError('minimum_cost must be between 0 and 100')
        if not 0.0 <= self.high_cost_threshold <= 100.0:
            raise ValueError('high_cost_threshold must be between 0 and 100')
        if self.high_cost_downsampling_factor <= 0:
            raise ValueError(
                'high_cost_downsampling_factor must be greater than zero'
            )
        if self.maximum_points <= 0:
            raise ValueError('maximum_points must be greater than zero')
        if self.statistics_window_cycles <= 0:
            raise ValueError(
                'statistics_window_cycles must be greater than zero'
            )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.statistics_count = 0
        self.statistics_first_cycle_time_ns = None
        self.statistics = {
            name: {'sum': 0.0, 'min': None, 'max': None}
            for name in ('cells', 'points', 'wall_ms', 'cpu_ms')
        }

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
            durability=DurabilityPolicy.VOLATILE,
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

        self.get_logger().info(
            f'CV grid to cloud: {self.input_topic} -> {self.output_topic}, '
            f'frame={self.output_frame}, voxel={self.voxel_size or "native"}'
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
    def _apply_planar_transform(x, y, transform):
        yaw = CvToPointCloud._yaw(transform.transform.rotation)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return (
            transform.transform.translation.x + cos_yaw * x - sin_yaw * y,
            transform.transform.translation.y + sin_yaw * x + cos_yaw * y,
        )

    def _lookup_transform(self, source_frame, stamp):
        if not source_frame:
            raise TransformException('Input grid has an empty frame_id')
        try:
            return self.tf_buffer.lookup_transform(
                self.output_frame,
                source_frame,
                Time.from_msg(stamp),
            )
        except TransformException:
            return self.tf_buffer.lookup_transform(
                self.output_frame,
                source_frame,
                Time(),
            )

    def _aggregate_source_blocks(self, msg):
        """First reduction: pool eligible source cells in fixed blocks."""
        costs = np.asarray(msg.data, dtype=np.float32)
        expected_size = msg.info.width * msg.info.height
        if costs.size != expected_size:
            raise ValueError(
                'OccupancyGrid data size does not match its geometry'
            )

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
            block_counts >= self.minimum_cells_per_block
        )
        selected_costs = block_costs[block_rows, block_cols]
        if selected_costs.size == 0:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty(0, dtype=np.float32),
            )

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
            col_starts.astype(np.float64) + 0.5 * col_sizes
        ) * msg.info.resolution
        local_y = (
            row_starts.astype(np.float64) + 0.5 * row_sizes
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

        if msg.header.frame_id == self.output_frame:
            output_x, output_y = source_x, source_y
        else:
            transform = self._lookup_transform(
                msg.header.frame_id,
                msg.header.stamp,
            )
            output_x, output_y = self._apply_planar_transform(
                source_x,
                source_y,
                transform,
            )
        return np.column_stack((output_x, output_y)), selected_costs

    @staticmethod
    def _voxelize(points_xy, costs, resolution):
        """Second reduction: collapse block samples on a metric voxel grid."""
        if points_xy.size == 0:
            return (
                np.empty((0, 2), dtype=np.int64),
                np.empty(0, dtype=np.float32),
            )
        finite = np.all(np.isfinite(points_xy), axis=1) & np.isfinite(costs)
        points_xy = points_xy[finite]
        costs = costs[finite]
        if points_xy.size == 0:
            return (
                np.empty((0, 2), dtype=np.int64),
                np.empty(0, dtype=np.float32),
            )

        indices = np.floor(points_xy / resolution).astype(np.int64)
        unique_indices, inverse = np.unique(
            indices,
            axis=0,
            return_inverse=True,
        )
        ordered = np.lexsort((-costs, inverse))
        first_in_voxel = np.concatenate((
            np.array([True]),
            inverse[ordered][1:] != inverse[ordered][:-1],
        ))
        representatives = ordered[first_in_voxel]
        return (
            unique_indices,
            costs[representatives].astype(np.float32, copy=False),
        )

    def _downsample_high_cost(self, indices, costs):
        """Apply the extra coarse pass only to costs above the threshold."""
        factor = self.high_cost_downsampling_factor
        high = costs > self.high_cost_threshold
        if factor <= 1 or not np.any(high):
            return indices, costs

        low_indices = indices[~high]
        low_costs = costs[~high]
        high_indices = indices[high]
        high_costs = costs[high]
        coarse_indices = np.floor_divide(high_indices, factor)
        _, inverse = np.unique(
            coarse_indices,
            axis=0,
            return_inverse=True,
        )
        ordered = np.lexsort((-high_costs, inverse))
        first_in_group = np.concatenate((
            np.array([True]),
            inverse[ordered][1:] != inverse[ordered][:-1],
        ))
        representatives = ordered[first_in_group]
        return (
            np.concatenate((low_indices, high_indices[representatives])),
            np.concatenate((low_costs, high_costs[representatives])),
        )

    def _make_cloud(self, indices, costs, resolution, stamp):
        cloud_data = np.empty(costs.size, dtype=POINT_DTYPE)
        cloud_data['x'] = (indices[:, 0] + 0.5) * resolution
        cloud_data['y'] = (indices[:, 1] + 0.5) * resolution
        cloud_data['z'] = 0.0
        cloud_data['cost'] = costs
        cloud_data['confidence'] = 1.0

        cloud = PointCloud2()
        cloud.header.stamp = stamp
        cloud.header.frame_id = self.output_frame
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

    def _update_statistics(
        self,
        cell_count,
        point_count,
        wall_time_ms,
        cpu_time_ms,
        cycle_time_ns,
    ):
        """Accumulate and periodically log CV conversion workload."""
        values = {
            'cells': float(cell_count),
            'points': float(point_count),
            'wall_ms': wall_time_ms,
            'cpu_ms': cpu_time_ms,
        }
        self.statistics_count += 1
        if self.statistics_first_cycle_time_ns is None:
            self.statistics_first_cycle_time_ns = cycle_time_ns
        for name, value in values.items():
            statistic = self.statistics[name]
            statistic['sum'] += value
            statistic['min'] = (
                value if statistic['min'] is None
                else min(statistic['min'], value)
            )
            statistic['max'] = (
                value if statistic['max'] is None
                else max(statistic['max'], value)
            )

        if self.statistics_count < self.statistics_window_cycles:
            return

        elapsed_sec = (
            cycle_time_ns - self.statistics_first_cycle_time_ns
        ) * 1.0e-9
        frequency_hz = 0.0
        if elapsed_sec > 0.0 and self.statistics_count > 1:
            frequency_hz = (self.statistics_count - 1) / elapsed_sec

        def summary(name, precision):
            statistic = self.statistics[name]
            average = statistic['sum'] / self.statistics_count
            return (
                f'{average:.{precision}f}/'
                f'{statistic["min"]:.{precision}f}/'
                f'{statistic["max"]:.{precision}f}'
            )

        self.get_logger().info(
            f'CV to cloud over {self.statistics_count} cycles: '
            f'cells avg/min/max={summary("cells", 1)}; '
            f'points avg/min/max={summary("points", 1)}; '
            f'wall_ms avg/min/max={summary("wall_ms", 3)}; '
            f'cpu_ms avg/min/max={summary("cpu_ms", 3)}; '
            f'frequency={frequency_hz:.2f} Hz'
        )
        self.statistics_count = 0
        self.statistics_first_cycle_time_ns = None
        self.statistics = {
            name: {'sum': 0.0, 'min': None, 'max': None}
            for name in self.statistics
        }

    def grid_callback(self, msg):
        cycle_time_ns = time.perf_counter_ns()
        cpu_start_ns = time.process_time_ns()
        resolution = self.voxel_size or float(msg.info.resolution)
        if resolution <= 0.0:
            self.get_logger().warn(
                'Cannot convert a grid with non-positive resolution',
                throttle_duration_sec=2.0,
            )
            return
        try:
            points_xy, costs = self._aggregate_source_blocks(msg)
            indices, costs = self._voxelize(
                points_xy,
                costs,
                resolution,
            )
            indices, costs = self._downsample_high_cost(indices, costs)
        except (TransformException, ValueError) as exc:
            self.get_logger().warn(
                f'Cannot convert CV grid: {exc}',
                throttle_duration_sec=2.0,
            )
            return

        if costs.size > self.maximum_points:
            strongest = np.argpartition(
                costs,
                -self.maximum_points,
            )[-self.maximum_points:]
            indices = indices[strongest]
            costs = costs[strongest]
        self.cloud_pub.publish(
            self._make_cloud(indices, costs, resolution, msg.header.stamp)
        )
        wall_time_ms = (time.perf_counter_ns() - cycle_time_ns) * 1.0e-6
        cpu_time_ms = (time.process_time_ns() - cpu_start_ns) * 1.0e-6
        self._update_statistics(
            msg.info.width * msg.info.height,
            costs.size,
            wall_time_ms,
            cpu_time_ms,
            cycle_time_ns,
        )


def main(args=None):
    rclpy.init(args=args)
    node = CvToPointCloud()
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
