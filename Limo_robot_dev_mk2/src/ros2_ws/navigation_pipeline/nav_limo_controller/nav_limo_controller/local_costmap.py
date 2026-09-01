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

"""Convert the persistent local pointcloud into a graded OccupancyGrid."""

import array
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
from sensor_msgs.msg import PointCloud2, PointField


REQUIRED_FIELDS = ('x', 'y', 'cost', 'confidence')


class LocalCostmap(Node):
    """Rasterize local point costs for the Nav2 controller costmap."""

    def __init__(self):
        super().__init__('local_costmap_converter')

        self.declare_parameter(
            'input_topic',
            '/limo/nav_map_package/online/local_ptcld',
        )
        self.declare_parameter(
            'output_topic',
            '/limo/nav_map_package/online/local_costmap',
        )
        self.declare_parameter('output_frame', 'base_link')
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('length', 2.46)
        self.declare_parameter('width', 2.66)
        self.declare_parameter('minimum_cost', 1.0)
        self.declare_parameter('minimum_confidence', 0.30)
        self.declare_parameter('scale_cost_by_confidence', False)
        self.declare_parameter('statistics_window_cycles', 30)

        self.input_topic = str(self.get_parameter('input_topic').value)
        self.output_topic = str(self.get_parameter('output_topic').value)
        self.output_frame = str(self.get_parameter('output_frame').value)
        self.resolution = float(self.get_parameter('resolution').value)
        self.length = float(self.get_parameter('length').value)
        self.width = float(self.get_parameter('width').value)
        self.minimum_cost = float(
            self.get_parameter('minimum_cost').value
        )
        self.minimum_confidence = float(
            self.get_parameter('minimum_confidence').value
        )
        self.scale_cost_by_confidence = bool(
            self.get_parameter('scale_cost_by_confidence').value
        )
        self.statistics_window_cycles = int(
            self.get_parameter('statistics_window_cycles').value
        )

        if not self.input_topic or not self.output_topic:
            raise ValueError('input_topic and output_topic must not be empty')
        if not self.output_frame:
            raise ValueError('output_frame must not be empty')
        for name, value in (
            ('resolution', self.resolution),
            ('length', self.length),
            ('width', self.width),
        ):
            if value <= 0.0:
                raise ValueError(f'{name} must be greater than zero')
        if not 0.0 <= self.minimum_cost <= 100.0:
            raise ValueError('minimum_cost must be between 0 and 100')
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError(
                'minimum_confidence must be between 0 and 1'
            )
        if self.statistics_window_cycles <= 0:
            raise ValueError(
                'statistics_window_cycles must be greater than zero'
            )

        self.cells_x = int(math.ceil(self.length / self.resolution))
        self.cells_y = int(math.ceil(self.width / self.resolution))
        self.grid_length = self.cells_x * self.resolution
        self.grid_width = self.cells_y * self.resolution
        self.origin_x = 0.0
        self.origin_y = -0.5 * self.grid_width

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
            self.pointcloud_callback,
            input_qos,
        )
        self.grid_pub = self.create_publisher(
            OccupancyGrid,
            self.output_topic,
            output_qos,
        )

        self.statistics_count = 0
        self.input_points_sum = 0
        self.occupied_cells_sum = 0
        self.wall_time_sum_ms = 0.0
        self.wall_time_min_ms = None
        self.wall_time_max_ms = None

        self.get_logger().info(
            f'Local costmap converter: {self.input_topic} -> '
            f'{self.output_topic}, frame={self.output_frame}, '
            f'geometry={self.cells_x}x{self.cells_y} at '
            f'{self.resolution:.3f}m, confidence>='
            f'{self.minimum_confidence:.2f}, '
            f'scale_by_confidence={self.scale_cost_by_confidence}'
        )

    def _read_cloud(self, msg):
        if msg.header.frame_id != self.output_frame:
            raise ValueError(
                f'Input frame must be {self.output_frame!r}, got '
                f'{msg.header.frame_id!r}'
            )
        point_count = int(msg.width) * int(msg.height)
        if point_count == 0:
            empty = np.empty(0, dtype=np.float32)
            return empty, empty, empty, empty
        if msg.point_step <= 0 or msg.row_step < msg.point_step * msg.width:
            raise ValueError('Input cloud has invalid point or row stride')
        if len(msg.data) < msg.row_step * msg.height:
            raise ValueError('Input cloud data is shorter than its geometry')

        fields = {field.name: field for field in msg.fields}
        for name in REQUIRED_FIELDS:
            field = fields.get(name)
            if field is None:
                raise ValueError(f'Input cloud is missing field {name!r}')
            if field.datatype != PointField.FLOAT32 or field.count != 1:
                raise ValueError(f'Input field {name!r} must be FLOAT32[1]')
            if field.offset < 0 or field.offset + 4 > msg.point_step:
                raise ValueError(f'Input field {name!r} has invalid offset')

        byte_order = '>' if msg.is_bigendian else '<'
        dtype = np.dtype({
            'names': list(REQUIRED_FIELDS),
            'formats': [f'{byte_order}f4'] * len(REQUIRED_FIELDS),
            'offsets': [fields[name].offset for name in REQUIRED_FIELDS],
            'itemsize': msg.point_step,
        })
        points = np.ndarray(
            shape=(msg.height, msg.width),
            dtype=dtype,
            buffer=msg.data,
            strides=(msg.row_step, msg.point_step),
        ).reshape(-1)
        return tuple(
            points[name].astype(np.float32, copy=False)
            for name in REQUIRED_FIELDS
        )

    def _rasterize(self, x, y, costs, confidences):
        finite = (
            np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(costs)
            & np.isfinite(confidences)
        )
        keep = (
            finite
            & (x >= self.origin_x)
            & (x < self.origin_x + self.grid_length)
            & (y >= self.origin_y)
            & (y < self.origin_y + self.grid_width)
            & (costs >= self.minimum_cost)
            & (confidences >= self.minimum_confidence)
        )
        grid = np.zeros((self.cells_y, self.cells_x), dtype=np.int16)
        if not np.any(keep):
            return grid

        x = x[keep]
        y = y[keep]
        costs = np.clip(costs[keep], 0.0, 100.0)
        if self.scale_cost_by_confidence:
            costs = costs * np.clip(confidences[keep], 0.0, 1.0)

        cell_x = np.floor(
            (x - self.origin_x) / self.resolution
        ).astype(np.int64)
        cell_y = np.floor(
            (y - self.origin_y) / self.resolution
        ).astype(np.int64)
        flat_indices = cell_y * self.cells_x + cell_x
        cell_costs = np.rint(costs).astype(np.int16)
        np.maximum.at(grid.ravel(), flat_indices, cell_costs)
        return grid

    def _make_grid(self, source, costs):
        grid = OccupancyGrid()
        grid.header.stamp = source.header.stamp
        grid.header.frame_id = self.output_frame
        grid.info.map_load_time = source.header.stamp
        grid.info.resolution = self.resolution
        grid.info.width = self.cells_x
        grid.info.height = self.cells_y
        grid.info.origin.position.x = self.origin_x
        grid.info.origin.position.y = self.origin_y
        grid.info.origin.orientation.w = 1.0
        grid.data = array.array(
            'b',
            costs.astype(np.int8, copy=False).ravel().tobytes(),
        )
        return grid

    def _update_statistics(
        self,
        input_points,
        occupied_cells,
        wall_time_ms,
    ):
        self.statistics_count += 1
        self.input_points_sum += input_points
        self.occupied_cells_sum += occupied_cells
        self.wall_time_sum_ms += wall_time_ms
        if self.wall_time_min_ms is None:
            self.wall_time_min_ms = wall_time_ms
            self.wall_time_max_ms = wall_time_ms
        else:
            self.wall_time_min_ms = min(
                self.wall_time_min_ms,
                wall_time_ms,
            )
            self.wall_time_max_ms = max(
                self.wall_time_max_ms,
                wall_time_ms,
            )
        if self.statistics_count < self.statistics_window_cycles:
            return

        count = self.statistics_count
        self.get_logger().info(
            f'Local costmap over {count} cycles: '
            f'input_points_avg={self.input_points_sum / count:.1f}; '
            f'occupied_cells_avg={self.occupied_cells_sum / count:.1f}; '
            f'wall_ms avg/min/max={self.wall_time_sum_ms / count:.3f}/'
            f'{self.wall_time_min_ms:.3f}/{self.wall_time_max_ms:.3f}'
        )
        self.statistics_count = 0
        self.input_points_sum = 0
        self.occupied_cells_sum = 0
        self.wall_time_sum_ms = 0.0
        self.wall_time_min_ms = None
        self.wall_time_max_ms = None

    def pointcloud_callback(self, msg):
        start_ns = time.perf_counter_ns()
        try:
            x, y, costs, confidences = self._read_cloud(msg)
            grid_costs = self._rasterize(x, y, costs, confidences)
        except ValueError as exc:
            self.get_logger().warn(
                f'Cannot update local costmap: {exc}',
                throttle_duration_sec=2.0,
            )
            return

        self.grid_pub.publish(self._make_grid(msg, grid_costs))
        wall_time_ms = (time.perf_counter_ns() - start_ns) * 1.0e-6
        self._update_statistics(
            int(msg.width) * int(msg.height),
            int(np.count_nonzero(grid_costs)),
            wall_time_ms,
        )


def main(args=None):
    rclpy.init(args=args)
    node = LocalCostmap()
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
