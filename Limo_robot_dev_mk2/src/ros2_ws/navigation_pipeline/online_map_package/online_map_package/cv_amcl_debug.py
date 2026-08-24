#!/usr/bin/env python3
"""Publish subsampled debug images for AMCL's two local CV grids."""

import math
from functools import partial

import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image


class CvAmclDebug(Node):
    """Discretize and render obstacle and street grids; nothing else."""

    def __init__(self):
        super().__init__('cv_amcl_debug')
        self.bridge = CvBridge()

        self.declare_parameter(
            'obstacle_grid_topic',
            '/limo/nav_map_package/online/metric_bev/'
            'cost_grid_binary_obstacles',
        )
        self.declare_parameter(
            'street_grid_topic',
            '/limo/nav_map_package/online/metric_bev/'
            'cost_grid_binary_street',
        )
        self.declare_parameter(
            'obstacle_grid_subsampled_topic',
            '/limo/nav_map_package/online/cv_amcl_debug/'
            'subsampled_obstacles',
        )
        self.declare_parameter(
            'street_grid_subsampled_topic',
            '/limo/nav_map_package/online/cv_amcl_debug/'
            'subsampled_street',
        )
        self.declare_parameter('occupied_threshold', 50)
        self.declare_parameter('sad_cell_size', 0.05)

        self.occupied_threshold = int(
            self.get_parameter('occupied_threshold').value
        )
        self.cell_size = float(self.get_parameter('sad_cell_size').value)
        if not 0 <= self.occupied_threshold <= 100:
            raise ValueError('occupied_threshold must be between 0 and 100')
        if self.cell_size <= 0.0:
            raise ValueError('sad_cell_size must be positive')

        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.obstacle_pub = self.create_publisher(
            Image,
            str(self.get_parameter('obstacle_grid_subsampled_topic').value),
            latched_qos,
        )
        self.street_pub = self.create_publisher(
            Image,
            str(self.get_parameter('street_grid_subsampled_topic').value),
            latched_qos,
        )
        self.create_subscription(
            OccupancyGrid,
            str(self.get_parameter('obstacle_grid_topic').value),
            partial(self._grid_callback, publisher=self.obstacle_pub),
            latched_qos,
        )
        self.create_subscription(
            OccupancyGrid,
            str(self.get_parameter('street_grid_topic').value),
            partial(self._grid_callback, publisher=self.street_pub),
            latched_qos,
        )

        self.get_logger().info(
            f'Publishing only the two CV grids subsampled at '
            f'{self.cell_size:g} m'
        )

    def _grid_callback(self, msg: OccupancyGrid, publisher) -> None:
        width = int(msg.info.width)
        height = int(msg.info.height)
        resolution = float(msg.info.resolution)
        if (
            width <= 0
            or height <= 0
            or resolution <= 0.0
            or len(msg.data) != width * height
        ):
            self.get_logger().warning('Ignoring malformed local CV grid')
            return

        # Match C++ std::lround for positive values.
        block_size = max(
            1,
            int(math.floor(self.cell_size / resolution + 0.5)),
        )
        occupancy = np.asarray(msg.data, dtype=np.int16).reshape(
            height,
            width,
        )
        subsampled = np.full(
            (
                math.ceil(height / block_size),
                math.ceil(width / block_size),
            ),
            -1,
            dtype=np.int16,
        )

        for output_row, first_row in enumerate(range(0, height, block_size)):
            last_row = min(first_row + block_size, height)
            for output_col, first_col in enumerate(
                range(0, width, block_size)
            ):
                last_col = min(first_col + block_size, width)
                block = occupancy[first_row:last_row, first_col:last_col]
                known = block >= 0
                if np.any(known):
                    subsampled[output_row, output_col] = int(round(
                        100.0 * np.mean(
                            block[known] >= self.occupied_threshold
                        )
                    ))

        # Occupied is black, free is white and unknown is mid-gray.
        debug_image = np.full(subsampled.shape, 127, dtype=np.uint8)
        known = subsampled >= 0
        debug_image[known] = np.rint(
            255.0 * (1.0 - subsampled[known] / 100.0)
        ).astype(np.uint8)
        rotated_debug_image = np.rot90(
            np.flipud(debug_image),
            k=1,
        )
        output = self.bridge.cv2_to_imgmsg(
            rotated_debug_image,
            encoding='mono8',
        )
        output.header = msg.header
        publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = CvAmclDebug()
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
