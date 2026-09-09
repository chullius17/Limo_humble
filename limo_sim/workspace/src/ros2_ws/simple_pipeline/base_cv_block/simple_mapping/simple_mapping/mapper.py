#!/usr/bin/env python3
"""Project and temporally filter the combined metric BEV on a global canvas."""

import math

from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class Mapper(Node):
    """Build a categorical global map from combined local costmaps."""

    def __init__(self):
        super().__init__('simple_mapper')

        self.declare_parameter(
            'input_topic',
            '/limo/simple_mapping/metric_bev/costmap_grid_combined',
        )
        self.declare_parameter(
            'output_topic',
            '/limo/simple_mapping/mapper/map',
        )
        self.declare_parameter('global_frame', 'odom')
        self.declare_parameter('resolution', 0.02)
        self.declare_parameter('map_size_meters', 15.0)
        self.declare_parameter('roi_x_min_m', 0.0)
        self.declare_parameter('roi_x_max_m', 1.85)
        self.declare_parameter('roi_width_near_m', 0.60)
        self.declare_parameter('roi_width_far_m', 2.65)
        self.declare_parameter(
            'roi_debug_topic',
            '/limo/simple_mapping/mapper/roi_debug',
        )
        self.declare_parameter('free_update_gain', 0.35)
        self.declare_parameter('occupied_update_gain', 0.60)

        # Class centers are inherited from MetricBev: 0, 40, 60 and 100.
        # Boundaries are their midpoints; 50 is reserved for unseen cells.
        self.declare_parameter('white_threshold', 20.0)
        self.declare_parameter('turquoise_threshold', 50.0)
        self.declare_parameter('magenta_threshold', 80.0)
        self.declare_parameter('free_value', 0)
        self.declare_parameter('white_value', 40)
        self.declare_parameter('unknown_value', 50)
        self.declare_parameter('turquoise_value', 60)
        self.declare_parameter('magenta_value', 100)
        self.declare_parameter('update_rate_hz', 20.0)

        self.input_topic = str(self.get_parameter('input_topic').value)
        self.output_topic = str(self.get_parameter('output_topic').value)
        self.global_frame = str(self.get_parameter('global_frame').value)
        self.resolution = float(self.get_parameter('resolution').value)
        self.map_size_meters = float(
            self.get_parameter('map_size_meters').value)
        self.roi_x_min_m = float(
            self.get_parameter('roi_x_min_m').value)
        self.roi_x_max_m = float(
            self.get_parameter('roi_x_max_m').value)
        self.roi_width_near_m = float(
            self.get_parameter('roi_width_near_m').value)
        self.roi_width_far_m = float(
            self.get_parameter('roi_width_far_m').value)
        self.roi_debug_topic = str(
            self.get_parameter('roi_debug_topic').value)
        self.free_update_gain = float(
            self.get_parameter('free_update_gain').value)
        self.occupied_update_gain = float(
            self.get_parameter('occupied_update_gain').value)

        self.white_threshold = float(
            self.get_parameter('white_threshold').value)
        self.turquoise_threshold = float(
            self.get_parameter('turquoise_threshold').value)
        self.magenta_threshold = float(
            self.get_parameter('magenta_threshold').value)
        self.free_value = int(self.get_parameter('free_value').value)
        self.white_value = int(self.get_parameter('white_value').value)
        self.unknown_value = int(self.get_parameter('unknown_value').value)
        self.turquoise_value = int(
            self.get_parameter('turquoise_value').value)
        self.magenta_value = int(
            self.get_parameter('magenta_value').value)
        update_rate_hz = float(
            self.get_parameter('update_rate_hz').value)

        self._validate_parameters(update_rate_hz)
        self.map_size_pixels = int(round(
            self.map_size_meters / self.resolution))

        # Continuous temporal state. Classification happens only at output.
        shape = (self.map_size_pixels, self.map_size_pixels)
        self.filtered_cost = np.zeros(shape, dtype=np.float32)
        self.seen = np.zeros(shape, dtype=bool)
        self.frame_observation = np.full(shape, -1.0, dtype=np.float32)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pending_costmap = None
        self.cached_geometry_key = None
        self.cached_local_x = None
        self.cached_local_y = None
        self.cached_roi_mask = None

        map_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.map_publisher = self.create_publisher(
            OccupancyGrid,
            self.output_topic,
            map_qos,
        )
        self.roi_debug_publisher = self.create_publisher(
            Image,
            self.roi_debug_topic,
            10,
        )
        self.costmap_subscription = self.create_subscription(
            OccupancyGrid,
            self.input_topic,
            self._costmap_callback,
            map_qos,
        )
        self.update_timer = self.create_timer(
            1.0 / update_rate_hz,
            self._process_pending_costmap,
        )
        self.output_message = self._new_output_message()

        self.get_logger().info(
            f'Simple mapper listening on {self.input_topic}; publishing '
            f'{self.output_topic} in {self.global_frame}.')
        self.get_logger().info(
            'Cost classes: [0,20)=free, [20,50)=white, '
            '[50,80)=turquoise, [80,100]=magenta; unseen=50.')
        self.get_logger().info(
            f'Update ROI: x=[{self.roi_x_min_m:.2f}, '
            f'{self.roi_x_max_m:.2f}] m, width '
            f'{self.roi_width_near_m:.2f} -> '
            f'{self.roi_width_far_m:.2f} m.')

    def _validate_parameters(self, update_rate_hz: float) -> None:
        if self.resolution <= 0.0 or self.map_size_meters <= 0.0:
            raise ValueError('resolution and map_size_meters must be positive')
        if update_rate_hz <= 0.0:
            raise ValueError('update_rate_hz must be positive')
        if self.roi_x_max_m <= self.roi_x_min_m:
            raise ValueError('roi_x_max_m must exceed roi_x_min_m')
        if self.roi_width_near_m < 0.0 or self.roi_width_far_m < 0.0:
            raise ValueError('ROI widths must be non-negative')
        gains = (
            self.free_update_gain,
            self.occupied_update_gain,
        )
        if any(gain < 0.0 or gain > 1.0 for gain in gains):
            raise ValueError('temporal gains must be in [0, 1]')
        if not (
            0.0 <= self.white_threshold
            < self.turquoise_threshold
            < self.magenta_threshold
            <= 100.0
        ):
            raise ValueError('class thresholds must be ordered in [0, 100]')
        output_values = (
            self.free_value,
            self.white_value,
            self.unknown_value,
            self.turquoise_value,
            self.magenta_value,
        )
        if any(value < 0 or value > 100 for value in output_values):
            raise ValueError('output class values must be in [0, 100]')

    def _new_output_message(self) -> OccupancyGrid:
        message = OccupancyGrid()
        message.header.frame_id = self.global_frame
        message.info.resolution = self.resolution
        message.info.width = self.map_size_pixels
        message.info.height = self.map_size_pixels
        half_size = self.map_size_meters / 2.0
        message.info.origin.position.x = -half_size
        message.info.origin.position.y = -half_size
        message.info.origin.orientation.w = 1.0
        return message

    def _costmap_callback(self, message: OccupancyGrid) -> None:
        # A single-slot mailbox prevents old local observations from building up.
        self.pending_costmap = message

    def _process_pending_costmap(self) -> None:
        message = self.pending_costmap
        if message is None:
            return
        source_frame = message.header.frame_id
        if not source_frame:
            self.get_logger().warn(
                'Combined costmap has no frame_id.',
                throttle_duration_sec=2.0,
            )
            self.pending_costmap = None
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.global_frame,
                source_frame,
                rclpy.time.Time(),
            )
        except TransformException as error:
            self.get_logger().warn(
                f'TF {self.global_frame} <- {source_frame} unavailable: '
                f'{error}',
                throttle_duration_sec=2.0,
            )
            return

        try:
            self._integrate_costmap(message, transform)
        except (TypeError, ValueError) as error:
            self.get_logger().error(f'Invalid combined costmap: {error}')
        finally:
            # Clear only if a newer callback did not replace this message.
            if self.pending_costmap is message:
                self.pending_costmap = None

    def _local_coordinates(self, message: OccupancyGrid):
        width = int(message.info.width)
        height = int(message.info.height)
        resolution = float(message.info.resolution)
        origin_x = float(message.info.origin.position.x)
        origin_y = float(message.info.origin.position.y)
        geometry_key = (
            width,
            height,
            resolution,
            origin_x,
            origin_y,
        )
        if geometry_key != self.cached_geometry_key:
            columns = np.arange(width, dtype=np.float32)
            rows = np.arange(height, dtype=np.float32)
            column_grid, row_grid = np.meshgrid(columns, rows)
            self.cached_local_x = (
                origin_x + (column_grid + 0.5) * resolution)
            self.cached_local_y = (
                origin_y + (row_grid + 0.5) * resolution)
            roi_length = self.roi_x_max_m - self.roi_x_min_m
            interpolation = (
                (self.cached_local_x - self.roi_x_min_m) / roi_length)
            allowed_width = (
                self.roi_width_near_m
                + interpolation
                * (self.roi_width_far_m - self.roi_width_near_m)
            )
            self.cached_roi_mask = (
                (self.cached_local_x >= self.roi_x_min_m)
                & (self.cached_local_x <= self.roi_x_max_m)
                & (np.abs(self.cached_local_y) <= allowed_width / 2.0)
            )
            self.cached_geometry_key = geometry_key
        return self.cached_local_x, self.cached_local_y

    def _integrate_costmap(self, message, transform) -> None:
        width = int(message.info.width)
        height = int(message.info.height)
        if width <= 0 or height <= 0:
            raise ValueError('grid dimensions must be positive')
        data = np.asarray(message.data, dtype=np.int16)
        if data.size != width * height:
            raise ValueError(
                f'expected {width * height} cells, received {data.size}')
        data = data.reshape((height, width))

        local_x, local_y = self._local_coordinates(message)
        if self.roi_debug_publisher.get_subscription_count() > 0:
            self._publish_roi_debug(message.header)
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = math.atan2(
            2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
            1.0 - 2.0 * (
                rotation.y * rotation.y + rotation.z * rotation.z),
        )
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        world_x = translation.x + cosine * local_x - sine * local_y
        world_y = translation.y + sine * local_x + cosine * local_y

        half_size = self.map_size_meters / 2.0
        canvas_x = np.floor(
            (world_x + half_size) / self.resolution).astype(np.int32)
        canvas_y = np.floor(
            (world_y + half_size) / self.resolution).astype(np.int32)
        valid = (
            (data >= 0)
            & self.cached_roi_mask
            & (canvas_x >= 0)
            & (canvas_x < self.map_size_pixels)
            & (canvas_y >= 0)
            & (canvas_y < self.map_size_pixels)
        )
        if not np.any(valid):
            return

        # Resolve spatial collisions inside this frame with the highest cost.
        linear_indexes = (
            canvas_y[valid] * self.map_size_pixels + canvas_x[valid])
        observations = np.clip(data[valid], 0, 100).astype(np.float32)
        frame_flat = self.frame_observation.ravel()
        frame_flat.fill(-1.0)
        np.maximum.at(frame_flat, linear_indexes, observations)
        unique_indexes = np.flatnonzero(frame_flat >= 0.0)
        frame_costs = frame_flat[unique_indexes]

        filtered_flat = self.filtered_cost.ravel()
        seen_flat = self.seen.ravel()
        previous = filtered_flat[unique_indexes]

        # EMA evidence is proportional to the measured cell cost. Starting
        # from zero, a measurement c adds occupied_update_gain * c; repeated
        # measurements converge to c instead of saturating every class at 100.
        gains = np.where(
            frame_costs > 0.0,
            self.occupied_update_gain,
            self.free_update_gain,
        )
        filtered = previous + gains * (frame_costs - previous)
        filtered_flat[unique_indexes] = filtered
        seen_flat[unique_indexes] = True

        self._publish_map(message.header.stamp)

    def _publish_map(self, stamp) -> None:
        classified = np.full(
            self.filtered_cost.shape,
            self.unknown_value,
            dtype=np.int8,
        )
        observed = self.seen
        classified[observed & (
            self.filtered_cost < self.white_threshold
        )] = self.free_value
        classified[observed & (
            self.filtered_cost >= self.white_threshold
        ) & (
            self.filtered_cost < self.turquoise_threshold
        )] = self.white_value
        classified[observed & (
            self.filtered_cost >= self.turquoise_threshold
        ) & (
            self.filtered_cost < self.magenta_threshold
        )] = self.turquoise_value
        classified[observed & (
            self.filtered_cost >= self.magenta_threshold
        )] = self.magenta_value

        self.output_message.header.stamp = stamp
        self.output_message.data = classified.ravel().tolist()
        self.map_publisher.publish(self.output_message)

    def _publish_roi_debug(self, header) -> None:
        """Publish the update ROI as a rotated black-and-white image."""
        debug = (self.cached_roi_mask.astype(np.uint8) * 255)
        debug = np.ascontiguousarray(np.rot90(debug))
        message = Image()
        message.header = header
        message.height = debug.shape[0]
        message.width = debug.shape[1]
        message.encoding = 'mono8'
        message.is_bigendian = False
        message.step = debug.shape[1]
        message.data = debug.tobytes()
        self.roi_debug_publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = Mapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
