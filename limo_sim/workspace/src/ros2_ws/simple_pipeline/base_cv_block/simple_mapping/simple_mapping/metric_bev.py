#!/usr/bin/env python3
"""Build one combined metric costmap from the classified BEV image."""

import array
import queue
import threading
from typing import Tuple

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
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
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster


class MetricBev(Node):
    """Convert classified BEV colors into a single combined costmap."""

    COLOR_MAP = {
        'TURQUOISE': np.array([255, 255, 0], dtype=np.uint8),
        'WHITE': np.array([255, 255, 255], dtype=np.uint8),
        'MAGENTA': np.array([255, 0, 255], dtype=np.uint8),
    }
    CONFIG_MAP = {
        'TURQUOISE': {'peak_cost': 60.0, 'radius': 5},
        'WHITE': {
            'peak_cost': 40.0,
            'radius': 10,
            # Keep white below the value 50 reserved for unknown cells in the
            # categorical global map. This also keeps the color-cost bands
            # disjoint: white 40-49, turquoise 60, magenta 100.
            'interior_max_cost': 49.0,
            'interior_radius': 220,
        },
        'MAGENTA': {'peak_cost': 100.0, 'radius': 2},
    }

    TOLERANCE = 30
    DECAY = 8.0
    ROI_FRACTION = 0.7
    RULER_FRACTION = 0.08
    VERTICAL_SCALE_PER_METER = 8.0

    def __init__(self):
        super().__init__('metric_bev')
        self.bridge = CvBridge()

        self.declare_parameter(
            'input_topic',
            'limo/cv_package/classification/output/raw',
        )
        self.declare_parameter(
            'costmap_topic',
            '/limo/simple_mapping/metric_bev/costmap_grid_combined',
        )
        self.declare_parameter(
            'roi_debug_topic',
            '/limo/simple_mapping/metric_bev/roi_debug',
        )
        self.declare_parameter('fixed_frame', 'base_link')
        self.declare_parameter('costmap_frame', 'cv_origin_combined')
        self.declare_parameter('resolution', 0.0092)
        self.declare_parameter('publish_debug', True)

        self.fixed_frame = str(self.get_parameter('fixed_frame').value)
        self.costmap_frame = str(self.get_parameter('costmap_frame').value)
        self.resolution = float(self.get_parameter('resolution').value)
        self.publish_debug = bool(
            self.get_parameter('publish_debug').value)

        map_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.costmap_pub = self.create_publisher(
            OccupancyGrid,
            str(self.get_parameter('costmap_topic').value),
            map_qos,
        )
        self.roi_debug_pub = None
        if self.publish_debug:
            self.roi_debug_pub = self.create_publisher(
                Image,
                str(self.get_parameter('roi_debug_topic').value),
                10,
            )

        self.tf_broadcaster = TransformBroadcaster(self)
        self.latest_stamp = None
        self.tf_timer = self.create_timer(0.1, self._publish_transform)

        self.bev_queue = queue.Queue(maxsize=1)
        self.bev_sub = self.create_subscription(
            Image,
            str(self.get_parameter('input_topic').value),
            self._bev_callback,
            10,
        )
        self.worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
        )
        self.worker.start()

        self.get_logger().info(
            'Metric BEV initialized: publishing one combined costmap')

    def _publish_transform(self) -> None:
        transform = TransformStamped()
        transform.header.stamp = (
            self.latest_stamp
            if self.latest_stamp is not None
            else self.get_clock().now().to_msg()
        )
        transform.header.frame_id = self.fixed_frame
        transform.child_frame_id = self.costmap_frame
        transform.transform.translation.x = 0.6
        transform.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(transform)

    def _crop_to_roi(
        self,
        bgr: np.ndarray,
    ) -> Tuple[np.ndarray, int, int]:
        height = bgr.shape[0]
        roi_bottom = int(round(height * (1.0 - self.ROI_FRACTION)))
        roi_top = int(round(height * (1.0 - self.RULER_FRACTION)))
        return bgr[roi_bottom:roi_top, :], roi_bottom, roi_top

    def _make_mask(
        self,
        bgr: np.ndarray,
        exact_color: np.ndarray,
    ) -> np.ndarray:
        color = exact_color.astype(np.int16)
        lower = np.clip(color - self.TOLERANCE, 0, 255).astype(np.uint8)
        upper = np.clip(color + self.TOLERANCE, 0, 255).astype(np.uint8)
        return cv2.inRange(bgr, lower, upper)

    @staticmethod
    def _axis_distance_to_zero(
        binary: np.ndarray,
        axis: int,
    ) -> np.ndarray:
        foreground = binary > 0
        size = binary.shape[axis]
        index_shape = [1, 1]
        index_shape[axis] = size
        indexes = np.broadcast_to(
            np.arange(size).reshape(index_shape),
            binary.shape,
        )

        backward_source = np.where(foreground, -1, indexes)
        backward = np.maximum.accumulate(backward_source, axis=axis)
        backward_distance = np.where(
            backward < 0,
            size,
            indexes - backward,
        )

        forward_source = np.where(foreground, size, indexes)
        forward = np.flip(
            np.minimum.accumulate(
                np.flip(forward_source, axis=axis),
                axis=axis,
            ),
            axis=axis,
        )
        forward_distance = np.where(
            forward >= size,
            size,
            forward - indexes,
        )
        return np.minimum(
            backward_distance,
            forward_distance,
        ).astype(np.float32)

    def _inflate_layer(
        self,
        mask: np.ndarray,
        peak_cost: float,
        radius_px: int,
        interior_max_cost: float = None,
        interior_radius: float = None,
    ) -> np.ndarray:
        obstacle = cv2.bitwise_not(mask)
        distance = cv2.distanceTransform(
            obstacle,
            cv2.DIST_L2,
            cv2.DIST_MASK_3,
        )
        cost_layer = np.zeros_like(distance, dtype=np.float32)
        within_radius = distance <= radius_px
        if np.any(within_radius):
            normalized = distance[within_radius] / max(radius_px, 1)
            cost_layer[within_radius] = (
                peak_cost * np.exp(-self.DECAY * normalized)
            )

        if interior_max_cost is not None:
            mask_pixels = mask > 0
            if np.any(mask_pixels):
                horizontal = self._axis_distance_to_zero(mask, axis=1)
                vertical = self._axis_distance_to_zero(mask, axis=0)
                height = mask.shape[0]
                row_indexes = np.arange(height).reshape(-1, 1)
                distance_from_robot = (
                    (height - 1) - row_indexes
                ) * self.resolution
                vertical_scale = (
                    1.0
                    + self.VERTICAL_SCALE_PER_METER * distance_from_robot
                )
                interior_distance = np.minimum(
                    horizontal,
                    vertical * vertical_scale,
                )
                normalized = (
                    interior_distance[mask_pixels]
                    / max(interior_radius, 1e-3)
                )
                cost_layer[mask_pixels] = (
                    peak_cost
                    + (interior_max_cost - peak_cost)
                    * (1.0 - np.exp(-self.DECAY * normalized))
                )

        return np.clip(cost_layer, 0, 100).astype(np.uint8)

    def _image_to_costmap(
        self,
        bgr: np.ndarray,
        color_name: str,
    ) -> np.ndarray:
        config = self.CONFIG_MAP[color_name]
        mask = self._make_mask(bgr, self.COLOR_MAP[color_name])
        return self._inflate_layer(
            mask,
            config['peak_cost'],
            config['radius'],
            interior_max_cost=config.get('interior_max_cost'),
            interior_radius=config.get('interior_radius'),
        )

    def _to_occupancy_grid(
        self,
        cost_image: np.ndarray,
        header: Header,
    ) -> OccupancyGrid:
        rotated = cv2.rotate(cost_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated = cv2.flip(rotated, 1)
        height, width = rotated.shape

        grid = OccupancyGrid()
        grid.header.stamp = header.stamp
        grid.header.frame_id = self.costmap_frame
        grid.info.resolution = self.resolution
        grid.info.width = width
        grid.info.height = height
        grid.info.origin.position.x = 0.0
        grid.info.origin.position.y = -(height * self.resolution) / 2.0
        grid.info.origin.orientation.w = 1.0
        grid.data = array.array('b', rotated.astype(np.int8).tobytes())
        return grid

    def _render_roi_debug(
        self,
        image: np.ndarray,
        roi_bottom: int,
        roi_top: int,
    ) -> np.ndarray:
        debug = image.copy()
        width = image.shape[1]
        dark = np.zeros_like(debug[:roi_bottom, :])
        cv2.addWeighted(
            debug[:roi_bottom, :],
            0.3,
            dark,
            0.7,
            0,
            debug[:roi_bottom, :],
        )
        cv2.line(
            debug,
            (0, roi_bottom),
            (width - 1, roi_bottom),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            debug,
            f'ROI: bottom {self.ROI_FRACTION:.0%}',
            (8, roi_bottom - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.line(
            debug,
            (0, roi_top),
            (width - 1, roi_top),
            (255, 255, 0),
            1,
        )
        for x_position in range(0, width, 10):
            major_tick = x_position % 50 == 0
            tick_size = 8 if major_tick else 4
            thickness = 2 if major_tick else 1
            cv2.line(
                debug,
                (x_position, roi_top - tick_size),
                (x_position, roi_top + tick_size),
                (255, 255, 0),
                thickness,
            )
            if major_tick and 0 < x_position < width - 20:
                cv2.putText(
                    debug,
                    str(x_position),
                    (x_position - 10, roi_top - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
        return debug

    def _bev_callback(self, message: Image) -> None:
        self.latest_stamp = message.header.stamp
        if self.bev_queue.full():
            try:
                self.bev_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.bev_queue.put_nowait(message)
        except queue.Full:
            pass

    def _worker_loop(self) -> None:
        while rclpy.ok():
            try:
                message = self.bev_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._process_frame(message)

    def _process_frame(self, message: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding='bgr8',
            )
        except Exception as error:
            self.get_logger().error(f'cv_bridge error: {error}')
            return

        roi, roi_bottom, roi_top = self._crop_to_roi(bgr)
        color_layers = [
            self._image_to_costmap(roi, color_name)
            for color_name in self.COLOR_MAP
        ]
        combined = np.maximum.reduce(color_layers)
        self.costmap_pub.publish(
            self._to_occupancy_grid(combined, message.header))

        if (
            self.roi_debug_pub is not None
            and self.roi_debug_pub.get_subscription_count() > 0
        ):
            debug_image = self._render_roi_debug(
                bgr,
                roi_bottom,
                roi_top,
            )
            debug_message = self.bridge.cv2_to_imgmsg(
                debug_image,
                encoding='bgr8',
            )
            debug_message.header = message.header
            self.roi_debug_pub.publish(debug_message)


def main(args=None):
    rclpy.init(args=args)
    node = MetricBev()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
