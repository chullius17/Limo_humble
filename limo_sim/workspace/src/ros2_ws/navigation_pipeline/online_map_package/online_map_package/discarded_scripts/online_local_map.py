"""Combine online filtered semantic layers into one robot-centric cost map."""

import copy
import math

import cv2
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image


class OnlineLocalMap(Node):
    """Fuse the three filtered maps using the offline cost policy."""

    COLORS = ('turquoise', 'white', 'magenta')

    def __init__(self):
        super().__init__('online_local_map')

        input_prefix = '/limo/nav_map_package/online/filtering'
        self.declare_parameter(
            'turquoise_topic',
            f'{input_prefix}/map_paper_turquoise',
        )
        self.declare_parameter(
            'white_topic',
            f'{input_prefix}/map_paper_white',
        )
        self.declare_parameter(
            'magenta_topic',
            f'{input_prefix}/map_paper_magenta',
        )
        self.declare_parameter(
            'output_topic',
            '/limo/nav_map_package/online/local_map/combined_grid',
        )
        self.declare_parameter(
            'debug_image_topic',
            '/limo/nav_map_package/online/local_map/combined_image/raw',
        )
        self.declare_parameter('turquoise_factor', 0.6)
        self.declare_parameter('white_factor', 0.3)
        self.declare_parameter('magenta_factor', 1.0)

        self.factors = {
            color: float(self.get_parameter(f'{color}_factor').value)
            for color in self.COLORS
        }
        if any(factor < 0.0 for factor in self.factors.values()):
            raise ValueError('Semantic cost factors must be non-negative')

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        debug_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.maps = {color: None for color in self.COLORS}
        self.updated_layers = set()
        self._map_subscriptions = []
        for color in self.COLORS:
            topic = str(self.get_parameter(f'{color}_topic').value)
            subscription = self.create_subscription(
                OccupancyGrid,
                topic,
                lambda msg, layer=color: self._map_callback(layer, msg),
                map_qos,
            )
            self._map_subscriptions.append(subscription)

        output_topic = str(self.get_parameter('output_topic').value)
        debug_image_topic = str(
            self.get_parameter('debug_image_topic').value
        )
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(
            OccupancyGrid,
            output_topic,
            map_qos,
        )
        self.debug_image_publisher = self.create_publisher(
            Image,
            debug_image_topic,
            debug_qos,
        )

        self.get_logger().info(
            'Online local map started: '
            f'white x{self.factors["white"]:.1f}, '
            f'turquoise x{self.factors["turquoise"]:.1f}, '
            f'magenta x{self.factors["magenta"]:.1f}; '
            f'event-driven output={output_topic}, debug={debug_image_topic}'
        )

    def _map_callback(self, color: str, msg: OccupancyGrid) -> None:
        """Cache the newest filtered map for one semantic color."""
        expected_size = msg.info.width * msg.info.height
        if len(msg.data) != expected_size:
            self.get_logger().warn(
                f'Ignoring malformed {color} grid: data={len(msg.data)}, '
                f'expected={expected_size}',
                throttle_duration_sec=2.0,
            )
            return
        self.maps[color] = msg
        self.updated_layers.add(color)
        if len(self.updated_layers) == len(self.COLORS):
            self.updated_layers.clear()
            self._publish_combined()

    @staticmethod
    def _same_geometry(first: OccupancyGrid, second: OccupancyGrid) -> bool:
        """Check all metadata needed for cell-wise map combination."""
        first_origin = first.info.origin
        second_origin = second.info.origin
        return (
            first.header.frame_id == second.header.frame_id
            and first.info.width == second.info.width
            and first.info.height == second.info.height
            and math.isclose(
                first.info.resolution,
                second.info.resolution,
            )
            and math.isclose(
                first_origin.position.x,
                second_origin.position.x,
            )
            and math.isclose(
                first_origin.position.y,
                second_origin.position.y,
            )
            and math.isclose(
                first_origin.position.z,
                second_origin.position.z,
            )
            and math.isclose(
                first_origin.orientation.x,
                second_origin.orientation.x,
            )
            and math.isclose(
                first_origin.orientation.y,
                second_origin.orientation.y,
            )
            and math.isclose(
                first_origin.orientation.z,
                second_origin.orientation.z,
            )
            and math.isclose(
                first_origin.orientation.w,
                second_origin.orientation.w,
            )
        )

    def _combine_maps(self):
        """Apply the offline maximum-of-scaled-layers fusion rule."""
        reference = next(
            (self.maps[color] for color in self.COLORS
             if self.maps[color] is not None),
            None,
        )
        if reference is None:
            return None, None

        shape = (reference.info.height, reference.info.width)
        combined = np.zeros(shape, dtype=np.float32)
        seen = np.zeros(shape, dtype=bool)

        for color in self.COLORS:
            msg = self.maps[color]
            if msg is None:
                continue
            if not self._same_geometry(reference, msg):
                self.get_logger().warn(
                    f'Skipping {color} grid: geometry or frame differs '
                    'from the reference filtering grid',
                    throttle_duration_sec=2.0,
                )
                continue

            grid = np.asarray(msg.data, dtype=np.int16).reshape(shape)
            known = grid >= 0
            scaled = (
                np.clip(grid, 0, 100).astype(np.float32)
                * self.factors[color]
            )
            np.maximum(
                combined,
                np.where(known, scaled, 0.0),
                out=combined,
            )
            np.logical_or(seen, known, out=seen)

        output = np.where(
            seen,
            np.clip(combined, 0.0, 100.0),
            -1.0,
        ).astype(np.int8)
        return reference, output

    def _publish_combined(self) -> None:
        """Publish after every semantic layer has supplied a fresh map."""
        reference, combined = self._combine_maps()
        if reference is None:
            return

        output = OccupancyGrid()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = reference.header.frame_id
        output.info = copy.deepcopy(reference.info)
        output.data = combined.ravel().tolist()
        self.publisher.publish(output)

        if self.debug_image_publisher.get_subscription_count() > 0:
            self._publish_debug_image(combined, output.header)

    def _publish_debug_image(self, grid: np.ndarray, header) -> None:
        """Render the combined costs with base_link at the bottom centre."""
        known = grid >= 0
        normalized = (
            np.clip(grid, 0, 100).astype(np.float32) * 2.55
        ).astype(np.uint8)
        image = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        image[known & (grid == 0)] = (30, 30, 30)
        image[~known] = (0, 0, 0)

        # OccupancyGrid columns are base X and rows are base Y. Rotating the
        # image puts +X upwards; the horizontal flip maps base +Y to the
        # viewer's left, removing the previous left/right mirroring.
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image = cv2.flip(image, 1)
        height, width = image.shape[:2]
        robot = (width // 2, height - 1)
        arrow_length = max(1, min(20, height // 3))
        cv2.arrowedLine(
            image,
            robot,
            (robot[0], robot[1] - arrow_length),
            (255, 255, 255),
            1,
            tipLength=0.3,
        )

        image_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        image_msg.header = header
        self.debug_image_publisher.publish(image_msg)


def main(args=None):
    """Run the online local semantic map node."""
    rclpy.init(args=args)
    node = OnlineLocalMap()
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
