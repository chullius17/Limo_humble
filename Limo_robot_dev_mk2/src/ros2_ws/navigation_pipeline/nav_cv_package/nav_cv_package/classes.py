#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class Classification(Node):
    """Classifies white BEV pixels according to their distance from blue pixels."""

    BLUE_BGR = np.array([255, 0, 0], dtype=np.uint8)
    WHITE_BGR = np.array([255, 255, 255], dtype=np.uint8)
    MAGENTA_BGR = np.array([255, 0, 255], dtype=np.uint8)

    def __init__(self):
        super().__init__('classification')
        self.bridge = CvBridge()

        self.declare_parameter('input_topic', 'limo/nav_cv_package/bev/bird_perspective/raw')
        self.declare_parameter('debug_topic', 'limo/nav_cv_package/classification/debug/raw')
        self.declare_parameter('output_topic', 'limo/nav_cv_package/classification/output/raw')
        self.declare_parameter('blue_distance_threshold_px', 8.0)
        self.declare_parameter('magenta_distance_threshold_px', 8.0)
        self.declare_parameter('color_tolerance', 30)

        self.input_topic = self.get_parameter('input_topic').value
        self.debug_topic = self.get_parameter('debug_topic').value
        self.output_topic = self.get_parameter('output_topic').value

        self.debug_pub = self.create_publisher(Image, self.debug_topic, 10)
        self.output_pub = self.create_publisher(Image, self.output_topic, 10)
        self.image_sub = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            10,
        )

        self.get_logger().info(
            f'Classification node listening on {self.input_topic}; '
            f'publishing debug images on {self.debug_topic} and final output on '
            f'{self.output_topic}'
        )

    @staticmethod
    def _color_mask(image: np.ndarray, color: np.ndarray, tolerance: int) -> np.ndarray:
        lower = np.clip(color.astype(np.int16) - tolerance, 0, 255).astype(np.uint8)
        upper = np.clip(color.astype(np.int16) + tolerance, 0, 255).astype(np.uint8)
        return cv2.inRange(image, lower, upper) > 0

    @classmethod
    def classify_images(
        cls,
        image: np.ndarray,
        blue_distance_threshold_px: float,
        magenta_distance_threshold_px: float,
        color_tolerance: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the first-pass debug image and the propagated final output."""
        white_mask = cls._color_mask(image, cls.WHITE_BGR, color_tolerance)
        blue_mask = cls._color_mask(image, cls.BLUE_BGR, color_tolerance)

        if np.any(blue_mask):
            # distanceTransform measures the distance of every non-zero pixel from
            # the nearest zero pixel. Only blue pixels are set to zero, therefore
            # turquoise and every other color cannot act as distance sources.
            distance_input = np.full(image.shape[:2], 255, dtype=np.uint8)
            distance_input[blue_mask] = 0
            distance_from_blue = cv2.distanceTransform(
                distance_input,
                cv2.DIST_L2,
                cv2.DIST_MASK_PRECISE,
            )
            far_white_mask = white_mask & (
                distance_from_blue > blue_distance_threshold_px
            )
        else:
            far_white_mask = white_mask

        debug_image = image.copy()
        debug_image[far_white_mask] = cls.MAGENTA_BGR

        # Apply the second distance transform to the first-pass image. Only
        # magenta pixels are distance sources; all other colors are ignored.
        remaining_white_mask = cls._color_mask(
            debug_image,
            cls.WHITE_BGR,
            color_tolerance,
        )
        magenta_mask = cls._color_mask(
            debug_image,
            cls.MAGENTA_BGR,
            color_tolerance,
        )

        output_image = debug_image.copy()
        if np.any(magenta_mask):
            distance_input = np.full(image.shape[:2], 255, dtype=np.uint8)
            distance_input[magenta_mask] = 0
            distance_from_magenta = cv2.distanceTransform(
                distance_input,
                cv2.DIST_L2,
                cv2.DIST_MASK_PRECISE,
            )
            close_white_mask = remaining_white_mask & (
                distance_from_magenta < magenta_distance_threshold_px
            )
            output_image[close_white_mask] = cls.MAGENTA_BGR

        return debug_image, output_image

    def image_callback(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Failed to convert BEV image: {exc}')
            return

        blue_threshold_px = float(
            self.get_parameter('blue_distance_threshold_px').value
        )
        magenta_threshold_px = float(
            self.get_parameter('magenta_distance_threshold_px').value
        )
        tolerance = int(self.get_parameter('color_tolerance').value)

        if blue_threshold_px < 0.0:
            self.get_logger().error(
                'blue_distance_threshold_px must be non-negative'
            )
            return
        if magenta_threshold_px < 0.0:
            self.get_logger().error(
                'magenta_distance_threshold_px must be non-negative'
            )
            return
        if not 0 <= tolerance <= 255:
            self.get_logger().error('color_tolerance must be between 0 and 255')
            return

        debug_image, output_image = self.classify_images(
            image,
            blue_distance_threshold_px=blue_threshold_px,
            magenta_distance_threshold_px=magenta_threshold_px,
            color_tolerance=tolerance,
        )

        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)

        output_msg = self.bridge.cv2_to_imgmsg(output_image, encoding='bgr8')
        output_msg.header = msg.header
        self.output_pub.publish(output_msg)


def main(args=None):
    rclpy.init(args=args)
    node = Classification()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
