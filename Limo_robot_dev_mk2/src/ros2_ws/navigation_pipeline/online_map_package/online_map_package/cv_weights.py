#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.msg import ParticleCloud
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image, PointCloud2


class CvWeights(Node):
    """Publish a normalized distance image from the static CV map."""

    def __init__(self):
        super().__init__('cv_weights')
        self.bridge = CvBridge()

        self.declare_parameter(
            'input_topic',
            '/limo/nav_map_package/online/maps/cv_map',
        )
        self.declare_parameter(
            'debug_topic',
            '/limo/nav_map_package/online/cv_weights/debug',
        )
        self.declare_parameter(
            'pointcloud_topic',
            '/limo/nav_map_package/online/cv_2_ptcld/points',
        )
        self.declare_parameter('particle_cloud_topic', '/particle_cloud')
        self.declare_parameter('occupied_threshold', 50)

        input_topic = str(self.get_parameter('input_topic').value)
        debug_topic = str(self.get_parameter('debug_topic').value)
        pointcloud_topic = str(self.get_parameter('pointcloud_topic').value)
        particle_cloud_topic = str(
            self.get_parameter('particle_cloud_topic').value
        )
        self.latest_cv_cloud = None
        self.latest_particle_cloud = None

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.debug_pub = self.create_publisher(
            Image,
            debug_topic,
            qos_profile_sensor_data,
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            input_topic,
            self.map_callback,
            map_qos,
        )
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            pointcloud_topic,
            self.cloud_callback,
            qos_profile_sensor_data,
        )
        self.particle_cloud_sub = self.create_subscription(
            ParticleCloud,
            particle_cloud_topic,
            self.particle_cloud_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f'Computing normalized distance from CV-map obstacles: '
            f'{input_topic} -> {debug_topic}; '
            f'CV cloud input: {pointcloud_topic}; '
            f'AMCL particles input: {particle_cloud_topic}'
        )

    def cloud_callback(self, msg: PointCloud2) -> None:
        """Cache the latest output produced by cv_2_ptcld."""
        self.latest_cv_cloud = msg

    def particle_cloud_callback(self, msg: ParticleCloud) -> None:
        """Cache AMCL particle poses and their associated weights."""
        self.latest_particle_cloud = msg

    @staticmethod
    def normalized_distance(
        occupancy: np.ndarray,
        occupied_threshold: int,
    ) -> np.ndarray:
        """Return a mono8 L2 distance image from occupied map cells."""
        obstacles = occupancy >= occupied_threshold
        if not np.any(obstacles):
            return np.zeros(occupancy.shape, dtype=np.uint8)

        # Black pixels in the source PGM become occupied cells (value 100) in
        # the OccupancyGrid. OpenCV expects distance sources to be zero.
        distance_input = np.full(occupancy.shape, 255, dtype=np.uint8)
        distance_input[obstacles] = 0
        distances = cv2.distanceTransform(
            distance_input,
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )
        maximum = float(np.max(distances))
        if maximum <= 0.0:
            return np.zeros(occupancy.shape, dtype=np.uint8)
        return np.rint(distances * (255.0 / maximum)).astype(np.uint8)

    def map_callback(self, msg: OccupancyGrid) -> None:
        threshold = int(self.get_parameter('occupied_threshold').value)
        if not 0 <= threshold <= 100:
            self.get_logger().error(
                'occupied_threshold must be between 0 and 100'
            )
            return

        width = int(msg.info.width)
        height = int(msg.info.height)
        if width <= 0 or height <= 0 or len(msg.data) != width * height:
            self.get_logger().error(
                f'Invalid CV map: {width}x{height}, {len(msg.data)} cells'
            )
            return

        try:
            occupancy = np.asarray(msg.data, dtype=np.int16).reshape(
                height,
                width,
            )
            debug_image = self.normalized_distance(occupancy, threshold)
        except cv2.error as error:
            self.get_logger().error(
                f'Cannot compute CV distance transform: {error}'
            )
            return

        output = self.bridge.cv2_to_imgmsg(debug_image, encoding='mono8')
        output.header = msg.header
        self.debug_pub.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = CvWeights()
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
