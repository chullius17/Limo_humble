import copy
import math

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener


class NavMap(Node):
    """Combine the SLAM map, the CV cost grid and current lidar endpoints."""

    def __init__(self):
        super().__init__('nav_map')

        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('static_map_topic', '/map')
        self.declare_parameter(
            'cv_grid_topic',
            '/limo/nav_map_package/cv_map_display/cv_map_occupancy_grid',
        )
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter(
            'output_topic',
            '/limo/nav_map_package/nav_map/combined_grid',
        )
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('lidar_cost', 100)

        self.global_frame = self.get_parameter('global_frame').value
        self.static_map_topic = self.get_parameter('static_map_topic').value
        self.cv_grid_topic = self.get_parameter('cv_grid_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.lidar_cost = int(self.get_parameter('lidar_cost').value)

        if publish_rate_hz <= 0.0:
            raise ValueError('publish_rate_hz must be greater than zero')
        if not 0 <= self.lidar_cost <= 100:
            raise ValueError('lidar_cost must be between 0 and 100')

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.static_map = None
        self.cv_grid = None
        self.scan = None

        self.create_subscription(
            OccupancyGrid,
            self.static_map_topic,
            self.static_map_callback,
            map_qos,
        )
        self.create_subscription(
            OccupancyGrid,
            self.cv_grid_topic,
            self.cv_grid_callback,
            10,
        )
        self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            self.output_topic,
            map_qos,
        )
        self.create_timer(1.0 / publish_rate_hz, self.publish_combined_map)

        self.get_logger().info(
            f'Combining static={self.static_map_topic}, cv={self.cv_grid_topic}, '
            f'laser={self.scan_topic} -> {self.output_topic}'
        )

    def static_map_callback(self, msg: OccupancyGrid) -> None:
        self.static_map = msg

    def cv_grid_callback(self, msg: OccupancyGrid) -> None:
        self.cv_grid = msg

    def scan_callback(self, msg: LaserScan) -> None:
        self.scan = msg

    @staticmethod
    def _yaw_from_quaternion(quaternion) -> float:
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

    def _lookup_transform(self, source_frame: str, stamp):
        if not source_frame:
            raise TransformException('Input message has an empty frame_id')
        return self.tf_buffer.lookup_transform(
            self.global_frame,
            source_frame,
            Time.from_msg(stamp),
        )

    def _transform_xy(self, x: np.ndarray, y: np.ndarray, transform):
        yaw = self._yaw_from_quaternion(transform.transform.rotation)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        tx = transform.transform.translation.x
        ty = transform.transform.translation.y
        return (
            tx + cos_yaw * x - sin_yaw * y,
            ty + sin_yaw * x + cos_yaw * y,
        )

    def _world_to_output_cells(self, x: np.ndarray, y: np.ndarray):
        info = self.static_map.info
        origin_yaw = self._yaw_from_quaternion(info.origin.orientation)
        cos_yaw = math.cos(origin_yaw)
        sin_yaw = math.sin(origin_yaw)
        dx = x - info.origin.position.x
        dy = y - info.origin.position.y

        # Inverse rotation from the global frame into the OccupancyGrid axes.
        grid_x = cos_yaw * dx + sin_yaw * dy
        grid_y = -sin_yaw * dx + cos_yaw * dy
        cols = np.floor(grid_x / info.resolution).astype(np.int32)
        rows = np.floor(grid_y / info.resolution).astype(np.int32)
        valid = (
            (cols >= 0)
            & (cols < info.width)
            & (rows >= 0)
            & (rows < info.height)
        )
        return rows[valid], cols[valid], valid

    def _overlay_cv_grid(self, combined: np.ndarray) -> None:
        msg = self.cv_grid
        if msg is None or not msg.data:
            return

        data = np.asarray(msg.data, dtype=np.int16).reshape(
            msg.info.height,
            msg.info.width,
        )
        # Only positive costs are overlaid. Unknown and free CV cells do not
        # erase the SLAM layer.
        source_rows, source_cols = np.nonzero(data > 0)
        if source_rows.size == 0:
            return

        resolution = msg.info.resolution
        local_x = (source_cols + 0.5) * resolution
        local_y = (source_rows + 0.5) * resolution

        origin_yaw = self._yaw_from_quaternion(msg.info.origin.orientation)
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
            world_x, world_y = self._transform_xy(source_x, source_y, transform)
        rows, cols, valid = self._world_to_output_cells(world_x, world_y)
        costs = data[source_rows, source_cols][valid]
        flat_indices = rows * self.static_map.info.width + cols
        np.maximum.at(combined.ravel(), flat_indices, costs)

    def _overlay_scan(self, combined: np.ndarray) -> None:
        msg = self.scan
        if msg is None or not msg.ranges:
            return

        ranges = np.asarray(msg.ranges, dtype=np.float32)
        valid = (
            np.isfinite(ranges)
            & (ranges >= msg.range_min)
            & (ranges <= msg.range_max)
        )
        if not np.any(valid):
            return

        angles = msg.angle_min + np.arange(ranges.size) * msg.angle_increment
        scan_x = ranges[valid] * np.cos(angles[valid])
        scan_y = ranges[valid] * np.sin(angles[valid])

        transform = self._lookup_transform(msg.header.frame_id, msg.header.stamp)
        world_x, world_y = self._transform_xy(scan_x, scan_y, transform)
        rows, cols, _ = self._world_to_output_cells(world_x, world_y)
        combined[rows, cols] = np.maximum(combined[rows, cols], self.lidar_cost)

    def publish_combined_map(self) -> None:
        if self.static_map is None:
            return

        combined = np.asarray(self.static_map.data, dtype=np.int16).reshape(
            self.static_map.info.height,
            self.static_map.info.width,
        ).copy()

        try:
            self._overlay_cv_grid(combined)
        except TransformException as exc:
            self.get_logger().warn(
                f'CV transform unavailable: {exc}',
                throttle_duration_sec=2.0,
            )

        try:
            self._overlay_scan(combined)
        except TransformException as exc:
            self.get_logger().warn(
                f'Laser transform unavailable: {exc}',
                throttle_duration_sec=2.0,
            )

        output = OccupancyGrid()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = self.global_frame
        output.info = copy.deepcopy(self.static_map.info)
        output.data = np.clip(combined, -1, 100).astype(np.int8).ravel().tolist()
        self.map_pub.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = NavMap()
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
