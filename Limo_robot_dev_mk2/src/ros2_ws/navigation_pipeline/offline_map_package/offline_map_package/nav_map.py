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
    """Build the offline source-specific and combined occupancy grids.

    The static SLAM map provides the geometry (size, resolution and origin) of
    every output. CV cells and lidar endpoints are transformed into that common
    grid. Three maps are published:

    * combined map: static SLAM map with CV and lidar obstacles overlaid;
    * laser map: current lidar endpoints only;
    * CV map: CV cells whose cost is above ``cv_cost_threshold`` only.

    The source-specific maps use zero (free) for cells not supplied by their
    source. They are snapshots of the latest received messages, not cumulative
    maps built over time.
    """

    def __init__(self):
        super().__init__('nav_map')

        # ------------------------------------------------------------------
        # Configuration
        # ------------------------------------------------------------------
        # All output coordinates are expressed in global_frame. The static map
        # also defines the width, height, resolution and origin of each output.
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('static_map_topic', '/map')
        self.declare_parameter(
            'cv_grid_topic',
            '/limo/nav_map_package/offline/cv_map_display/cv_map_occupancy_grid',
        )
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter(
            'output_topic',
            '/limo/nav_map_package/online/nav_map/combined_grid',
        )
        self.declare_parameter(
            'laser_map_topic',
            '/limo/nav_map_package/online/nav_map/laser_map',
        )
        self.declare_parameter(
            'cv_map_topic',
            '/limo/nav_map_package/online/nav_map/cv_map',
        )
        # lidar_cost is assigned to each valid scan endpoint. CV costs retain
        # their original 0--100 values when copied to an output map.
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('lidar_cost', 100)
        self.declare_parameter('cv_cost_threshold', 40.0)

        # Read parameters once at startup. Runtime parameter changes are not
        # supported by this node.
        self.global_frame = self.get_parameter('global_frame').value
        self.static_map_topic = self.get_parameter('static_map_topic').value
        self.cv_grid_topic = self.get_parameter('cv_grid_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.laser_map_topic = self.get_parameter('laser_map_topic').value
        self.cv_map_topic = self.get_parameter('cv_map_topic').value
        publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.lidar_cost = int(self.get_parameter('lidar_cost').value)
        self.cv_cost_threshold = float(
            self.get_parameter('cv_cost_threshold').value
        )

        # OccupancyGrid values must remain in the ROS-defined [-1, 100] range.
        # The source costs are therefore validated before publishers start.
        if publish_rate_hz <= 0.0:
            raise ValueError('publish_rate_hz must be greater than zero')
        if not 0 <= self.lidar_cost <= 100:
            raise ValueError('lidar_cost must be between 0 and 100')
        if not 0.0 <= self.cv_cost_threshold <= 100.0:
            raise ValueError('cv_cost_threshold must be between 0 and 100')

        # A transient-local map publisher retains its last sample. This lets a
        # late subscriber such as AMCL receive a map immediately on connection.
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # TF is required because CV grids and laser scans may be expressed in
        # sensor-local frames rather than directly in the map frame.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Callbacks only cache the most recent message. The timer below performs
        # all transformations and map construction at a fixed publication rate.
        self.static_map = None
        self.cv_grid = None
        self.scan = None

        # The static map uses latched map QoS; the scan uses the standard sensor
        # QoS profile. The CV stream keeps a small queue for regular updates.
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
        # All three offline maps use identical map QoS and geometry.
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            self.output_topic,
            map_qos,
        )
        self.laser_map_pub = self.create_publisher(
            OccupancyGrid,
            self.laser_map_topic,
            map_qos,
        )
        self.cv_map_pub = self.create_publisher(
            OccupancyGrid,
            self.cv_map_topic,
            map_qos,
        )
        self.create_timer(1.0 / publish_rate_hz, self.publish_combined_map)

        self.get_logger().info(
            f'Offline mapping: static={self.static_map_topic}, '
            f'cv={self.cv_grid_topic}, laser={self.scan_topic} -> '
            f'combined={self.output_topic}, laser_map={self.laser_map_topic}, '
            f'cv_map={self.cv_map_topic} '
            f'(CV cost > {self.cv_cost_threshold:g})'
        )

    def static_map_callback(self, msg: OccupancyGrid) -> None:
        """Cache the map used as the output geometry and combined-map base."""
        self.static_map = msg

    def cv_grid_callback(self, msg: OccupancyGrid) -> None:
        """Cache the latest computer-vision cost grid."""
        self.cv_grid = msg

    def scan_callback(self, msg: LaserScan) -> None:
        """Cache the latest lidar scan."""
        self.scan = msg

    @staticmethod
    def _yaw_from_quaternion(quaternion) -> float:
        """Extract the planar yaw angle from a ROS quaternion."""
        # Only rotation around Z affects a two-dimensional occupancy grid.
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
        """Return the source-frame to global-frame transform for a message."""
        if not source_frame:
            raise TransformException('Input message has an empty frame_id')
        try:
            # First use the transform corresponding to the sensor timestamp so
            # moving sensors are projected at the pose where data was acquired.
            return self.tf_buffer.lookup_transform(
                self.global_frame,
                source_frame,
                Time.from_msg(stamp),
            )
        except TransformException:
            # Sensor messages can fall between two simulated-clock TF updates.
            # Use the freshest transform rather than dropping the whole layer.
            return self.tf_buffer.lookup_transform(
                self.global_frame,
                source_frame,
                Time(),
            )

    def _transform_xy(self, x: np.ndarray, y: np.ndarray, transform):
        """Apply a planar rigid transform to arrays of x/y coordinates."""
        # Vectorized NumPy operations transform all selected points together.
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
        """Convert global coordinates to valid row/column output-map indices.

        Returns the filtered rows and columns plus a Boolean mask referring to
        the original input arrays. The mask is needed to filter matching costs.
        """
        info = self.static_map.info
        origin_yaw = self._yaw_from_quaternion(info.origin.orientation)
        cos_yaw = math.cos(origin_yaw)
        sin_yaw = math.sin(origin_yaw)
        dx = x - info.origin.position.x
        dy = y - info.origin.position.y

        # Apply the inverse map-origin transform: translate with respect to the
        # map origin, then rotate from the global frame into the grid axes.
        grid_x = cos_yaw * dx + sin_yaw * dy
        grid_y = -sin_yaw * dx + cos_yaw * dy
        # OccupancyGrid is row-major: x selects a column and y selects a row.
        cols = np.floor(grid_x / info.resolution).astype(np.int32)
        rows = np.floor(grid_y / info.resolution).astype(np.int32)
        valid = (
            (cols >= 0)
            & (cols < info.width)
            & (rows >= 0)
            & (rows < info.height)
        )
        return rows[valid], cols[valid], valid

    def _overlay_cv_grid(
        self,
        output_grid: np.ndarray,
        cost_threshold: float = 0.0,
    ) -> None:
        """Project CV cells above a threshold into an output-sized array."""
        msg = self.cv_grid
        if msg is None or not msg.data:
            return

        # ROS stores OccupancyGrid data as a flat, row-major sequence.
        data = np.asarray(msg.data, dtype=np.int16).reshape(
            msg.info.height,
            msg.info.width,
        )
        # Unknown, free and below-threshold CV cells do not affect the output.
        source_rows, source_cols = np.nonzero(data > cost_threshold)
        if source_rows.size == 0:
            return

        # Use cell centers rather than cell corners when converting indices to
        # metric coordinates in the CV grid's own coordinate system.
        resolution = msg.info.resolution
        local_x = (source_cols + 0.5) * resolution
        local_y = (source_rows + 0.5) * resolution

        # Apply the CV grid origin pose to obtain coordinates in the frame named
        # by msg.header.frame_id.
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

        # Avoid a TF lookup when the CV coordinates are already global.
        if msg.header.frame_id == self.global_frame:
            world_x, world_y = source_x, source_y
        else:
            transform = self._lookup_transform(
                msg.header.frame_id,
                msg.header.stamp,
            )
            world_x, world_y = self._transform_xy(source_x, source_y, transform)
        # Discard points outside the static-map bounds. If several CV cells map
        # to one output cell, keep the greatest cost with np.maximum.at.
        rows, cols, valid = self._world_to_output_cells(world_x, world_y)
        costs = data[source_rows, source_cols][valid]
        flat_indices = rows * self.static_map.info.width + cols
        np.maximum.at(output_grid.ravel(), flat_indices, costs)

    def _overlay_scan(self, output_grid: np.ndarray) -> None:
        """Project valid endpoints from the latest scan into an output array."""
        msg = self.scan
        if msg is None or not msg.ranges:
            return

        # NaN/Inf and measurements outside the sensor limits do not describe a
        # valid obstacle endpoint and are removed before projection.
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        valid = (
            np.isfinite(ranges)
            & (ranges >= msg.range_min)
            & (ranges <= msg.range_max)
        )
        if not np.any(valid):
            return

        # Convert polar LaserScan samples into Cartesian sensor-frame points.
        angles = msg.angle_min + np.arange(ranges.size) * msg.angle_increment
        scan_x = ranges[valid] * np.cos(angles[valid])
        scan_y = ranges[valid] * np.sin(angles[valid])

        # Move endpoints into the global frame, then into static-map cells.
        transform = self._lookup_transform(msg.header.frame_id, msg.header.stamp)
        world_x, world_y = self._transform_xy(scan_x, scan_y, transform)
        rows, cols, _ = self._world_to_output_cells(world_x, world_y)
        output_grid[rows, cols] = np.maximum(
            output_grid[rows, cols],
            self.lidar_cost,
        )

    def _make_output_message(self, grid: np.ndarray, stamp) -> OccupancyGrid:
        """Wrap a 2-D array in an OccupancyGrid using static-map metadata."""
        output = OccupancyGrid()
        output.header.stamp = stamp
        output.header.frame_id = self.global_frame
        # Deep-copy metadata so output messages do not share mutable ROS fields
        # with the cached static-map message.
        output.info = copy.deepcopy(self.static_map.info)
        output.data = np.clip(grid, -1, 100).astype(np.int8).ravel().tolist()
        return output

    def publish_combined_map(self) -> None:
        """Build and publish combined, laser-only and thresholded-CV maps."""
        # No output geometry is known until the first static map arrives.
        if self.static_map is None:
            return

        # Start the combined output from the complete SLAM map. Source-specific
        # layers instead start as free space, so they contain only their source.
        combined = np.asarray(self.static_map.data, dtype=np.int16).reshape(
            self.static_map.info.height,
            self.static_map.info.width,
        ).copy()
        laser_map = np.zeros_like(combined)
        cv_layer = np.zeros_like(combined)

        try:
            # First build the unfiltered CV layer. Positive CV costs are merged
            # into the static map without allowing free/unknown CV cells to
            # erase existing SLAM information.
            self._overlay_cv_grid(cv_layer)
            cv_cells = cv_layer > 0
            combined[cv_cells] = np.maximum(
                combined[cv_cells],
                cv_layer[cv_cells],
            )
        except TransformException as exc:
            self.get_logger().warn(
                f'CV transform unavailable: {exc}',
                throttle_duration_sec=2.0,
            )

        try:
            # The laser-only layer contains only endpoints from the latest scan.
            # Merge those occupied cells into the combined map as well.
            self._overlay_scan(laser_map)
            laser_cells = laser_map > 0
            combined[laser_cells] = np.maximum(
                combined[laser_cells],
                laser_map[laser_cells],
            )
        except TransformException as exc:
            self.get_logger().warn(
                f'Laser transform unavailable: {exc}',
                throttle_duration_sec=2.0,
            )

        stamp = self.get_clock().now().to_msg()
        self.map_pub.publish(self._make_output_message(combined, stamp))

        # Derive the dedicated CV map after the combined map so thresholding
        # does not change the combined output. "Greater than" is intentional:
        # a cost equal to the configured threshold is considered free.
        # AMCL-style binary layer: every retained CV obstacle is occupied and
        # every other cell is free. Do not preserve semantic 30/60/100 costs in
        # this dedicated map; those remain available in the combined map.
        cv_map = np.where(
            cv_layer > self.cv_cost_threshold,
            100,
            0,
        ).astype(np.int16)
        self.laser_map_pub.publish(self._make_output_message(laser_map, stamp))
        self.cv_map_pub.publish(self._make_output_message(cv_map, stamp))


def main(args=None):
    """Initialize ROS, run the node and shut it down cleanly."""
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
