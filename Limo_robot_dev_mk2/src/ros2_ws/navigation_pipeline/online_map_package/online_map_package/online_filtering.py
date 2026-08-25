"""Robot-centric Bayesian filtering for one online semantic cost layer."""

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
from tf2_ros import Buffer, TransformException, TransformListener


class OnlineFiltering(Node):
    """Filter one semantic channel on a rectangular base-linked canvas."""

    CHANNELS = {
        'TURQUOISE': 'turquoise',
        'WHITE': 'white',
        'MAGENTA': 'magenta',
    }

    def __init__(self):
        super().__init__('online_filtering')

        self.declare_parameter('color', 'TURQUOISE')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tracking_frame', 'odom')
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('cv_offset_x_m', 0.6)
        self.declare_parameter('roi_x_min_m', 0.0)
        self.declare_parameter('roi_x_max_m', 1.85)
        self.declare_parameter('roi_width_near_m', 0.6)
        self.declare_parameter('roi_width_far_m', 2.65)
        self.declare_parameter('publish_rate_hz', 20.0)

        self.color_flag = str(self.get_parameter('color').value).upper()
        if self.color_flag not in self.CHANNELS:
            options = ', '.join(self.CHANNELS)
            raise ValueError(
                f'Unsupported color {self.color_flag!r}; use {options}'
            )

        self.color_suffix = self.CHANNELS[self.color_flag]
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.tracking_frame = str(
            self.get_parameter('tracking_frame').value
        )
        self.resolution = float(self.get_parameter('resolution').value)
        self.cv_offset_x_m = float(
            self.get_parameter('cv_offset_x_m').value
        )
        self.roi_x_min_m = float(self.get_parameter('roi_x_min_m').value)
        self.roi_x_max_m = float(self.get_parameter('roi_x_max_m').value)
        self.roi_width_near_m = float(
            self.get_parameter('roi_width_near_m').value
        )
        self.roi_width_far_m = float(
            self.get_parameter('roi_width_far_m').value
        )
        publish_rate_hz = float(
            self.get_parameter('publish_rate_hz').value
        )

        if self.resolution <= 0.0:
            raise ValueError('resolution must be greater than zero')
        if self.cv_offset_x_m < 0.0:
            raise ValueError('cv_offset_x_m must be non-negative')
        if self.roi_x_max_m <= self.roi_x_min_m:
            raise ValueError('roi_x_max_m must be greater than roi_x_min_m')
        if self.roi_width_near_m < 0.0 or self.roi_width_far_m <= 0.0:
            raise ValueError('ROI widths must be non-negative and non-zero')
        if publish_rate_hz <= 0.0:
            raise ValueError('publish_rate_hz must be greater than zero')

        # OccupancyGrid columns follow base X; rows follow base Y. The robot is
        # at x=0 and centered laterally on the rear edge of the local canvas.
        requested_length_x = self.cv_offset_x_m + self.roi_x_max_m
        requested_width_y = max(
            self.roi_width_near_m,
            self.roi_width_far_m,
        )
        self.map_size_x = int(math.ceil(requested_length_x / self.resolution))
        self.map_size_y = int(math.ceil(requested_width_y / self.resolution))
        self.map_length_x_m = self.map_size_x * self.resolution
        self.map_width_y_m = self.map_size_y * self.resolution
        self.map_origin_x = 0.0
        self.map_origin_y = -0.5 * self.map_width_y_m

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()

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
        costmap_topic = (
            '/limo/nav_map_package/online/metric_bev/'
            f'cost_grid_{self.color_suffix}'
        )
        map_topic = (
            '/limo/nav_map_package/online/filtering/'
            f'map_paper_{self.color_suffix}'
        )
        debug_topic = (
            '/limo/nav_map_package/online/filtering/'
            f'roi_debug_{self.color_suffix}'
        )
        self.filtered_publisher = self.create_publisher(
            OccupancyGrid,
            map_topic,
            map_qos,
        )
        self.roi_debug_publisher = self.create_publisher(
            Image,
            debug_topic,
            debug_qos,
        )
        self.costmap_sub = self.create_subscription(
            OccupancyGrid,
            costmap_topic,
            self.costmap_callback,
            map_qos,
        )

        self.canvas_logodds = np.zeros(
            (self.map_size_y, self.map_size_x),
            dtype=np.float32,
        )
        self.seen_canvas = np.zeros(
            (self.map_size_y, self.map_size_x),
            dtype=bool,
        )
        canvas_rows, canvas_cols = np.indices(
            (self.map_size_y, self.map_size_x),
            dtype=np.float32,
        )
        self.canvas_local_x = (
            self.map_origin_x + (canvas_cols + 0.5) * self.resolution
        )
        self.canvas_local_y = (
            self.map_origin_y + (canvas_rows + 0.5) * self.resolution
        )
        self.previous_base_pose = None
        self.L_OCC = 1.0
        self.L_FREE = 0.35
        self.L_MAX = 5.0
        self.L_MIN = -3.0

        self.costmap = None
        self.cached_geometry = None
        self.cached_local_x = None
        self.cached_local_y = None
        self.cached_roi_mask = None
        self.cached_sensor_frame = None
        self.cached_sensor_transform = None
        self.cached_projection_key = None
        self.projected_source_indices = None
        self.projected_output_indices = None
        self.visible_output_indices = None
        self.grid_msg_filtered = self.get_default_occupancy_grid()
        self.timer = self.create_timer(
            1.0 / publish_rate_hz,
            self.timer_callback,
        )

        self.get_logger().info(
            f'Online Bayesian filtering [{self.color_flag}]: '
            f'{costmap_topic} -> {map_topic}; '
            f'grid={self.map_size_x}x{self.map_size_y} '
            f'({self.map_length_x_m:.2f}x{self.map_width_y_m:.2f} m) '
            f'resolution={self.resolution:.3f} m, frame={self.base_frame}, '
            f'memory_tracking_frame={self.tracking_frame}'
        )

    def get_default_occupancy_grid(self) -> OccupancyGrid:
        """Build metadata for the rectangular robot-centric output map."""
        msg = OccupancyGrid()
        msg.header.frame_id = self.base_frame
        msg.info.resolution = self.resolution
        msg.info.width = self.map_size_x
        msg.info.height = self.map_size_y
        msg.info.origin.position.x = self.map_origin_x
        msg.info.origin.position.y = self.map_origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        return msg

    def costmap_callback(self, msg: OccupancyGrid) -> None:
        """Cache the newest costmap for this semantic channel."""
        self.costmap = msg

    def _cache_input_geometry(self, msg: OccupancyGrid) -> None:
        """Cache cell centers and the unchanged trapezoidal ROI mask."""
        geometry = (
            msg.header.frame_id,
            msg.info.height,
            msg.info.width,
            msg.info.resolution,
            msg.info.origin.position.x,
            msg.info.origin.position.y,
        )
        if geometry == self.cached_geometry:
            return

        rows, cols = np.indices(
            (msg.info.height, msg.info.width),
            dtype=np.float32,
        )
        resolution = msg.info.resolution
        self.cached_local_x = (
            msg.info.origin.position.x + (cols + 0.5) * resolution
        )
        self.cached_local_y = (
            msg.info.origin.position.y + (rows + 0.5) * resolution
        )

        roi_length = self.roi_x_max_m - self.roi_x_min_m
        interpolation = (
            (self.cached_local_x - self.roi_x_min_m) / roi_length
        )
        allowed_width = (
            self.roi_width_near_m
            + interpolation
            * (self.roi_width_far_m - self.roi_width_near_m)
        )
        self.cached_roi_mask = (
            (self.cached_local_x >= self.roi_x_min_m)
            & (self.cached_local_x <= self.roi_x_max_m)
            & (np.abs(self.cached_local_y) <= 0.5 * allowed_width)
        )
        self.cached_geometry = geometry

    def _cache_static_projection(self, transform) -> None:
        """Cache the static source-to-output cell correspondence."""
        transform_pose = self._pose_from_transform(transform)
        projection_key = (self.cached_geometry, transform_pose)
        if projection_key == self.cached_projection_key:
            return

        translation_x, translation_y, yaw = transform_pose
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        base_x = (
            translation_x
            + cos_yaw * self.cached_local_x
            - sin_yaw * self.cached_local_y
        )
        base_y = (
            translation_y
            + sin_yaw * self.cached_local_x
            + cos_yaw * self.cached_local_y
        )
        output_x = np.floor(
            (base_x - self.map_origin_x) / self.resolution
        ).astype(np.int32)
        output_y = np.floor(
            (base_y - self.map_origin_y) / self.resolution
        ).astype(np.int32)
        valid = (
            self.cached_roi_mask
            & (output_x >= 0)
            & (output_x < self.map_size_x)
            & (output_y >= 0)
            & (output_y < self.map_size_y)
        )

        self.projected_source_indices = np.flatnonzero(valid)
        self.projected_output_indices = (
            output_y[valid] * self.map_size_x + output_x[valid]
        )
        self.visible_output_indices = np.unique(
            self.projected_output_indices
        )
        self.cached_projection_key = projection_key

    def _build_observation_update(self, data: np.ndarray) -> np.ndarray:
        """Aggregate the source grid directly into output log-odds cells."""
        update = np.zeros_like(self.canvas_logodds)
        update_flat = update.ravel()
        update_flat[self.visible_output_indices] = -self.L_FREE

        source_values = data.ravel()[self.projected_source_indices]
        occupied_sources = source_values > 0
        if np.any(occupied_sources):
            occupied_outputs = np.unique(
                self.projected_output_indices[occupied_sources]
            )
            # An occupied source wins over free sources mapped to the same
            # lower-resolution output cell.
            update_flat[occupied_outputs] = self.L_OCC

        self.seen_canvas.ravel()[self.visible_output_indices] = True
        return update

    @staticmethod
    def _pose_from_transform(transform) -> tuple:
        """Return planar x, y and yaw from a TransformStamped."""
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = math.atan2(
            2.0 * (
                rotation.w * rotation.z + rotation.x * rotation.y
            ),
            1.0 - 2.0 * (
                rotation.y * rotation.y + rotation.z * rotation.z
            ),
        )
        return translation.x, translation.y, yaw

    def _reproject_memory(self, current_pose: tuple) -> None:
        """Keep the full rectangular memory fixed in the tracking frame."""
        if self.previous_base_pose is None:
            self.previous_base_pose = current_pose
            return

        old_x, old_y, old_yaw = self.previous_base_pose
        new_x, new_y, new_yaw = current_pose
        self.previous_base_pose = current_pose

        if (
            math.isclose(old_x, new_x, abs_tol=1e-6)
            and math.isclose(old_y, new_y, abs_tol=1e-6)
            and math.isclose(old_yaw, new_yaw, abs_tol=1e-6)
        ):
            return

        cos_new = math.cos(new_yaw)
        sin_new = math.sin(new_yaw)
        tracking_x = (
            new_x
            + cos_new * self.canvas_local_x
            - sin_new * self.canvas_local_y
        )
        tracking_y = (
            new_y
            + sin_new * self.canvas_local_x
            + cos_new * self.canvas_local_y
        )

        delta_x = tracking_x - old_x
        delta_y = tracking_y - old_y
        cos_old = math.cos(old_yaw)
        sin_old = math.sin(old_yaw)
        old_local_x = cos_old * delta_x + sin_old * delta_y
        old_local_y = -sin_old * delta_x + cos_old * delta_y

        source_x = np.floor(
            (old_local_x - self.map_origin_x) / self.resolution
        ).astype(np.int32)
        source_y = np.floor(
            (old_local_y - self.map_origin_y) / self.resolution
        ).astype(np.int32)
        valid = (
            (source_x >= 0)
            & (source_x < self.map_size_x)
            & (source_y >= 0)
            & (source_y < self.map_size_y)
        )

        reprojected_logodds = np.zeros_like(self.canvas_logodds)
        reprojected_seen = np.zeros_like(self.seen_canvas)
        reprojected_logodds[valid] = self.canvas_logodds[
            source_y[valid],
            source_x[valid],
        ]
        reprojected_seen[valid] = self.seen_canvas[
            source_y[valid],
            source_x[valid],
        ]
        self.canvas_logodds = reprojected_logodds
        self.seen_canvas = reprojected_seen

    def timer_callback(self) -> None:
        """Project the latest layer, update log-odds and publish the map."""
        if self.costmap is None:
            return

        msg = self.costmap
        try:
            if self.cached_sensor_frame != msg.header.frame_id:
                self.cached_sensor_transform = (
                    self.tf_buffer.lookup_transform(
                        self.base_frame,
                        msg.header.frame_id,
                        rclpy.time.Time(),
                    )
                )
                self.cached_sensor_frame = msg.header.frame_id
                self.cached_projection_key = None
            base_transform = self.tf_buffer.lookup_transform(
                self.tracking_frame,
                self.base_frame,
                rclpy.time.Time(),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f'TF not available: {exc}',
                throttle_duration_sec=2.0,
            )
            return

        self._reproject_memory(
            self._pose_from_transform(base_transform)
        )

        self._cache_input_geometry(msg)
        self._cache_static_projection(self.cached_sensor_transform)
        data = np.asarray(msg.data, dtype=np.uint8).reshape(
            msg.info.height,
            msg.info.width,
        )
        update_matrix = self._build_observation_update(data)
        self.canvas_logodds = np.clip(
            self.canvas_logodds + update_matrix,
            self.L_MIN,
            self.L_MAX,
        )

        probability = 1.0 / (1.0 + np.exp(-self.canvas_logodds))
        filtered = (probability * 100.0).astype(np.int8)
        output = np.where(self.seen_canvas, filtered, np.int8(-1))
        self.grid_msg_filtered.header.stamp = self.get_clock().now().to_msg()
        self.grid_msg_filtered.data = output.ravel().tolist()
        self.filtered_publisher.publish(self.grid_msg_filtered)

        if self.roi_debug_publisher.get_subscription_count() > 0:
            self._publish_roi_debug(data, msg.header)

    def _publish_roi_debug(self, data: np.ndarray, header) -> None:
        """Publish the input layer with the active filter ROI highlighted."""
        gray = (np.clip(data.astype(np.float32), 0, 100) * 2.55).astype(
            np.uint8
        )
        debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        debug[~self.cached_roi_mask] = (
            debug[~self.cached_roi_mask] * 0.2
        ).astype(np.uint8)
        green_overlay = np.zeros_like(debug)
        green_overlay[:, :, 1] = 120
        debug[self.cached_roi_mask] = cv2.addWeighted(
            debug,
            0.65,
            green_overlay,
            0.35,
            0,
        )[self.cached_roi_mask]
        contours, _ = cv2.findContours(
            self.cached_roi_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(debug, contours, -1, (0, 255, 255), 2)
        debug = cv2.rotate(debug, cv2.ROTATE_90_COUNTERCLOCKWISE)
        debug = cv2.flip(debug, 1)
        debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        debug_msg.header = header
        self.roi_debug_publisher.publish(debug_msg)


def main(args=None):
    """Run the online semantic filtering node."""
    rclpy.init(args=args)
    node = OnlineFiltering()
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
