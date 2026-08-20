import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class Filtering(Node):
    CHANNELS = {
        'TURQUOISE': 'turquoise',
        'WHITE': 'white',
        'MAGENTA': 'magenta',
    }

    def __init__(self):
        super().__init__('filtering')

        # --- ROS2 PARAMETERS ---
        self.declare_parameter('color', 'TURQUOISE')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('resolution', 0.02)
        self.declare_parameter('map_size_meters', 10.0)
        self.declare_parameter('roi_x_min_m', 0.0)
        self.declare_parameter('roi_x_max_m', 1.85)
        self.declare_parameter('roi_width_near_m', 0.6)
        self.declare_parameter('roi_width_far_m', 2.65)

        self.set_parameters([rclpy.parameter.Parameter('use_sim_time',
                             rclpy.Parameter.Type.BOOL, True)])

        self.color_flag = self.get_parameter('color').value.upper()
        if self.color_flag not in self.CHANNELS:
            options = ', '.join(self.CHANNELS)
            raise ValueError(f'Unsupported color {self.color_flag!r}; use {options}')
        self.global_frame = self.get_parameter('global_frame').value
        self.resolution = self.get_parameter('resolution').value
        self.map_size_meters = self.get_parameter('map_size_meters').value
        self.roi_x_min_m = self.get_parameter('roi_x_min_m').value
        self.roi_x_max_m = self.get_parameter('roi_x_max_m').value
        self.roi_width_near_m = self.get_parameter('roi_width_near_m').value
        self.roi_width_far_m = self.get_parameter('roi_width_far_m').value

        if self.roi_x_max_m <= self.roi_x_min_m:
            raise ValueError('roi_x_max_m must be greater than roi_x_min_m')
        if self.roi_width_near_m < 0.0 or self.roi_width_far_m < 0.0:
            raise ValueError('ROI widths must be non-negative')

        self.map_size_pixels = int(self.map_size_meters / self.resolution)

        # --- TF2 CONFIGURATION ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()

        # --- DYNAMIC TOPIC CONFIGURATION ---
        self.color_suffix = self.color_flag.lower()
        self.costmap_suffix = self.CHANNELS[self.color_flag]
        costmap_topic = (
            '/limo/nav_map_package/offline/metric_bev/'
            f'cost_grid_{self.costmap_suffix}'
        )
        map_topic = (
            '/limo/nav_map_package/offline/filtering/'
            f'map_paper_{self.color_suffix}'
        )

        self.filtered_publisher = self.create_publisher(OccupancyGrid, map_topic, 10)
        self.roi_debug_publisher = self.create_publisher(
            Image,
            '/limo/nav_map_package/offline/filtering/'
            f'roi_debug_{self.color_suffix}',
            10,
        )

        # --- BAYESIAN LOG-ODDS FILTER CONFIGURATION ---
        # The global canvas stores log-odds to accelerate probabilistic calculations.
        self.canvas_logodds = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=np.float32)
        self.seen_canvas = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=bool)
        self.L_OCC = 1.0    # Certainty increment if an obstacle is detected
        self.L_FREE = 0.35    # Decrement if free space is detected (reduced value for smooth clearing)
        self.L_MAX = 5.0     # Maximum saturation point of the memory
        self.L_MIN = -3.0    # Minimum saturation point of the map

        self.costmap = None
        
        # Geometric Caching variables to avoid continuous allocations on the CPU
        self.cached_local_x = None
        self.cached_local_y = None
        self.cached_shape = None
        self.cached_roi_mask = None  # Local mask of the actual field of view cone

        # Subscription and Timer at 10Hz
        self.costmap_sub = self.create_subscription(OccupancyGrid, costmap_topic, self.costmap_callback, 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

        # Pre-allocation of the output message to optimize real-time performance
        self.grid_msg_filtered = self.get_default_occupancy_grid()

        self.get_logger().info(
            f'Probabilistic filtering initialized for channel [{self.color_flag}] '
            f'on topic {costmap_topic}; publishing on {map_topic}'
        )
        self.get_logger().info(
            'Bayesian ROI: '
            f'x=[{self.roi_x_min_m:.2f}, {self.roi_x_max_m:.2f}] m, '
            f'width={self.roi_width_near_m:.2f} m near -> '
            f'{self.roi_width_far_m:.2f} m far'
        )

    def get_default_occupancy_grid(self) -> OccupancyGrid:
        """Initializes standard metadata for the global occupancy grid map."""
        msg = OccupancyGrid()
        msg.header.frame_id = self.global_frame
        msg.info.resolution = self.resolution
        msg.info.width = self.map_size_pixels
        msg.info.height = self.map_size_pixels
        half = self.map_size_meters / 2.0
        msg.info.origin.position.x = -half
        msg.info.origin.position.y = -half
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        return msg

    def costmap_callback(self, msg: OccupancyGrid):
        self.costmap = msg

    def timer_callback(self):
        if self.costmap is None:
            return

        try:
            # Retrieving the transformation between the global world (odom) and the local sensor origin
            tf = self.tf_buffer.lookup_transform(
                self.global_frame,
                f'offline_metric_bev_origin_{self.costmap_suffix}',
                rclpy.time.Time(),
            )
            origin_x = tf.transform.translation.x
            origin_y = tf.transform.translation.y
            q = tf.transform.rotation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)

            cw = self.costmap.info.width
            ch = self.costmap.info.height
            
            data = np.array(self.costmap.data, dtype=np.uint8).reshape((ch, cw))

            # --- GEOMETRIC CACHING & ROI MASK GENERATION ---
            # Executed only if the dimensions of the incoming frame change
            if self.cached_shape != (ch, cw):
                cres = self.costmap.info.resolution
                cox = self.costmap.info.origin.position.x
                coy = self.costmap.info.origin.position.y
                
                cols = np.arange(cw)
                rows = np.arange(ch)
                cc, rr = np.meshgrid(cols, rows)

                # X and Y coordinates expressed in meters relative to the cv_origin_[color] frame
                self.cached_local_x = cox + cc * cres + cres / 2.0
                self.cached_local_y = coy + rr * cres + cres / 2.0
                self.cached_shape = (ch, cw)

                # Trapezoidal ROI. Width parameters represent the complete
                # lateral width, centered around local y=0.
                roi_length = self.roi_x_max_m - self.roi_x_min_m
                interpolation = (self.cached_local_x - self.roi_x_min_m) / roi_length
                allowed_width = (
                    self.roi_width_near_m
                    + interpolation * (self.roi_width_far_m - self.roi_width_near_m)
                )
                max_allowed_abs_y = allowed_width / 2.0

                # Generation of the local boolean mask to isolate the useful field of view
                self.cached_roi_mask = (
                    (self.cached_local_x >= self.roi_x_min_m)
                    & (self.cached_local_x <= self.roi_x_max_m)
                    & (np.abs(self.cached_local_y) <= max_allowed_abs_y)
                )

            # Vectorial projection of the local map onto global map coordinates (Rotation-translation)
            world_x = origin_x + cos_yaw * self.cached_local_x - sin_yaw * self.cached_local_y
            world_y = origin_y + sin_yaw * self.cached_local_x + cos_yaw * self.cached_local_y

            half = self.map_size_meters / 2.0
            px = ((world_x + half) / self.resolution).astype(np.int32)
            py = ((world_y + half) / self.resolution).astype(np.int32)

            # Filter to exclude pixels accidentally projected outside the global canvas
            inside_canvas = (px >= 0) & (px < self.map_size_pixels) & (py >= 0) & (py < self.map_size_pixels)

            # --- PROBABILISTIC BAYESIAN FILTER WITH ROI COVERAGE ---
            update_matrix = np.zeros_like(self.canvas_logodds)
            
            # 1. OBSTACLE ACCUMULATION: If the sensor detects an obstacle (>0), we add it to the map
            occ_mask = inside_canvas & (data > 0)
            update_matrix[py[occ_mask], px[occ_mask]] = self.L_OCC

            # 2. OBSTACLE REMOVAL (BUG SOLVED): We subtract certainty points ONLY if the cell is 0
            # AND it is physically inside the geometric ROI visible by the robot!
            free_mask = inside_canvas & (data == 0) & self.cached_roi_mask
            update_matrix[py[free_mask], px[free_mask]] = -self.L_FREE

            # Mark all pixels inside the ROI that are within the canvas as "seen"
            seen_mask = inside_canvas & self.cached_roi_mask
            self.seen_canvas[py[seen_mask], px[seen_mask]] = True

            # Applying the update and saturation within stability limits
            self.canvas_logodds = np.clip(self.canvas_logodds + update_matrix, self.L_MIN, self.L_MAX)
            
            # Inverse conversion: Log-Odds -> Probability via Sigmoid function
            prob_matrix = 1.0 / (1.0 + np.exp(-self.canvas_logodds))
            canvas_filtered = (prob_matrix * 100.0).astype(np.int8)

            if self.roi_debug_publisher.get_subscription_count() > 0:
                self._publish_roi_debug(data, self.costmap.header)

        except TransformException as e:
            self.get_logger().warn(f'TF not available: {e}', throttle_duration_sec=2.0)
            return

        # --- OCCUPANCY GRID MAP PUBLICATION ---
        # np.where(condition, value if true, value if false)
        canvas_out = np.where(self.seen_canvas, canvas_filtered, np.int8(-1))
        self.grid_msg_filtered.header.stamp = self.get_clock().now().to_msg()
        self.grid_msg_filtered.data = canvas_out.flatten().tolist()
        self.filtered_publisher.publish(self.grid_msg_filtered)

    def _publish_roi_debug(self, data: np.ndarray, header) -> None:
        """Publish the local costmap with the Bayesian update ROI highlighted."""
        cost_gray = np.clip(data.astype(np.float32), 0, 100)
        cost_gray = (cost_gray * 2.55).astype(np.uint8)
        debug = cv2.cvtColor(cost_gray, cv2.COLOR_GRAY2BGR)

        # Dim pixels outside the filter ROI and tint the active area green.
        debug[~self.cached_roi_mask] = (debug[~self.cached_roi_mask] * 0.2).astype(np.uint8)
        green_overlay = np.zeros_like(debug)
        green_overlay[:, :, 1] = 120
        debug[self.cached_roi_mask] = cv2.addWeighted(
            debug, 0.65, green_overlay, 0.35, 0
        )[self.cached_roi_mask]

        contours, _ = cv2.findContours(
            self.cached_roi_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(debug, contours, -1, (0, 255, 255), 2)

        # Landscape visualization; this affects only the debug topic.
        debug = cv2.rotate(debug, cv2.ROTATE_90_COUNTERCLOCKWISE)

        debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        debug_msg.header = header
        self.roi_debug_publisher.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = Filtering()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
