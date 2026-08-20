import copy

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
import time

class CVMapDisplay(Node):
    """
    Combines the complete filtering grids and renders a separate robot-centric view.
    """

    def __init__(self):
        super().__init__('cv_map_display')

        self.declare_parameter('robot_frame',         'base_link')
        self.declare_parameter('view_resolution',     0.02)     # Lightweight definition for images (2cm/px)
        self.declare_parameter('view_range_m',        3.0)      # Semi-side of ego canvas
        self.declare_parameter('turquoise_factor',    0.6)
        self.declare_parameter('white_factor',        0.3)
        self.declare_parameter('magenta_factor',      1.0)
        self.declare_parameter('enable_telemetry',    True)

        self.set_parameters([rclpy.parameter.Parameter('use_sim_time',
                             rclpy.Parameter.Type.BOOL, True)])

        self.robot_frame        = self.get_parameter('robot_frame').value
        self.view_resolution    = self.get_parameter('view_resolution').value
        self.view_range_m       = self.get_parameter('view_range_m').value
        self.turquoise_factor   = self.get_parameter('turquoise_factor').value
        self.white_factor       = self.get_parameter('white_factor').value
        self.magenta_factor     = self.get_parameter('magenta_factor').value
        self.debug_telemetry    = self.get_parameter('enable_telemetry').value

        # Dynamic canvas dimensions based on chosen resolutions
        self.canvas_px = int(2 * self.view_range_m / self.view_resolution)

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.bridge      = CvBridge()

        # Storage for incoming occupancy grid messages
        self.map_data_turquoise = None
        self.map_data_white = None
        self.map_data_magenta = None

        # --- SUBSCRIPTIONS WITH ROI PROJECTORS ---
        filtering_topic = '/limo/nav_map_package/offline/filtering'
        self.create_subscription(
            OccupancyGrid,
            f'{filtering_topic}/map_paper_turquoise',
            self.turquoise_map_callback,
            10,
        )
        self.create_subscription(
            OccupancyGrid,
            f'{filtering_topic}/map_paper_white',
            self.white_map_callback,
            10,
        )
        self.create_subscription(
            OccupancyGrid,
            f'{filtering_topic}/map_paper_magenta',
            self.magenta_map_callback,
            10,
        )
        self.get_logger().info('Subscribing to TURQUOISE, WHITE and MAGENTA maps')

        # --- COMBINED IMAGE PUBLISHERS ---
        self.firstp_img_pub = self.create_publisher(
            Image,
            '/limo/nav_map_package/offline/cv_map_display/cv_map_image/raw',
            10,
        )
        
        # --- NEW PUBLISHER: COMBINED OCCUPANCY GRID ---
        self.grid_pub = self.create_publisher(
            OccupancyGrid,
            '/limo/nav_map_package/offline/cv_map_display/cv_map_occupancy_grid',
            10,
        )

        self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('CVMapDisplay (Visualization & OccupancyGrid) node started')

    # ------------------------------------------------------------------

    def turquoise_map_callback(self, msg: OccupancyGrid):
        self.map_data_turquoise = msg

    def white_map_callback(self, msg: OccupancyGrid):
        self.map_data_white = msg

    def magenta_map_callback(self, msg: OccupancyGrid):
        self.map_data_magenta = msg

    @staticmethod
    def _same_geometry(first: OccupancyGrid, second: OccupancyGrid) -> bool:
        first_origin = first.info.origin
        second_origin = second.info.origin
        return (
            first.header.frame_id == second.header.frame_id
            and first.info.width == second.info.width
            and first.info.height == second.info.height
            and math.isclose(first.info.resolution, second.info.resolution)
            and math.isclose(first_origin.position.x, second_origin.position.x)
            and math.isclose(first_origin.position.y, second_origin.position.y)
            and math.isclose(first_origin.orientation.z, second_origin.orientation.z)
            and math.isclose(first_origin.orientation.w, second_origin.orientation.w)
        )

    def _combine_full_maps(self):
        layers = [
            (self.map_data_turquoise, self.turquoise_factor),
            (self.map_data_white, self.white_factor),
            (self.map_data_magenta, self.magenta_factor),
        ]
        reference = next((msg for msg, _ in layers if msg is not None), None)
        if reference is None:
            return None, None

        shape = (reference.info.height, reference.info.width)
        combined = np.zeros(shape, dtype=np.float32)
        seen = np.zeros(shape, dtype=bool)

        for msg, factor in layers:
            if msg is None:
                continue
            if not self._same_geometry(reference, msg):
                self.get_logger().warn(
                    'Skipping filtering grid with geometry different from the reference',
                    throttle_duration_sec=2.0,
                )
                continue

            grid = np.asarray(msg.data, dtype=np.int16).reshape(shape)
            known = grid >= 0
            scaled = np.clip(grid, 0, 100).astype(np.float32) * factor
            np.maximum(combined, np.where(known, scaled, 0.0), out=combined)
            np.logical_or(seen, known, out=seen)

        combined_grid = np.where(seen, np.clip(combined, 0, 100), -1).astype(np.int8)
        return reference, combined_grid

    # ------------------------------------------------------------------

    def _project_grid_layer_optimized(self, m: OccupancyGrid, robot_x: float, robot_y: float,
                                        cos_y: float, sin_y: float, canvas: np.ndarray, 
                                        seen_canvas: np.ndarray, factor: float = 1.0) -> None:
        grid = np.array(m.data, dtype=np.int8).reshape((m.info.height, m.info.width)).astype(np.float32)
        unknown_mask = (grid < 0)
        grid = np.clip(grid, 0, 100) * factor
        grid[unknown_mask] = -1.0 

        res_src = m.info.resolution
        x0_src  = m.info.origin.position.x
        y0_src  = m.info.origin.position.y

        M_map_to_odom = np.array([[res_src, 0.0,     x0_src],
                                [0.0,     res_src, y0_src],
                                [0.0,     0.0,     1.0   ]], dtype=np.float32)

        M_odom_to_robot = np.array([[cos_y,  sin_y, -cos_y * robot_x - sin_y * robot_y],
                                    [-sin_y, cos_y,  sin_y * robot_x - cos_y * robot_y],
                                    [0.0,    0.0,    1.0                              ]], dtype=np.float32)

        cx = self.canvas_px / 2.0
        cy = self.canvas_px / 2.0
        M_robot_to_canvas = np.array([[0.0,                    -1.0 / self.view_resolution, cx],
                                    [-1.0 / self.view_resolution, 0.0,                     cy],
                                    [0.0,                    0.0,                     1.0]], dtype=np.float32)

        M_warp = (M_robot_to_canvas @ M_odom_to_robot @ M_map_to_odom)[0:2, :]

        local_layer = cv2.warpAffine(grid, M_warp, (self.canvas_px, self.canvas_px),
                                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)

        observed = (local_layer >= 0)
        np.maximum(canvas, np.where(observed, local_layer, 0), out=canvas)
        np.logical_or(seen_canvas, observed, out=seen_canvas)

    # ------------------------------------------------------------------

    def timer_callback(self):
        t_timer_start = time.perf_counter()

        reference, combined_grid = self._combine_full_maps()
        if reference is None:
            return

        try:
            tf = self.tf_buffer.lookup_transform(
                reference.header.frame_id,
                self.robot_frame,
                rclpy.time.Time(),
            )
        except TransformException as e:
            self.get_logger().warn(f'TF not available: {e}', throttle_duration_sec=2.0)
            return

        robot_x = tf.transform.translation.x
        robot_y = tf.transform.translation.y
        q = tf.transform.rotation
        robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        cos_y, sin_y = math.cos(robot_yaw), math.sin(robot_yaw)

        # 1. Local Ego Canvas Projection (Fast because it uses view_resolution)
        ego_canvas = np.zeros((self.canvas_px, self.canvas_px), dtype=np.float32)
        seen_canvas = np.zeros((self.canvas_px, self.canvas_px), dtype=bool)

        if self.map_data_turquoise is not None:
            self._project_grid_layer_optimized(self.map_data_turquoise, robot_x, robot_y, cos_y, sin_y, ego_canvas, seen_canvas, self.turquoise_factor)
        if self.map_data_white is not None:
            self._project_grid_layer_optimized(self.map_data_white, robot_x, robot_y, cos_y, sin_y, ego_canvas, seen_canvas, self.white_factor)
        if self.map_data_magenta is not None:
            self._project_grid_layer_optimized(self.map_data_magenta, robot_x, robot_y, cos_y, sin_y, ego_canvas, seen_canvas, self.magenta_factor)

        t_ego_done = time.perf_counter()

        # Publish the complete filtering map without ego cropping or axis
        # conversion. Geometry and frame stay identical to the input grids.
        current_time = self.get_clock().now().to_msg()
        output_grid = OccupancyGrid()
        output_grid.header.stamp = current_time
        output_grid.header.frame_id = reference.header.frame_id
        output_grid.info = copy.deepcopy(reference.info)
        output_grid.data = combined_grid.ravel().tolist()
        self.grid_pub.publish(output_grid)

        # 2. Rendering firstp from ego canvas
        norm_e   = (np.clip(ego_canvas, 0, 100) / 100.0 * 255.0).astype(np.uint8)
        firstp   = cv2.applyColorMap(norm_e, cv2.COLORMAP_JET)
        firstp[ego_canvas == 0] = (30, 30, 30)
        half = self.canvas_px // 2
        cv2.arrowedLine(firstp, (half, half), (half, half - 25), (0, 0, 255), 2, tipLength=0.3)
        cv2.arrowedLine(firstp, (half, half), (half - 25, half), (0, 255, 0), 2, tipLength=0.3)
        cv2.circle(firstp, (half, half), 2, (255, 255, 255), -1)

        firstp_msg = self.bridge.cv2_to_imgmsg(firstp, encoding='bgr8')
        firstp_msg.header.stamp    = current_time
        firstp_msg.header.frame_id = self.robot_frame
        self.firstp_img_pub.publish(firstp_msg)

        t_end = time.perf_counter()

        if self.debug_telemetry:
            self.get_logger().info(
                f"[TIMER] "
                f"ego={(t_ego_done-t_timer_start)*1000:.1f}ms "
                f"grid_pub={(t_end-t_ego_done)*1000:.1f}ms "
                f"total={(t_end-t_timer_start)*1000:.1f}ms"
            )

def main(args=None):
    rclpy.init(args=args)
    node = CVMapDisplay()
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
