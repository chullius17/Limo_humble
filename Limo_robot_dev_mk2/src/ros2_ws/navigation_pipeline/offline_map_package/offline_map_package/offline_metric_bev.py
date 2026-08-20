#!/usr/bin/env python3
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
import cv2
from cv_bridge import CvBridge
import numpy as np
import time
import threading
import queue
import array
from std_msgs.msg import Header
from typing import Tuple
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster
from collections import deque
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy


class OfflineMetricBEV(Node):
    """
    Converts the turquoise, white and magenta channels into metric cost grids.
    """

    COLORS = ['TURQUOISE', 'WHITE', 'MAGENTA']

    COLOR_MAP = {
        'TURQUOISE': np.array([255, 255,   0], dtype=np.uint8),
        'WHITE': np.array([255, 255, 255], dtype=np.uint8),
        'MAGENTA': np.array([255,   0, 255], dtype=np.uint8),
    }

    TOLERANCE = 30          # Pixel-value tolerance for color matching

    CONFIG_MAP = {
        'TURQUOISE': {'peak_cost': 60.0, 'radius': 2},
        'WHITE': {'peak_cost': 30.0, 'radius': 5},
        'MAGENTA': {'peak_cost': 100.0, 'radius': 5},
    }

    DECAY = 8.0
    ROI_FRACTION = 0.4
    RULER_FRACTION = 0.08

    def __init__(self):
        super().__init__('offline_metric_bev')
        self.bridge = CvBridge()
        self.publish_individual = True
        self.publish_combined = False
        self.topic_namespace = 'offline/metric_bev'
        self.frame_prefix = 'offline_metric_bev_origin'

        self.latest_bev_stamp = None

        # Configure QoS profile compatible with RViz Map display
        map_qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        # Debug images are realtime data: stale frames are less useful than a
        # dropped frame, so never let a slow visualizer build up a backlog.
        debug_image_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # Force use_sim_time parameter configuration
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time',
                             rclpy.Parameter.Type.BOOL, True)])

        self.colors = self.COLORS
        self.color_map = self.COLOR_MAP
        self.config_map = self.CONFIG_MAP
        self.get_logger().info('Using TURQUOISE, WHITE and MAGENTA color maps')

        # Global (color-independent) parameters
        self.declare_parameter('fixed_frame', 'base_link')
        self.declare_parameter('global_frame', 'odom')
        self.declare_parameter('resolution', 0.0092)
        self.declare_parameter('publish_debug', True)

        self.fixed_frame   = self.get_parameter('fixed_frame').value
        self.global_frame  = self.get_parameter('global_frame').value
        self.resolution    = self.get_parameter('resolution').value
        self.publish_debug = self.get_parameter('publish_debug').value

        # Pre-build 256-entry BGR Lookup Table for heatmap rendering (shared across colors)
        lut_inputs = np.arange(256, dtype=np.uint8).reshape(256, 1)
        jet_inputs = np.clip(lut_inputs.astype(np.float32) * (255.0 / 100.0), 0, 255).astype(np.uint8)
        jet_colors = cv2.applyColorMap(jet_inputs, cv2.COLORMAP_JET).reshape(256, 3)
        jet_colors[0] = [0, 0, 0]  # Cost 0 = black background
        self.debug_lut = jet_colors

        # TF2 Listener and Broadcaster setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer at 10Hz to publish TF frames continuously
        self.tf_timer = self.create_timer(0.1, self.tf_callback)

        # Single subscription shared by all active colors
        self.bev_sub = self.create_subscription(
            Image,
            'limo/nav_cv_package/classification/output/raw',
            self.bev_callback,
            10
        )

        # Global ROI debug publishers
        if self.publish_debug:
            self.roi_debug_pub = self.create_publisher(
                Image,
                f'/limo/nav_map_package/{self.topic_namespace}/roi_debug',
                debug_image_qos,
            )

        # Per-color publisher bundle.
        self.color_state = {}
        for color in self.colors:
            suffixes = [color.lower()]
            state = {'outputs': {}}
            for suffix in suffixes if self.publish_individual else []:
                output = {'topic_suffix': suffix}
                output['costmap_pub'] = self.create_publisher(
                    OccupancyGrid,
                    f'/limo/nav_map_package/{self.topic_namespace}/cost_grid_{suffix}',
                    map_qos_profile,
                )
                if self.publish_debug:
                    output['debug_pub'] = self.create_publisher(
                        Image,
                        f'/limo/nav_map_package/{self.topic_namespace}/debug_{suffix}',
                        10,
                    )
                state['outputs'][suffix] = output

            self.color_state[color] = state

        self.combined_costmap_pub = None
        if self.publish_combined:
            self.combined_costmap_pub = self.create_publisher(
                OccupancyGrid,
                f'/limo/nav_map_package/{self.topic_namespace}/cost_grid_combined',
                map_qos_profile,
            )

        # Debug parameter to enable/disable telemetry logging
        self.declare_parameter('enable_telemetry', True)
        self.debug_telemetry = self.get_parameter('enable_telemetry').value

        # Frame counter to throttle logging output
        self.frame_count = 0

        # Sliding window buffer (30 frames) for telemetry metrics
        self.window_size = 30
        self.telemetry_stats_shared = {
            'conversion': deque(maxlen=self.window_size),
            'roi_crop':   deque(maxlen=self.window_size),
            'roi_debug':  deque(maxlen=self.window_size),
            'total':      deque(maxlen=self.window_size),
        }
        self.telemetry_stats_color = {
            color: {
                'mask_inflate':      deque(maxlen=self.window_size),
                'occupancy_publish': deque(maxlen=self.window_size),
                'debug_render':      deque(maxlen=self.window_size),
            }
            for color in self.colors
        }

        self.get_logger().info(
            f'offline_metric_bev initialized for channels {self.colors}; '
            f'individual={self.publish_individual}, combined={self.publish_combined}'
        )

        self.bev_queue = queue.Queue(maxsize=1)
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def tf_callback(self):
        """
        Publishes the static transform between base_footprint and cv_origin_[color].
        Uses the exact timestamp header retrieved from base_footprint via TF2 buffer.
        """
        try:
            tf_base = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.fixed_frame,
                rclpy.time.Time()
            )
            stamp = tf_base.header.stamp
        except TransformException:
            if self.latest_bev_stamp is not None:
                stamp = self.latest_bev_stamp
            else:
                stamp = self.get_clock().now().to_msg()

        output_suffixes = []
        if self.publish_individual:
            output_suffixes.extend(color.lower() for color in self.colors)
        if self.publish_combined:
            output_suffixes.append('combined')
        for suffix in output_suffixes:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.fixed_frame   # base_link
            t.child_frame_id = f'{self.frame_prefix}_{suffix}'
            t.transform.translation.x = 0.6
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.0
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t)

    def _crop_to_roi(self, bgr: np.ndarray) -> Tuple[np.ndarray, int, int]:
        h = bgr.shape[0]
        roi_bottom = int(round(h * (1.0 - self.ROI_FRACTION)))
        roi_top = int(round(h * (1.0 - self.RULER_FRACTION)))
        return bgr[roi_bottom:roi_top, :], roi_bottom, roi_top

    def _make_mask(self, bgr: np.ndarray, exact_color: np.ndarray) -> np.ndarray:
        lo = np.clip(exact_color.astype(np.int16) - self.TOLERANCE, 0, 255).astype(np.uint8)
        hi = np.clip(exact_color.astype(np.int16) + self.TOLERANCE, 0, 255).astype(np.uint8)
        return cv2.inRange(bgr, lo, hi)

    def _build_cost_layer(self, mask: np.ndarray, peak_cost: float,
                          radius_px: int) -> np.ndarray:
        # Outward inflation: distance from every external pixel to the nearest
        # pixel belonging to this color mask.
        obstacle = cv2.bitwise_not(mask)
        distance_from_mask = cv2.distanceTransform(
            obstacle,
            cv2.DIST_L2,
            cv2.DIST_MASK_3,
        )

        cost_layer = np.zeros(mask.shape, dtype=np.float32)
        within_radius = distance_from_mask <= radius_px
        if np.any(within_radius):
            normalized_distance = (
                distance_from_mask[within_radius] / max(radius_px, 1)
            )
            cost_layer[within_radius] = peak_cost * np.exp(
                -self.DECAY * normalized_distance
            )

        return np.clip(cost_layer, 0, 100).astype(np.uint8)

    def _image_to_costmap(self, bgr: np.ndarray, color: str) -> np.ndarray:
        target_color = self.color_map[color]
        config = self.config_map[color]

        mask = self._make_mask(bgr, target_color)

        return self._build_cost_layer(
            mask,
            config['peak_cost'],
            config['radius'],
        )

    def _costmap_to_occupancy_grid(self, cost_img: np.ndarray, header: Header, topic_suffix: str) -> OccupancyGrid:
        rotated_cost = cv2.rotate(cost_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated_cost = cv2.flip(rotated_cost, 1)

        h, w = rotated_cost.shape
        grid = OccupancyGrid()

        grid.header.stamp = header.stamp
        grid.header.frame_id = f'{self.frame_prefix}_{topic_suffix}'

        grid.info.resolution = self.resolution
        grid.info.width = w
        grid.info.height = h

        grid.info.origin.position.x = 0.0
        grid.info.origin.position.y = -(h * self.resolution) / 2.0
        grid.info.origin.orientation.w = 1.0

        grid.data = array.array('b', rotated_cost.astype(np.int8).tobytes())
        return grid

    def _render_debug(self, cost_img: np.ndarray) -> np.ndarray:
        return self.debug_lut[cost_img]

    def _render_roi_debug(self, full_bgr: np.ndarray, roi_bottom: int, roi_top: int) -> np.ndarray:
        debug = full_bgr.copy()
        h, w = full_bgr.shape[0], full_bgr.shape[1]

        overlay = debug[:roi_bottom, :].copy()
        cv2.addWeighted(overlay, 0.3, np.zeros_like(overlay), 0.7, 0, debug[:roi_bottom, :])

        cv2.line(debug, (0, roi_bottom), (w - 1, roi_bottom), color=(0, 255, 0), thickness=2)
        cv2.putText(debug, f'ROI: bottom {self.ROI_FRACTION:.0%}', (8, roi_bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        ruler_y = roi_top
        cv2.line(debug, (0, ruler_y), (w - 1, ruler_y), color=(255, 255, 0), thickness=1)

        for x in range(0, w, 10):
            if x % 50 == 0:
                cv2.line(debug, (x, ruler_y - 8), (x, ruler_y + 8), color=(255, 255, 0), thickness=2)
                if x > 0 and x < w - 20:
                    cv2.putText(debug, str(x), (x - 10, ruler_y - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
            else:
                cv2.line(debug, (x, ruler_y - 4), (x, ruler_y + 4), color=(255, 255, 0), thickness=1)

        return debug

    def bev_callback(self, msg: Image):
        self.latest_bev_stamp = msg.header.stamp
        if self.bev_queue.full():
            try:
                self.bev_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.bev_queue.put_nowait(msg)
        except queue.Full:
            pass

    def _worker_loop(self):
        while rclpy.ok():
            try:
                msg = self.bev_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._process_frame(msg)

    def _process_frame(self, msg: Image):
        t = {'conversion': 0.0, 'roi_crop': 0.0, 'roi_debug': 0.0, 'total': 0.0}
        t_color = {color: {'mask_inflate': 0.0, 'occupancy_publish': 0.0, 'debug_render': 0.0}
                   for color in self.colors}
        start_total = time.perf_counter()

        # --- Shared: CvBridge conversion ---
        t_start = time.perf_counter()
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return
        t['conversion'] = (time.perf_counter() - t_start) * 1000

        # --- Shared: ROI crop ---
        t_start = time.perf_counter()
        roi_bgr, roi_bottom, roi_top = self._crop_to_roi(bgr)
        t['roi_crop'] = (time.perf_counter() - t_start) * 1000

        # --- Shared: Single ROI debug overlay publishing ---
        has_roi_raw = self.publish_debug and self.roi_debug_pub.get_subscription_count() > 0

        if has_roi_raw:
            t_start = time.perf_counter()
            roi_debug_bgr = self._render_roi_debug(bgr, roi_bottom, roi_top)
            roi_debug_msg = self.bridge.cv2_to_imgmsg(roi_debug_bgr, encoding='bgr8')
            roi_debug_msg.header = msg.header
            self.roi_debug_pub.publish(roi_debug_msg)
            t['roi_debug'] = (time.perf_counter() - t_start) * 1000

        # --- Per-color processing ---
        cost_layers = []
        for color in self.colors:
            state = self.color_state[color]

            t_start = time.perf_counter()
            cost_img_cropped = self._image_to_costmap(roi_bgr, color)
            cost_layers.append(cost_img_cropped)
            t_color[color]['mask_inflate'] = (time.perf_counter() - t_start) * 1000

            if self.publish_individual:
                t_start = time.perf_counter()
                output_layers = {color.lower(): cost_img_cropped}
                for suffix, cost_layer in output_layers.items():
                    state['outputs'][suffix]['costmap_pub'].publish(
                        self._costmap_to_occupancy_grid(
                            cost_layer,
                            msg.header,
                            suffix,
                        )
                    )
                t_color[color]['occupancy_publish'] = (
                    time.perf_counter() - t_start
                ) * 1000

            if self.publish_individual and self.publish_debug:
                debug_outputs = [
                    (suffix, cost_layer)
                    for suffix, cost_layer in output_layers.items()
                    if state['outputs'][suffix]['debug_pub'].get_subscription_count() > 0
                ]
                if debug_outputs:
                    t_start = time.perf_counter()
                    for suffix, cost_layer in debug_outputs:
                        debug_bgr = self._render_debug(cost_layer)
                        debug_msg = self.bridge.cv2_to_imgmsg(debug_bgr, encoding='bgr8')
                        debug_msg.header = msg.header
                        state['outputs'][suffix]['debug_pub'].publish(debug_msg)
                    t_color[color]['debug_render'] = (time.perf_counter() - t_start) * 1000

        # The semantic costs match CVMapDisplay: turquoise=60, white=30 and
        # magenta=100. Taking the cell-wise maximum preserves the dominant
        # class wherever inflated layers overlap.
        if self.publish_combined:
            combined_cost = np.maximum.reduce(cost_layers)
            self.combined_costmap_pub.publish(
                self._costmap_to_occupancy_grid(
                    combined_cost,
                    msg.header,
                    'combined',
                )
            )

        t['total'] = (time.perf_counter() - start_total) * 1000
        self.log_diagnostics(t, t_color)

    def log_diagnostics(self, t, t_color):
        for key, val in t.items():
            if key in self.telemetry_stats_shared:
                self.telemetry_stats_shared[key].append(val)

        for color in self.colors:
            for key, val in t_color[color].items():
                if key in self.telemetry_stats_color[color]:
                    self.telemetry_stats_color[color][key].append(val)

        self.frame_count += 1
        if self.frame_count % 30 == 0 and self.debug_telemetry:
            avg_shared = {
                key: float(np.mean(self.telemetry_stats_shared[key])) if len(self.telemetry_stats_shared[key]) > 0 else 0.0
                for key in self.telemetry_stats_shared
            }
            fps = 1000.0 / avg_shared['total'] if avg_shared['total'] > 0 else 0.0

            active_colors_str = "+".join(self.colors)
            lines = [
                f"\n================ METRIC BEV [{active_colors_str}] WORKER @ {fps:.1f} FPS ================",
                f"  Frames processed: {self.frame_count}",
                f"  [Total Latency]        Current: {t['total']:.2f} ms | Avg ({self.window_size}f): {avg_shared['total']:.2f} ms",
                f"  -----------------------------------------------------------------",
                f"  [CvBridge Conversion]  Current: {t['conversion']:.2f} ms | Avg ({self.window_size}f): {avg_shared['conversion']:.2f} ms",
                f"  [ROI Crop]             Current: {t['roi_crop']:.2f} ms | Avg ({self.window_size}f): {avg_shared['roi_crop']:.2f} ms",
                f"  [ROI Debug Pub]        Current: {t['roi_debug']:.2f} ms | Avg ({self.window_size}f): {avg_shared['roi_debug']:.2f} ms",
            ]
            for color in self.colors:
                avg_c = {
                    k: float(np.mean(self.telemetry_stats_color[color][k])) if len(self.telemetry_stats_color[color][k]) > 0 else 0.0
                    for k in self.telemetry_stats_color[color]
                }
                lines.append(f"  ---------------- {color} ----------------")
                lines.append(f"  [Mask + Inflate]       Current: {t_color[color]['mask_inflate']:.2f} ms | Avg ({self.window_size}f): {avg_c['mask_inflate']:.2f} ms")
                lines.append(f"  [Occupancy Publish]    Current: {t_color[color]['occupancy_publish']:.2f} ms | Avg ({self.window_size}f): {avg_c['occupancy_publish']:.2f} ms")
                lines.append(f"  [Debug Render]         Current: {t_color[color]['debug_render']:.2f} ms | Avg ({self.window_size}f): {avg_c['debug_render']:.2f} ms")
            lines.append("=========================================================================")

            self.get_logger().info("\n".join(lines))


def main(args=None):
    rclpy.init(args=args)
    node = OfflineMetricBEV()
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
