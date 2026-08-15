#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import threading
import queue
from collections import deque
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class BirdPerspective(Node):

    def __init__(self):
        super().__init__('bird_perspective')

        self.bridge = CvBridge()

        # Storage variables for camera state
        self.depth_img = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # BEV rasterization configuration
        # 1 pixel = 1 cm (0.01 m). Canvas 600x600 = 6m x 6m
        self.res = 0.01
        self.side = 600
        self.kernel_inflate = np.ones((3, 3), dtype=np.uint8)

        # QoS Profiles setup to avoid mismatched subscription errors
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ROS 2 Subscriptions
        self.rgb_sub = self.create_subscription(
            Image, '/detection/lines_and_curbs/raw', self.rgb_callback, 10
        )
        self.info_sub = self.create_subscription(
            CameraInfo, '/limo/color/camera_info', self.camera_info_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/limo/depth/image_raw', self.depth_callback, sensor_qos
        )

        # ROS 2 Publisher
        self.bird_pub = self.create_publisher(
            Image, '/limo/color/image_raw_bird_perspective', 10
        )

        # Queue and worker thread setup for multithreaded processing
        self.rgb_queue = queue.Queue(maxsize=1)
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        # Telemetry & Diagnostics setup
        self.declare_parameter('enable_telemetry', True)
        self.debug_telemetry = self.get_parameter('enable_telemetry').value
        self.frame_count = 0
        self.window_size = 30
        self.telemetry_stats = {
            'conversion': deque(maxlen=self.window_size),
            'bev_pipeline': deque(maxlen=self.window_size),
            'publish': deque(maxlen=self.window_size),
            'total': deque(maxlen=self.window_size),
        }

        self.get_logger().info("Multithreaded BirdPerspective node initialized successfully.")

    def camera_info_callback(self, msg):
        """Extracts intrinsic parameters from CameraInfo topic."""
        if self.fx is None:
            self.get_logger().info("CameraInfo received successfully!")
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_callback(self, msg):
        """Converts raw depth image to float32 meters array."""
        try:
            tmp_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")
            return

        if tmp_depth.dtype == np.uint16:
            self.depth_img = tmp_depth.astype(np.float32) / 1000.0
        else:
            self.depth_img = tmp_depth

    def rgb_callback(self, msg):
        """Non-blocking queue producer for incoming RGB frames."""
        if self.rgb_queue.full():
            try:
                self.rgb_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.rgb_queue.put_nowait(msg)
        except queue.Full:
            pass

    def _worker_loop(self):
        """Worker thread loop processing frames asynchronously."""
        while rclpy.ok():
            try:
                msg = self.rgb_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._process_frame(msg)

    def _process_frame(self, msg):
        """Executes full BEV projection pipeline and updates diagnostic telemetry."""
        t = {'conversion': 0.0, 'bev_pipeline': 0.0, 'publish': 0.0, 'total': 0.0}
        start_total = time.perf_counter()

        depth_img = self.depth_img
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        if depth_img is None or fx is None:
            missing = []
            if depth_img is None: missing.append("Depth")
            if fx is None: missing.append("CameraInfo")
            self.get_logger().warn(f"Waiting for {', '.join(missing)}; skipping frame.", throttle_duration_sec=2.0)
            return

        # --- 1. CONVERSION ---
        t_start = time.perf_counter()
        try:
            rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")
            return
        t['conversion'] = (time.perf_counter() - t_start) * 1000.0

        # --- 2. BEV PIPELINE ---
        t_start = time.perf_counter()

        # Rescale depth map and camera intrinsics if RGB overlay resolution differs
        h_rgb, w_rgb = rgb_img.shape[:2]
        h_depth, w_depth = depth_img.shape[:2]

        if (h_rgb, w_rgb) != (h_depth, w_depth):
            scale_x = w_rgb / float(w_depth)
            scale_y = h_rgb / float(h_depth)
            curr_fx = fx * scale_x
            curr_fy = fy * scale_y
            curr_cx = cx * scale_x
            curr_cy = cy * scale_y
            curr_depth = cv2.resize(depth_img, (w_rgb, h_rgb), interpolation=cv2.INTER_NEAREST)
        else:
            curr_fx, curr_fy, curr_cx, curr_cy = fx, fy, cx, cy
            curr_depth = depth_img

        # Identify non-black pixels from overlay
        valid_color_mask = (rgb_img[:, :, 0] > 0) | (rgb_img[:, :, 1] > 0) | (rgb_img[:, :, 2] > 0)
        v_indices, u_indices = np.where(valid_color_mask)

        if len(u_indices) == 0:
            empty_bev = np.zeros((self.side, self.side, 3), dtype=np.uint8)
            self._publish_bev(empty_bev, msg.header)
            return

        colors = rgb_img[v_indices, u_indices]
        z = curr_depth[v_indices, u_indices]

        # Filter out invalid depth values (between 10cm and 5m)
        valid_depth = (z > 0.1) & (z < 5.0) & (~np.isnan(z)) & (~np.isinf(z))
        u = u_indices[valid_depth]
        v = v_indices[valid_depth]
        z = z[valid_depth]
        colors = colors[valid_depth]

        if len(u) == 0:
            empty_bev = np.zeros((self.side, self.side, 3), dtype=np.uint8)
            self._publish_bev(empty_bev, msg.header)
            return

        # 3D Back-projection to Camera Frame
        # X_cam: Right, Y_cam: Down, Z_cam: Forward
        x_cam = (u - curr_cx) * z / curr_fx
        
        # Ground / Robot Frame Transformation
        # X_robot (Forward) = Z_cam
        # Y_robot (Left)    = -X_cam
        x_robot = z
        y_robot = -x_cam

        # Rasterization onto BEV Canvas
        bev_img = np.zeros((self.side, self.side, 3), dtype=np.uint8)

        # Center of robot is at bottom-middle of the image
        u_bev = (self.side / 2.0 - (y_robot / self.res)).astype(np.int32)
        v_bev = (self.side - (x_robot / self.res)).astype(np.int32)

        # Filter points within canvas boundaries
        mask = (u_bev >= 0) & (u_bev < self.side) & (v_bev >= 0) & (v_bev < self.side)

        if np.any(mask):
            bev_img[v_bev[mask], u_bev[mask]] = colors[mask]
            # Inflate lines to increase visibility
            bev_img = cv2.dilate(bev_img, self.kernel_inflate, iterations=1)

        t['bev_pipeline'] = (time.perf_counter() - t_start) * 1000.0

        # --- 3. PUBLISH ---
        t_start = time.perf_counter()
        self._publish_bev(bev_img, msg.header)
        t['publish'] = (time.perf_counter() - t_start) * 1000.0

        t['total'] = (time.perf_counter() - start_total) * 1000.0
        self.log_diagnostics(t, np.sum(mask))

    def _publish_bev(self, bev_img, header):
        """Converts and publishes the BEV OpenCV image to ROS topic."""
        bird_msg = self.bridge.cv2_to_imgmsg(bev_img, encoding='bgr8')
        bird_msg.header = header
        self.bird_pub.publish(bird_msg)

    def log_diagnostics(self, t, point_count):
        """Records telemetry timing statistics and outputs moving average profiles."""
        for key, val in t.items():
            if key in self.telemetry_stats:
                self.telemetry_stats[key].append(val)

        self.frame_count += 1
        if self.frame_count % 30 == 0 and self.debug_telemetry:
            avg = {
                key: float(np.mean(self.telemetry_stats[key])) if len(self.telemetry_stats[key]) > 0 else 0.0
                for key in self.telemetry_stats
            }
            fps = 1000.0 / avg['total'] if avg['total'] > 0 else 0.0

            self.get_logger().info(
                f"\n================ BEV WORKER TELEMETRY ({fps:.1f} FPS) ================\n"
                f"  Frames processed: {self.frame_count} | Active projected points: {point_count}\n"
                f"  [Total Latency]     Current: {t['total']:.2f} ms | Avg ({self.window_size}f): {avg['total']:.2f} ms\n"
                f"  -----------------------------------------------------------------\n"
                f"  [Conversion]        Current: {t['conversion']:.2f} ms | Avg ({self.window_size}f): {avg['conversion']:.2f} ms\n"
                f"  [BEV Pipeline]      Current: {t['bev_pipeline']:.2f} ms | Avg ({self.window_size}f): {avg['bev_pipeline']:.2f} ms\n"
                f"  [Publish]           Current: {t['publish']:.2f} ms | Avg ({self.window_size}f): {avg['publish']:.2f} ms\n"
                f"========================================================================="
            )


def main(args=None):
    rclpy.init(args=args)
    node = BirdPerspective()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()