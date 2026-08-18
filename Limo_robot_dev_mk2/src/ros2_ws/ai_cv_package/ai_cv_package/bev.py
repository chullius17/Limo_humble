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

        # Pre-calculated 4x4 Extrinsic Transformation Matrix (RotX(-pi/2))
        R = np.array([
            [1.0,  0.0,  0.0],
            [0.0,  0.0,  1.0],
            [0.0, -1.0,  0.0]
        ], dtype=np.float32)
        t = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.extrinsic_matrix = np.eye(4, dtype=np.float32)
        self.extrinsic_matrix[:3, :3] = R.T
        self.extrinsic_matrix[:3, 3] = -R.T @ t

        # BEV rasterization configuration
        self.res = 0.01
        self.side = 600
        self.output_height = self.side // 2
        self.kernel_inflate = np.ones((3, 3), dtype=np.uint8)

        # Camera streams normally use the ROS sensor-data QoS policy.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ROS 2 Subscriptions
        self.rgb_sub = self.create_subscription(
            Image, 'limo/ai_cv_package/boundaries/lines_and_curbs/raw', self.rgb_callback, 10
        )

        self.declare_parameter('camera_info_topic', '/rgb/camera_info')
        self.declare_parameter('depth_topic', '/depth_camera/depth/image_raw')
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value

        self.info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, sensor_qos
        )

        # ROS 2 Publisher
        self.bird_pub = self.create_publisher(
            Image, 'limo/ai_cv_package/bev/bird_perspective/raw', 10
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
        self.get_logger().info("CPU worker thread started.")
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
            self.get_logger().warn('Waiting for Depth and CameraInfo; skipping frame.', throttle_duration_sec=2.0)
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

        # Identify non-black pixels
        valid_color_mask = (rgb_img[:, :, 0] > 0) | (rgb_img[:, :, 1] > 0) | (rgb_img[:, :, 2] > 0)
        v_indices, u_indices = np.where(valid_color_mask)

        if len(u_indices) == 0:
            empty_bev = np.zeros((self.output_height, self.side, 3), dtype=np.uint8)
            self._publish_bev(empty_bev, msg.header)
            return

        colors = rgb_img[v_indices, u_indices]
        z = depth_img[v_indices, u_indices]

        # Filter out invalid depth values
        valid_depth = (z > 0.1) & (~np.isnan(z)) & (~np.isinf(z))
        u = u_indices[valid_depth]
        v = v_indices[valid_depth]
        z = z[valid_depth]
        colors = colors[valid_depth]

        if len(u) == 0:
            return

        # 3D Back-projection to camera frame
        x_c = (u - cx) * z / fx
        y_c = (v - cy) * z / fy
        camera_points = np.vstack((x_c, y_c, z, np.ones_like(x_c)))

        # Apply extrinsic transformation matrix
        world_points = np.dot(self.extrinsic_matrix, camera_points)

        # Rasterization onto BEV canvas
        # Publish only the upper half of the BEV (the area in front of the
        # robot). The robot origin remains at row self.side / 2, just below
        # the output image, so the metric coordinate convention is unchanged.
        bev_img = np.zeros((self.output_height, self.side, 3), dtype=np.uint8)

        u_bev = (world_points[0, :] / self.res + self.side / 2.0).astype(np.int32)
        v_bev = (world_points[1, :] / self.res + self.side / 2.0).astype(np.int32)

        mask = (
            (u_bev >= 0)
            & (u_bev < self.side)
            & (v_bev >= 0)
            & (v_bev < self.output_height)
        )
        bev_img[v_bev[mask], u_bev[mask]] = colors[mask]

        # Apply dilation to inflate projected pixels
        bev_img = cv2.dilate(bev_img, self.kernel_inflate, iterations=1)
        t['bev_pipeline'] = (time.perf_counter() - t_start) * 1000.0

        # --- 3. PUBLISH ---
        t_start = time.perf_counter()
        self._publish_bev(bev_img, msg.header)
        t['publish'] = (time.perf_counter() - t_start) * 1000.0

        t['total'] = (time.perf_counter() - start_total) * 1000.0
        self.log_diagnostics(t, len(u))

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
                f"  Frames processed: {self.frame_count} | Active points: {point_count}\n"
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
