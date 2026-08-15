#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

import time
from collections import deque
import threading
import queue
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class ColorLaneDetector(Node):

    def __init__(self):
        super().__init__('color_lane_detector')

        # ROS 2 Publishers
        self.raw_mask_pub = self.create_publisher(Image, 'limo/base_cv/detection/lane_masks/raw', 10)
        self.image_pub = self.create_publisher(Image, 'limo/base_cv/detection/lane_overlay/raw', 10)
        
        self.bridge = CvBridge()

        # Telemetry control parameters
        self.declare_parameter('enable_telemetry', True)
        self.debug_telemetry = self.get_parameter('enable_telemetry').value
        self.frame_counter = 0

        # Target processing size for low-latency operations
        self.target_size = 300

        # HSV Threshold parameters for Yellow and Black colors
        self.yellow_lower = np.array([15, 80, 80], dtype=np.uint8)
        self.yellow_upper = np.array([35, 255, 255], dtype=np.uint8)

        self.black_lower = np.array([0, 0, 0], dtype=np.uint8)
        self.black_upper = np.array([180, 255, 150], dtype=np.uint8)

        # Best effort QoS matching high-rate video streams
        latest_frame_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscriber to Limo's camera
        self.declare_parameter('rgb_topic', '/rgb/image_raw')
        self.camera_topic = self.get_parameter('rgb_topic').value
        self.rgb_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            latest_frame_qos
        )

        # ROI parameters for cropping
        self.declare_parameter('roi_y_min', 0.1)
        self.declare_parameter('roi_y_max', 1.0)
        self.roi_y_min = self.get_parameter('roi_y_min').value
        self.roi_y_max = self.get_parameter('roi_y_max').value

        # Telemetry metrics window (sliding window of 30 frames)
        self.window_size = 30
        self.telemetry_stats = {
            '0_transport_delay': deque(maxlen=self.window_size),
            '1_convert_time': deque(maxlen=self.window_size),
            '2_queue_waiting_time': deque(maxlen=self.window_size),
            '3_hsv_segmentation': deque(maxlen=self.window_size),
            '9_post_canvas': deque(maxlen=self.window_size),
            '11_ros_publish_enqueue': deque(maxlen=self.window_size),
            'total_pipeline': deque(maxlen=self.window_size),
            'async_encode_publish': deque(maxlen=self.window_size),
            '12_msg_age_final_publish': deque(maxlen=self.window_size),
        }

        # Threading queues
        self.processing_queue = queue.Queue(maxsize=1)
        self.pub_queue = queue.Queue(maxsize=1)

        # Start async worker threads
        self.worker_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.pub_thread = threading.Thread(target=self._publish_worker, daemon=True)
        self.worker_thread.start()
        self.pub_thread.start()

        self.get_logger().info("HSV Color Segmentation Node initialized successfully.")

    def image_callback(self, msg):
        """Producer Callback: Non-blocking enqueue to prevent ROS executor queue delays."""
        # Update parameter value dynamically
        self.debug_telemetry = self.get_parameter('enable_telemetry').value

        if self.debug_telemetry:
            t_now_ros = self.get_clock().now().nanoseconds / 1e9
            t_msg_ros = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            transport_delay = (t_now_ros - t_msg_ros) * 1000.0
            self.telemetry_stats['0_transport_delay'].append(transport_delay)

        # Drop older frames if worker thread is occupied to minimize ingress latency
        if self.processing_queue.full():
            try:
                self.processing_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            time_entering = time.perf_counter() if self.debug_telemetry else 0.0
            self.processing_queue.put_nowait((msg, time_entering))
        except queue.Full:
            pass

    def _processing_worker(self):
        """Worker thread executing frame conversion and color thresholding."""
        while rclpy.ok():
            try:
                item = self.processing_queue.get(timeout=0.5)
                time_exiting = time.perf_counter() if self.debug_telemetry else 0.0
            except queue.Empty:
                continue

            msg, time_entering = item

            if self.debug_telemetry:
                queue_delay = (time_exiting - time_entering) * 1000.0
                self.telemetry_stats['2_queue_waiting_time'].append(queue_delay)

            t_convert_start = time.perf_counter() if self.debug_telemetry else 0.0
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception as e:
                self.get_logger().error(f"CvBridge image conversion failed: {str(e)}")
                continue

            if self.debug_telemetry:
                convert_time = (time.perf_counter() - t_convert_start) * 1000.0
                self.telemetry_stats['1_convert_time'].append(convert_time)

            self._process_frame(cv_image, msg.header.stamp)

    def _process_frame(self, cv_image, stamp):
        """Processes BGR frame using HSV color space thresholds with ROI cropping and low-res scaling."""
        t_start = time.perf_counter() if self.debug_telemetry else 0.0

        # Step 1: Downsampling frame to target size (300x300)
        low_res = cv2.resize(cv_image, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)

        # Step 2: Convert to HSV color space
        t_hsv_start = time.perf_counter() if self.debug_telemetry else 0.0
        hsv_image = cv2.cvtColor(low_res, cv2.COLOR_BGR2HSV)

        # Step 3: Create binary masks for Yellow and Black colors
        yellow_mask = cv2.inRange(hsv_image, self.yellow_lower, self.yellow_upper)
        black_mask = cv2.inRange(hsv_image, self.black_lower, self.black_upper)

        # Step 4: Geometric ROI Cropping (Keep band between Y_min = 54% and Y_max = 91%)
        y_min = int(self.target_size * self.roi_y_min)
        y_max = int(self.target_size * self.roi_y_max)

        yellow_mask[:y_min, :] = 0
        yellow_mask[y_max:, :] = 0
        black_mask[:y_min, :] = 0
        black_mask[y_max:, :] = 0

        if self.debug_telemetry:
            t_hsv_end = time.perf_counter()
            self.telemetry_stats['3_hsv_segmentation'].append((t_hsv_end - t_hsv_start) * 1000.0)

        # Step 5: Construct raw color mask image on low-res scale
        t_canvas_start = time.perf_counter() if self.debug_telemetry else 0.0
        mask_overlay = np.zeros_like(low_res)

        # Assign Blue color (255, 0, 0) to black-detected pixels
        mask_overlay[black_mask > 0] = (255, 0, 0)

        # Assign Green color (0, 255, 0) to yellow-detected pixels
        mask_overlay[yellow_mask > 0] = (0, 255, 0)

        # Create blended debug image overlay
        overlay_image = cv2.addWeighted(low_res, 0.7, mask_overlay, 0.5, 0)

        if self.debug_telemetry:
            t_canvas_end = time.perf_counter()
            self.telemetry_stats['9_post_canvas'].append((t_canvas_end - t_canvas_start) * 1000.0)

        # Step 6: Enqueue frames for publisher worker thread
        t_pub_start = time.perf_counter() if self.debug_telemetry else 0.0
        if self.pub_queue.full():
            try:
                self.pub_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.pub_queue.put_nowait((mask_overlay, overlay_image, stamp))
        except queue.Full:
            pass

        if self.debug_telemetry:
            dt_pub_enqueue = (time.perf_counter() - t_pub_start) * 1000.0
            self.telemetry_stats['11_ros_publish_enqueue'].append(dt_pub_enqueue)

            dt_total_pipe = (time.perf_counter() - t_start) * 1000.0
            self.telemetry_stats['total_pipeline'].append(dt_total_pipe)

        # Log diagnostics every 30 frames if telemetry flag is enabled
        self.frame_counter += 1
        if self.frame_counter % 30 == 0 and self.debug_telemetry:
            self._log_telemetry_report()

    def _publish_worker(self):
        """Worker thread for ROS 2 frame serialization and publishing."""
        while rclpy.ok():
            try:
                item = self.pub_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            mask_overlay, overlay_image, stamp = item
            t_pub_start = time.perf_counter() if self.debug_telemetry else 0.0

            try:
                # 1. Publish raw mask image
                ros_mask_msg = self.bridge.cv2_to_imgmsg(mask_overlay, encoding='bgr8')
                ros_mask_msg.header.stamp = stamp
                self.raw_mask_pub.publish(ros_mask_msg)

                # 2. Publish debug overlay image
                ros_overlay_msg = self.bridge.cv2_to_imgmsg(overlay_image, encoding='bgr8')
                ros_overlay_msg.header.stamp = stamp
                self.image_pub.publish(ros_overlay_msg)

            except Exception as e:
                self.get_logger().error(f"Publishing failed: {str(e)}")

            if self.debug_telemetry:
                dt_async_pub = (time.perf_counter() - t_pub_start) * 1000.0
                self.telemetry_stats['async_encode_publish'].append(dt_async_pub)

                t_now_final = self.get_clock().now().nanoseconds / 1e9
                t_msg_final = stamp.sec + stamp.nanosec * 1e-9
                final_age = (t_now_final - t_msg_final) * 1000.0
                self.telemetry_stats['12_msg_age_final_publish'].append(final_age)

    def _log_telemetry_report(self):
        """Prints sliding window performance statistics averaged over the last 30 frames."""
        avg_transport = np.mean(self.telemetry_stats['0_transport_delay']) if len(self.telemetry_stats['0_transport_delay']) > 0 else 0.0
        avg_convert = np.mean(self.telemetry_stats['1_convert_time']) if len(self.telemetry_stats['1_convert_time']) > 0 else 0.0
        avg_queue_wait = np.mean(self.telemetry_stats['2_queue_waiting_time']) if len(self.telemetry_stats['2_queue_waiting_time']) > 0 else 0.0
        avg_hsv = np.mean(self.telemetry_stats['3_hsv_segmentation']) if len(self.telemetry_stats['3_hsv_segmentation']) > 0 else 0.0
        avg_canvas = np.mean(self.telemetry_stats['9_post_canvas']) if len(self.telemetry_stats['9_post_canvas']) > 0 else 0.0
        avg_pub_enqueue = np.mean(self.telemetry_stats['11_ros_publish_enqueue']) if len(self.telemetry_stats['11_ros_publish_enqueue']) > 0 else 0.0
        avg_total = np.mean(self.telemetry_stats['total_pipeline']) if len(self.telemetry_stats['total_pipeline']) > 0 else 0.0
        avg_async_pub = np.mean(self.telemetry_stats['async_encode_publish']) if len(self.telemetry_stats['async_encode_publish']) > 0 else 0.0
        avg_final_age = np.mean(self.telemetry_stats['12_msg_age_final_publish']) if len(self.telemetry_stats['12_msg_age_final_publish']) > 0 else 0.0

        fps = 1000.0 / avg_total if avg_total > 0 else 0.0

        self.get_logger().info(
            f"\n"
            f"====== COLOR SEGMENTATION PERFORMANCE REPORT (AVG {self.window_size} frames) ======\n"
            f"  Frames Processed: {self.frame_counter}\n"
            f"-----------------------------------------\n"
            f"[TIMESTAMP LATENCY]\n"
            f" Camera -> Node ingress:          {avg_transport:.2f} ms\n"
            f" Frame age final publish:          {avg_final_age:.2f} ms\n"
            f"-----------------------------------------\n"
            f"[EXECUTION BREAKDOWN]\n"
            f" Queue Waiting Delay:              {avg_queue_wait:.2f} ms\n"
            f" CvBridge Conversion:              {avg_convert:.2f} ms\n"
            f" HSV Color Thresholding (ROI/300px): {avg_hsv:.2f} ms\n"
            f" Canvas Rendering (Masks/Blend):   {avg_canvas:.2f} ms\n"
            f" ROS Publish Enqueue:              {avg_pub_enqueue:.2f} ms\n"
            f" Async Publish:                    {avg_async_pub:.2f} ms\n"
            f"-----------------------------------------\n"
            f" TOTAL PIPELINE TIME:              {avg_total:.2f} ms\n"
            f" ESTIMATED INTERNAL FPS:           {fps:.1f}\n"
            f"=========================================\n"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ColorLaneDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Color Lane Segmenter Node.")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()