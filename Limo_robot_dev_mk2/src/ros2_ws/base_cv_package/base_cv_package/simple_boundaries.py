#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import threading
from queue import Queue, Empty
from collections import deque


class CurbDetector(Node):

    def __init__(self):
        super().__init__('curb_detector')
        self.bridge = CvBridge()

        # ROI parameters for cropping
        self.declare_parameter('roi_y_min', 0.5)
        self.declare_parameter('roi_y_max', 1.0)
        self.roi_y_min = self.get_parameter('roi_y_min').value
        self.roi_y_max = self.get_parameter('roi_y_max').value

        # Topic Subscription
        self.image_sub = self.create_subscription(
            Image,
            'limo/base_cv/detection/lane_masks/raw',
            self.image_callback,
            10
        )

        # Publishers
        self.debug_pub = self.create_publisher(Image, 'limo/base_cv/boundaries/curb_points_debug/raw', 10)
        self.lines_pub = self.create_publisher(Image, 'limo/base_cv/boundaries/lines_and_curbs/raw', 10)

        # Threading and Queue Setup
        self.frame_queue = Queue(maxsize=1)
        self.is_running = True

        # Structuring elements for multi-zone background cleaning
        # Bottom third: Large kernel for heavy noise removal
        self.kernel_bottom = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        # Middle third: Small kernel for moderate noise removal
        self.kernel_middle = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        # Top third: No opening performed

        self.sampling_step = 3

        # Telemetry & Diagnostics setup
        self.declare_parameter('enable_telemetry', True)
        self.debug_telemetry = self.get_parameter('enable_telemetry').value
        self.frame_count = 0
        self.window_size = 30
        self.telemetry_stats = {
            'decomp': deque(maxlen=self.window_size),
            'step1_masks': deque(maxlen=self.window_size),
            'step3_crop': deque(maxlen=self.window_size),
            'step4_bg_clean': deque(maxlen=self.window_size),
            'step5_color_iso': deque(maxlen=self.window_size),
            'step7_points': deque(maxlen=self.window_size),
            'step8_draw_publish': deque(maxlen=self.window_size),
            'total': deque(maxlen=self.window_size),
        }

        # Start background worker thread
        self.worker_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.worker_thread.start()
        self.get_logger().info("CurbDetector node initialized: Multi-zone MORPH_OPEN Background Extraction.")

    def image_callback(self, msg):
        """ROS 2 Callback: Enqueues incoming frames, dropping stale frames if queue is full."""
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                pass
        self.frame_queue.put(msg)

    def _processing_worker(self):
        """Worker Thread executing image processing pipeline asynchronously."""
        while self.is_running and rclpy.ok():
            try:
                msg = self.frame_queue.get(timeout=0.1)
            except Empty:
                continue

            self.process_image(msg)
            self.frame_queue.task_done()

    def process_image(self, msg):
        start_total = time.perf_counter()
        t = {
            'decomp': 0.0,
            'step1_masks': 0.0,
            'step3_crop': 0.0,
            'step4_bg_clean': 0.0,
            'step5_color_iso': 0.0,
            'step7_points': 0.0,
            'step8_draw_publish': 0.0,
            'total': 0.0,
        }

        # Image Decompression
        try:
            t_start = time.perf_counter()
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            t['decomp'] = (time.perf_counter() - t_start) * 1000.0
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {str(e)}")
            return

        high_h, high_w, _ = cv_image.shape

        # --- STEP 1: DOWN-SAMPLING & UNCLASSIFIED MASK ---
        t_start = time.perf_counter()
        LOW_RES_SIZE = 300
        low_res = cv2.resize(cv_image, (LOW_RES_SIZE, LOW_RES_SIZE), interpolation=cv2.INTER_NEAREST)

        # Foreground: Pixels that are classified (non-black)
        foreground_mask = ((low_res[:, :, 0] > 0) |
                           (low_res[:, :, 1] > 0) |
                           (low_res[:, :, 2] > 0)).astype(np.uint8) * 255

        # Background: Pixels that are NOT classified (black in raw lane_masks)
        background_mask = cv2.bitwise_not(foreground_mask)
        t['step1_masks'] = (time.perf_counter() - t_start) * 1000.0

        if not np.any(background_mask):
            empty_frame = np.zeros_like(cv_image)
            t_start = time.perf_counter()
            self.publish_image(self.debug_pub, empty_frame, msg.header.stamp)
            self.publish_image(self.lines_pub, empty_frame, msg.header.stamp)
            t['step8_draw_publish'] = (time.perf_counter() - t_start) * 1000.0
            t['total'] = (time.perf_counter() - start_total) * 1000.0
            self.log_diagnostics(high_w, high_h, t)
            return

        # --- STEP 3: FIXED GEOMETRIC CROP OF ROI ---
        t_start = time.perf_counter()
        background_mask[int(LOW_RES_SIZE * self.roi_y_max):, :] = False
        background_mask[:int(LOW_RES_SIZE * self.roi_y_min), :] = False

        background_pure = background_mask.copy()
        
        t['step3_crop'] = (time.perf_counter() - t_start) * 1000.0

        # --- STEP 4: MULTI-ZONE BACKGROUND CLEANING ---
        t_start = time.perf_counter()
        
        # Calculate horizontal band indices (3 equal vertical zones)
        y_third = LOW_RES_SIZE // 3
        y_two_thirds = 2 * y_third - 10     # Fine tuning by hand

        # Top Third [0 : y_third]: Zero opening (keep mask as is)
        
        # Middle Third [y_third : y_two_thirds]: Small kernel opening
        background_mask[y_third:y_two_thirds, :] = cv2.morphologyEx(
            background_mask, cv2.MORPH_OPEN, self.kernel_middle, iterations=1
        )[y_third:y_two_thirds, :]

        # Bottom Third [y_two_thirds : LOW_RES_SIZE]: Large kernel opening
        background_mask[y_two_thirds:, :] = cv2.morphologyEx(
            background_mask, cv2.MORPH_OPEN, self.kernel_bottom, iterations=2
        )[y_two_thirds:, :]

        # Isola il rumore/tratteggio rimosso con le aperture morfologiche
        dashed_mask = cv2.bitwise_xor(background_pure, background_mask)
        
        t['step4_bg_clean'] = (time.perf_counter() - t_start) * 1000.0

        # --- STEP 5: TWO-COLOR ISOLATION ---
        t_start = time.perf_counter()
        full_overlay_frame = cv_image.copy()
        only_lines_frame = np.zeros_like(cv_image)

        b_low = low_res[:, :, 0]
        g_low = low_res[:, :, 1]
        r_low = low_res[:, :, 2]
    
        is_green_low = (r_low < 50) & (g_low > 200) & (b_low < 50)
        is_blue_low = (b_low > 200) & (g_low < 50) & (r_low < 50)
        t['step5_color_iso'] = (time.perf_counter() - t_start) * 1000.0

        # --- STEP 7: POINT EXTRACTION AND RESCALING ---
        t_start = time.perf_counter()
        raw_points_green = np.argwhere(is_green_low)
        raw_points_blue = np.argwhere(is_blue_low)
        raw_points_background = np.argwhere(background_mask > 0)
        raw_points_dashed = np.argwhere(dashed_mask > 0)

        scale_x = high_w / LOW_RES_SIZE
        scale_y = high_h / LOW_RES_SIZE

        scale_points = lambda raw_pts: np.stack([raw_pts[:, 1] * scale_x, raw_pts[:, 0] * scale_y], axis=-1).astype(np.int32) \
                        if len(raw_pts) > 0 else np.empty((0, 2), dtype=np.int32)

        pts_green = scale_points(raw_points_green)
        pts_blue = scale_points(raw_points_blue)
        pts_unclassified = scale_points(raw_points_background)
        pts_dashed = scale_points(raw_points_dashed)
        t['step7_points'] = (time.perf_counter() - t_start) * 1000.0

        # --- STEP 8: DRAW AND PUBLISH ---
        t_start = time.perf_counter()
        if len(pts_green) > 0:
            u_g = np.clip(pts_green[:, 0], 0, high_w - 1)
            v_g = np.clip(pts_green[:, 1], 0, high_h - 1)
            only_lines_frame[v_g, u_g] = [255, 255, 0]

        if len(pts_blue) > 0:
            u_b = np.clip(pts_blue[:, 0], 0, high_w - 1)
            v_b = np.clip(pts_blue[:, 1], 0, high_h - 1)
            only_lines_frame[v_b, u_b] = [255, 0, 0]

        # Draw unclassified background points in White BGR [255, 255, 255].
        if len(pts_unclassified) > 0:
            pts_sampled = pts_unclassified[::self.sampling_step]
            u_c = np.clip(pts_sampled[:, 0], 0, high_w - 1)
            v_c = np.clip(pts_sampled[:, 1], 0, high_h - 1)
            
            only_lines_frame[v_c, u_c] = [255, 255, 255]
            full_overlay_frame[v_c, u_c] = [255, 255, 255]

        # Draw dashed mask points in White BGR [255, 255, 255]
        if len(pts_dashed) > 0:
            pts_dashed_sampled = pts_dashed[::self.sampling_step]
            u_d = np.clip(pts_dashed_sampled[:, 0], 0, high_w - 1)
            v_d = np.clip(pts_dashed_sampled[:, 1], 0, high_h - 1)

            only_lines_frame[v_d, u_d] = [255, 255, 255]
            full_overlay_frame[v_d, u_d] = [255, 255, 255]

        self.publish_image(self.debug_pub, full_overlay_frame, msg.header.stamp)
        self.publish_image(self.lines_pub, only_lines_frame, msg.header.stamp)
        t['step8_draw_publish'] = (time.perf_counter() - t_start) * 1000.0

        t['total'] = (time.perf_counter() - start_total) * 1000.0
        self.log_diagnostics(high_w, high_h, t)

    def log_diagnostics(self, w, h, t):
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
                f"\n"
                f"================ CURB DETECTOR PROFILE ({w}x{h} @ {fps:.1f} WORKER FPS) ================\n"
                f"  Frames processed: {self.frame_count} | Queue depth: {self.frame_queue.qsize()}\n"
                f"  [Total Latency]                 Current: {t['total']:.2f} ms | Avg ({self.window_size}f): {avg['total']:.2f} ms\n"
                f"  -----------------------------------------------------------------\n"
                f"  [Decompression]                 Current: {t['decomp']:.2f} ms | Avg ({self.window_size}f): {avg['decomp']:.2f} ms\n"
                f"  [Step 1: Mask Extraction]       Current: {t['step1_masks']:.2f} ms | Avg ({self.window_size}f): {avg['step1_masks']:.2f} ms\n"
                f"  [Step 3: Crop]                  Current: {t['step3_crop']:.2f} ms | Avg ({self.window_size}f): {avg['step3_crop']:.2f} ms\n"
                f"  [Step 4: Multi-Zone BG Clean]   Current: {t['step4_bg_clean']:.2f} ms | Avg ({self.window_size}f): {avg['step4_bg_clean']:.2f} ms\n"
                f"  [Step 5: Color Isolation]       Current: {t['step5_color_iso']:.2f} ms | Avg ({self.window_size}f): {avg['step5_color_iso']:.2f} ms\n"
                f"  [Step 7: Point Extraction]      Current: {t['step7_points']:.2f} ms | Avg ({self.window_size}f): {avg['step7_points']:.2f} ms\n"
                f"  [Step 8: Draw & Publish]        Current: {t['step8_draw_publish']:.2f} ms | Avg ({self.window_size}f): {avg['step8_draw_publish']:.2f} ms\n"
                f"======================================================================"
            )

    def publish_image(self, publisher, frame, timestamp):
        try:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.stamp = timestamp
            publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish image: {str(e)}")

    def destroy_node(self):
        self.is_running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CurbDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
