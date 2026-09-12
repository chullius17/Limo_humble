#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import (
    CameraInfo,
    CompressedImage,
    Image,
    PointCloud2,
    PointField,
)
import cv2
from cv_bridge import CvBridge
import numpy as np
from turbojpeg import TJPF_BGR, TurboJPEG
from tf2_ros import Buffer, TransformException, TransformListener
import time
import threading
from queue import Queue, Empty
from collections import deque


class CurbDetector(Node):

    LABEL_INVALID = np.uint8(0)
    LABEL_BLUE = np.uint8(1)
    LABEL_TURQUOISE = np.uint8(2)
    LABEL_BACKGROUND = np.uint8(3)
    CLOUD_DTYPE = np.dtype({
        'names': ('x', 'y', 'z', 'class_id'),
        'formats': ('<f4', '<f4', '<f4', 'u1'),
        'offsets': (0, 4, 8, 12),
        'itemsize': 16,
    })
    CLOUD_FIELDS = [
        PointField(
            name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(
            name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(
            name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(
            name='class_id', offset=12, datatype=PointField.UINT8, count=1),
    ]

    def __init__(self):
        super().__init__('curb_detector')
        self.bridge = CvBridge()
        self.jpeg = TurboJPEG()

        self.declare_parameter('debug_jpeg_quality', 85)
        self.debug_jpeg_quality = int(
            self.get_parameter('debug_jpeg_quality').value)
        if not 1 <= self.debug_jpeg_quality <= 100:
            raise ValueError('debug_jpeg_quality must be in [1, 100]')

        # ROI parameters for cropping
        self.declare_parameter('roi_y_min', 0.0)
        self.declare_parameter('roi_y_max', 1.0)
        self.roi_y_min = self.get_parameter('roi_y_min').value
        self.roi_y_max = self.get_parameter('roi_y_max').value
        if not 0.0 <= self.roi_y_min < self.roi_y_max <= 1.0:
            raise ValueError(
                'roi_y_min and roi_y_max must define a range in [0, 1]')
        self.declare_parameter('point_voxel_size', 3)
        self.point_voxel_size = int(
            self.get_parameter('point_voxel_size').value)
        if self.point_voxel_size <= 0:
            raise ValueError('point_voxel_size must be positive')

        self.declare_parameter(
            'camera_info_topic', '/rgb/camera_info')
        self.declare_parameter(
            'depth_topic',
            'limo/cv_package/depth_correction/depth_corrected/raw')
        self.declare_parameter(
            'fallback_depth_topic', '/depth_camera/depth/image_raw')
        self.declare_parameter('corrected_depth_timeout_sec', 1.0)
        self.declare_parameter('fallback_depth_width', 320)
        self.declare_parameter('fallback_depth_height', 120)
        self.declare_parameter(
            'pointcloud_topic', 'limo/cv_package/boundaries/points')
        self.declare_parameter('bev_frame', 'base_link')
        self.declare_parameter('input_crop_y_min', 0.5)
        self.declare_parameter('pointcloud_min_depth_m', 0.1)
        self.declare_parameter('pointcloud_max_depth_m', 5.0)
        self.declare_parameter('blue_radius_min_m', 0.10)
        self.declare_parameter('blue_radius_max_m', 0.16)

        self.input_crop_y_min = float(
            self.get_parameter('input_crop_y_min').value)
        self.cloud_min_depth = float(
            self.get_parameter('pointcloud_min_depth_m').value)
        self.cloud_max_depth = float(
            self.get_parameter('pointcloud_max_depth_m').value)
        self.corrected_depth_timeout = float(
            self.get_parameter('corrected_depth_timeout_sec').value)
        self.fallback_depth_width = int(
            self.get_parameter('fallback_depth_width').value)
        self.fallback_depth_height = int(
            self.get_parameter('fallback_depth_height').value)
        self.blue_radius_min = float(
            self.get_parameter('blue_radius_min_m').value)
        self.blue_radius_max = float(
            self.get_parameter('blue_radius_max_m').value)
        self.bev_frame = str(self.get_parameter('bev_frame').value)
        if not 0.0 <= self.input_crop_y_min < 1.0:
            raise ValueError('input_crop_y_min must be in [0, 1)')
        if not 0.0 < self.cloud_min_depth < self.cloud_max_depth:
            raise ValueError('Point-cloud depth range is invalid')
        if self.corrected_depth_timeout <= 0.0:
            raise ValueError('corrected_depth_timeout_sec must be positive')
        if self.fallback_depth_width <= 0 or self.fallback_depth_height <= 0:
            raise ValueError('Fallback depth dimensions must be positive')
        if not 0.0 <= self.blue_radius_min <= self.blue_radius_max:
            raise ValueError('Blue point radii are invalid')
        if not self.bev_frame:
            raise ValueError('bev_frame cannot be empty')

        pipeline_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        cloud_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Topic Subscription
        self.image_sub = self.create_subscription(
            Image,
            'limo/cv_package/detection/lane_labels/raw',
            self.image_callback,
            pipeline_qos,
        )
        self.depth_sub = self.create_subscription(
            Image,
            self.get_parameter('depth_topic').value,
            self.depth_callback,
            sensor_qos,
        )
        self.fallback_depth_sub = self.create_subscription(
            Image,
            self.get_parameter('fallback_depth_topic').value,
            self.fallback_depth_callback,
            sensor_qos,
        )
        self.info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self.camera_info_callback,
            sensor_qos,
        )

        # Publishers
        self.debug_pub = self.create_publisher(
            CompressedImage,
            'limo/cv_package/boundaries/curb_points_debug/compressed',
            pipeline_qos,
        )
        self.lines_pub = self.create_publisher(
            CompressedImage,
            'limo/cv_package/boundaries/lines_and_curbs/compressed',
            pipeline_qos,
        )
        self.pointcloud_pub = self.create_publisher(
            PointCloud2,
            self.get_parameter('pointcloud_topic').value,
            cloud_qos,
        )

        self.sensor_lock = threading.Lock()
        self.depth_image = None
        self.depth_source = None
        self.corrected_depth_received_at = None
        self.camera_intrinsics = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Threading and Queue Setup
        self.frame_queue = Queue(maxsize=1)
        self.is_running = True

        # Telemetry & Diagnostics setup
        self.declare_parameter('enable_telemetry', True)
        self.debug_telemetry = self.get_parameter('enable_telemetry').value
        self.frame_count = 0
        self.window_size = 30
        self.telemetry_stats = {
            'decomp': deque(maxlen=self.window_size),
            'step1_masks': deque(maxlen=self.window_size),
            'step3_crop': deque(maxlen=self.window_size),
            'step5_color_iso': deque(maxlen=self.window_size),
            'step7_points': deque(maxlen=self.window_size),
            'step8_cloud_prepare': deque(maxlen=self.window_size),
            'step8_lock': deque(maxlen=self.window_size),
            'step8_tf': deque(maxlen=self.window_size),
            'step8_math': deque(maxlen=self.window_size),
            'step8_cloud_publish': deque(maxlen=self.window_size),
            'step9_draw_publish': deque(maxlen=self.window_size),
            'point_count': deque(maxlen=self.window_size),
            'total': deque(maxlen=self.window_size),
        }

        # Start background worker thread
        self.worker_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.worker_thread.start()
        self.get_logger().info(
            f'CurbDetector initialized: compact 2D cloud in '
            f'{self.bev_frame}.')

    def voxelize_points(self, raw_points, image_width):
        """Replace all points in each 2D cell with their centroid."""
        voxel_size = self.point_voxel_size
        if voxel_size == 1 or len(raw_points) == 0:
            return raw_points

        voxel_columns = (image_width + voxel_size - 1) // voxel_size
        voxel_ids = (
            (raw_points[:, 0] // voxel_size) * voxel_columns
            + raw_points[:, 1] // voxel_size
        )
        counts = np.bincount(voxel_ids)
        sum_y = np.bincount(voxel_ids, weights=raw_points[:, 0])
        sum_x = np.bincount(voxel_ids, weights=raw_points[:, 1])
        occupied = np.flatnonzero(counts)

        centroids_y = np.rint(
            sum_y[occupied] / counts[occupied]).astype(np.int32)
        centroids_x = np.rint(
            sum_x[occupied] / counts[occupied]).astype(np.int32)
        return np.column_stack((centroids_y, centroids_x))

    @staticmethod
    def quaternion_to_rotation(quaternion):
        """Return the rotation matrix represented by a ROS quaternion."""
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w
        norm = x * x + y * y + z * z + w * w
        if norm < 1e-12:
            raise ValueError('TF contains an invalid zero quaternion')

        scale = 2.0 / norm
        return np.array([
            [1.0 - scale * (y * y + z * z),
             scale * (x * y - z * w),
             scale * (x * z + y * w)],
            [scale * (x * y + z * w),
             1.0 - scale * (x * x + z * z),
             scale * (y * z - x * w)],
            [scale * (x * z - y * w),
             scale * (y * z + x * w),
             1.0 - scale * (x * x + y * y)],
        ], dtype=np.float32)

    def image_callback(self, msg):
        """ROS 2 Callback: Enqueues incoming frames, dropping stale frames if queue is full."""
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                pass
        self.frame_queue.put(msg)

    def camera_info_callback(self, msg):
        """Store the RGB calibration used to back-project label pixels."""
        intrinsics = (
            float(msg.k[0]),
            float(msg.k[4]),
            float(msg.k[2]),
            float(msg.k[5]),
            int(msg.width),
            int(msg.height),
        )
        with self.sensor_lock:
            self.camera_intrinsics = intrinsics

    def convert_depth(self, msg):
        """Convert a ROS depth image to float32 metres."""
        try:
            depth = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough')
        except Exception as error:
            self.get_logger().error(f'Depth conversion failed: {error}')
            return None
        if depth.dtype == np.uint16 or msg.encoding in ('16UC1', 'mono16'):
            return depth.astype(np.float32) * 0.001
        elif depth.dtype != np.float32:
            return depth.astype(np.float32)
        return depth

    def store_depth(self, depth, msg, source, received_at):
        """Atomically replace the selected depth frame and report switches."""
        with self.sensor_lock:
            previous_source = self.depth_source
            self.depth_image = depth
            self.depth_source = source
            if source == 'corrected':
                self.corrected_depth_received_at = received_at
        if previous_source != source:
            self.get_logger().info(f'Using {source} depth: {msg.header.frame_id}')

    def depth_callback(self, msg):
        """Always prefer a valid frame from the corrected depth topic."""
        depth = self.convert_depth(msg)
        if depth is None:
            return
        received_at = time.monotonic()
        self.store_depth(depth, msg, 'corrected', received_at)

    def fallback_depth_callback(self, msg):
        """Use raw depth only while corrected depth is unavailable or stale."""
        received_at = time.monotonic()
        with self.sensor_lock:
            corrected_at = self.corrected_depth_received_at
        if (corrected_at is not None and
                received_at - corrected_at <= self.corrected_depth_timeout):
            return

        depth = self.convert_depth(msg)
        if depth is None:
            return

        crop_start = int(depth.shape[0] * self.input_crop_y_min)
        cropped_depth = depth[crop_start:, :]
        if cropped_depth.size == 0:
            self.get_logger().warning(
                f'Cannot crop raw depth with input_crop_y_min='
                f'{self.input_crop_y_min:.3f}',
                throttle_duration_sec=2.0,
            )
            return
        depth = cv2.resize(
            cropped_depth,
            (self.fallback_depth_width, self.fallback_depth_height),
            interpolation=cv2.INTER_NEAREST,
        )

        # A corrected frame may have arrived while the raw image was decoded.
        with self.sensor_lock:
            corrected_at = self.corrected_depth_received_at
        if (corrected_at is not None and
                time.monotonic() - corrected_at <=
                self.corrected_depth_timeout):
            return
        self.store_depth(depth, msg, 'raw fallback', received_at)

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
            'step5_color_iso': 0.0,
            'step7_points': 0.0,
            'step8_cloud_prepare': 0.0,
            'step8_lock': 0.0,
            'step8_tf': 0.0,
            'step8_math': 0.0,
            'step8_cloud_publish': 0.0,
            'step9_draw_publish': 0.0,
            'point_count': float('nan'),
            'total': 0.0,
        }

        # Decode the compact class-label image.
        try:
            t_start = time.perf_counter()
            labels = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='mono8',
            )
            t['decomp'] = (time.perf_counter() - t_start) * 1000.0
        except Exception as e:
            self.get_logger().error(
                f"Failed to convert label image: {str(e)}")
            return

        if labels.ndim != 2:
            self.get_logger().error('Expected a single-channel label image')
            return
        height, width = labels.shape

        # --- STEP 1: DIRECT LABEL MASKS ---
        t_start = time.perf_counter()
        blue_mask = labels == self.LABEL_BLUE
        turquoise_mask = labels == self.LABEL_TURQUOISE
        background_mask = labels == self.LABEL_BACKGROUND
        t['step1_masks'] = (time.perf_counter() - t_start) * 1000.0

        # --- STEP 3: OPTIONAL ADDITIONAL BOUNDARY ROI ---
        t_start = time.perf_counter()
        roi_mask = np.zeros((height, width), dtype=bool)
        roi_start = int(height * self.roi_y_min)
        roi_stop = int(height * self.roi_y_max)
        roi_mask[roi_start:roi_stop, :] = True
        blue_mask &= roi_mask
        turquoise_mask &= roi_mask
        background_mask &= roi_mask
        t['step3_crop'] = (time.perf_counter() - t_start) * 1000.0

        # --- STEP 5: LABEL VALIDATION ---
        t_start = time.perf_counter()
        invalid_values = labels > self.LABEL_BACKGROUND
        if np.any(invalid_values):
            self.get_logger().warn(
                'Ignoring unsupported detector label values',
                throttle_duration_sec=2.0,
            )
        t['step5_color_iso'] = (time.perf_counter() - t_start) * 1000.0

        # --- STEP 7: POINT EXTRACTION ---
        t_start = time.perf_counter()
        raw_points_turquoise = np.argwhere(turquoise_mask)
        raw_points_blue = np.argwhere(blue_mask)
        raw_points_white = np.argwhere(background_mask)

        raw_points_blue = self.voxelize_points(
            raw_points_blue, width)
        raw_points_white = self.voxelize_points(
            raw_points_white, width)
        t['step7_points'] = (time.perf_counter() - t_start) * 1000.0

        # --- STEP 8: METRIC 2D BEV POINT CLOUD ---
        (point_count, prepare_ms, publish_ms,
         lock_ms, tf_ms, math_ms) = self.publish_pointcloud(
            raw_points_blue,
            raw_points_turquoise,
            raw_points_white,
            width,
            height,
            msg.header,
        )
        t['step8_cloud_prepare'] = prepare_ms
        t['step8_lock'] = lock_ms
        t['step8_tf'] = tf_ms
        t['step8_math'] = math_ms
        t['step8_cloud_publish'] = publish_ms
        if point_count is not None:
            t['point_count'] = point_count

        # --- STEP 9: DRAW AND PUBLISH COMPATIBILITY IMAGE ---
        t_start = time.perf_counter()
        publish_debug = self.debug_pub.get_subscription_count() > 0
        publish_lines = self.lines_pub.get_subscription_count() > 0
        if publish_debug or publish_lines:
            only_lines_frame = np.zeros(
                (height, width, 3), dtype=np.uint8)
            if len(raw_points_turquoise) > 0:
                v_t, u_t = raw_points_turquoise.T
                only_lines_frame[v_t, u_t] = [255, 255, 0]

            if len(raw_points_blue) > 0:
                v_b, u_b = raw_points_blue.T
                only_lines_frame[v_b, u_b] = [255, 0, 0]

            # Keep one valid-background representative per spatial voxel.
            if len(raw_points_white) > 0:
                v_w, u_w = raw_points_white.T
                only_lines_frame[v_w, u_w] = [255, 255, 255]

            try:
                debug_msg = self.encode_debug_image(
                    only_lines_frame, msg.header)
                if publish_debug:
                    self.debug_pub.publish(debug_msg)
                if publish_lines:
                    self.lines_pub.publish(debug_msg)
            except Exception as e:
                self.get_logger().error(
                    f"Failed to publish compressed debug image: {str(e)}")
        t['step9_draw_publish'] = (time.perf_counter() - t_start) * 1000.0

        t['total'] = (time.perf_counter() - start_total) * 1000.0
        self.log_diagnostics(width, height, t)

    def log_diagnostics(self, w, h, t):
        for key, val in t.items():
            if key in self.telemetry_stats and np.isfinite(val):
                self.telemetry_stats[key].append(val)

        self.frame_count += 1
        if self.frame_count % 30 == 0 and self.debug_telemetry:
            avg = {
                key: float(np.mean(self.telemetry_stats[key])) if len(self.telemetry_stats[key]) > 0 else 0.0
                for key in self.telemetry_stats
            }
            fps = 1000.0 / avg['total'] if avg['total'] > 0 else 0.0
            point_count_avg = (
                f'{avg["point_count"]:.0f}'
                if self.telemetry_stats['point_count'] else 'n/a'
            )

            self.get_logger().info(
                f"\n"
                f"================ CURB DETECTOR PROFILE ({w}x{h} @ {fps:.1f} WORKER FPS) ================\n"
                f"  Frames processed: {self.frame_count} | Queue depth: {self.frame_queue.qsize()}\n"
                f"  [Total Latency]                 Current: {t['total']:.2f} ms | Avg ({self.window_size}f): {avg['total']:.2f} ms\n"
                f"  -----------------------------------------------------------------\n"
                f"  [Decompression]                 Current: {t['decomp']:.2f} ms | Avg ({self.window_size}f): {avg['decomp']:.2f} ms\n"
                f"  [Step 1: Mask Extraction]       Current: {t['step1_masks']:.2f} ms | Avg ({self.window_size}f): {avg['step1_masks']:.2f} ms\n"
                f"  [Step 3: Crop]                  Current: {t['step3_crop']:.2f} ms | Avg ({self.window_size}f): {avg['step3_crop']:.2f} ms\n"
                f"  [Step 5: Label Validation]      Current: {t['step5_color_iso']:.2f} ms | Avg ({self.window_size}f): {avg['step5_color_iso']:.2f} ms\n"
                f"  [Step 7: Point Extraction]      Current: {t['step7_points']:.2f} ms | Avg ({self.window_size}f): {avg['step7_points']:.2f} ms\n"
                f"  [Step 8a: Cloud Preparation]   Current: {t['step8_cloud_prepare']:.2f} ms | Avg ({self.window_size}f): {avg['step8_cloud_prepare']:.2f} ms\n"
                f"    [8a.1 Lock/Read]              Current: {t['step8_lock']:.2f} ms | Avg ({self.window_size}f): {avg['step8_lock']:.2f} ms\n"
                f"    [8a.2 TF Lookup]              Current: {t['step8_tf']:.2f} ms | Avg ({self.window_size}f): {avg['step8_tf']:.2f} ms\n"
                f"    [8a.3 Projection/Serialize]   Current: {t['step8_math']:.2f} ms | Avg ({self.window_size}f): {avg['step8_math']:.2f} ms\n"
                f"  [Step 8b: DDS Publish]         Current: {t['step8_cloud_publish']:.2f} ms | Avg ({self.window_size}f): {avg['step8_cloud_publish']:.2f} ms\n"
                f"  [Step 9: Draw & Publish]        Current: {t['step9_draw_publish']:.2f} ms | Avg ({self.window_size}f): {avg['step9_draw_publish']:.2f} ms\n"
                f"  [Cloud Points]                  Avg ({self.window_size} clouds): {point_count_avg}\n"
                f"======================================================================"
            )

    def publish_pointcloud(
            self, blue_points, turquoise_points, background_points,
            width, height, header):
        """Back-project the pixel classes directly into a 2D BEV cloud."""
        prepare_started_at = time.perf_counter()
        lock_ms = 0.0
        tf_ms = 0.0
        math_ms = 0.0

        def skipped_result():
            prepare_ms = (
                time.perf_counter() - prepare_started_at) * 1000.0
            return None, prepare_ms, 0.0, lock_ms, tf_ms, math_ms

        lock_started_at = time.perf_counter()
        with self.sensor_lock:
            depth = self.depth_image
            intrinsics = self.camera_intrinsics
        lock_ms = (time.perf_counter() - lock_started_at) * 1000.0

        if depth is None or intrinsics is None:
            self.get_logger().warning(
                'Waiting for depth and RGB CameraInfo',
                throttle_duration_sec=2.0,
            )
            return skipped_result()
        if depth.shape[:2] != (height, width):
            self.get_logger().warning(
                f'Cannot align label map {width}x{height} with depth '
                f'{depth.shape[1]}x{depth.shape[0]}',
                throttle_duration_sec=2.0,
            )
            return skipped_result()

        fx_raw, fy_raw, cx_raw, cy_raw, info_width, info_height = intrinsics
        if fx_raw <= 0.0 or fy_raw <= 0.0 or not info_width or not info_height:
            self.get_logger().error('Invalid RGB CameraInfo intrinsics')
            return skipped_result()

        math_started_at = time.perf_counter()
        point_groups = (
            (blue_points, self.LABEL_BLUE),
            (turquoise_points, self.LABEL_TURQUOISE),
            (background_points, self.LABEL_BACKGROUND),
        )
        nonempty_groups = [
            (points, label) for points, label in point_groups if len(points)
        ]
        if nonempty_groups:
            pixels = np.concatenate(
                [group[0] for group in nonempty_groups], axis=0)
            class_ids = np.concatenate([
                np.full(len(group[0]), group[1], dtype=np.uint8)
                for group in nonempty_groups
            ])
        else:
            pixels = np.empty((0, 2), dtype=np.int32)
            class_ids = np.empty(0, dtype=np.uint8)

        rows = pixels[:, 0]
        cols = pixels[:, 1]
        z = depth[rows, cols]
        valid = (
            np.isfinite(z)
            & (z >= self.cloud_min_depth)
            & (z <= self.cloud_max_depth)
        )
        rows = rows[valid]
        cols = cols[valid]
        z = z[valid].astype(np.float32, copy=False)
        class_ids = class_ids[valid]

        crop_start = int(info_height * self.input_crop_y_min)
        crop_height = info_height - crop_start
        source_cols = (
            (cols.astype(np.float32) + 0.5) * info_width / width - 0.5)
        source_rows = (
            crop_start
            + (rows.astype(np.float32) + 0.5) * crop_height / height
            - 0.5
        )
        math_ms += (time.perf_counter() - math_started_at) * 1000.0

        if not header.frame_id:
            self.get_logger().warning(
                'Cannot create BEV points with an empty input frame_id',
                throttle_duration_sec=2.0,
            )
            return skipped_result()

        tf_started_at = time.perf_counter()
        try:
            transform = self.tf_buffer.lookup_transform(
                self.bev_frame,
                header.frame_id,
                Time.from_msg(header.stamp),
            )
            rotation = self.quaternion_to_rotation(
                transform.transform.rotation)
        except (TransformException, TypeError, ValueError) as error:
            tf_ms = (time.perf_counter() - tf_started_at) * 1000.0
            self.get_logger().warning(
                f'Waiting for TF {header.frame_id} -> '
                f'{self.bev_frame}: {error}',
                throttle_duration_sec=2.0,
            )
            return skipped_result()
        tf_ms = (time.perf_counter() - tf_started_at) * 1000.0

        math_started_at = time.perf_counter()
        camera_points = np.column_stack((
            (source_cols - cx_raw) * z / fx_raw,
            (source_rows - cy_raw) * z / fy_raw,
            z,
        )).astype(np.float32, copy=False)
        bev_points = camera_points @ rotation[:2, :].T
        bev_points += np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
        ], dtype=np.float32)

        cloud_points = np.empty(len(rows), dtype=self.CLOUD_DTYPE)
        cloud_points['x'] = bev_points[:, 0]
        cloud_points['y'] = bev_points[:, 1]
        cloud_points['z'] = 0.0
        cloud_points['class_id'] = class_ids

        cloud = PointCloud2()
        cloud.header.stamp = header.stamp
        cloud.header.frame_id = self.bev_frame
        cloud.height = 1
        cloud.width = len(cloud_points)
        cloud.fields = self.CLOUD_FIELDS
        cloud.is_bigendian = False
        cloud.point_step = self.CLOUD_DTYPE.itemsize
        cloud.row_step = cloud.point_step * cloud.width
        cloud.data = cloud_points.tobytes()
        cloud.is_dense = True
        math_ms += (time.perf_counter() - math_started_at) * 1000.0
        prepare_ms = (
            time.perf_counter() - prepare_started_at) * 1000.0
        publish_started_at = time.perf_counter()
        self.pointcloud_pub.publish(cloud)
        publish_ms = (
            time.perf_counter() - publish_started_at) * 1000.0
        return (
            len(cloud_points), prepare_ms, publish_ms,
            lock_ms, tf_ms, math_ms,
        )

    def encode_debug_image(self, frame, header):
        """Encode a BGR debug frame as JPEG using libjpeg-turbo."""
        msg = CompressedImage()
        msg.header = header
        msg.format = 'bgr8; jpeg compressed bgr8'
        msg.data = self.jpeg.encode(
            frame,
            quality=self.debug_jpeg_quality,
            pixel_format=TJPF_BGR,
        )
        return msg

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
