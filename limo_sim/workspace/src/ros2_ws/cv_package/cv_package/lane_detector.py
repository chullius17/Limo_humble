#!/usr/bin/env python3
"""HSV-based lane/curb color segmentation node.

Optimized ingress: the camera frame is never fully color-converted or fully
thresholded. A zero-copy view is taken over the raw message buffer, only the
configured vertical ROI band is resized/converted/thresholded, and the
label-map composition uses ``cv2.copyTo`` instead of boolean fancy indexing.
"""
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import CompressedImage, Image
import cv2
from cv_bridge import CvBridge
import numpy as np
from turbojpeg import TJPF_BGR, TurboJPEG

import time
from collections import deque
import threading
import queue
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class ColorLaneDetector(Node):

    LABEL_INVALID = np.uint8(0)
    LABEL_BLUE = np.uint8(1)
    LABEL_TURQUOISE = np.uint8(2)
    LABEL_BACKGROUND = np.uint8(3)

    # Encodings the zero-copy ingress path understands directly, mapped to
    # the cv2 code that converts that native channel order to HSV.
    _FAST_HSV_CODE = {
        'bgr8': cv2.COLOR_BGR2HSV,
        'rgb8': cv2.COLOR_RGB2HSV,
    }
    # Same encodings, mapped to the code that brings them to BGR for the
    # (debug-only) visual overlay, or None when already BGR.
    _FAST_TO_BGR_CODE = {
        'bgr8': None,
        'rgb8': cv2.COLOR_RGB2BGR,
    }

    def __init__(self):
        super().__init__('color_lane_detector')

        # Accept the camera's best-effort stream and keep only its latest frame.
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Reliable output remains compatible with image_view while limiting backlog.
        output_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ROS 2 Publishers
        self.label_pub = self.create_publisher(
            Image, 'limo/cv_package/detection/lane_labels/raw', output_qos
        )
        self.debug_mask_pub = self.create_publisher(
            CompressedImage,
            'limo/cv_package/detection/lane_masks/compressed',
            output_qos,
        )
        self.debug_overlay_pub = self.create_publisher(
            CompressedImage,
            'limo/cv_package/detection/lane_overlay/compressed',
            output_qos,
        )

        self.bridge = CvBridge()
        self.jpeg = TurboJPEG()

        self.declare_parameter('debug_jpeg_quality', 85)
        self.debug_jpeg_quality = int(
            self.get_parameter('debug_jpeg_quality').value)
        if not 1 <= self.debug_jpeg_quality <= 100:
            raise ValueError('debug_jpeg_quality must be in [1, 100]')

        # Telemetry control parameters
        self.declare_parameter('enable_telemetry', True)
        self.debug_telemetry = self.get_parameter('enable_telemetry').value
        self.frame_counter = 0

        # Number of OpenCV worker threads. At this node's resolutions the
        # per-op parallel fan-out/join overhead exceeds the work itself, so
        # the default keeps OpenCV single-threaded rather than competing
        # with the executor and the daemon worker threads for CPU cores.
        self.declare_parameter('opencv_num_threads', 1)
        cv2.setNumThreads(int(self.get_parameter('opencv_num_threads').value))

        # How often (in frames) to refresh the cached subscriber-count check
        # for the debug topics, instead of querying it on every frame.
        self.declare_parameter('debug_probe_interval_frames', 30)
        self.debug_probe_interval_frames = max(
            1, int(self.get_parameter('debug_probe_interval_frames').value))
        self._publish_debug_mask = False
        self._publish_overlay = False

        self.add_on_set_parameters_callback(self._on_set_parameters)

        # Lower half of the frame, published without vertical stretching.
        self.output_size = (320, 120)
        self.last_logged_resolution = None
        self.last_logged_ingress_path = None

        # HSV Threshold parameters for Yellow and Black colors
        self.yellow_lower = np.array([15, 80, 80], dtype=np.uint8)
        self.yellow_upper = np.array([35, 255, 255], dtype=np.uint8)

        self.black_lower = np.array([0, 0, 0], dtype=np.uint8)
        self.black_upper = np.array([180, 255, 150], dtype=np.uint8)

        # Subscriber to Limo's camera
        self.declare_parameter('rgb_topic', '/rgb/image_raw')
        self.camera_topic = self.get_parameter('rgb_topic').value
        self.rgb_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            camera_qos
        )

        # ROI parameters for cropping
        self.declare_parameter('roi_y_min', 0.1)
        self.declare_parameter('roi_y_max', 1.0)
        self.roi_y_min = self.get_parameter('roi_y_min').value
        self.roi_y_max = self.get_parameter('roi_y_max').value
        if not 0.0 <= self.roi_y_min < self.roi_y_max <= 1.0:
            raise ValueError(
                'roi_y_min and roi_y_max must define a range in [0, 1]')

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

        # Frame accounting: how many frames actually arrive, and how many
        # are discarded by the drop-oldest queues below. Together with the
        # sliding-window stats this tells apart "CPU-bound" (drops, low
        # measured FPS) from "input-starved" (no drops, FPS == camera rate).
        self.frames_received = 0
        self.frames_dropped_ingress = 0
        self.frames_dropped_publish = 0
        self._ingress_arrival_times = deque(maxlen=self.window_size + 1)
        self._last_report_wall_time = time.perf_counter()
        self._last_report_frame_counter = 0

        # Reusable working buffers for the ROI band, allocated on first use
        # and only reallocated when the input/ROI geometry changes. These
        # are private to the worker thread and never escape it, so reusing
        # them across frames is safe. Arrays that are handed to the
        # publisher thread through pub_queue (labels, mask_overlay,
        # overlay_image) are intentionally NOT part of this cache: see the
        # aliasing note in `_process_frame`.
        self._geometry_key = None
        self._band_bounds = None  # (y_min, y_max, src_y0, src_y1, exact)
        self._buf_band = None
        self._buf_hsv = None
        self._buf_v = None
        self._buf_black = None
        self._buf_yellow = None
        self._const_blue = None
        self._const_turquoise = None
        self._buf_low_res_full = None

        # Runtime self-check: confirm this OpenCV build writes cv2.copyTo
        # results in place into a contiguous row-slice view. If it does not
        # (returns a freshly allocated array instead), fall back to the
        # numpy equivalent, which is slower but still correct.
        self._use_copyto = self._probe_copyto_in_place()
        self.get_logger().info(
            f'cv2 {cv2.__version__}: copyTo in-place composition '
            f'{"enabled" if self._use_copyto else "unavailable, using numpy fallback"}.'
        )

        # Threading queues
        self.processing_queue = queue.Queue(maxsize=1)
        self.pub_queue = queue.Queue(maxsize=1)

        # Start async worker threads
        self.worker_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.pub_thread = threading.Thread(target=self._publish_worker, daemon=True)
        self.worker_thread.start()
        self.pub_thread.start()

        self.get_logger().info("HSV Color Segmentation Node initialized successfully.")

    def _probe_copyto_in_place(self):
        """Check that cv2.copyTo writes through a contiguous row-slice view.

        Some OpenCV builds silently allocate a new array instead of writing
        into `dst` when shapes/strides do not match their fast path. This
        probe catches that case once at startup rather than trusting it.
        """
        try:
            probe = np.zeros((4, 4), dtype=np.uint8)
            view = probe[1:3, :]
            mask = np.zeros((2, 4), dtype=np.uint8)
            mask[0, 0] = 255
            result = cv2.copyTo(np.full((2, 4), 7, dtype=np.uint8), mask, view)
            return (
                result is view
                and probe[1, 0] == 7
                and probe[0, 0] == 0
                and probe[3, 0] == 0
            )
        except Exception:
            return False

    def _on_set_parameters(self, params):
        """Apply runtime parameter changes without a per-frame lookup."""
        for param in params:
            if param.name == 'enable_telemetry':
                self.debug_telemetry = bool(param.value)
            elif param.name == 'debug_jpeg_quality':
                quality = int(param.value)
                if not 1 <= quality <= 100:
                    return SetParametersResult(
                        successful=False,
                        reason='debug_jpeg_quality must be in [1, 100]')
                self.debug_jpeg_quality = quality
        return SetParametersResult(successful=True)

    def image_callback(self, msg):
        """Producer Callback: Non-blocking enqueue to prevent ROS executor queue delays."""
        self.frames_received += 1

        if self.debug_telemetry:
            t_now_ros = self.get_clock().now().nanoseconds / 1e9
            t_msg_ros = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            transport_delay = (t_now_ros - t_msg_ros) * 1000.0
            self.telemetry_stats['0_transport_delay'].append(transport_delay)
            self._ingress_arrival_times.append(time.perf_counter())

        # Drop older frames if worker thread is occupied to minimize ingress latency
        if self.processing_queue.full():
            try:
                self.processing_queue.get_nowait()
                self.frames_dropped_ingress += 1
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
            # Total pipeline time is measured from here, including ingress
            # conversion below, so the reported FPS reflects real throughput.
            t_start = time_exiting if self.debug_telemetry else 0.0

            if self.debug_telemetry:
                queue_delay = (time_exiting - time_entering) * 1000.0
                self.telemetry_stats['2_queue_waiting_time'].append(queue_delay)

            t_convert_start = time.perf_counter() if self.debug_telemetry else 0.0
            try:
                cv_image, hsv_code, to_bgr_code = self._view_source(msg)
            except Exception as e:
                self.get_logger().error(f"Frame ingress failed: {str(e)}")
                continue

            if self.debug_telemetry:
                convert_time = (time.perf_counter() - t_convert_start) * 1000.0
                self.telemetry_stats['1_convert_time'].append(convert_time)

            self._process_frame(cv_image, hsv_code, to_bgr_code, msg.header, t_start)

    def _view_source(self, msg):
        """Return (frame_view, hsv_code, to_bgr_code) without a full-frame color conversion.

        For 'bgr8'/'rgb8' this is a zero-copy view over msg.data honoring
        msg.step (row padding), left in its native channel order so the
        colour work happens only on the small downscaled crop later. Any
        other encoding falls back to CvBridge on the full frame.
        """
        hsv_code = self._FAST_HSV_CODE.get(msg.encoding)
        if hsv_code is None:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self._log_ingress_path('cvbridge-fallback', msg)
            return cv_image, cv2.COLOR_BGR2HSV, None

        width, height, step = msg.width, msg.height, msg.step
        row_bytes = width * 3
        flat = np.frombuffer(msg.data, dtype=np.uint8)
        if step == row_bytes:
            view = flat.reshape(height, width, 3)
        else:
            # Row padding present: keep the row stride, drop the pad bytes.
            # This stays a zero-copy view over `flat` (a strided reshape of
            # a contiguous-per-row slice does not require a copy).
            view = flat.reshape(height, step)[:, :row_bytes].reshape(height, width, 3)

        self._log_ingress_path('zero-copy', msg)
        return view, hsv_code, self._FAST_TO_BGR_CODE[msg.encoding]

    def _log_ingress_path(self, path, msg):
        if path != self.last_logged_ingress_path:
            self.get_logger().info(
                f"Ingress path: {path} (encoding='{msg.encoding}', "
                f"step={msg.step}, padded={msg.step != msg.width * 3})"
            )
            self.last_logged_ingress_path = path

    def _ensure_buffers(self, input_width, input_height):
        """(Re)allocate ROI-band working buffers when the geometry changes.

        These buffers are private to the worker thread: they are fully
        consumed within a single call to `_process_frame` and never placed
        on `pub_queue`, so reusing them across frames is safe.
        """
        out_w, out_h = self.output_size
        key = (input_width, input_height, out_w, out_h,
               self.roi_y_min, self.roi_y_max)
        if key == self._geometry_key:
            return

        crop_y = input_height // 2
        crop_height = input_height - crop_y
        y_min = int(out_h * self.roi_y_min)
        y_max = int(out_h * self.roi_y_max)
        band_h = y_max - y_min

        # Under INTER_AREA, output row j is built strictly from source rows
        # [j*sy, (j+1)*sy). Resizing only the source rows that map onto the
        # ROI band therefore reproduces exactly what "resize full crop, then
        # slice" would produce, provided the band edges land on integer
        # source rows. Verify that condition instead of assuming it.
        scale_y = crop_height / out_h
        src_y0 = crop_y + y_min * scale_y
        src_y1 = crop_y + y_max * scale_y
        exact = (
            abs(src_y0 - round(src_y0)) < 1e-6
            and abs(src_y1 - round(src_y1)) < 1e-6
        )
        src_y0_i, src_y1_i = int(round(src_y0)), int(round(src_y1))

        self._band_bounds = (y_min, y_max, crop_y, src_y0_i, src_y1_i, exact)
        self._buf_band = np.empty((band_h, out_w, 3), dtype=np.uint8)
        self._buf_hsv = np.empty((band_h, out_w, 3), dtype=np.uint8)
        self._buf_v = np.empty((band_h, out_w), dtype=np.uint8)
        self._buf_black = np.empty((band_h, out_w), dtype=np.uint8)
        self._buf_yellow = np.empty((band_h, out_w), dtype=np.uint8)
        self._const_blue = np.full((band_h, out_w), self.LABEL_BLUE, dtype=np.uint8)
        self._const_turquoise = np.full(
            (band_h, out_w), self.LABEL_TURQUOISE, dtype=np.uint8)
        self._geometry_key = key

        self.get_logger().info(
            f'Image resolution: input={input_width}x{input_height}, '
            f'output={out_w}x{out_h}, ROI band rows [{y_min}:{y_max}] '
            f'(exact source alignment: {exact})'
        )

    def _resize_band(self, cv_image):
        """Resize just the ROI band into `self._buf_band` and return it.

        Falls back to resizing the full lower-half crop and slicing the
        band out of it when the ROI edges do not land on integer source
        rows (see `_ensure_buffers`); this is always correct, just not the
        fast path.
        """
        y_min, y_max, crop_y, src_y0, src_y1, exact = self._band_bounds
        out_w, out_h = self.output_size
        if exact:
            cv2.resize(
                cv_image[src_y0:src_y1, :],
                (out_w, y_max - y_min),
                dst=self._buf_band,
                interpolation=cv2.INTER_AREA,
            )
        else:
            full = cv2.resize(
                cv_image[crop_y:, :],
                (out_w, out_h),
                interpolation=cv2.INTER_AREA,
            )
            self._buf_band[:, :, :] = full[y_min:y_max, :]
        return self._buf_band

    def _compose_label_band(self, band_view, black_mask, yellow_mask):
        """Write BLUE then TURQUOISE into `band_view` according to the masks.

        `band_view` must already hold LABEL_BACKGROUND everywhere; overlap
        precedence (TURQUOISE overwrites BLUE) is preserved by ordering.
        """
        if self._use_copyto:
            cv2.copyTo(self._const_blue, black_mask, band_view)
            cv2.copyTo(self._const_turquoise, yellow_mask, band_view)
        else:
            np.copyto(band_view, self.LABEL_BLUE, where=(black_mask != 0))
            np.copyto(band_view, self.LABEL_TURQUOISE, where=(yellow_mask != 0))

    def _process_frame(self, cv_image, hsv_code, to_bgr_code, header, t_start):
        """Process a frame using HSV thresholds restricted to the ROI band."""
        input_height, input_width = cv_image.shape[:2]
        self._ensure_buffers(input_width, input_height)
        y_min, y_max, crop_y, _, _, _ = self._band_bounds
        out_w, out_h = self.output_size

        # Step 1+2: resize only the ROI band, then convert that small crop
        # to HSV (instead of converting/resizing the full camera frame).
        band_bgr_native = self._resize_band(cv_image)
        cv2.cvtColor(band_bgr_native, hsv_code, dst=self._buf_hsv)

        # Step 3: yellow needs all three HSV channels; black is exactly
        # V <= 150 (H spans [0,179] and S spans [0,255] in 8-bit HSV, so
        # both bounds of the original 3-channel inRange are tautologies).
        cv2.inRange(self._buf_hsv, self.yellow_lower, self.yellow_upper,
                    dst=self._buf_yellow)
        cv2.extractChannel(self._buf_hsv, 2, dst=self._buf_v)
        cv2.threshold(self._buf_v, 150, 255, cv2.THRESH_BINARY_INV,
                      dst=self._buf_black)

        if self.debug_telemetry:
            t_hsv_end = time.perf_counter()
            self.telemetry_stats['3_hsv_segmentation'].append(
                (t_hsv_end - t_start) * 1000.0)

        # Step 4/5: compose the label map. `labels` is allocated fresh every
        # frame (~38 KB, ~1 microsecond) rather than reused: it is handed to
        # the publisher thread through pub_queue, which can still be
        # serializing frame N while this frame N+1 is being composed here.
        # Reusing a single buffer across that boundary would be a silent
        # data race. Any array placed on pub_queue must be freshly
        # allocated in this frame; do not turn these into cached buffers.
        t_canvas_start = time.perf_counter() if self.debug_telemetry else 0.0
        labels = np.zeros((out_h, out_w), dtype=np.uint8)
        band_view = labels[y_min:y_max, :]
        band_view.fill(self.LABEL_BACKGROUND)
        self._compose_label_band(band_view, self._buf_black, self._buf_yellow)

        if self.frame_counter % self.debug_probe_interval_frames == 0:
            self._publish_debug_mask = self.debug_mask_pub.get_subscription_count() > 0
            self._publish_overlay = self.debug_overlay_pub.get_subscription_count() > 0
        publish_debug_mask = self._publish_debug_mask
        publish_overlay = self._publish_overlay

        mask_overlay = None
        overlay_image = None
        if publish_debug_mask or publish_overlay:
            # Debug-only path: resize the full crop (not just the band) and
            # bring it to BGR so the overlay stays visually identical to the
            # non-optimized version, at the cost of an extra resize/convert
            # that nobody pays unless something is actually subscribed.
            low_res_full = cv2.resize(
                cv_image[crop_y:, :], (out_w, out_h), interpolation=cv2.INTER_AREA)
            if to_bgr_code is not None:
                low_res_full = cv2.cvtColor(low_res_full, to_bgr_code)
            mask_overlay = np.zeros_like(low_res_full)
            mask_overlay[labels == self.LABEL_BLUE] = (255, 0, 0)
            mask_overlay[labels == self.LABEL_TURQUOISE] = (0, 255, 0)
            if publish_overlay:
                overlay_image = cv2.addWeighted(
                    low_res_full, 0.7, mask_overlay, 0.5, 0)

        if self.debug_telemetry:
            t_canvas_end = time.perf_counter()
            self.telemetry_stats['9_post_canvas'].append(
                (t_canvas_end - t_canvas_start) * 1000.0)

        # Step 6: Enqueue frames for publisher worker thread
        t_pub_start = time.perf_counter() if self.debug_telemetry else 0.0
        if self.pub_queue.full():
            try:
                self.pub_queue.get_nowait()
                self.frames_dropped_publish += 1
            except queue.Empty:
                pass
        try:
            self.pub_queue.put_nowait(
                (
                    labels,
                    mask_overlay,
                    overlay_image,
                    header,
                    publish_debug_mask,
                    publish_overlay,
                )
            )
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

            (
                labels,
                mask_overlay,
                overlay_image,
                header,
                publish_debug_mask,
                publish_overlay,
            ) = item
            t_pub_start = time.perf_counter() if self.debug_telemetry else 0.0

            try:
                label_msg = self.bridge.cv2_to_imgmsg(
                    labels,
                    encoding='mono8',
                )
                label_msg.header = header
                self.label_pub.publish(label_msg)

                if publish_debug_mask:
                    self.debug_mask_pub.publish(
                        self._encode_debug_image(mask_overlay, header))

                if publish_overlay:
                    self.debug_overlay_pub.publish(
                        self._encode_debug_image(overlay_image, header))

            except Exception as e:
                self.get_logger().error(f"Publishing failed: {str(e)}")

            if self.debug_telemetry:
                dt_async_pub = (time.perf_counter() - t_pub_start) * 1000.0
                self.telemetry_stats['async_encode_publish'].append(dt_async_pub)

                t_now_final = self.get_clock().now().nanoseconds / 1e9
                stamp = header.stamp
                t_msg_final = stamp.sec + stamp.nanosec * 1e-9
                final_age = (t_now_final - t_msg_final) * 1000.0
                self.telemetry_stats['12_msg_age_final_publish'].append(final_age)

    def _encode_debug_image(self, image, header):
        """Encode a BGR debug frame as JPEG using libjpeg-turbo."""
        msg = CompressedImage()
        msg.header = header
        msg.format = 'bgr8; jpeg compressed bgr8'
        msg.data = self.jpeg.encode(
            image,
            quality=self.debug_jpeg_quality,
            pixel_format=TJPF_BGR,
        )
        return msg

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

        # Measured worker throughput over the wall-clock time between
        # reports, and the actual camera input rate, so a low internal FPS
        # can be told apart from simply being input-starved.
        now = time.perf_counter()
        report_span = now - self._last_report_wall_time
        frames_since_report = self.frame_counter - self._last_report_frame_counter
        measured_fps = frames_since_report / report_span if report_span > 0 else 0.0
        self._last_report_wall_time = now
        self._last_report_frame_counter = self.frame_counter

        input_hz = 0.0
        if len(self._ingress_arrival_times) >= 2:
            span = self._ingress_arrival_times[-1] - self._ingress_arrival_times[0]
            if span > 0:
                input_hz = (len(self._ingress_arrival_times) - 1) / span

        self.get_logger().info(
            f"\n"
            f"====== COLOR SEGMENTATION PERFORMANCE REPORT (AVG {self.window_size} frames) ======\n"
            f"  Frames Processed: {self.frame_counter} / Received: {self.frames_received}\n"
            f"  Dropped (ingress/publish): "
            f"{self.frames_dropped_ingress}/{self.frames_dropped_publish}\n"
            f"-----------------------------------------\n"
            f"[RATE]\n"
            f" Camera input rate:                {input_hz:.1f} Hz\n"
            f" Measured worker throughput:        {measured_fps:.1f} FPS\n"
            f" Estimated FPS (1000/total):        {fps:.1f} FPS\n"
            f"-----------------------------------------\n"
            f"[TIMESTAMP LATENCY]\n"
            f" Camera -> Node ingress:          {avg_transport:.2f} ms\n"
            f" Frame age final publish:          {avg_final_age:.2f} ms\n"
            f"-----------------------------------------\n"
            f"[EXECUTION BREAKDOWN]\n"
            f" Queue Waiting Delay:              {avg_queue_wait:.2f} ms\n"
            f" Frame Ingress (view/CvBridge):    {avg_convert:.2f} ms\n"
            f" HSV Color Thresholding (ROI band): {avg_hsv:.2f} ms\n"
            f" Label-map Encoding:               {avg_canvas:.2f} ms\n"
            f" ROS Publish Enqueue:              {avg_pub_enqueue:.2f} ms\n"
            f" Async Publish:                    {avg_async_pub:.2f} ms\n"
            f"-----------------------------------------\n"
            f" TOTAL PIPELINE TIME:              {avg_total:.2f} ms\n"
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
