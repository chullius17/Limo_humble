#!/usr/bin/env python3
"""Adaptive mean segmentation with an absolute fallback in low-variance regions."""

from collections import deque
import math
import queue
import threading
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from turbojpeg import TJPF_BGR, TurboJPEG


class BinaryLaneDetector(Node):
    """Label dark road pixels using mean(neighborhood) - C in the resized ROI."""

    LABEL_INVALID = np.uint8(0)
    LABEL_BLUE = np.uint8(1)
    LABEL_BACKGROUND = np.uint8(3)

    _ENCODINGS = {
        'bgr8': (3, cv2.COLOR_BGR2GRAY, None),
        'rgb8': (3, cv2.COLOR_RGB2GRAY, cv2.COLOR_RGB2BGR),
        'mono8': (1, None, cv2.COLOR_GRAY2BGR),
    }
    _TIMINGS = (
        'transport_delay', 'queue_wait', 'ingress', 'adaptive_threshold',
        'canvas', 'publish_enqueue', 'total_pipeline', 'async_encode_publish',
        'final_message_age',
    )

    def __init__(self):
        super().__init__('binary_lane_detector')
        readonly = ParameterDescriptor(read_only=True)
        for name, default in (
            ('rgb_topic', '/rgb/image_raw'),
            ('roi_y_min', 0.1), ('roi_y_max', 1.0),
            ('opencv_num_threads', 1), ('debug_probe_interval_frames', 30),
        ):
            self.declare_parameter(name, default, readonly)
        for name, default in (
            ('adaptive_block_size', 61), ('adaptive_c', 5.0),
            ('enable_variance_fallback', True),
            ('fallback_variance_threshold', 25.0), ('fallback_gray_threshold', 150),
            ('debug_jpeg_quality', 85), ('enable_telemetry', True),
        ):
            self.declare_parameter(name, default)
        self._settings = {
            name: self.get_parameter(name).value for name in (
                'adaptive_block_size', 'adaptive_c', 'debug_jpeg_quality',
                'enable_telemetry', 'enable_variance_fallback',
                'fallback_variance_threshold', 'fallback_gray_threshold',
            )
        }
        self._validate_settings(self._settings)
        self.output_size = (320, 120)
        self.roi_y_min = self.get_parameter('roi_y_min').value
        self.roi_y_max = self.get_parameter('roi_y_max').value
        if not 0.0 <= self.roi_y_min < self.roi_y_max <= 1.0:
            raise ValueError('roi_y_min and roi_y_max must define a range in [0, 1]')
        self._y_min = int(self.output_size[1] * self.roi_y_min)
        self._y_max = int(self.output_size[1] * self.roi_y_max)
        if self._y_min == self._y_max:
            raise ValueError('ROI must contain at least one output row')
        threads = self.get_parameter('opencv_num_threads').value
        self.debug_probe_interval_frames = self.get_parameter(
            'debug_probe_interval_frames').value
        if threads < 1 or self.debug_probe_interval_frames < 1:
            raise ValueError('Thread count and debug probe interval must be positive')
        cv2.setNumThreads(threads)

        self.bridge = CvBridge()
        self.jpeg = TurboJPEG()
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        output_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.label_pub = self.create_publisher(
            Image, 'limo/cv_package/detection/lane_binary_labels/raw', output_qos)
        self.debug_mask_pub = self.create_publisher(
            CompressedImage, 'limo/cv_package/detection/lane_binary_masks/compressed',
            output_qos)
        self.debug_overlay_pub = self.create_publisher(
            CompressedImage, 'limo/cv_package/detection/lane_binary_overlay/compressed',
            output_qos)
        self.debug_fallback_overlay_pub = self.create_publisher(
            CompressedImage,
            'limo/cv_package/detection/lane_binary_fallback_overlay/compressed',
            output_qos)
        self._publish_fallback_overlay = False
        self._publish_debug_mask = False
        self._publish_overlay = False
        self._geometry_key = None
        self._last_ingress_path = None
        self.window_size = 30
        self.telemetry_stats = {
            name: deque(maxlen=self.window_size)
            for name in self._TIMINGS + ('fallback_percent',)}
        self._arrival_times = deque(maxlen=self.window_size + 1)
        self.frames_received = 0
        self.frame_counter = 0
        self.frames_dropped_ingress = 0
        self.frames_dropped_publish = 0
        self._last_report_time = time.perf_counter()
        self._last_report_count = 0
        self.processing_queue = queue.Queue(maxsize=1)
        self.pub_queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self.rgb_sub = self.create_subscription(
            Image, self.get_parameter('rgb_topic').value,
            self.image_callback, camera_qos)
        self.worker_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.pub_thread = threading.Thread(target=self._publish_worker, daemon=True)
        self.worker_thread.start()
        self.pub_thread.start()
        self.get_logger().info('Adaptive mean binary lane detector initialized.')

    @staticmethod
    def _validate_settings(settings):
        block_size = settings['adaptive_block_size']
        if (isinstance(block_size, bool) or not isinstance(block_size, int)
                or block_size < 3 or block_size % 2 == 0):
            raise ValueError('adaptive_block_size must be an odd integer >= 3')
        c = settings['adaptive_c']
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c):
            raise ValueError('adaptive_c must be a finite number')
        quality = settings['debug_jpeg_quality']
        if (isinstance(quality, bool) or not isinstance(quality, int)
                or not 1 <= quality <= 100):
            raise ValueError('debug_jpeg_quality must be an integer in [1, 100]')
        for name in ('enable_telemetry', 'enable_variance_fallback'):
            if not isinstance(settings[name], bool):
                raise ValueError(f'{name} must be a boolean')
        variance = settings['fallback_variance_threshold']
        if (isinstance(variance, bool) or not isinstance(variance, (int, float))
                or not math.isfinite(variance) or variance < 0):
            raise ValueError('fallback_variance_threshold must be finite and >= 0')
        gray = settings['fallback_gray_threshold']
        if isinstance(gray, bool) or not isinstance(gray, int) or not 0 <= gray <= 255:
            raise ValueError('fallback_gray_threshold must be an integer in [0, 255]')

    def _on_set_parameters(self, params):
        # Validate the whole update before replacing the snapshot used by workers.
        settings = self._settings.copy()
        for param in params:
            if param.name in settings:
                settings[param.name] = param.value
        try:
            self._validate_settings(settings)
        except ValueError as exc:
            return SetParametersResult(successful=False, reason=str(exc))
        self._settings = settings
        return SetParametersResult(successful=True)

    @staticmethod
    def _put_latest(target, item):
        """Enqueue without blocking; return how many frames were discarded."""
        dropped = 0
        try:
            target.put_nowait(item)
            return dropped
        except queue.Full:
            pass
        try:
            target.get_nowait()
            dropped += 1
        except queue.Empty:
            pass
        try:
            target.put_nowait(item)
        except queue.Full:
            dropped += 1
        return dropped

    def _message_age_ms(self, header):
        stamp = header.stamp
        return (self.get_clock().now().nanoseconds
                - (stamp.sec * 1_000_000_000 + stamp.nanosec)) * 1e-6

    def image_callback(self, msg):
        self.frames_received += 1
        arrival = None
        if self._settings['enable_telemetry']:
            arrival = time.perf_counter()
            self._arrival_times.append(arrival)
            self.telemetry_stats['transport_delay'].append(self._message_age_ms(msg.header))
        self.frames_dropped_ingress += self._put_latest(
            self.processing_queue, (msg, arrival))

    def _view_source(self, msg):
        """View common camera encodings without a copy, including padded rows."""
        encoding = self._ENCODINGS.get(msg.encoding)
        if encoding is None:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray_code, bgr_code = cv2.COLOR_BGR2GRAY, None
            path = 'CvBridge fallback'
        else:
            channels, gray_code, bgr_code = encoding
            if (msg.width < 1 or msg.height < 1 or msg.step < msg.width * channels
                    or len(msg.data) < msg.height * msg.step):
                raise ValueError('Invalid image dimensions, step or data length')
            shape = (msg.height, msg.width)
            strides = (msg.step, channels)
            if channels > 1:
                shape += (channels,)
                strides += (1,)
            frame = np.ndarray(shape, np.uint8, buffer=msg.data, strides=strides)
            path = 'zero-copy'
        ingress = (path, msg.encoding, msg.step)
        if ingress != self._last_ingress_path:
            self.get_logger().info(
                f'Ingress: {path}, encoding={msg.encoding}, step={msg.step}')
            self._last_ingress_path = ingress
        return frame, gray_code, bgr_code

    def _resize_band(self, frame):
        """Reuse buffers; resize only the ROI when source row alignment allows it."""
        height, width = frame.shape[:2]
        key = frame.shape
        out_w, out_h = self.output_size
        if key != self._geometry_key:
            crop_y = height // 2
            scale = (height - crop_y) / out_h
            src_y0 = crop_y + self._y_min * scale
            src_y1 = crop_y + self._y_max * scale
            exact = (scale >= 1.0 and width >= out_w
                     and abs(src_y0 - round(src_y0)) < 1e-6
                     and abs(src_y1 - round(src_y1)) < 1e-6)
            self._band_bounds = (crop_y, int(round(src_y0)), int(round(src_y1)), exact)
            band_shape = (self._y_max - self._y_min, out_w)
            self._buf_band = np.empty(band_shape + frame.shape[2:], dtype=np.uint8)
            self._buf_full = np.empty((out_h, out_w) + frame.shape[2:], dtype=np.uint8)
            self._buf_gray = np.empty(band_shape, dtype=np.uint8)
            self._buf_road = np.empty(band_shape, dtype=np.uint8)
            self._buf_mean = np.empty(band_shape, dtype=np.float32)
            self._buf_variance = np.empty(band_shape, dtype=np.float32)
            self._buf_square = np.empty(band_shape, dtype=np.float32)
            self._buf_mean_u8 = np.empty(band_shape, dtype=np.uint8)
            self._buf_delta = np.empty(band_shape, dtype=np.int16)
            self._buf_low_variance = np.empty(band_shape, dtype=np.uint8)
            self._buf_absolute = np.empty(band_shape, dtype=np.uint8)
            self._geometry_key = key
            self.get_logger().info(
                f'Image resolution: input={width}x{height}, output={out_w}x{out_h}, '
                f'ROI rows [{self._y_min}:{self._y_max}], direct band resize={exact}')
        crop_y, src_y0, src_y1, exact = self._band_bounds
        if exact:
            cv2.resize(frame[src_y0:src_y1], (out_w, self._y_max - self._y_min),
                       dst=self._buf_band, interpolation=cv2.INTER_AREA)
        else:
            cv2.resize(frame[crop_y:], self.output_size, dst=self._buf_full,
                       interpolation=cv2.INTER_AREA)
            np.copyto(self._buf_band, self._buf_full[self._y_min:self._y_max])
        return self._buf_band

    def _threshold_road(self, gray, settings):
        """Use the absolute threshold only where local population variance is low."""
        block_size = settings['adaptive_block_size']
        c = settings['adaptive_c']
        if not settings['enable_variance_fallback']:
            cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                block_size, c, dst=self._buf_road)
            return

        kernel = (block_size, block_size)
        # The grayscale band already owns its ROI-sized buffer. Older OpenCV
        # sqrBoxFilter builds do not accept BORDER_ISOLATED.
        border = cv2.BORDER_REPLICATE
        cv2.boxFilter(gray, cv2.CV_32F, kernel, dst=self._buf_mean,
                      normalize=True, borderType=border)
        cv2.sqrBoxFilter(gray, cv2.CV_32F, kernel, dst=self._buf_variance,
                         normalize=True, borderType=border)
        cv2.multiply(self._buf_mean, self._buf_mean, dst=self._buf_square)
        cv2.subtract(self._buf_variance, self._buf_square, dst=self._buf_variance)
        # Cancellation can produce tiny negative values in uniform bright areas.
        cv2.max(self._buf_variance, 0.0, dst=self._buf_variance)

        # Reuse the mean, including the uint8 rounding and floor(C) convention
        # of OpenCV's adaptive mean inverse threshold. Signed subtraction avoids
        # saturation. Clamping C outside the possible delta range [-255, 255]
        # preserves the result and keeps the OpenCV scalar representable.
        cv2.convertScaleAbs(self._buf_mean, dst=self._buf_mean_u8)
        cv2.subtract(gray, self._buf_mean_u8, dst=self._buf_delta, dtype=cv2.CV_16S)
        cutoff = float(max(-256, min(256, -math.floor(c))))
        cv2.compare(self._buf_delta, cutoff, cv2.CMP_LE, dst=self._buf_road)
        cv2.compare(self._buf_variance, settings['fallback_variance_threshold'],
                    cv2.CMP_LT, dst=self._buf_low_variance)
        cv2.threshold(gray, settings['fallback_gray_threshold'], 255,
                      cv2.THRESH_BINARY_INV, dst=self._buf_absolute)
        cv2.copyTo(self._buf_absolute, self._buf_low_variance, self._buf_road)

    def _process_frame(self, frame, gray_code, bgr_code, settings):
        """Return independent label/debug arrays safe for asynchronous publishing."""
        telemetry = settings['enable_telemetry']
        started = time.perf_counter() if telemetry else 0.0
        band = self._resize_band(frame)
        if gray_code is None:
            gray = band
        else:
            gray = cv2.cvtColor(band, gray_code, dst=self._buf_gray)
        self._threshold_road(gray, settings)
        if telemetry:
            fallback_percent = (100.0 * cv2.countNonZero(self._buf_low_variance)
                                / gray.size if settings['enable_variance_fallback'] else 0.0)
            self.telemetry_stats['fallback_percent'].append(fallback_percent)
            self.telemetry_stats['adaptive_threshold'].append(
                (time.perf_counter() - started) * 1000.0)
            started = time.perf_counter()

        out_w, out_h = self.output_size
        labels = np.full((out_h, out_w), self.LABEL_INVALID, dtype=np.uint8)
        label_band = labels[self._y_min:self._y_max]
        label_band.fill(self.LABEL_BACKGROUND)
        np.copyto(label_band, self.LABEL_BLUE, where=self._buf_road != 0)
        if self.frame_counter % self.debug_probe_interval_frames == 0:
            self._publish_debug_mask = self.debug_mask_pub.get_subscription_count() > 0
            self._publish_overlay = self.debug_overlay_pub.get_subscription_count() > 0
            self._publish_fallback_overlay = (
                self.debug_fallback_overlay_pub.get_subscription_count() > 0)

        mask = None
        overlay = None
        fallback_overlay = None
        if (self._publish_debug_mask or self._publish_overlay
                or self._publish_fallback_overlay):
            mask = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            # BGR blue. No resized camera image is needed for the mask alone.
            mask[self._y_min:self._y_max, :, 0] = self._buf_road
            if self._publish_overlay or self._publish_fallback_overlay:
                crop_y, _, _, exact = self._band_bounds
                if exact:
                    cv2.resize(frame[crop_y:], self.output_size, dst=self._buf_full,
                               interpolation=cv2.INTER_AREA)
                background = self._buf_full
                if bgr_code is not None:
                    background = cv2.cvtColor(background, bgr_code)
                if self._publish_overlay:
                    overlay = cv2.addWeighted(background, 0.7, mask, 0.5, 0)
                if self._publish_fallback_overlay:
                    # Keep the blue-only mask independent for asynchronous publishing.
                    # BGR blue + red gives purple where road and fallback overlap.
                    fallback_mask = mask.copy()
                    if settings['enable_variance_fallback']:
                        fallback_mask[self._y_min:self._y_max, :, 2] = self._buf_low_variance
                    fallback_overlay = cv2.addWeighted(
                        background, 0.7, fallback_mask, 0.5, 0)
        if telemetry:
            self.telemetry_stats['canvas'].append((time.perf_counter() - started) * 1000.0)
        return labels, mask if self._publish_debug_mask else None, overlay, fallback_overlay

    def _processing_worker(self):
        while not self._stop.is_set():
            try:
                msg, arrival = self.processing_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            settings = self._settings
            telemetry = settings['enable_telemetry']
            started = time.perf_counter() if telemetry else 0.0
            if telemetry and arrival is not None:
                self.telemetry_stats['queue_wait'].append((started - arrival) * 1000.0)
            try:
                frame, gray_code, bgr_code = self._view_source(msg)
                if telemetry:
                    self.telemetry_stats['ingress'].append(
                        (time.perf_counter() - started) * 1000.0)
                images = self._process_frame(frame, gray_code, bgr_code, settings)
                enqueue_start = time.perf_counter() if telemetry else 0.0
                self.frames_dropped_publish += self._put_latest(
                    self.pub_queue, (images, msg.header, settings))
                if telemetry:
                    self.telemetry_stats['publish_enqueue'].append(
                        (time.perf_counter() - enqueue_start) * 1000.0)
                    self.telemetry_stats['total_pipeline'].append(
                        (time.perf_counter() - started) * 1000.0)
                self.frame_counter += 1
                if telemetry and self.frame_counter % self.window_size == 0:
                    self._log_telemetry_report()
            except Exception as exc:
                self.get_logger().error(f'Frame processing failed: {exc}')

    def _encode_debug_image(self, image, header, quality):
        msg = CompressedImage()
        msg.header = header
        msg.format = 'bgr8; jpeg compressed bgr8'
        msg.data = self.jpeg.encode(image, quality=quality, pixel_format=TJPF_BGR)
        return msg

    def _publish_worker(self):
        while not self._stop.is_set():
            try:
                images, header, settings = self.pub_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            telemetry = settings['enable_telemetry']
            started = time.perf_counter() if telemetry else 0.0
            labels, mask, overlay, fallback_overlay = images
            try:
                msg = self.bridge.cv2_to_imgmsg(labels, encoding='mono8')
                msg.header = header
                self.label_pub.publish(msg)
                for publisher, image in (
                    (self.debug_mask_pub, mask), (self.debug_overlay_pub, overlay),
                    (self.debug_fallback_overlay_pub, fallback_overlay),
                ):
                    if image is not None:
                        publisher.publish(self._encode_debug_image(
                            image, header, settings['debug_jpeg_quality']))
                if telemetry:
                    self.telemetry_stats['async_encode_publish'].append(
                        (time.perf_counter() - started) * 1000.0)
                    self.telemetry_stats['final_message_age'].append(
                        self._message_age_ms(header))
            except Exception as exc:
                self.get_logger().error(f'Publishing failed: {exc}')

    def _log_telemetry_report(self):
        # Snapshot deques before aggregating; workers append concurrently.
        snapshots = {name: tuple(values) for name, values in self.telemetry_stats.items()}
        averages = {name: sum(values) / len(values) if values else 0.0
                    for name, values in snapshots.items()}
        now = time.perf_counter()
        span = now - self._last_report_time
        fps = (self.frame_counter - self._last_report_count) / span if span > 0 else 0.0
        self._last_report_time = now
        self._last_report_count = self.frame_counter
        arrivals = tuple(self._arrival_times)
        input_hz = 0.0
        if len(arrivals) > 1 and arrivals[-1] > arrivals[0]:
            input_hz = (len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
        total = averages['total_pipeline']
        estimated_fps = 1000.0 / total if total > 0 else 0.0
        breakdown = '\n'.join(
            f'  {name}: {value:.2f} {"%" if name == "fallback_percent" else "ms"}'
            for name, value in averages.items())
        self.get_logger().info(
            f'\n====== BINARY SEGMENTATION PERFORMANCE (AVG {self.window_size} frames) ======\n'
            f'  Processed/received: {self.frame_counter}/{self.frames_received}\n'
            f'  Dropped (ingress/publish): '
            f'{self.frames_dropped_ingress}/{self.frames_dropped_publish}\n'
            f'  Camera: {input_hz:.1f} Hz; worker: {fps:.1f} FPS; '
            f'estimated (1000/total): {estimated_fps:.1f} FPS\n'
            f'{breakdown}\n'
            f'================================================================')

    def destroy_node(self):
        """Stop workers before destroying the publishers they use."""
        self._stop.set()
        self.worker_thread.join()
        self.pub_thread.join()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = BinaryLaneDetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
