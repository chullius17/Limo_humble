#!/usr/bin/env python3
"""Seeded road segmentation with gradient barriers and native connected components."""

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


class WaterfallLaneDetector(Node):
    """Grow dark seeds through regions bounded by strong grayscale gradients."""

    LABEL_INVALID = np.uint8(0)
    LABEL_BLUE = np.uint8(1)
    LABEL_BACKGROUND = np.uint8(3)

    _ENCODINGS = {
        'bgr8': (3, cv2.COLOR_BGR2GRAY, None),
        'rgb8': (3, cv2.COLOR_RGB2GRAY, cv2.COLOR_RGB2BGR),
        'mono8': (1, None, cv2.COLOR_GRAY2BGR),
    }
    _TIMINGS = (
        'transport_delay', 'queue_wait', 'ingress', 'grayscale_resize',
        'gradient', 'seed_erosion', 'region_growing',
        'canvas', 'publish_enqueue', 'total_pipeline', 'async_encode_publish',
        'final_message_age',
    )

    def __init__(self):
        super().__init__('waterfall_lane_detector')
        readonly = ParameterDescriptor(read_only=True)
        for name, default in (
            ('rgb_topic', '/rgb/image_raw'),
            ('roi_y_min', 0.1), ('roi_y_max', 1.0),
            ('opencv_num_threads', 1), ('debug_probe_interval_frames', 30),
        ):
            self.declare_parameter(name, default, readonly)
        defaults = {
            'seed_max_gray': 80,
            'gradient_threshold': 15.0,
            'seed_y_min': 0.0,
            'seed_erosion_iterations': 0,
            'barrier_dilation_iterations': 1,
            'debug_jpeg_quality': 85,
            'enable_telemetry': True,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        self._settings = {name: self.get_parameter(name).value for name in defaults}
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
            Image, 'limo/cv_package/detection/lane_waterfall_labels/raw', output_qos)
        self.debug_mask_pub = self.create_publisher(
            CompressedImage, 'limo/cv_package/detection/lane_waterfall_masks/compressed',
            output_qos)
        self.debug_overlay_pub = self.create_publisher(
            CompressedImage, 'limo/cv_package/detection/lane_waterfall_overlay/compressed',
            output_qos)
        self.debug_seeds_overlay_pub = self.create_publisher(
            CompressedImage,
            'limo/cv_package/detection/lane_waterfall_seeds_overlay/compressed',
            output_qos)
        self._publish_seeds_overlay = False
        self._publish_debug_mask = False
        self._publish_overlay = False
        self._geometry_key = None
        self._morphology_kernel = np.ones((3, 3), dtype=np.uint8)
        self._last_ingress_path = None
        self.window_size = 30
        self.telemetry_stats = {
            name: deque(maxlen=self.window_size)
            for name in self._TIMINGS + (
                'seed_percent', 'seed_removed_percent', 'barrier_percent',
                'road_percent', 'seeded_components')}
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
        self.get_logger().info(
            'Waterfall lane detector initialized: Sobel barriers and 4-connected seed growth.')

    @staticmethod
    def _validate_settings(settings):
        for name, low, high in (
            ('seed_max_gray', 0, 255), ('debug_jpeg_quality', 1, 100),
            ('barrier_dilation_iterations', 0, 5), ('seed_erosion_iterations', 0, 5),
        ):
            value = settings[name]
            if (isinstance(value, bool) or not isinstance(value, int)
                    or not low <= value <= high):
                raise ValueError(f'{name} must be an integer in [{low}, {high}]')
        gradient = settings['gradient_threshold']
        if (isinstance(gradient, bool) or not isinstance(gradient, (int, float))
                or not math.isfinite(gradient) or not 0 <= gradient <= 255):
            raise ValueError('gradient_threshold must be finite and in [0, 255]')
        seed_y = settings['seed_y_min']
        if (isinstance(seed_y, bool) or not isinstance(seed_y, (int, float))
                or not math.isfinite(seed_y) or not 0 <= seed_y < 1):
            raise ValueError('seed_y_min must be finite and in [0, 1)')
        if not isinstance(settings['enable_telemetry'], bool):
            raise ValueError('enable_telemetry must be a boolean')

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
            self._buf_dx = np.empty(band_shape, dtype=np.float32)
            self._buf_dy = np.empty(band_shape, dtype=np.float32)
            self._buf_gradient = np.empty(band_shape, dtype=np.float32)
            self._buf_barriers = np.empty(band_shape, dtype=np.uint8)
            self._buf_free = np.empty(band_shape, dtype=np.uint8)
            self._buf_seeds = np.empty(band_shape, dtype=np.uint8)
            self._buf_components = np.empty(band_shape, dtype=np.int32)
            self._buf_component_road = np.empty(self._buf_road.size + 1, dtype=np.uint8)
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

    def _propagate_seeds(self):
        """Select all 4-connected free-space components containing an admitted seed."""
        if cv2.countNonZero(self._buf_seeds) == 0:
            self._buf_road.fill(0)
            return 0
        # Explicit SAUF/Wu algorithm keeps 4-connectivity predictable across
        # OpenCV builds. The component array stays private to this worker.
        count, components = cv2.connectedComponentsWithAlgorithm(
            self._buf_free, 4, cv2.CV_32S, cv2.CCL_WU, labels=self._buf_components)
        selected = self._buf_component_road[:count]
        selected.fill(0)
        selected[components[self._buf_seeds != 0]] = 255
        selected[0] = 0  # Barrier/background ID must never be road.
        np.take(selected, components, out=self._buf_road, mode='clip')
        return int(np.count_nonzero(selected))

    def _segment_road(self, gray, settings):
        """Build gradient barriers, admit dark seeds, then grow through free regions."""
        telemetry = settings['enable_telemetry']
        started = time.perf_counter() if telemetry else 0.0
        # Scale 1/8 gives a unit response to a one-level-per-pixel axis ramp.
        # Keep float derivatives: converting them to uint8 would clip strong edges.
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, dst=self._buf_dx,
                  ksize=3, scale=0.125, borderType=cv2.BORDER_REPLICATE)
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, dst=self._buf_dy,
                  ksize=3, scale=0.125, borderType=cv2.BORDER_REPLICATE)
        cv2.absdiff(self._buf_dx, 0.0, dst=self._buf_dx)
        cv2.absdiff(self._buf_dy, 0.0, dst=self._buf_dy)
        cv2.add(self._buf_dx, self._buf_dy, dst=self._buf_gradient)
        cv2.compare(self._buf_gradient, settings['gradient_threshold'],
                    cv2.CMP_GT, dst=self._buf_barriers)
        iterations = settings['barrier_dilation_iterations']
        if iterations:
            cv2.dilate(self._buf_barriers, self._morphology_kernel,
                       dst=self._buf_barriers, iterations=iterations)
        if telemetry:
            self.telemetry_stats['gradient'].append((time.perf_counter() - started) * 1000)
            started = time.perf_counter()
        cv2.bitwise_not(self._buf_barriers, dst=self._buf_free)
        cv2.threshold(gray, settings['seed_max_gray'], 255,
                      cv2.THRESH_BINARY_INV, dst=self._buf_seeds)
        cv2.bitwise_and(self._buf_seeds, self._buf_free, dst=self._buf_seeds)
        seed_y0 = int(gray.shape[0] * settings['seed_y_min'])
        self._buf_seeds[:seed_y0].fill(0)
        # Erode admitted seeds, never the traversable mask: surviving groups
        # still grow through their entire region. Zero padding also removes
        # tiny seed groups touching an image/ROI boundary.
        raw_seed_count = cv2.countNonZero(self._buf_seeds) if telemetry else 0
        erosion_started = time.perf_counter() if telemetry else 0.0
        iterations = settings['seed_erosion_iterations']
        if iterations:
            cv2.erode(self._buf_seeds, self._morphology_kernel,
                      dst=self._buf_seeds, iterations=iterations,
                      borderType=cv2.BORDER_CONSTANT, borderValue=0)
        if telemetry:
            self.telemetry_stats['seed_erosion'].append(
                (time.perf_counter() - erosion_started) * 1000)
            remaining_seeds = cv2.countNonZero(self._buf_seeds)
            removed_percent = (100.0 * (raw_seed_count - remaining_seeds) / raw_seed_count
                               if raw_seed_count else 0.0)
            self.telemetry_stats['seed_removed_percent'].append(removed_percent)
        seeded_components = self._propagate_seeds()
        if telemetry:
            self.telemetry_stats['region_growing'].append(
                (time.perf_counter() - started) * 1000)
            for name, mask in (('seed_percent', self._buf_seeds),
                               ('barrier_percent', self._buf_barriers),
                               ('road_percent', self._buf_road)):
                self.telemetry_stats[name].append(100.0 * cv2.countNonZero(mask) / gray.size)
            self.telemetry_stats['seeded_components'].append(seeded_components)

    def _process_frame(self, frame, gray_code, bgr_code, settings):
        """Return independent label/debug arrays safe for asynchronous publishing."""
        telemetry = settings['enable_telemetry']
        started = time.perf_counter() if telemetry else 0.0
        band = self._resize_band(frame)
        if gray_code is None:
            gray = band
        else:
            gray = cv2.cvtColor(band, gray_code, dst=self._buf_gray)
        if telemetry:
            self.telemetry_stats['grayscale_resize'].append(
                (time.perf_counter() - started) * 1000.0)
        self._segment_road(gray, settings)
        started = time.perf_counter() if telemetry else 0.0

        out_w, out_h = self.output_size
        labels = np.full((out_h, out_w), self.LABEL_INVALID, dtype=np.uint8)
        label_band = labels[self._y_min:self._y_max]
        label_band.fill(self.LABEL_BACKGROUND)
        np.copyto(label_band, self.LABEL_BLUE, where=self._buf_road != 0)
        if self.frame_counter % self.debug_probe_interval_frames == 0:
            self._publish_debug_mask = self.debug_mask_pub.get_subscription_count() > 0
            self._publish_overlay = self.debug_overlay_pub.get_subscription_count() > 0
            self._publish_seeds_overlay = (
                self.debug_seeds_overlay_pub.get_subscription_count() > 0)

        mask = None
        overlay = None
        seeds_overlay = None
        if (self._publish_debug_mask or self._publish_overlay
                or self._publish_seeds_overlay):
            mask = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            # BGR blue. No resized camera image is needed for the mask alone.
            mask[self._y_min:self._y_max, :, 0] = self._buf_road
            if self._publish_overlay or self._publish_seeds_overlay:
                crop_y, _, _, exact = self._band_bounds
                if exact:
                    cv2.resize(frame[crop_y:], self.output_size, dst=self._buf_full,
                               interpolation=cv2.INTER_AREA)
                background = self._buf_full
                if bgr_code is not None:
                    background = cv2.cvtColor(background, bgr_code)
                if self._publish_overlay:
                    overlay = cv2.addWeighted(background, 0.7, mask, 0.5, 0)
                if self._publish_seeds_overlay:
                    # Show actual barriers (after dilation) and surviving seeds (after erosion).
                    # Regions without either retain the original camera pixels.
                    seeds_overlay = background.copy()
                    band_view = seeds_overlay[self._y_min:self._y_max]
                    band_view[self._buf_barriers != 0] = (0, 0, 255)
                    band_view[self._buf_seeds != 0] = (0, 255, 0)
        if telemetry:
            self.telemetry_stats['canvas'].append((time.perf_counter() - started) * 1000.0)
        return labels, mask if self._publish_debug_mask else None, overlay, seeds_overlay

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
            labels, mask, overlay, seeds_overlay = images
            try:
                msg = self.bridge.cv2_to_imgmsg(labels, encoding='mono8')
                msg.header = header
                self.label_pub.publish(msg)
                for publisher, image in (
                    (self.debug_mask_pub, mask), (self.debug_overlay_pub, overlay),
                    (self.debug_seeds_overlay_pub, seeds_overlay),
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
        lines = []
        for name, value in averages.items():
            unit = '%' if name.endswith('_percent') else 'ms'
            if name == 'seeded_components':
                unit = 'count'
            lines.append(f'  {name}: {value:.2f} {unit}')
        breakdown = '\n'.join(lines)
        self.get_logger().info(
            f'\n====== WATERFALL SEGMENTATION PERFORMANCE (AVG {self.window_size} frames) ======\n'
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
        node = WaterfallLaneDetector()
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
