#!/usr/bin/env python3
"""Combined, GPU-first BEV projection and color classification node."""

from collections import deque
import threading
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image

class BevAndClassification(Node):
    """Projects boundary pixels to BEV and classifies them in one process."""

    BLUE_BGR = np.array([255, 0, 0], dtype=np.uint8)
    WHITE_BGR = np.array([255, 255, 255], dtype=np.uint8)
    MAGENTA_BGR = np.array([255, 0, 255], dtype=np.uint8)

    def __init__(self):
        super().__init__('bev_and_classification')
        self.bridge = CvBridge()

        self.declare_parameter('camera_info_topic', '/rgb/camera_info')
        self.declare_parameter(
            'depth_topic',
            'limo/cv_package/depth_correction/depth_corrected/raw',
        )
        self.declare_parameter(
            'rgb_topic',
            'limo/cv_package/boundaries/lines_and_curbs/raw',
        )
        self.declare_parameter(
            'bev_topic',
            'limo/cv_package/bev/bird_perspective/raw',
        )
        self.declare_parameter(
            'debug_topic',
            'limo/cv_package/classification/debug/raw',
        )
        self.declare_parameter(
            'output_topic',
            'limo/cv_package/classification/output/raw',
        )
        self.declare_parameter('input_crop_y_min', 0.5)
        self.declare_parameter('bev_width', 600)
        self.declare_parameter('bev_height', 300)
        self.declare_parameter('bev_resolution', 0.01)
        self.declare_parameter('projection_stride', 1)
        self.declare_parameter('point_inflation_size', 3)
        self.declare_parameter('blue_distance_threshold_px', 10.0)
        self.declare_parameter('blue_max_distance_threshold_px', 16.0)
        self.declare_parameter('magenta_distance_threshold_px', 10.0)
        self.declare_parameter('enable_second_distance_transform', True)
        self.declare_parameter('color_tolerance', 30)
        self.declare_parameter('use_gpu', True)
        self.declare_parameter('max_processing_fps', 15.0)
        self.declare_parameter('enable_telemetry', True)
        self.declare_parameter('telemetry_window_size', 60)
        self.declare_parameter('telemetry_log_interval_frames', 30)

        self.width = int(self.get_parameter('bev_width').value)
        self.height = int(self.get_parameter('bev_height').value)
        self.resolution = float(self.get_parameter('bev_resolution').value)
        self.stride = int(self.get_parameter('projection_stride').value)
        self.point_inflation_size = int(
            self.get_parameter('point_inflation_size').value)
        self.crop_y_min = float(
            self.get_parameter('input_crop_y_min').value)
        self.use_gpu = bool(self.get_parameter('use_gpu').value)
        self.enable_second_distance_transform = bool(
            self.get_parameter('enable_second_distance_transform').value)
        self.max_processing_fps = float(
            self.get_parameter('max_processing_fps').value)
        self.telemetry_enabled = bool(
            self.get_parameter('enable_telemetry').value)
        self.telemetry_window = int(
            self.get_parameter('telemetry_window_size').value)
        self.telemetry_interval = int(
            self.get_parameter('telemetry_log_interval_frames').value)
        if self.width <= 0 or self.height <= 0 or self.resolution <= 0.0:
            raise ValueError('BEV dimensions and resolution must be positive')
        if self.stride <= 0:
            raise ValueError('projection_stride must be positive')
        if (self.point_inflation_size <= 0 or
                self.point_inflation_size % 2 == 0):
            raise ValueError('point_inflation_size must be positive and odd')
        if not 0.0 <= self.crop_y_min < 1.0:
            raise ValueError('input_crop_y_min must be in [0, 1)')
        if self.telemetry_window <= 0 or self.telemetry_interval <= 0:
            raise ValueError('telemetry sizes must be positive')
        if self.max_processing_fps < 0.0:
            raise ValueError('max_processing_fps must be non-negative')
        self.max_forward = self.height * self.resolution
        self.minimum_frame_period = (
            1.0 / self.max_processing_fps
            if self.max_processing_fps > 0.0 else 0.0
        )

        pipeline_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.bev_pub = self.create_publisher(
            Image, self.get_parameter('bev_topic').value, pipeline_qos)
        self.debug_pub = self.create_publisher(
            Image, self.get_parameter('debug_topic').value, pipeline_qos)
        self.output_pub = self.create_publisher(
            Image, self.get_parameter('output_topic').value, pipeline_qos)
        self.rgb_sub = self.create_subscription(
            Image,
            self.get_parameter('rgb_topic').value,
            self.rgb_callback,
            pipeline_qos,
        )
        self.depth_sub = self.create_subscription(
            Image,
            self.get_parameter('depth_topic').value,
            self.depth_callback,
            sensor_qos,
        )
        self.info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self.camera_info_callback,
            10,
        )

        self.depth_img = None
        self.depth_received_at = None
        self.fx = None
        self.cx = None
        self.info_width = None
        self.info_height = None
        # Single-slot mailbox: callbacks always overwrite the pending frame,
        # so the worker can never process an old queue backlog.
        self.frame_condition = threading.Condition()
        self.latest_frame = None
        self.running = True

        self.received = 0
        self.dropped = 0
        self.processed = 0
        self.last_report_at = time.perf_counter()
        self.last_report_received = 0
        self.last_report_processed = 0
        self.last_report_dropped = 0
        timing_names = (
            'queue', 'conversion', 'resize', 'telemetry_probe',
            'gpu_allocation', 'staging_rgb', 'staging_depth',
            'gpu_clear', 'gpu_upload', 'gpu_project', 'gpu_unpack',
            'gpu_download', 'gpu_wall',
            'cpu_sample', 'cpu_valid_mask', 'cpu_nonzero',
            'cpu_projection', 'cpu_rasterize',
            'inflation', 'classification', 'class_clear',
            'class_grayscale', 'class_bounds', 'class_masks_initial',
            'class_blue_distance', 'class_first_pass',
            'class_masks_second', 'class_magenta_distance',
            'class_finalize', 'class_copy_output',
            'class_gpu_upload', 'class_gpu_download',
            'publish_bev_convert', 'publish_bev_send',
            'publish_debug_convert', 'publish_debug_send',
            'publish_output_convert', 'publish_output_send',
            'publish', 'total', 'rgb_age', 'depth_age',
        )
        self.timings = {
            name: deque(maxlen=self.telemetry_window)
            for name in timing_names
        }
        workload_names = (
            'input_points', 'valid_depth_points', 'projected_points',
            'bev_points', 'classification_roi_pixels',
        )
        self.workloads = {
            name: deque(maxlen=self.telemetry_window)
            for name in workload_names
        }
        self.cpu_bev = np.zeros(
            (self.height, self.width, 3), dtype=np.uint8)
        self.cpu_debug = np.zeros_like(self.cpu_bev)
        self.cpu_output = np.zeros_like(self.cpu_bev)
        self.point_inflation_kernel = np.ones(
            (self.point_inflation_size, self.point_inflation_size),
            dtype=np.uint8,
        )

        self.cuda = None
        self.cuda_context = None
        self.cuda_module = None
        self.cuda_stream = None
        self.gpu_enabled = False
        self.gpu_input_shape = None
        self.host_rgb = None
        self.host_depth = None
        self.host_bev = None
        self.gpu_rgb = None
        self.gpu_depth = None
        self.gpu_bev_raw = None
        self.gpu_bev = None
        self.gpu_bev_inflated = None
        self.gpu_debug = None
        self.gpu_output = None
        self.host_debug = None
        self.host_output = None
        if self.use_gpu:
            self._initialize_cuda()

        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()
        self.get_logger().info(
            f'Combined BEV+classification ready: {self.width}x{self.height}, '
            f'{self.resolution:.3f} m/px, stride {self.stride}, backend '
            f'{"CUDA" if self.gpu_enabled else "CPU"}, '
            f'classification '
            f'{"PyCUDA thresholds" if self.gpu_enabled else "OpenCV CPU"}, '
            f'second pass '
            f'{"enabled" if self.enable_second_distance_transform else "disabled"}, '
            f'max {self.max_processing_fps:.1f} FPS.')

    def _initialize_cuda(self):
        """Create one CUDA context and all persistent output allocations."""
        pushed = False
        try:
            import pycuda.driver as cuda
            from pycuda.compiler import SourceModule

            cuda.init()
            if cuda.Device.count() == 0:
                raise RuntimeError('no CUDA device found')
            device = cuda.Device(0)
            self.cuda = cuda
            if hasattr(device, 'retain_primary_context'):
                self.cuda_context = device.retain_primary_context()
                self.cuda_context.push()
            else:
                self.cuda_context = device.make_context()
            pushed = True

            self.cuda_module = SourceModule(r'''
                #include <math.h>

                extern "C" __global__ void project_to_bev(
                    const unsigned char *__restrict__ rgb,
                    const float *__restrict__ depth,
                    unsigned int *__restrict__ bev,
                    int input_width, int input_height, int stride,
                    float fx, float cx,
                    int bev_width, int bev_height,
                    float resolution, float max_forward)
                {
                    const int sample_width =
                        (input_width + stride - 1) / stride;
                    const int sample_height =
                        (input_height + stride - 1) / stride;
                    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < sample_width * sample_height) {
                        const int x = (idx % sample_width) * stride;
                        const int y = (idx / sample_width) * stride;
                        const int source = y * input_width + x;
                        const int color = source * 3;
                        const unsigned char b = rgb[color];
                        const unsigned char g = rgb[color + 1];
                        const unsigned char r = rgb[color + 2];
                        if ((b | g | r) != 0) {
                            const float z = depth[source];
                            if (isfinite(z) && z > 0.1f &&
                                z < max_forward) {
                                const float x_camera =
                                    ((float)x - cx) * z / fx;
                                const int u = __float2int_rz(
                                    0.5f * (float)bev_width
                                    + x_camera / resolution);
                                const int v = __float2int_rz(
                                    (float)(bev_height - 1)
                                    - z / resolution);
                                if (u >= 0 && u < bev_width &&
                                    v >= 0 && v < bev_height) {
                                    const int target = v * bev_width + u;
                                    // One aligned store prevents torn BGR
                                    // colors when several rays hit one cell.
                                    bev[target] = (unsigned int)b
                                        | ((unsigned int)g << 8)
                                        | ((unsigned int)r << 16);
                                }
                            }
                        }
                    }
                }

                extern "C" __global__ void unpack_bev(
                    const unsigned int *__restrict__ input,
                    unsigned char *__restrict__ output,
                    int width, int height)
                {
                    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx >= width * height) return;
                    const unsigned int color = input[idx];
                    const int target = idx * 3;
                    output[target] = (unsigned char)(color & 255U);
                    output[target + 1] =
                        (unsigned char)((color >> 8) & 255U);
                    output[target + 2] =
                        (unsigned char)((color >> 16) & 255U);
                }

                __device__ __forceinline__ bool color_matches(
                    const unsigned char *image, int pixel,
                    int target_b, int target_g, int target_r, int tolerance)
                {
                    const int offset = pixel * 3;
                    const int db = (int)image[offset] - target_b;
                    const int dg = (int)image[offset + 1] - target_g;
                    const int dr = (int)image[offset + 2] - target_r;
                    return db >= -tolerance && db <= tolerance
                        && dg >= -tolerance && dg <= tolerance
                        && dr >= -tolerance && dr <= tolerance;
                }

                extern "C" __global__ void inflate_bgr(
                    const unsigned char *__restrict__ input,
                    unsigned char *__restrict__ output,
                    int width, int height, int radius)
                {
                    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx >= width * height) return;
                    const int x = idx % width;
                    const int y = idx / width;
                    unsigned char max_b = 0;
                    unsigned char max_g = 0;
                    unsigned char max_r = 0;
                    for (int dy = -radius; dy <= radius; ++dy) {
                        const int source_y = y + dy;
                        if (source_y < 0 || source_y >= height) continue;
                        for (int dx = -radius; dx <= radius; ++dx) {
                            const int source_x = x + dx;
                            if (source_x < 0 || source_x >= width) continue;
                            const int source = (source_y * width + source_x) * 3;
                            max_b = input[source] > max_b
                                ? input[source] : max_b;
                            max_g = input[source + 1] > max_g
                                ? input[source + 1] : max_g;
                            max_r = input[source + 2] > max_r
                                ? input[source + 2] : max_r;
                        }
                    }
                    const int target = idx * 3;
                    output[target] = max_b;
                    output[target + 1] = max_g;
                    output[target + 2] = max_r;
                }

                extern "C" __global__ void classify_blue_thresholds(
                    const unsigned char *__restrict__ input,
                    unsigned char *__restrict__ debug,
                    int width, int height, int tolerance,
                    float minimum_squared, float maximum_squared,
                    int maximum_radius)
                {
                    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx >= width * height) return;
                    const int target = idx * 3;
                    const unsigned char b = input[target];
                    const unsigned char g = input[target + 1];
                    const unsigned char r = input[target + 2];
                    debug[target] = b;
                    debug[target + 1] = g;
                    debug[target + 2] = r;

                    if (!color_matches(
                            input, idx, 255, 255, 255, tolerance)) return;

                    const int x = idx % width;
                    const int y = idx / width;
                    bool blue_within_maximum = false;
                    bool blue_within_minimum = false;
                    for (int dy = -maximum_radius;
                            dy <= maximum_radius && !blue_within_minimum;
                            ++dy) {
                        const int source_y = y + dy;
                        if (source_y < 0 || source_y >= height) continue;
                        for (int dx = -maximum_radius;
                                dx <= maximum_radius; ++dx) {
                            const float distance_squared =
                                (float)(dx * dx + dy * dy);
                            if (distance_squared > maximum_squared) continue;
                            const int source_x = x + dx;
                            if (source_x < 0 || source_x >= width) continue;
                            const int source = source_y * width + source_x;
                            if (!color_matches(
                                    input, source, 255, 0, 0,
                                    tolerance)) continue;
                            blue_within_maximum = true;
                            if (distance_squared <= minimum_squared) {
                                blue_within_minimum = true;
                                break;
                            }
                        }
                    }

                    if (blue_within_minimum) return;
                    if (blue_within_maximum) {
                        debug[target] = 255;
                        debug[target + 1] = 0;
                        debug[target + 2] = 255;
                    } else {
                        debug[target] = 0;
                        debug[target + 1] = 0;
                        debug[target + 2] = 0;
                    }
                }

                extern "C" __global__ void propagate_magenta_threshold(
                    const unsigned char *__restrict__ debug,
                    unsigned char *__restrict__ output,
                    int width, int height, int tolerance,
                    float threshold_squared, int threshold_radius)
                {
                    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx >= width * height) return;
                    const int target = idx * 3;
                    output[target] = debug[target];
                    output[target + 1] = debug[target + 1];
                    output[target + 2] = debug[target + 2];
                    if (threshold_squared <= 0.0f || !color_matches(
                            debug, idx, 255, 255, 255, tolerance)) return;

                    const int x = idx % width;
                    const int y = idx / width;
                    for (int dy = -threshold_radius;
                            dy <= threshold_radius; ++dy) {
                        const int source_y = y + dy;
                        if (source_y < 0 || source_y >= height) continue;
                        for (int dx = -threshold_radius;
                                dx <= threshold_radius; ++dx) {
                            const float distance_squared =
                                (float)(dx * dx + dy * dy);
                            if (distance_squared >= threshold_squared) continue;
                            const int source_x = x + dx;
                            if (source_x < 0 || source_x >= width) continue;
                            const int source = source_y * width + source_x;
                            if (color_matches(
                                    debug, source, 255, 0, 255,
                                    tolerance)) {
                                output[target] = 255;
                                output[target + 1] = 0;
                                output[target + 2] = 255;
                                return;
                            }
                        }
                    }
                }

            ''', no_extern_c=True)

            self.project_kernel = self.cuda_module.get_function(
                'project_to_bev')
            self.unpack_kernel = self.cuda_module.get_function('unpack_bev')
            self.inflation_cuda_kernel = self.cuda_module.get_function(
                'inflate_bgr')
            self.blue_threshold_kernel = self.cuda_module.get_function(
                'classify_blue_thresholds')
            self.magenta_threshold_kernel = self.cuda_module.get_function(
                'propagate_magenta_threshold')
            self.cuda_stream = cuda.Stream()
            image_shape = (self.height, self.width, 3)
            image_bytes = int(np.prod(image_shape))
            self.host_bev = cuda.pagelocked_empty(image_shape, np.uint8)
            self.host_debug = cuda.pagelocked_empty(image_shape, np.uint8)
            self.host_output = cuda.pagelocked_empty(image_shape, np.uint8)
            self.gpu_bev_raw = cuda.mem_alloc(
                self.width * self.height * np.dtype(np.uint32).itemsize)
            self.gpu_bev = cuda.mem_alloc(image_bytes)
            self.gpu_bev_inflated = cuda.mem_alloc(image_bytes)
            self.gpu_debug = cuda.mem_alloc(image_bytes)
            self.gpu_output = cuda.mem_alloc(image_bytes)
            self.cuda_events = [cuda.Event() for _ in range(10)]
            self.gpu_enabled = True
            self.get_logger().info(
                f'CUDA fusion enabled on {device.name()}')
        except Exception as exc:
            self.gpu_enabled = False
            self.get_logger().warn(
                f'CUDA unavailable ({exc}); using combined CPU fallback.')
        finally:
            if pushed:
                self.cuda_context.pop()

    def _ensure_gpu_inputs(self, shape):
        if self.gpu_input_shape == shape:
            return
        for name in ('gpu_rgb', 'gpu_depth'):
            allocation = getattr(self, name)
            if allocation is not None:
                allocation.free()
        height, width = shape
        self.host_rgb = self.cuda.pagelocked_empty(
            (height, width, 3), np.uint8)
        self.host_depth = self.cuda.pagelocked_empty(
            (height, width), np.float32)
        self.gpu_rgb = self.cuda.mem_alloc(self.host_rgb.nbytes)
        self.gpu_depth = self.cuda.mem_alloc(self.host_depth.nbytes)
        self.gpu_input_shape = shape

    def camera_info_callback(self, msg):
        self.fx = float(msg.k[0])
        self.cx = float(msg.k[2])
        self.info_width = int(msg.width)
        self.info_height = int(msg.height)

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough')
        except Exception as exc:
            self.get_logger().error(f'Depth conversion failed: {exc}')
            return
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) * 0.001
        elif depth.dtype != np.float32:
            depth = depth.astype(np.float32)
        self.depth_img = depth
        self.depth_received_at = time.perf_counter()

    def rgb_callback(self, msg):
        self.received += 1
        queued_at = time.perf_counter()
        with self.frame_condition:
            if self.latest_frame is not None:
                self.dropped += 1
            self.latest_frame = (msg, queued_at)
            self.frame_condition.notify()

    def _worker_loop(self):
        pushed = False
        next_frame_at = 0.0
        if self.gpu_enabled:
            try:
                self.cuda_context.push()
                pushed = True
            except Exception as exc:
                self.gpu_enabled = False
                self.get_logger().error(
                    f'CUDA context activation failed ({exc}); using CPU.')
        try:
            while self.running and rclpy.ok():
                with self.frame_condition:
                    while self.running and self.latest_frame is None:
                        self.frame_condition.wait(timeout=0.5)
                    if not self.running:
                        break

                    # Wait for the next FPS slot while callbacks keep replacing
                    # latest_frame. On expiry we consume exactly the newest.
                    remaining = next_frame_at - time.perf_counter()
                    if remaining > 0.0:
                        self.frame_condition.wait(timeout=remaining)
                        continue
                    msg, queued_at = self.latest_frame
                    self.latest_frame = None

                next_frame_at = (
                    time.perf_counter() + self.minimum_frame_period)
                try:
                    self._process_frame(msg, queued_at)
                except Exception as exc:
                    self.get_logger().error(
                        f'Combined pipeline frame failed: {exc}')
        finally:
            if pushed:
                self.cuda_context.pop()

    def _thresholds(self):
        values = (
            float(self.get_parameter('blue_distance_threshold_px').value),
            float(self.get_parameter(
                'blue_max_distance_threshold_px').value),
            float(self.get_parameter(
                'magenta_distance_threshold_px').value),
            int(self.get_parameter('color_tolerance').value),
        )
        minimum, maximum, magenta, tolerance = values
        if minimum < 0.0 or maximum < minimum or magenta < 0.0:
            raise ValueError('classification distances are inconsistent')
        if not 0 <= tolerance <= 255:
            raise ValueError('color_tolerance must be in [0, 255]')
        return values

    def _wanted_outputs(self):
        return (
            self.bev_pub.get_subscription_count() > 0,
            self.debug_pub.get_subscription_count() > 0,
            self.output_pub.get_subscription_count() > 0,
        )

    def _process_frame(self, msg, queued_at):
        wanted = self._wanted_outputs()
        if not any(wanted):
            return
        started = time.perf_counter()
        timing = {name: 0.0 for name in self.timings}
        workload = {name: float('nan') for name in self.workloads}
        timing['queue'] = (started - queued_at) * 1000.0
        stamp = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
        if stamp:
            timing['rgb_age'] = max(
                0.0,
                (self.get_clock().now().nanoseconds - stamp) / 1e6,
            )
        if self.depth_received_at is not None:
            timing['depth_age'] = (
                started - self.depth_received_at) * 1000.0

        depth = self.depth_img
        if (depth is None or self.fx is None or not self.info_width or
                not self.info_height):
            self.get_logger().warn(
                'Waiting for depth and CameraInfo.',
                throttle_duration_sec=2.0,
            )
            return
        stage = time.perf_counter()
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            thresholds = (
                self._thresholds()
                if wanted[1] or wanted[2]
                else (0.0, 0.0, 0.0, 0)
            )
        except Exception as exc:
            self.get_logger().error(f'Input conversion failed: {exc}')
            return
        timing['conversion'] = (time.perf_counter() - stage) * 1000.0

        probe_workload = (
            self.telemetry_enabled
            and (self.processed + 1) % self.telemetry_interval == 0
        )
        if probe_workload:
            stage = time.perf_counter()
            occupied_input = np.bitwise_or.reduce(rgb, axis=2) != 0
            workload['input_points'] = int(np.count_nonzero(occupied_input))
            timing['telemetry_probe'] += (
                time.perf_counter() - stage) * 1000.0

        height, width = rgb.shape[:2]
        stage = time.perf_counter()
        if depth.shape[:2] != (height, width):
            depth = cv2.resize(
                depth, (width, height), interpolation=cv2.INTER_NEAREST)
        timing['resize'] = (time.perf_counter() - stage) * 1000.0
        scale_x = width / float(self.info_width)
        fx = self.fx * scale_x
        cx = self.cx * scale_x

        if self.gpu_enabled:
            try:
                images = self._process_cuda(
                    rgb, depth, fx, cx, thresholds, wanted, timing,
                    workload, probe_workload)
            except Exception as exc:
                self.gpu_enabled = False
                self.get_logger().error(
                    f'CUDA pipeline failed ({exc}); switching to CPU.')
                images = self._process_cpu(
                    rgb, depth, fx, cx, thresholds, wanted, timing,
                    workload)
        else:
            images = self._process_cpu(
                rgb, depth, fx, cx, thresholds, wanted, timing, workload)

        stage = time.perf_counter()
        self._publish_requested(images, wanted, msg.header, timing)
        timing['publish'] = (time.perf_counter() - stage) * 1000.0
        timing['total'] = (time.perf_counter() - started) * 1000.0
        self._record_telemetry(timing, workload)

    def _process_cuda(
            self, rgb, depth, fx, cx, thresholds, wanted, timing,
            workload, probe_workload):
        """Project, inflate and classify without a CPU round trip."""
        height, width = rgb.shape[:2]
        stage = time.perf_counter()
        self._ensure_gpu_inputs((height, width))
        timing['gpu_allocation'] = (
            time.perf_counter() - stage) * 1000.0

        stage = time.perf_counter()
        np.copyto(self.host_rgb, rgb, casting='unsafe')
        timing['staging_rgb'] = (time.perf_counter() - stage) * 1000.0
        stage = time.perf_counter()
        np.copyto(self.host_depth, depth, casting='unsafe')
        timing['staging_depth'] = (
            time.perf_counter() - stage) * 1000.0

        cuda = self.cuda
        stream = self.cuda_stream
        (
            start_event,
            clear_event,
            upload_event,
            project_event,
            unpack_event,
            inflation_event,
            blue_event,
            magenta_event,
            bev_download_event,
            download_event,
        ) = self.cuda_events
        pixels = self.width * self.height
        raster_bytes = pixels * np.dtype(np.uint32).itemsize
        block = 256
        sample_width = (width + self.stride - 1) // self.stride
        sample_height = (height + self.stride - 1) // self.stride
        project_grid = (sample_width * sample_height + block - 1) // block
        image_grid = (pixels + block - 1) // block
        wall_started = time.perf_counter()
        start_event.record(stream)
        cuda.memset_d8_async(self.gpu_bev_raw, 0, raster_bytes, stream)
        clear_event.record(stream)
        cuda.memcpy_htod_async(self.gpu_rgb, self.host_rgb, stream)
        cuda.memcpy_htod_async(self.gpu_depth, self.host_depth, stream)
        upload_event.record(stream)

        self.project_kernel(
            self.gpu_rgb, self.gpu_depth, self.gpu_bev_raw,
            np.int32(width), np.int32(height), np.int32(self.stride),
            np.float32(fx), np.float32(cx),
            np.int32(self.width), np.int32(self.height),
            np.float32(self.resolution), np.float32(self.max_forward),
            block=(block, 1, 1), grid=(project_grid, 1, 1), stream=stream,
        )
        project_event.record(stream)
        self.unpack_kernel(
            self.gpu_bev_raw, self.gpu_bev,
            np.int32(self.width), np.int32(self.height),
            block=(block, 1, 1), grid=(image_grid, 1, 1), stream=stream,
        )
        unpack_event.record(stream)

        self.inflation_cuda_kernel(
            self.gpu_bev, self.gpu_bev_inflated,
            np.int32(self.width), np.int32(self.height),
            np.int32(self.point_inflation_size // 2),
            block=(block, 1, 1), grid=(image_grid, 1, 1), stream=stream,
        )
        inflation_event.record(stream)

        debug_image = None
        output_image = None
        if wanted[1] or wanted[2]:
            minimum, maximum, magenta, tolerance = thresholds
            self.blue_threshold_kernel(
                self.gpu_bev_inflated, self.gpu_debug,
                np.int32(self.width), np.int32(self.height),
                np.int32(tolerance), np.float32(minimum * minimum),
                np.float32(maximum * maximum),
                np.int32(int(np.ceil(maximum))),
                block=(block, 1, 1), grid=(image_grid, 1, 1), stream=stream,
            )
        blue_event.record(stream)

        if wanted[2] and self.enable_second_distance_transform:
            self.magenta_threshold_kernel(
                self.gpu_debug, self.gpu_output,
                np.int32(self.width), np.int32(self.height),
                np.int32(tolerance), np.float32(magenta * magenta),
                np.int32(int(np.ceil(magenta))),
                block=(block, 1, 1), grid=(image_grid, 1, 1), stream=stream,
            )
        magenta_event.record(stream)

        bev_image = None
        if wanted[0]:
            cuda.memcpy_dtoh_async(
                self.host_bev, self.gpu_bev_inflated, stream)
            bev_image = self.host_bev
        bev_download_event.record(stream)
        if wanted[1]:
            cuda.memcpy_dtoh_async(self.host_debug, self.gpu_debug, stream)
            debug_image = self.host_debug
        if wanted[2]:
            if self.enable_second_distance_transform:
                cuda.memcpy_dtoh_async(
                    self.host_output, self.gpu_output, stream)
                output_image = self.host_output
            elif wanted[1]:
                output_image = self.host_debug
            else:
                cuda.memcpy_dtoh_async(
                    self.host_output, self.gpu_debug, stream)
                output_image = self.host_output
        download_event.record(stream)
        stream.synchronize()
        timing['gpu_wall'] = (
            time.perf_counter() - wall_started) * 1000.0

        timing['gpu_clear'] = start_event.time_till(clear_event)
        timing['gpu_upload'] = clear_event.time_till(upload_event)
        timing['gpu_project'] = upload_event.time_till(project_event)
        timing['gpu_unpack'] = project_event.time_till(unpack_event)
        timing['inflation'] = unpack_event.time_till(inflation_event)
        timing['class_blue_distance'] = inflation_event.time_till(blue_event)
        if wanted[2] and self.enable_second_distance_transform:
            timing['class_magenta_distance'] = blue_event.time_till(
                magenta_event)
        timing['gpu_download'] = magenta_event.time_till(download_event)
        if wanted[1] or wanted[2]:
            timing['class_gpu_download'] = bev_download_event.time_till(
                download_event)
            timing['classification'] = (
                timing['class_blue_distance']
                + timing['class_magenta_distance']
                + timing['class_gpu_download']
            )
        else:
            timing['class_blue_distance'] = 0.0
            timing['class_magenta_distance'] = 0.0
        if probe_workload:
            if wanted[1] or wanted[2]:
                workload['classification_roi_pixels'] = (
                    self.width * self.height)
            if bev_image is not None:
                stage = time.perf_counter()
                occupied_bev = np.bitwise_or.reduce(bev_image, axis=2) != 0
                workload['bev_points'] = int(np.count_nonzero(occupied_bev))
                timing['telemetry_probe'] += (
                    time.perf_counter() - stage) * 1000.0
        return bev_image, debug_image, output_image

    def _process_cpu(
            self, rgb, depth, fx, cx, thresholds, wanted, timing,
            workload):
        stage = time.perf_counter()
        sampled_rgb = rgb[::self.stride, ::self.stride]
        sampled_depth = depth[::self.stride, ::self.stride]
        timing['cpu_sample'] = (time.perf_counter() - stage) * 1000.0

        stage = time.perf_counter()
        color_mask = np.bitwise_or.reduce(sampled_rgb, axis=2) != 0
        valid = (
            color_mask & np.isfinite(sampled_depth)
            & (sampled_depth > 0.1) & (sampled_depth < self.max_forward)
        )
        timing['cpu_valid_mask'] = (
            time.perf_counter() - stage) * 1000.0

        stage = time.perf_counter()
        rows, cols = np.nonzero(valid)
        timing['cpu_nonzero'] = (time.perf_counter() - stage) * 1000.0
        workload['valid_depth_points'] = len(rows)

        stage = time.perf_counter()
        z = sampled_depth[rows, cols]
        x_pixels = cols.astype(np.float32) * self.stride
        u = (
            self.width * 0.5 + ((x_pixels - cx) * z / fx)
            / self.resolution
        ).astype(np.int32)
        v = (self.height - 1 - z / self.resolution).astype(np.int32)
        inside = (
            (u >= 0) & (u < self.width)
            & (v >= 0) & (v < self.height)
        )
        timing['cpu_projection'] = (
            time.perf_counter() - stage) * 1000.0
        workload['projected_points'] = int(np.count_nonzero(inside))

        stage = time.perf_counter()
        bev = self.cpu_bev
        bev.fill(0)
        if np.any(inside):
            indices = v[inside] * self.width + u[inside]
            bev.reshape(-1, 3)[indices] = sampled_rgb[rows, cols][inside]
        timing['cpu_rasterize'] = (
            time.perf_counter() - stage) * 1000.0
        self._inflate_bev(bev, timing)
        debug, output = self._classify(
            bev, thresholds, wanted, timing, workload)
        return bev, debug, output

    def _inflate_bev(self, bev, timing):
        """Enlarge projected BEV points before classification and publishing."""
        started = time.perf_counter()
        if self.point_inflation_size > 1:
            cv2.dilate(
                bev,
                self.point_inflation_kernel,
                dst=bev,
                iterations=1,
            )
        timing['inflation'] = (time.perf_counter() - started) * 1000.0

    @staticmethod
    def _color_mask(image, color, tolerance):
        lower = np.clip(
            color.astype(np.int16) - tolerance, 0, 255).astype(np.uint8)
        upper = np.clip(
            color.astype(np.int16) + tolerance, 0, 255).astype(np.uint8)
        return cv2.inRange(image, lower, upper) > 0

    @classmethod
    def _classify_images(
            cls, image, minimum, maximum, magenta, tolerance, timing,
            enable_second_distance_transform=True):
        """Classify a BEV ROI using fast approximate distance transforms."""
        stage = time.perf_counter()
        white_mask = cls._color_mask(image, cls.WHITE_BGR, tolerance)
        blue_mask = cls._color_mask(image, cls.BLUE_BGR, tolerance)
        timing['class_masks_initial'] = (
            time.perf_counter() - stage) * 1000.0

        stage = time.perf_counter()
        if np.any(blue_mask):
            distance_input = np.full(image.shape[:2], 255, dtype=np.uint8)
            distance_input[blue_mask] = 0
            distance_from_blue = cv2.distanceTransform(
                distance_input,
                cv2.DIST_L2,
                cv2.DIST_MASK_3,
            )
            eligible_white_mask = white_mask & (
                distance_from_blue <= maximum)
            far_white_mask = eligible_white_mask & (
                distance_from_blue > minimum)
            discarded_white_mask = white_mask & (
                distance_from_blue > maximum)
        else:
            far_white_mask = np.zeros_like(white_mask)
            discarded_white_mask = white_mask
        timing['class_blue_distance'] = (
            time.perf_counter() - stage) * 1000.0

        stage = time.perf_counter()
        debug_image = image.copy()
        debug_image[discarded_white_mask] = 0
        debug_image[far_white_mask] = cls.MAGENTA_BGR
        timing['class_first_pass'] = (
            time.perf_counter() - stage) * 1000.0

        if not enable_second_distance_transform:
            return debug_image, debug_image

        stage = time.perf_counter()
        remaining_white_mask = cls._color_mask(
            debug_image, cls.WHITE_BGR, tolerance)
        magenta_mask = cls._color_mask(
            debug_image, cls.MAGENTA_BGR, tolerance)
        timing['class_masks_second'] = (
            time.perf_counter() - stage) * 1000.0

        stage = time.perf_counter()
        output_image = debug_image.copy()
        if np.any(magenta_mask):
            distance_input = np.full(image.shape[:2], 255, dtype=np.uint8)
            distance_input[magenta_mask] = 0
            distance_from_magenta = cv2.distanceTransform(
                distance_input,
                cv2.DIST_L2,
                cv2.DIST_MASK_3,
            )
            close_white_mask = remaining_white_mask & (
                distance_from_magenta < magenta)
            output_image[close_white_mask] = cls.MAGENTA_BGR
        timing['class_magenta_distance'] = (
            time.perf_counter() - stage) * 1000.0

        stage = time.perf_counter()
        output_image[discarded_white_mask] = 0
        timing['class_finalize'] = (
            time.perf_counter() - stage) * 1000.0
        return debug_image, output_image

    def _classify(self, bev, thresholds, wanted, timing, workload):
        """Classify on CPU when the CUDA pipeline is unavailable."""
        return self._classify_cpu(
            bev, thresholds, wanted, timing, workload)

    def _classify_cpu(self, bev, thresholds, wanted, timing, workload):
        """Classify the occupied BEV rectangle with OpenCV EDTs."""
        if not (wanted[1] or wanted[2]):
            return None, None
        started = time.perf_counter()
        stage = time.perf_counter()
        self.cpu_debug.fill(0)
        self.cpu_output.fill(0)
        timing['class_clear'] = (time.perf_counter() - stage) * 1000.0

        # All pixels outside this rectangle are black. Blue/magenta sources
        # and white targets are therefore entirely contained in the ROI, so
        # cropping does not alter any distance-transform result.
        stage = time.perf_counter()
        grayscale = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        timing['class_grayscale'] = (
            time.perf_counter() - stage) * 1000.0
        workload['bev_points'] = int(cv2.countNonZero(grayscale))

        stage = time.perf_counter()
        x, y, width, height = cv2.boundingRect(grayscale)
        timing['class_bounds'] = (
            time.perf_counter() - stage) * 1000.0
        workload['classification_roi_pixels'] = width * height
        if width == 0 or height == 0:
            timing['classification'] = (
                time.perf_counter() - started) * 1000.0
            return self.cpu_debug, self.cpu_output

        minimum, maximum, magenta, tolerance = thresholds
        roi = bev[y:y + height, x:x + width]
        debug_roi, output_roi = self._classify_images(
            roi, minimum, maximum, magenta, tolerance, timing,
            self.enable_second_distance_transform)
        stage = time.perf_counter()
        self.cpu_debug[y:y + height, x:x + width] = debug_roi
        self.cpu_output[y:y + height, x:x + width] = output_roi
        timing['class_copy_output'] = (
            time.perf_counter() - stage) * 1000.0
        timing['classification'] = (
            time.perf_counter() - started) * 1000.0
        return self.cpu_debug, self.cpu_output

    def _publish_requested(self, images, wanted, header, timing):
        outputs = (
            ('bev', images[0], wanted[0], self.bev_pub),
            ('debug', images[1], wanted[1], self.debug_pub),
            ('output', images[2], wanted[2], self.output_pub),
        )
        for name, image, requested, publisher in outputs:
            if not requested or not self.running or not rclpy.ok():
                continue
            stage = time.perf_counter()
            message = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            message.header = header
            timing[f'publish_{name}_convert'] = (
                time.perf_counter() - stage) * 1000.0
            stage = time.perf_counter()
            publisher.publish(message)
            timing[f'publish_{name}_send'] = (
                time.perf_counter() - stage) * 1000.0

    def _record_telemetry(self, timing, workload):
        self.processed += 1
        if not self.telemetry_enabled:
            return
        for name, value in timing.items():
            if np.isfinite(value):
                self.timings[name].append(value)
        for name, value in workload.items():
            if np.isfinite(value):
                self.workloads[name].append(value)
        if self.processed % self.telemetry_interval:
            return

        def average(name):
            values = self.timings[name]
            return float(np.mean(values)) if values else 0.0

        def percentile_95(name):
            values = self.timings[name]
            return float(np.percentile(values, 95)) if values else 0.0

        def workload_average(name):
            values = self.workloads[name]
            return f'{float(np.mean(values)):.0f}' if values else 'n/a'

        now = time.perf_counter()
        elapsed = max(now - self.last_report_at, 1e-9)
        input_rate = (
            self.received - self.last_report_received) / elapsed
        output_rate = (
            self.processed - self.last_report_processed) / elapsed
        dropped = self.dropped - self.last_report_dropped
        backend = (
            'CUDA BEV' if self.gpu_enabled else 'CPU BEV'
        )
        classification_backend = (
            'PyCUDA threshold kernels'
            if self.gpu_enabled else 'OpenCV CPU EDT')
        if self.gpu_enabled:
            classification_stages = (
                'Fused blue mask + thresholds '
                f'{average("class_blue_distance"):.3f} ms | '
                'fused magenta mask + threshold '
                f'{average("class_magenta_distance"):.3f} ms\n'
            )
        else:
            classification_stages = (
                f'Initial masks {average("class_masks_initial"):.3f} ms | '
                f'blue distance {average("class_blue_distance"):.3f} ms | '
                f'first pass {average("class_first_pass"):.3f} ms\n'
                f'Second masks {average("class_masks_second"):.3f} ms | '
                f'magenta distance '
                f'{average("class_magenta_distance"):.3f} ms | finalize '
                f'{average("class_finalize"):.3f} ms | copy output '
                f'{average("class_copy_output"):.3f} ms\n'
            )
        self.get_logger().info(
            '\n================ BEV+CLASSIFICATION PROFILE =================\n'
            f'Backend {backend} | input {input_rate:.1f} Hz | output '
            f'{output_rate:.1f} Hz | dropped {dropped}\n'
            f'Classification backend: {classification_backend}\n'
            '--- End-to-end [avg / p95] ---\n'
            f'Queue {average("queue"):.2f}/{percentile_95("queue"):.2f} ms | '
            f'total {average("total"):.2f}/{percentile_95("total"):.2f} ms | '
            f'RGB/depth age {average("rgb_age"):.1f}/'
            f'{average("depth_age"):.1f} ms\n'
            '--- Input preparation ---\n'
            f'CvBridge {average("conversion"):.3f} ms | depth resize '
            f'{average("resize"):.3f} ms | telemetry probe '
            f'{average("telemetry_probe"):.3f} ms\n'
            '--- CUDA pipeline (event time; wall includes submission + wait) ---\n'
            f'Allocation {average("gpu_allocation"):.3f} ms | host staging '
            f'RGB/depth {average("staging_rgb"):.3f}/'
            f'{average("staging_depth"):.3f} ms\n'
            f'Clear {average("gpu_clear"):.3f} ms | upload '
            f'{average("gpu_upload"):.3f} ms | project '
            f'{average("gpu_project"):.3f} ms | unpack '
            f'{average("gpu_unpack"):.3f} ms | download '
            f'{average("gpu_download"):.3f} ms | CUDA wall '
            f'{average("gpu_wall"):.3f} ms\n'
            '--- CPU BEV fallback ---\n'
            f'Sample {average("cpu_sample"):.3f} ms | valid mask '
            f'{average("cpu_valid_mask"):.3f} ms | nonzero '
            f'{average("cpu_nonzero"):.3f} ms | projection '
            f'{average("cpu_projection"):.3f} ms | rasterize '
            f'{average("cpu_rasterize"):.3f} ms\n'
            f'Point inflation {average("inflation"):.3f} ms\n'
            '--- Classification ---\n'
            f'Total {average("classification"):.3f} ms | clear '
            f'{average("class_clear"):.3f} ms | grayscale '
            f'{average("class_grayscale"):.3f} ms | bounds '
            f'{average("class_bounds"):.3f} ms\n'
            f'GPU classification upload/download '
            f'{average("class_gpu_upload"):.3f}/'
            f'{average("class_gpu_download"):.3f} ms\n'
            f'{classification_stages}'
            '--- ROS image publication ---\n'
            f'BEV convert/send {average("publish_bev_convert"):.3f}/'
            f'{average("publish_bev_send"):.3f} ms | debug '
            f'{average("publish_debug_convert"):.3f}/'
            f'{average("publish_debug_send"):.3f} ms | output '
            f'{average("publish_output_convert"):.3f}/'
            f'{average("publish_output_send"):.3f} ms | total '
            f'{average("publish"):.3f} ms\n'
            '--- Average workload ---\n'
            f'Input colored points {workload_average("input_points")} | '
            f'valid depth {workload_average("valid_depth_points")} | '
            f'projected {workload_average("projected_points")}\n'
            f'Inflated BEV points {workload_average("bev_points")} | '
            f'classification ROI '
            f'{workload_average("classification_roi_pixels")} px\n'
            '================================================================='
        )
        self.last_report_at = now
        self.last_report_received = self.received
        self.last_report_processed = self.processed
        self.last_report_dropped = self.dropped

    def destroy_node(self):
        self.running = False
        with self.frame_condition:
            self.frame_condition.notify_all()
        if self.worker.is_alive():
            self.worker.join(timeout=2.0)
        if not self.worker.is_alive():
            self._release_cuda()
        super().destroy_node()

    def _release_cuda(self):
        if self.cuda_context is None or self.cuda is None:
            return
        pushed = False
        try:
            self.cuda_context.push()
            pushed = True
            for name in (
                    'gpu_rgb', 'gpu_depth', 'gpu_bev_raw', 'gpu_bev',
                    'gpu_bev_inflated', 'gpu_debug', 'gpu_output'):
                allocation = getattr(self, name, None)
                if allocation is not None:
                    allocation.free()
                    setattr(self, name, None)
            self.project_kernel = None
            self.unpack_kernel = None
            self.inflation_cuda_kernel = None
            self.blue_threshold_kernel = None
            self.magenta_threshold_kernel = None
            self.cuda_events = None
            self.cuda_stream = None
            self.cuda_module = None
        except Exception as exc:
            self.get_logger().warn(f'CUDA cleanup warning: {exc}')
        finally:
            if pushed:
                self.cuda_context.pop()
            try:
                self.cuda_context.detach()
            except Exception:
                pass
            self.cuda_context = None


def main(args=None):
    rclpy.init(args=args)
    node = BevAndClassification()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
