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

from cv_package.classes import Classification


class BevAndClassification(Node):
    """Projects boundary pixels to BEV and classifies them in one process."""

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
        self.declare_parameter('projection_stride', 3)
        self.declare_parameter('blue_distance_threshold_px', 10.0)
        self.declare_parameter('blue_max_distance_threshold_px', 16.0)
        self.declare_parameter('magenta_distance_threshold_px', 10.0)
        self.declare_parameter('color_tolerance', 30)
        self.declare_parameter('use_gpu', True)
        self.declare_parameter('max_processing_fps', 12.0)
        self.declare_parameter('enable_telemetry', True)
        self.declare_parameter('telemetry_window_size', 60)
        self.declare_parameter('telemetry_log_interval_frames', 30)

        self.width = int(self.get_parameter('bev_width').value)
        self.height = int(self.get_parameter('bev_height').value)
        self.resolution = float(self.get_parameter('bev_resolution').value)
        self.stride = int(self.get_parameter('projection_stride').value)
        self.crop_y_min = float(
            self.get_parameter('input_crop_y_min').value)
        self.use_gpu = bool(self.get_parameter('use_gpu').value)
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
            'queue', 'conversion', 'resize', 'staging', 'upload',
            'bev_kernels', 'classification', 'download', 'publish',
            'total', 'rgb_age', 'depth_age',
        )
        self.timings = {
            name: deque(maxlen=self.telemetry_window)
            for name in timing_names
        }
        self.cpu_bev = np.zeros(
            (self.height, self.width, 3), dtype=np.uint8)
        self.cpu_debug = np.zeros_like(self.cpu_bev)
        self.cpu_output = np.zeros_like(self.cpu_bev)

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
        if self.use_gpu:
            self._initialize_cuda()

        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()
        self.get_logger().info(
            f'Combined BEV+classification ready: {self.width}x{self.height}, '
            f'{self.resolution:.3f} m/px, stride {self.stride}, backend '
            f'{"CUDA" if self.gpu_enabled else "CPU"}, '
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

            ''', no_extern_c=True)

            self.project_kernel = self.cuda_module.get_function(
                'project_to_bev')
            self.unpack_kernel = self.cuda_module.get_function('unpack_bev')
            self.cuda_stream = cuda.Stream()
            image_shape = (self.height, self.width, 3)
            image_bytes = int(np.prod(image_shape))
            self.host_bev = cuda.pagelocked_empty(image_shape, np.uint8)
            self.gpu_bev_raw = cuda.mem_alloc(
                self.width * self.height * np.dtype(np.uint32).itemsize)
            self.gpu_bev = cuda.mem_alloc(image_bytes)
            self.cuda_events = [cuda.Event() for _ in range(4)]
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

                    # Wait for the 12 FPS slot while callbacks keep replacing
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
                bev = self._process_cuda(
                    rgb, depth, fx, cx, timing)
                debug, output = self._classify_cpu(
                    bev, thresholds, wanted, timing)
                images = bev, debug, output
            except Exception as exc:
                self.gpu_enabled = False
                self.get_logger().error(
                    f'CUDA pipeline failed ({exc}); switching to CPU.')
                images = self._process_cpu(
                    rgb, depth, fx, cx, thresholds, wanted, timing)
        else:
            images = self._process_cpu(
                rgb, depth, fx, cx, thresholds, wanted, timing)

        stage = time.perf_counter()
        self._publish_requested(images, wanted, msg.header)
        timing['publish'] = (time.perf_counter() - stage) * 1000.0
        timing['total'] = (time.perf_counter() - started) * 1000.0
        self._record_telemetry(timing)

    def _process_cuda(self, rgb, depth, fx, cx, timing):
        height, width = rgb.shape[:2]
        self._ensure_gpu_inputs((height, width))
        stage = time.perf_counter()
        np.copyto(self.host_rgb, rgb, casting='unsafe')
        np.copyto(self.host_depth, depth, casting='unsafe')
        timing['staging'] = (time.perf_counter() - stage) * 1000.0

        cuda = self.cuda
        stream = self.cuda_stream
        start_event, upload_event, bev_event, end_event = \
            self.cuda_events
        pixels = self.width * self.height
        raster_bytes = pixels * np.dtype(np.uint32).itemsize
        block = 256
        sample_width = (width + self.stride - 1) // self.stride
        sample_height = (height + self.stride - 1) // self.stride
        project_grid = (sample_width * sample_height + block - 1) // block
        image_grid = (pixels + block - 1) // block
        start_event.record(stream)
        cuda.memset_d8_async(self.gpu_bev_raw, 0, raster_bytes, stream)
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
        self.unpack_kernel(
            self.gpu_bev_raw, self.gpu_bev,
            np.int32(self.width), np.int32(self.height),
            block=(block, 1, 1), grid=(image_grid, 1, 1), stream=stream,
        )
        bev_event.record(stream)

        cuda.memcpy_dtoh_async(self.host_bev, self.gpu_bev, stream)
        end_event.record(stream)
        stream.synchronize()

        timing['upload'] = start_event.time_till(upload_event)
        timing['bev_kernels'] = upload_event.time_till(bev_event)
        timing['download'] = bev_event.time_till(end_event)
        return self.host_bev

    def _process_cpu(
            self, rgb, depth, fx, cx, thresholds, wanted, timing):
        sampled_rgb = rgb[::self.stride, ::self.stride]
        sampled_depth = depth[::self.stride, ::self.stride]
        color_mask = np.bitwise_or.reduce(sampled_rgb, axis=2) != 0
        valid = (
            color_mask & np.isfinite(sampled_depth)
            & (sampled_depth > 0.1) & (sampled_depth < self.max_forward)
        )
        rows, cols = np.nonzero(valid)
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
        bev = self.cpu_bev
        bev.fill(0)
        if np.any(inside):
            indices = v[inside] * self.width + u[inside]
            bev.reshape(-1, 3)[indices] = sampled_rgb[rows, cols][inside]
        debug, output = self._classify_cpu(
            bev, thresholds, wanted, timing)
        return bev, debug, output

    def _classify_cpu(self, bev, thresholds, wanted, timing):
        """Classify only the occupied BEV rectangle with exact OpenCV EDT."""
        if not (wanted[1] or wanted[2]):
            return None, None
        started = time.perf_counter()
        self.cpu_debug.fill(0)
        self.cpu_output.fill(0)

        # All pixels outside this rectangle are black. Blue/magenta sources
        # and white targets are therefore entirely contained in the ROI, so
        # cropping does not alter any distance-transform result.
        grayscale = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        x, y, width, height = cv2.boundingRect(grayscale)
        if width == 0 or height == 0:
            timing['classification'] = (
                time.perf_counter() - started) * 1000.0
            return self.cpu_debug, self.cpu_output

        minimum, maximum, magenta, tolerance = thresholds
        roi = bev[y:y + height, x:x + width]
        debug_roi, output_roi = Classification.classify_images(
            roi,
            blue_distance_threshold_px=minimum,
            blue_max_distance_threshold_px=maximum,
            magenta_distance_threshold_px=magenta,
            color_tolerance=tolerance,
        )
        self.cpu_debug[y:y + height, x:x + width] = debug_roi
        self.cpu_output[y:y + height, x:x + width] = output_roi
        timing['classification'] = (
            time.perf_counter() - started) * 1000.0
        return self.cpu_debug, self.cpu_output

    def _publish_requested(self, images, wanted, header):
        for image, requested, publisher in zip(
                images, wanted,
                (self.bev_pub, self.debug_pub, self.output_pub)):
            if not requested or not self.running or not rclpy.ok():
                continue
            message = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            message.header = header
            publisher.publish(message)

    def _record_telemetry(self, timing):
        self.processed += 1
        if not self.telemetry_enabled:
            return
        for name, value in timing.items():
            if np.isfinite(value):
                self.timings[name].append(value)
        if self.processed % self.telemetry_interval:
            return

        def average(name):
            values = self.timings[name]
            return float(np.mean(values)) if values else 0.0

        now = time.perf_counter()
        elapsed = max(now - self.last_report_at, 1e-9)
        input_rate = (
            self.received - self.last_report_received) / elapsed
        output_rate = (
            self.processed - self.last_report_processed) / elapsed
        dropped = self.dropped - self.last_report_dropped
        backend = (
            'CUDA-BEV + CPU-classification'
            if self.gpu_enabled else 'CPU'
        )
        self.get_logger().info(
            '\n================ BEV+CLASSIFICATION =================\n'
            f'Backend {backend} | input {input_rate:.1f} Hz | output '
            f'{output_rate:.1f} Hz | dropped {dropped}\n'
            f'Queue {average("queue"):.2f} ms | conversion '
            f'{average("conversion"):.2f} ms | resize '
            f'{average("resize"):.2f} ms\n'
            f'GPU staging/upload {average("staging"):.2f}/'
            f'{average("upload"):.2f} ms | BEV kernels '
            f'{average("bev_kernels"):.2f} ms\n'
            f'CPU classification '
            f'{average("classification"):.2f} ms | GPU download '
            f'{average("download"):.2f} ms\n'
            f'Publish {average("publish"):.2f} ms | total '
            f'{average("total"):.2f} ms | RGB/depth age '
            f'{average("rgb_age"):.1f}/{average("depth_age"):.1f} ms\n'
            '======================================================='
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
                    'gpu_rgb', 'gpu_depth', 'gpu_bev_raw', 'gpu_bev'):
                allocation = getattr(self, name, None)
                if allocation is not None:
                    allocation.free()
                    setattr(self, name, None)
            self.project_kernel = None
            self.unpack_kernel = None
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
