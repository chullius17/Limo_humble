#!/usr/bin/env python3
"""Publish cropped depth images and fill missing values from a planar LUT.

Optimized hot path: the crop happens before any dtype conversion (a
zero-copy view over the raw message buffer, honoring row padding), the
resize runs on the crop's native dtype so INTER_NEAREST only ever moves
half-width samples, and the LUT fill happens in place in a single
preallocated buffer. The two JET debug views are computed and published
only when something is actually subscribed to them.
"""

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

import time
from collections import deque


class DepthCorrection(Node):
    """Create once and use a ground-plane depth lookup table."""

    # Encodings the zero-copy ingress path understands directly, mapped to
    # (numpy dtype, needs_millimetre_scale). Anything else falls back to
    # CvBridge on the full frame.
    _RAW_DTYPE = {
        '16UC1': (np.uint16, True),
        'mono16': (np.uint16, True),
        '32FC1': (np.float32, False),
    }
    _MM_TO_M = np.float32(0.001)

    def __init__(self):
        super().__init__('depth_correction')

        self.declare_parameter(
            'input_topic', '/depth_camera/depth/image_raw')
        self.declare_parameter(
            'camera_info_topic', '/depth_camera/depth/camera_info')
        self.declare_parameter(
            'output_topic',
            'limo/cv_package/depth_correction/image_jet/raw')
        self.declare_parameter(
            'corrected_output_topic',
            'limo/cv_package/depth_correction/image_jet_corrected/raw')
        self.declare_parameter(
            'depth_output_topic',
            'limo/cv_package/depth_correction/depth_corrected/raw')
        self.declare_parameter('plane_frame', 'camera_link')
        self.declare_parameter('output_width', 320)
        self.declare_parameter('output_height', 120)
        self.declare_parameter('min_depth_m', 0.1)
        self.declare_parameter('max_depth_m', 5.0)
        self.declare_parameter('calibration_frames', 30)
        self.declare_parameter('calibration_rows', 10)

        input_topic = self.get_parameter('input_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        output_topic = self.get_parameter('output_topic').value
        corrected_topic = self.get_parameter(
            'corrected_output_topic').value
        depth_output_topic = self.get_parameter('depth_output_topic').value
        self.plane_frame = self.get_parameter('plane_frame').value
        self.output_width = int(self.get_parameter('output_width').value)
        self.output_height = int(self.get_parameter('output_height').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.calibration_frames = int(
            self.get_parameter('calibration_frames').value)
        self.calibration_rows = int(
            self.get_parameter('calibration_rows').value)

        if self.output_width <= 0 or self.output_height <= 0:
            raise ValueError('Output dimensions must be greater than zero')
        if self.max_depth_m <= self.min_depth_m:
            raise ValueError('max_depth_m must be greater than min_depth_m')
        if self.calibration_frames <= 0:
            raise ValueError('calibration_frames must be greater than zero')
        if self.calibration_rows <= 0:
            raise ValueError('calibration_rows must be greater than zero')

        # How often (in frames) to refresh the cached subscriber-count check
        # for the two JET debug topics, instead of querying it every frame.
        self.declare_parameter('debug_probe_interval_frames', 30)
        self.debug_probe_interval_frames = max(
            1, int(self.get_parameter('debug_probe_interval_frames').value))
        self._publish_raw_jet = False
        self._publish_corrected_jet = False

        # Telemetry control parameters, off by default: this node ran with
        # no visibility at all before this pass.
        self.declare_parameter('enable_telemetry', False)
        self.debug_telemetry = bool(
            self.get_parameter('enable_telemetry').value)
        self.declare_parameter('telemetry_window_size', 30)
        self.window_size = max(
            1, int(self.get_parameter('telemetry_window_size').value))
        self.declare_parameter('telemetry_log_interval_frames', 30)
        self.telemetry_log_interval_frames = max(
            1, int(self.get_parameter(
                'telemetry_log_interval_frames').value))
        self.add_on_set_parameters_callback(self._on_set_parameters)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        output_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.bridge = CvBridge()
        self.camera_info = None
        self.ray_plane_denominator = None
        self.optical_to_plane_translation_z = None
        self.geometry_signature = None
        self.plane_height_samples = []
        self.depth_lut = None
        self.lut_valid_u8 = None
        self.lut_zero_filled = None

        self.last_logged_ingress_path = None

        # Reusable working buffers, sized once from the (fixed) output
        # dimensions. `buf_resized` is filled in place by the LUT step, so
        # it is both "resized" and "completed" from the original code at
        # different points of one call. Nothing here crosses a thread
        # boundary: this node has no worker threads, one executor callback
        # does the whole chain, and `cv2_to_imgmsg`/CvBridge copy the data
        # out (via `tobytes()`) before the callback returns, so the next
        # invocation is free to overwrite these buffers.
        out_shape = (self.output_height, self.output_width)
        self._buf_resized_u16 = np.empty(out_shape, dtype=np.uint16)
        self.buf_resized = np.empty(out_shape, dtype=np.float32)
        self._buf_fill_mask = np.empty(out_shape, dtype=np.uint8)
        self._buf_col_safe = np.empty(out_shape, dtype=np.float32)
        self._buf_col_u8 = np.empty(out_shape, dtype=np.uint8)
        self._buf_bgr = np.empty(
            (self.output_height, self.output_width, 3), dtype=np.uint8)
        self._col_scale = 255.0 / (self.max_depth_m - self.min_depth_m)

        # Telemetry metrics window.
        self.telemetry_stats = {
            'ingress': deque(maxlen=self.window_size),
            'resize_convert': deque(maxlen=self.window_size),
            'lut_fill': deque(maxlen=self.window_size),
            'depth_publish': deque(maxlen=self.window_size),
            'raw_jet': deque(maxlen=self.window_size),
            'corrected_jet': deque(maxlen=self.window_size),
            'total_callback': deque(maxlen=self.window_size),
        }
        self.frame_counter = 0
        self.frames_received = 0
        self._ingress_arrival_times = deque(maxlen=self.window_size + 1)
        self._last_report_wall_time = time.perf_counter()
        self._last_report_frame_counter = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.publisher = self.create_publisher(
            Image, output_topic, output_qos)
        self.corrected_publisher = self.create_publisher(
            Image, corrected_topic, output_qos)
        # Consumers of this topic (simple_boundaries.py, bev_and_clas.py)
        # already subscribe BEST_EFFORT; a reliable writer here can only
        # block on its single history slot for no benefit, and a stale
        # depth frame is worth less than a dropped one on a sensor stream.
        self.depth_publisher = self.create_publisher(
            Image, depth_output_topic, sensor_qos)
        self.info_subscription = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            sensor_qos,
        )
        self.depth_subscription = self.create_subscription(
            Image, input_topic, self.depth_callback, sensor_qos)

        self.get_logger().info(
            f'Listening on {input_topic}; publishing metric depth on '
            f'{depth_output_topic} and JET views on {output_topic} and '
            f'{corrected_topic} at {self.output_width}x{self.output_height}. '
            f'Waiting for {self.calibration_frames} calibration frames.')

    def _on_set_parameters(self, params):
        """Apply runtime parameter changes without a per-frame lookup."""
        for param in params:
            if param.name == 'enable_telemetry':
                self.debug_telemetry = bool(param.value)
        return SetParametersResult(successful=True)

    def camera_info_callback(self, msg):
        """Store the latest intrinsic calibration until the LUT is frozen."""
        if self.depth_lut is None:
            self.camera_info = msg

    @staticmethod
    def quaternion_to_rotation(quaternion):
        """Return a 3x3 rotation matrix for a geometry_msgs quaternion."""
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
        ], dtype=np.float64)

    def prepare_geometry(self, msg, input_height, input_width):
        """Precompute each output ray's coefficient along camera_link z."""
        info = self.camera_info
        if info is None or info.width == 0 or info.height == 0:
            self.get_logger().warning(
                'Waiting for depth CameraInfo', throttle_duration_sec=2.0)
            return False

        optical_frame = msg.header.frame_id
        if not optical_frame:
            self.get_logger().warning(
                'Depth image has an empty frame_id',
                throttle_duration_sec=2.0)
            return False

        signature = (
            optical_frame,
            input_height,
            input_width,
            info.width,
            info.height,
            tuple(info.k),
        )
        if self.geometry_signature == signature:
            return True

        try:
            transform = self.tf_buffer.lookup_transform(
                self.plane_frame, optical_frame, Time())
        except TransformException as error:
            self.get_logger().warning(
                f'Waiting for TF {optical_frame} -> {self.plane_frame}: '
                f'{error}',
                throttle_duration_sec=2.0,
            )
            return False

        scale_x = input_width / float(info.width)
        scale_y = input_height / float(info.height)
        fx = info.k[0] * scale_x
        fy = info.k[4] * scale_y
        cx = info.k[2] * scale_x
        cy = info.k[5] * scale_y
        if fx == 0.0 or fy == 0.0:
            self.get_logger().error('Invalid focal length in CameraInfo')
            return False

        crop_start = input_height // 2
        crop_height = input_height - crop_start
        output_u, output_v = np.meshgrid(
            np.arange(self.output_width, dtype=np.float64),
            np.arange(self.output_height, dtype=np.float64),
        )

        # Pixel-centre mapping equivalent to OpenCV's resize operation.
        source_u = (
            (output_u + 0.5) * input_width / self.output_width - 0.5)
        source_v = (
            crop_start
            + (output_v + 0.5) * crop_height / self.output_height
            - 0.5
        )
        rays = np.stack((
            (source_u - cx) / fx,
            (source_v - cy) / fy,
            np.ones_like(source_u),
        ), axis=0).reshape(3, -1)

        rotation = self.quaternion_to_rotation(
            transform.transform.rotation)
        # For p_plane = R * (depth * ray) + t, this is the coefficient
        # multiplying depth in the z coordinate of camera_link.
        denominator = (rotation[2, :] @ rays).reshape(
            self.output_height, self.output_width)

        self.ray_plane_denominator = denominator
        self.optical_to_plane_translation_z = (
            transform.transform.translation.z)
        self.geometry_signature = signature
        self.plane_height_samples.clear()
        self.get_logger().info(
            f'Geometry ready from {optical_frame} to {self.plane_frame}; '
            'starting offline LUT calibration.')
        return True

    def update_offline_calibration(
            self, depth, msg, input_height, input_width):
        """Estimate the plane once from valid pixels in the bottom rows."""
        if self.depth_lut is not None:
            return
        if not self.prepare_geometry(msg, input_height, input_width):
            return

        row_count = min(self.calibration_rows, depth.shape[0])
        bottom_depth = depth[-row_count:, :]
        bottom_denominator = self.ray_plane_denominator[-row_count:, :]
        valid = (
            np.isfinite(bottom_depth)
            & (bottom_depth >= self.min_depth_m)
            & (bottom_depth <= self.max_depth_m)
            & (np.abs(bottom_denominator) > 1e-8)
        )
        if not np.any(valid):
            self.get_logger().warning(
                f'No valid depth sample in the bottom {row_count} rows for '
                'calibration',
                throttle_duration_sec=2.0)
            return

        # Every point in the bottom rows votes for the plane coordinate z in
        # camera_link. A median rejects isolated obstacles and depth outliers.
        plane_z = (
            self.optical_to_plane_translation_z
            + bottom_depth[valid] * bottom_denominator[valid]
        )
        self.plane_height_samples.append(float(np.median(plane_z)))

        if len(self.plane_height_samples) < self.calibration_frames:
            return

        plane_height = float(np.median(self.plane_height_samples))
        numerator = plane_height - self.optical_to_plane_translation_z
        denominator = self.ray_plane_denominator
        lut = np.full(denominator.shape, np.nan, dtype=np.float32)
        usable = np.abs(denominator) > 1e-8
        lut[usable] = (numerator / denominator[usable]).astype(np.float32)
        lut[(lut < self.min_depth_m) | (lut > self.max_depth_m)] = np.nan

        # Freeze the table: no plane fitting or ray intersection is performed
        # again in the live processing path. The valid/zero-filled views are
        # precomputed once here too, so the per-frame fill never recomputes
        # `np.isfinite(self.depth_lut)`.
        self.depth_lut = lut
        self.lut_valid_u8 = (
            np.isfinite(lut).astype(np.uint8) * np.uint8(255))
        self.lut_zero_filled = np.where(
            np.isfinite(lut), lut, np.float32(0.0)).astype(np.float32)
        valid_entries = int(np.count_nonzero(np.isfinite(lut)))
        self.get_logger().info(
            f'Offline LUT frozen: plane z={plane_height:.4f} m in '
            f'{self.plane_frame}, {valid_entries}/{lut.size} valid entries.')

    def _view_source(self, msg):
        """Return (view, needs_mm_scale) without converting the full frame.

        For '16UC1'/'mono16'/'32FC1' this is a zero-copy view over
        msg.data honoring msg.step (row padding). Any other encoding falls
        back to CvBridge on the full frame, logged once.
        """
        info = self._RAW_DTYPE.get(msg.encoding)
        if info is None:
            try:
                depth = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding='passthrough')
            except CvBridgeError as error:
                raise ValueError(f'Cannot convert depth image: {error}')
            self._log_ingress_path('cvbridge-fallback', msg)
            return depth, depth.dtype == np.uint16

        dtype, needs_scale = info
        try:
            itemsize = np.dtype(dtype).itemsize
            width, height, step = msg.width, msg.height, msg.step
            row_bytes = width * itemsize
            flat = np.frombuffer(msg.data, dtype=dtype)
            if step == row_bytes:
                view = flat.reshape(height, width)
            else:
                row_elems = step // itemsize
                view = flat.reshape(height, row_elems)[:, :width]
        except (ValueError, TypeError) as error:
            raise ValueError(f'Cannot view depth buffer: {error}')

        self._log_ingress_path('zero-copy', msg)
        return view, needs_scale

    def _log_ingress_path(self, path, msg):
        if path != self.last_logged_ingress_path:
            self.get_logger().info(
                f"Ingress path: {path} (encoding='{msg.encoding}', "
                f'width={msg.width}, step={msg.step})'
            )
            self.last_logged_ingress_path = path

    def _resize_into(self, depth_view, needs_scale):
        """Crop and resize `depth_view` into `self.buf_resized` in place.

        The crop and resize happen on the source dtype first; INTER_NEAREST
        is a pure per-pixel selection, so scaling uint16 millimetres to
        float32 metres before or after resizing produces the identical
        value at every output pixel. Resizing the smaller uint16 crop
        (rather than a pre-converted float32 copy of it) halves the bytes
        the resize has to move.
        """
        crop_start = depth_view.shape[0] // 2
        lower = depth_view[crop_start:, :]
        out_size = (self.output_width, self.output_height)
        if needs_scale:
            cv2.resize(
                lower, out_size, dst=self._buf_resized_u16,
                interpolation=cv2.INTER_NEAREST)
            np.multiply(
                self._buf_resized_u16, self._MM_TO_M,
                out=self.buf_resized, casting='unsafe')
        else:
            cv2.resize(
                lower, out_size, dst=self.buf_resized,
                interpolation=cv2.INTER_NEAREST)
        return self.buf_resized

    def _colorize_into(self, depth, out_bgr):
        """Write the JET visualization of `depth` into `out_bgr` in place.

        `depth` is read only: this is called on `buf_resized` both before
        and after the LUT fill, and must not disturb it. The arithmetic is
        plain NumPy on preallocated scratch buffers so it stays bit-for-bit
        identical to the original allocating implementation; only
        `cv2.applyColorMap` (deterministic regardless of `dst`) is done
        with OpenCV.
        """
        valid = np.isfinite(depth) & (depth > 0.0)
        np.copyto(self._buf_col_safe, depth)
        self._buf_col_safe[~valid] = self.min_depth_m
        np.clip(
            self._buf_col_safe, self.min_depth_m, self.max_depth_m,
            out=self._buf_col_safe)
        np.subtract(
            self._buf_col_safe, self.min_depth_m, out=self._buf_col_safe)
        np.multiply(
            self._buf_col_safe, self._col_scale, out=self._buf_col_safe)
        self._buf_col_u8[:] = self._buf_col_safe.astype(np.uint8)
        cv2.applyColorMap(self._buf_col_u8, cv2.COLORMAP_JET, dst=out_bgr)
        out_bgr[~valid] = 0
        return out_bgr

    def publish_image(self, publisher, image, header):
        """Publish one BGR visualization preserving the source header."""
        output_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        output_msg.header = header
        publisher.publish(output_msg)

    def publish_depth(self, depth, header):
        """Publish corrected metric depth as a 32-bit floating-point image."""
        output_msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        output_msg.header = header
        self.depth_publisher.publish(output_msg)

    def _fill_from_lut(self, completed):
        """Fill unfillable-free pixels of `completed` in place from the LUT."""
        # `inRange` reproduces `isfinite(x) & (x > 0)` in one pass: NaN
        # fails every comparison, and +inf is excluded by the FLT_MAX
        # upper bound, matching the original's `~isfinite(...)` treatment
        # of both NaN and +/-inf as missing.
        cv2.inRange(
            completed,
            (float(np.nextafter(np.float32(0), np.float32(1))),),
            (float(np.finfo(np.float32).max),),
            dst=self._buf_fill_mask,
        )
        cv2.bitwise_not(self._buf_fill_mask, dst=self._buf_fill_mask)
        cv2.bitwise_and(
            self._buf_fill_mask, self.lut_valid_u8, dst=self._buf_fill_mask)
        cv2.copyTo(self.lut_zero_filled, self._buf_fill_mask, completed)

    def depth_callback(self, msg):
        """Crop, resize, calibrate if needed, then publish both views."""
        self.frames_received += 1
        t_start = time.perf_counter() if self.debug_telemetry else 0.0
        if self.debug_telemetry:
            self._ingress_arrival_times.append(t_start)

        t0 = time.perf_counter() if self.debug_telemetry else 0.0
        try:
            depth_view, needs_scale = self._view_source(msg)
        except ValueError as error:
            self.get_logger().error(str(error))
            return

        if depth_view.ndim != 2 or depth_view.shape[0] < 2:
            self.get_logger().warning(
                f'Invalid depth image shape: {depth_view.shape}',
                throttle_duration_sec=2.0)
            return
        if self.debug_telemetry:
            self.telemetry_stats['ingress'].append(
                (time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter() if self.debug_telemetry else 0.0
        resized = self._resize_into(depth_view, needs_scale)
        if self.debug_telemetry:
            self.telemetry_stats['resize_convert'].append(
                (time.perf_counter() - t0) * 1000.0)

        if self.frame_counter % self.debug_probe_interval_frames == 0:
            self._publish_raw_jet = (
                self.publisher.get_subscription_count() > 0)
            self._publish_corrected_jet = (
                self.corrected_publisher.get_subscription_count() > 0)

        if self._publish_raw_jet:
            t0 = time.perf_counter() if self.debug_telemetry else 0.0
            self.publish_image(
                self.publisher, self._colorize_into(resized, self._buf_bgr),
                msg.header)
            if self.debug_telemetry:
                self.telemetry_stats['raw_jet'].append(
                    (time.perf_counter() - t0) * 1000.0)

        self.update_offline_calibration(
            resized, msg, depth_view.shape[0], depth_view.shape[1])

        # From here `resized` (== self.buf_resized) is filled in place and
        # becomes "completed"; the raw JET view and the calibration update
        # above both needed the pre-fill values, so this order must not
        # change.
        t0 = time.perf_counter() if self.debug_telemetry else 0.0
        if self.depth_lut is not None:
            self._fill_from_lut(resized)
        if self.debug_telemetry:
            self.telemetry_stats['lut_fill'].append(
                (time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter() if self.debug_telemetry else 0.0
        self.publish_depth(resized, msg.header)
        if self.debug_telemetry:
            self.telemetry_stats['depth_publish'].append(
                (time.perf_counter() - t0) * 1000.0)

        if self._publish_corrected_jet:
            t0 = time.perf_counter() if self.debug_telemetry else 0.0
            self.publish_image(
                self.corrected_publisher,
                self._colorize_into(resized, self._buf_bgr), msg.header)
            if self.debug_telemetry:
                self.telemetry_stats['corrected_jet'].append(
                    (time.perf_counter() - t0) * 1000.0)

        self.frame_counter += 1
        if self.debug_telemetry:
            self.telemetry_stats['total_callback'].append(
                (time.perf_counter() - t_start) * 1000.0)
            if self.frame_counter % self.telemetry_log_interval_frames == 0:
                self._log_telemetry_report()

    def _log_telemetry_report(self):
        """Print sliding-window per-stage timing and input/throughput rate."""
        def avg(key):
            values = self.telemetry_stats[key]
            return float(np.mean(values)) if values else 0.0

        now = time.perf_counter()
        report_span = now - self._last_report_wall_time
        frames_since_report = (
            self.frame_counter - self._last_report_frame_counter)
        measured_fps = (
            frames_since_report / report_span if report_span > 0 else 0.0)
        self._last_report_wall_time = now
        self._last_report_frame_counter = self.frame_counter

        input_hz = 0.0
        if len(self._ingress_arrival_times) >= 2:
            span = (
                self._ingress_arrival_times[-1]
                - self._ingress_arrival_times[0])
            if span > 0:
                input_hz = (len(self._ingress_arrival_times) - 1) / span

        self.get_logger().info(
            "\n"
            f"====== DEPTH CORRECTION PERFORMANCE REPORT (AVG "
            f"{self.window_size} frames) ======\n"
            f"  Frames Processed: {self.frame_counter} / Received: "
            f"{self.frames_received}\n"
            "-----------------------------------------\n"
            "[RATE]\n"
            f" Input rate:                 {input_hz:.1f} Hz\n"
            f" Measured throughput:        {measured_fps:.1f} FPS\n"
            "-----------------------------------------\n"
            "[EXECUTION BREAKDOWN]\n"
            f" Ingress (view/CvBridge):    {avg('ingress'):.3f} ms\n"
            f" Crop+resize+convert:        {avg('resize_convert'):.3f} ms\n"
            f" LUT fill:                   {avg('lut_fill'):.3f} ms\n"
            f" Depth publish:              {avg('depth_publish'):.3f} ms\n"
            f" Raw JET (when subscribed):  {avg('raw_jet'):.3f} ms\n"
            f" Corrected JET (when sub.):  {avg('corrected_jet'):.3f} ms\n"
            "-----------------------------------------\n"
            f" TOTAL CALLBACK TIME:        {avg('total_callback'):.3f} ms\n"
            "=========================================\n"
        )


def main(args=None):
    rclpy.init(args=args)
    node = DepthCorrection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
