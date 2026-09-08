#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import cv2
from cv_bridge import CvBridge
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
        self.camera_info_width = None
        self.camera_info_height = None
        self.last_projection_geometry = None
        self.x_factor_lut = None

        # The 600x300 canvas at 1 cm/pixel covers an area of 6x3 m.
        self.declare_parameter('bev_width', 600)
        self.declare_parameter('bev_height', 300)
        self.declare_parameter('bev_resolution', 0.01)
        self.declare_parameter('projection_stride', 2)
        self.declare_parameter('use_gpu', True)
        self.width = int(self.get_parameter('bev_width').value)
        self.height = int(self.get_parameter('bev_height').value)
        self.res = float(self.get_parameter('bev_resolution').value)
        self.projection_stride = int(
            self.get_parameter('projection_stride').value)
        self.use_gpu = bool(self.get_parameter('use_gpu').value)
        if self.width <= 0 or self.height <= 0 or self.res <= 0.0:
            raise ValueError('BEV dimensions and resolution must be positive')
        if self.projection_stride <= 0:
            raise ValueError('projection_stride must be positive')
        self.max_forward_distance = self.height * self.res
        self.kernel_inflate = np.ones((3, 3), dtype=np.uint8)
        self.bev_buffer = np.zeros(
            (self.height, self.width, 3), dtype=np.uint8)

        # CUDA is initialized lazily and is completely optional.  Keeping the
        # imports out of module scope lets the same node run on machines where
        # PyCUDA is not installed, using the existing NumPy/OpenCV path.
        self.gpu_enabled = False
        self.cuda = None
        self.cuda_context = None
        self.cuda_module = None
        self.cuda_stream = None
        self.gpu_input_shape = None
        self.host_rgb_pinned = None
        self.host_depth_pinned = None
        self.host_bev_pinned = None
        self.host_counts_pinned = None
        self.gpu_rgb_input = None
        self.gpu_depth_input = None
        self.gpu_bev_raw = None
        self.gpu_bev_dilated = None
        self.gpu_counts = None

        # QoS Profiles setup to avoid mismatched subscription errors
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        pipeline_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ROS 2 Subscriptions
        self.rgb_sub = self.create_subscription(
            Image,
            'limo/cv_package/boundaries/lines_and_curbs/raw',
            self.rgb_callback,
            pipeline_qos,
        )

        self.declare_parameter('camera_info_topic', '/rgb/camera_info')
        self.declare_parameter(
            'depth_topic',
            'limo/cv_package/depth_correction/depth_corrected/raw')
        self.declare_parameter('input_crop_y_min', 0.5)
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.input_crop_y_min = float(
            self.get_parameter('input_crop_y_min').value)
        if not 0.0 <= self.input_crop_y_min < 1.0:
            raise ValueError('input_crop_y_min must be in [0.0, 1.0)')

        self.info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, sensor_qos
        )

        # ROS 2 Publisher
        self.bird_pub = self.create_publisher(
            Image,
            'limo/cv_package/bev/bird_perspective/raw',
            pipeline_qos,
        )

        # Queue and worker thread setup for multithreaded processing
        self.rgb_queue = queue.Queue(maxsize=1)

        # Telemetry & Diagnostics setup
        self.declare_parameter('enable_telemetry', True)
        self.declare_parameter('telemetry_window_size', 60)
        self.declare_parameter('telemetry_log_interval_frames', 30)
        self.debug_telemetry = self.get_parameter('enable_telemetry').value
        self.window_size = int(
            self.get_parameter('telemetry_window_size').value)
        self.log_interval = int(
            self.get_parameter('telemetry_log_interval_frames').value)
        if self.window_size <= 0 or self.log_interval <= 0:
            raise ValueError('Telemetry window and interval must be positive')

        self.frame_count = 0
        self.rgb_received_count = 0
        self.rgb_dropped_count = 0
        self.depth_received_count = 0
        self.last_report_time = time.perf_counter()
        self.last_report_frame_count = 0
        self.last_report_rgb_count = 0
        self.last_report_depth_count = 0
        self.last_report_dropped_count = 0
        self.depth_received_at = None
        self.is_running = True
        self.telemetry_stats = {
            'queue_wait': deque(maxlen=self.window_size),
            'conversion': deque(maxlen=self.window_size),
            'geometry': deque(maxlen=self.window_size),
            'depth_resize': deque(maxlen=self.window_size),
            'mask_extract': deque(maxlen=self.window_size),
            'gather_filter': deque(maxlen=self.window_size),
            'projection': deque(maxlen=self.window_size),
            'rasterize': deque(maxlen=self.window_size),
            'dilate': deque(maxlen=self.window_size),
            'gpu_staging': deque(maxlen=self.window_size),
            'gpu_upload': deque(maxlen=self.window_size),
            'gpu_kernel': deque(maxlen=self.window_size),
            'gpu_download': deque(maxlen=self.window_size),
            'gpu_pipeline': deque(maxlen=self.window_size),
            'bev_pipeline': deque(maxlen=self.window_size),
            'publish': deque(maxlen=self.window_size),
            'total': deque(maxlen=self.window_size),
            'message_age': deque(maxlen=self.window_size),
            'depth_age': deque(maxlen=self.window_size),
            'depth_conversion': deque(maxlen=self.window_size),
        }
        self.telemetry_counts = {
            'colored': deque(maxlen=self.window_size),
            'valid_depth': deque(maxlen=self.window_size),
            'projected': deque(maxlen=self.window_size),
        }

        if self.use_gpu:
            self._initialize_gpu()

        self.worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        self.get_logger().info(
            f'BirdPerspective initialized: {self.width}x{self.height}, '
            f'{self.res:.3f} m/px, '
            f'{self.max_forward_distance:.2f} m forward range, '
            f'projection stride {self.projection_stride}, '
            f'backend {"CUDA" if self.gpu_enabled else "CPU"}.')

    def _initialize_gpu(self):
        """Compile CUDA kernels and allocate persistent BEV buffers."""
        context_pushed = False
        try:
            import pycuda.driver as cuda
            from pycuda.compiler import SourceModule

            cuda.init()
            if cuda.Device.count() == 0:
                raise RuntimeError('no CUDA device found')

            self.cuda = cuda
            device = cuda.Device(0)
            if hasattr(device, 'retain_primary_context'):
                self.cuda_context = device.retain_primary_context()
                self.cuda_context.push()
            else:
                # Compatibility with older PyCUDA releases shipped on some
                # JetPack images. make_context() also makes it current.
                self.cuda_context = device.make_context()
            context_pushed = True

            self.cuda_module = SourceModule(r'''
                #include <math.h>

                extern "C" __global__ void project_to_bev(
                    const unsigned char *__restrict__ rgb,
                    const float *__restrict__ depth,
                    unsigned char *__restrict__ bev,
                    unsigned int *__restrict__ counts,
                    int input_width,
                    int input_height,
                    int stride,
                    float fx,
                    float cx,
                    int bev_width,
                    int bev_height,
                    float resolution,
                    float max_forward)
                {
                    // Accumulate telemetry per block. This reduces global
                    // atomics from one per point to at most three per block.
                    __shared__ unsigned int block_counts[3];
                    if (threadIdx.x < 3) {
                        block_counts[threadIdx.x] = 0;
                    }
                    __syncthreads();

                    const int sample_width =
                        (input_width + stride - 1) / stride;
                    const int sample_height =
                        (input_height + stride - 1) / stride;
                    const int sample_idx =
                        blockIdx.x * blockDim.x + threadIdx.x;
                    if (sample_idx < sample_width * sample_height) {
                        const int sample_x = sample_idx % sample_width;
                        const int sample_y = sample_idx / sample_width;
                        const int x = sample_x * stride;
                        const int y = sample_y * stride;
                        const int src_idx = y * input_width + x;
                        const int rgb_idx = src_idx * 3;
                        const unsigned char blue = rgb[rgb_idx];
                        const unsigned char green = rgb[rgb_idx + 1];
                        const unsigned char red = rgb[rgb_idx + 2];

                        if ((blue | green | red) != 0) {
                            atomicAdd(&block_counts[0], 1U);
                            const float z = depth[src_idx];

                            if (isfinite(z) && z > 0.1f &&
                                z < max_forward) {
                                atomicAdd(&block_counts[1], 1U);

                                // Optical frame: X right, Z forward. Robot:
                                // X forward = Z_cam, Y left = -X_cam.
                                const float x_cam =
                                    ((float)x - cx) * z / fx;
                                const int u_bev = __float2int_rz(
                                    0.5f * (float)bev_width
                                    + x_cam / resolution);
                                const int v_bev = __float2int_rz(
                                    (float)(bev_height - 1)
                                    - z / resolution);

                                if (u_bev >= 0 && u_bev < bev_width &&
                                    v_bev >= 0 && v_bev < bev_height) {
                                    atomicAdd(&block_counts[2], 1U);

                                    // Collisions are harmless here: the CPU
                                    // path also retains one source color.
                                    const int dst_idx =
                                        (v_bev * bev_width + u_bev) * 3;
                                    bev[dst_idx] = blue;
                                    bev[dst_idx + 1] = green;
                                    bev[dst_idx + 2] = red;
                                }
                            }
                        }
                    }

                    __syncthreads();
                    if (threadIdx.x < 3 &&
                        block_counts[threadIdx.x] != 0) {
                        atomicAdd(
                            &counts[threadIdx.x],
                            block_counts[threadIdx.x]);
                    }
                }

                extern "C" __global__ void dilate_bev_3x3(
                    const unsigned char *__restrict__ input,
                    unsigned char *__restrict__ output,
                    int width,
                    int height)
                {
                    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    const int pixels = width * height;
                    if (idx >= pixels) {
                        return;
                    }

                    const int x = idx % width;
                    const int y = idx / width;
                    unsigned char max_b = 0;
                    unsigned char max_g = 0;
                    unsigned char max_r = 0;

                    #pragma unroll
                    for (int dy = -1; dy <= 1; ++dy) {
                        const int ny = y + dy;
                        if (ny < 0 || ny >= height) continue;
                        #pragma unroll
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int nx = x + dx;
                            if (nx < 0 || nx >= width) continue;
                            const int src = (ny * width + nx) * 3;
                            max_b = max(max_b, input[src]);
                            max_g = max(max_g, input[src + 1]);
                            max_r = max(max_r, input[src + 2]);
                        }
                    }

                    const int dst = idx * 3;
                    output[dst] = max_b;
                    output[dst + 1] = max_g;
                    output[dst + 2] = max_r;
                }
            ''', no_extern_c=True)

            self.gpu_projection_kernel = self.cuda_module.get_function(
                'project_to_bev')
            self.gpu_dilate_kernel = self.cuda_module.get_function(
                'dilate_bev_3x3')
            self.cuda_stream = cuda.Stream()

            bev_shape = (self.height, self.width, 3)
            bev_nbytes = int(np.prod(bev_shape))
            self.host_bev_pinned = cuda.pagelocked_empty(
                bev_shape, dtype=np.uint8)
            self.host_counts_pinned = cuda.pagelocked_empty(
                3, dtype=np.uint32)
            self.gpu_bev_raw = cuda.mem_alloc(bev_nbytes)
            self.gpu_bev_dilated = cuda.mem_alloc(bev_nbytes)
            self.gpu_counts = cuda.mem_alloc(3 * np.dtype(np.uint32).itemsize)

            self.gpu_events = [cuda.Event() for _ in range(4)]
            self.gpu_enabled = True
            self.get_logger().info(
                f'CUDA BEV enabled on {device.name()}')
        except Exception as exc:
            self.gpu_enabled = False
            self.get_logger().warn(
                f'CUDA BEV unavailable ({exc}); using CPU fallback.')
        finally:
            if context_pushed:
                self.cuda_context.pop()

    def _ensure_gpu_input_buffers(self, shape):
        """Allocate page-locked host and device inputs when shape changes."""
        if shape == self.gpu_input_shape:
            return

        height, width = shape
        for allocation_name in ('gpu_rgb_input', 'gpu_depth_input'):
            allocation = getattr(self, allocation_name)
            if allocation is not None:
                allocation.free()

        self.host_rgb_pinned = self.cuda.pagelocked_empty(
            (height, width, 3), dtype=np.uint8)
        self.host_depth_pinned = self.cuda.pagelocked_empty(
            (height, width), dtype=np.float32)
        self.gpu_rgb_input = self.cuda.mem_alloc(
            self.host_rgb_pinned.nbytes)
        self.gpu_depth_input = self.cuda.mem_alloc(
            self.host_depth_pinned.nbytes)
        self.gpu_input_shape = shape

    def _process_gpu(self, rgb_img, depth_img, fx, cx, timings, counts):
        """Run fused projection/rasterization and dilation on CUDA."""
        height, width = rgb_img.shape[:2]
        self._ensure_gpu_input_buffers((height, width))

        stage_start = time.perf_counter()
        np.copyto(self.host_rgb_pinned, rgb_img, casting='unsafe')
        np.copyto(self.host_depth_pinned, depth_img, casting='unsafe')
        timings['gpu_staging'] = (
            time.perf_counter() - stage_start) * 1000.0

        cuda = self.cuda
        stream = self.cuda_stream
        event_start, event_upload, event_kernel, event_download = \
            self.gpu_events
        bev_nbytes = self.width * self.height * 3

        event_start.record(stream)
        cuda.memset_d8_async(self.gpu_bev_raw, 0, bev_nbytes, stream)
        cuda.memset_d32_async(self.gpu_counts, 0, 3, stream)
        cuda.memcpy_htod_async(
            self.gpu_rgb_input, self.host_rgb_pinned, stream)
        cuda.memcpy_htod_async(
            self.gpu_depth_input, self.host_depth_pinned, stream)
        event_upload.record(stream)

        sample_width = (
            width + self.projection_stride - 1
        ) // self.projection_stride
        sample_height = (
            height + self.projection_stride - 1
        ) // self.projection_stride
        block_size = 256
        projection_grid = (
            sample_width * sample_height + block_size - 1
        ) // block_size
        self.gpu_projection_kernel(
            self.gpu_rgb_input,
            self.gpu_depth_input,
            self.gpu_bev_raw,
            self.gpu_counts,
            np.int32(width),
            np.int32(height),
            np.int32(self.projection_stride),
            np.float32(fx),
            np.float32(cx),
            np.int32(self.width),
            np.int32(self.height),
            np.float32(self.res),
            np.float32(self.max_forward_distance),
            block=(block_size, 1, 1),
            grid=(projection_grid, 1, 1),
            stream=stream,
        )

        dilation_grid = (
            self.width * self.height + block_size - 1
        ) // block_size
        self.gpu_dilate_kernel(
            self.gpu_bev_raw,
            self.gpu_bev_dilated,
            np.int32(self.width),
            np.int32(self.height),
            block=(block_size, 1, 1),
            grid=(dilation_grid, 1, 1),
            stream=stream,
        )
        event_kernel.record(stream)

        cuda.memcpy_dtoh_async(
            self.host_bev_pinned, self.gpu_bev_dilated, stream)
        cuda.memcpy_dtoh_async(
            self.host_counts_pinned, self.gpu_counts, stream)
        event_download.record(stream)
        stream.synchronize()

        timings['gpu_upload'] = event_start.time_till(event_upload)
        timings['gpu_kernel'] = event_upload.time_till(event_kernel)
        timings['gpu_download'] = event_kernel.time_till(event_download)
        timings['gpu_pipeline'] = (
            timings['gpu_staging']
            + event_start.time_till(event_download)
        )
        counts['colored'] = int(self.host_counts_pinned[0])
        counts['valid_depth'] = int(self.host_counts_pinned[1])
        counts['projected'] = int(self.host_counts_pinned[2])
        return self.host_bev_pinned

    def camera_info_callback(self, msg):
        """Extracts intrinsic parameters from CameraInfo topic."""
        if self.fx is None:
            self.get_logger().info("CameraInfo received successfully!")
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.camera_info_width = msg.width
        self.camera_info_height = msg.height

    def depth_callback(self, msg):
        """Converts raw depth image to float32 meters array."""
        started_at = time.perf_counter()
        try:
            tmp_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")
            return

        if tmp_depth.dtype == np.uint16:
            self.depth_img = tmp_depth.astype(np.float32) / 1000.0
        else:
            self.depth_img = tmp_depth

        self.depth_received_at = time.perf_counter()
        self.depth_received_count += 1
        if self.debug_telemetry:
            self.telemetry_stats['depth_conversion'].append(
                (self.depth_received_at - started_at) * 1000.0)

    def rgb_callback(self, msg):
        """Non-blocking queue producer for incoming RGB frames."""
        self.rgb_received_count += 1
        enqueued_at = time.perf_counter()
        if self.rgb_queue.full():
            try:
                self.rgb_queue.get_nowait()
                self.rgb_dropped_count += 1
            except queue.Empty:
                pass
        try:
            self.rgb_queue.put_nowait((msg, enqueued_at))
        except queue.Full:
            self.rgb_dropped_count += 1

    def _worker_loop(self):
        """Worker thread loop processing frames asynchronously."""
        context_pushed = False
        if self.gpu_enabled:
            try:
                self.cuda_context.push()
                context_pushed = True
            except Exception as exc:
                self.gpu_enabled = False
                self.get_logger().error(
                    f'Cannot activate CUDA context in BEV worker ({exc}); '
                    'using CPU fallback.')

        try:
            while self.is_running and rclpy.ok():
                try:
                    msg, enqueued_at = self.rgb_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                self._process_frame(msg, enqueued_at)
        finally:
            if context_pushed:
                self.cuda_context.pop()

    def _process_frame(self, msg, enqueued_at):
        """Executes full BEV projection pipeline and updates diagnostic telemetry."""
        start_total = time.perf_counter()
        t = {
            key: 0.0 for key in self.telemetry_stats
            if key != 'depth_conversion'
        }
        counts = {'colored': 0, 'valid_depth': 0, 'projected': 0}
        t['queue_wait'] = (start_total - enqueued_at) * 1000.0

        stamp_ns = msg.header.stamp.sec * 1000000000
        stamp_ns += msg.header.stamp.nanosec
        if stamp_ns > 0:
            t['message_age'] = max(
                0.0,
                (self.get_clock().now().nanoseconds - stamp_ns) / 1e6,
            )
        if self.depth_received_at is not None:
            t['depth_age'] = (
                start_total - self.depth_received_at
            ) * 1000.0

        depth_img = self.depth_img
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        info_width = self.camera_info_width
        info_height = self.camera_info_height

        if depth_img is None or fx is None or not info_width or not info_height:
            missing = []
            if depth_img is None: missing.append("Depth")
            if fx is None or not info_width or not info_height:
                missing.append("CameraInfo")
            self.get_logger().warn(
                f"Waiting for {', '.join(missing)}; skipping frame.",
                throttle_duration_sec=2.0,
            )
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
        pipeline_start = time.perf_counter()
        t_start = pipeline_start

        # Adapt the original camera intrinsics to the lower-half crop and to the
        # actual RGB overlay resolution. This must be done even when RGB and
        # depth already have matching sizes: both streams are 320x120, whereas
        # CameraInfo normally describes the original 640x480 image.
        h_rgb, w_rgb = rgb_img.shape[:2]
        h_depth, w_depth = depth_img.shape[:2]

        crop_start_y = info_height * self.input_crop_y_min
        cropped_source_height = info_height - crop_start_y
        scale_x = w_rgb / float(info_width)
        scale_y = h_rgb / float(cropped_source_height)

        curr_fx = fx * scale_x
        curr_fy = fy * scale_y
        curr_cx = cx * scale_x
        curr_cy = (cy - crop_start_y) * scale_y

        projection_geometry = (
            info_width, info_height, w_rgb, h_rgb,
            curr_fx, curr_fy, curr_cx, curr_cy,
            self.projection_stride,
        )
        if projection_geometry != self.last_projection_geometry:
            sampled_u = np.arange(
                0, w_rgb, self.projection_stride, dtype=np.float32)
            sampled_height = (
                h_rgb + self.projection_stride - 1
            ) // self.projection_stride
            x_factor_row = (sampled_u - curr_cx) / curr_fx
            self.x_factor_lut = np.broadcast_to(
                x_factor_row,
                (sampled_height, sampled_u.size),
            ).copy()
            self.get_logger().info(
                f'Projection intrinsics adapted from '
                f'{info_width}x{info_height} to {w_rgb}x{h_rgb}: '
                f'fx={curr_fx:.2f}, fy={curr_fy:.2f}, '
                f'cx={curr_cx:.2f}, cy={curr_cy:.2f}')
            self.last_projection_geometry = projection_geometry

        t['geometry'] = (time.perf_counter() - t_start) * 1000.0

        t_start = time.perf_counter()
        if (h_rgb, w_rgb) != (h_depth, w_depth):
            curr_depth = cv2.resize(depth_img, (w_rgb, h_rgb), interpolation=cv2.INTER_NEAREST)
        else:
            curr_depth = depth_img
        t['depth_resize'] = (time.perf_counter() - t_start) * 1000.0

        if self.gpu_enabled:
            try:
                bev_img = self._process_gpu(
                    rgb_img, curr_depth, curr_fx, curr_cx, t, counts)
                self._finish_frame(
                    bev_img, msg.header, t, counts,
                    start_total, pipeline_start)
                return
            except Exception as exc:
                # A runtime CUDA error must not take the perception pipeline
                # down. The already prepared CPU LUT allows an immediate
                # fallback without waiting for another CameraInfo message.
                self.gpu_enabled = False
                self.get_logger().error(
                    f'CUDA BEV processing failed ({exc}); switching to CPU.')

        # Regular sampling reduces the point count before all expensive NumPy
        # gathers. The final dilation closes the small gaps in the BEV canvas.
        stride = self.projection_stride
        sampled_rgb = rgb_img[::stride, ::stride]
        sampled_depth = curr_depth[::stride, ::stride]

        # Identify non-black pixels without creating three temporary masks.
        t_start = time.perf_counter()
        valid_color_mask = np.bitwise_or.reduce(sampled_rgb, axis=2) != 0
        counts['colored'] = int(np.count_nonzero(valid_color_mask))
        t['mask_extract'] = (time.perf_counter() - t_start) * 1000.0

        if counts['colored'] == 0:
            empty_bev = self.bev_buffer
            empty_bev.fill(0)
            self._finish_frame(
                empty_bev, msg.header, t, counts,
                start_total, pipeline_start)
            return

        t_start = time.perf_counter()
        valid_depth_mask = (
            valid_color_mask
            & np.isfinite(sampled_depth)
            & (sampled_depth > 0.1)
            & (sampled_depth < self.max_forward_distance)
        )
        counts['valid_depth'] = int(np.count_nonzero(valid_depth_mask))
        z = sampled_depth[valid_depth_mask]
        colors = sampled_rgb[valid_depth_mask]
        x_factors = self.x_factor_lut[valid_depth_mask]
        t['gather_filter'] = (time.perf_counter() - t_start) * 1000.0

        if counts['valid_depth'] == 0:
            empty_bev = self.bev_buffer
            empty_bev.fill(0)
            self._finish_frame(
                empty_bev, msg.header, t, counts,
                start_total, pipeline_start)
            return

        # 3D Back-projection to Camera Frame
        # X_cam: Right, Y_cam: Down, Z_cam: Forward
        t_start = time.perf_counter()
        x_cam = x_factors * z

        # Ground / Robot Frame Transformation
        # X_robot (Forward) = Z_cam
        # Y_robot (Left)    = -X_cam
        x_robot = z
        y_robot = -x_cam
        t['projection'] = (time.perf_counter() - t_start) * 1000.0

        # Rasterization onto BEV Canvas
        t_start = time.perf_counter()
        bev_img = self.bev_buffer
        bev_img.fill(0)

        # Center of robot is at bottom-middle of the image
        u_bev = (self.width / 2.0 - y_robot / self.res).astype(np.int32)
        v_bev = (
            self.height - 1 - x_robot / self.res
        ).astype(np.int32)

        # Filter points within canvas boundaries
        mask = (
            (u_bev >= 0)
            & (u_bev < self.width)
            & (v_bev >= 0)
            & (v_bev < self.height)
        )

        projected_count = int(np.count_nonzero(mask))
        counts['projected'] = projected_count
        if projected_count:
            flat_indices = (
                v_bev[mask] * self.width + u_bev[mask]
            )
            bev_img.reshape(-1, 3)[flat_indices] = colors[mask]
        t['rasterize'] = (time.perf_counter() - t_start) * 1000.0

        t_start = time.perf_counter()
        if projected_count:
            cv2.dilate(
                bev_img,
                self.kernel_inflate,
                dst=bev_img,
                iterations=1,
            )
        t['dilate'] = (time.perf_counter() - t_start) * 1000.0

        self._finish_frame(
            bev_img, msg.header, t, counts, start_total, pipeline_start)

    def _finish_frame(
            self, bev_img, header, timings, counts,
            start_total, pipeline_start):
        """Publish a frame and finalize its telemetry sample."""
        timings['bev_pipeline'] = (
            time.perf_counter() - pipeline_start
        ) * 1000.0
        publish_start = time.perf_counter()
        self._publish_bev(bev_img, header)
        timings['publish'] = (
            time.perf_counter() - publish_start
        ) * 1000.0
        timings['total'] = (
            time.perf_counter() - start_total
        ) * 1000.0
        self.log_diagnostics(timings, counts)

    def _publish_bev(self, bev_img, header):
        """Converts and publishes the BEV OpenCV image to ROS topic."""
        if not self.is_running or not rclpy.ok():
            return
        bird_msg = self.bridge.cv2_to_imgmsg(bev_img, encoding='bgr8')
        bird_msg.header = header
        try:
            self.bird_pub.publish(bird_msg)
        except Exception:
            if self.is_running and rclpy.ok():
                raise

    def log_diagnostics(self, t, counts):
        """Record timings and periodically report throughput and percentiles."""
        for key, val in t.items():
            if key in self.telemetry_stats and np.isfinite(val):
                self.telemetry_stats[key].append(val)
        for key, val in counts.items():
            self.telemetry_counts[key].append(val)

        self.frame_count += 1
        if not self.debug_telemetry:
            return
        if self.frame_count % self.log_interval != 0:
            return

        def stats(values):
            samples = np.asarray(list(values), dtype=np.float64)
            if samples.size == 0:
                return 0.0, 0.0
            return float(np.mean(samples)), float(np.percentile(samples, 95))

        avg = {}
        p95 = {}
        for key, values in self.telemetry_stats.items():
            avg[key], p95[key] = stats(values)

        avg_count = {
            key: stats(values)[0]
            for key, values in self.telemetry_counts.items()
        }

        report_time = time.perf_counter()
        report_period = max(report_time - self.last_report_time, 1e-9)
        processed_delta = self.frame_count - self.last_report_frame_count
        rgb_delta = self.rgb_received_count - self.last_report_rgb_count
        depth_delta = self.depth_received_count - self.last_report_depth_count
        dropped_delta = (
            self.rgb_dropped_count - self.last_report_dropped_count)

        processing_fps = processed_delta / report_period
        rgb_input_fps = rgb_delta / report_period
        depth_input_fps = depth_delta / report_period
        worker_capacity_fps = (
            1000.0 / avg['total'] if avg['total'] > 0.0 else 0.0)
        valid_ratio = (
            100.0 * avg_count['valid_depth'] / avg_count['colored']
            if avg_count['colored'] > 0.0 else 0.0)
        projected_ratio = (
            100.0 * avg_count['projected'] / avg_count['valid_depth']
            if avg_count['valid_depth'] > 0.0 else 0.0)

        rows = [
            '',
            '==================== BEV TELEMETRY ====================',
            f'Canvas: {self.width}x{self.height} @ {self.res:.3f} m/px | '
            f'projection stride {self.projection_stride} | backend '
            f'{"CUDA" if self.gpu_enabled else "CPU"}',
            f'Rates: RGB in {rgb_input_fps:.1f} Hz | depth in '
            f'{depth_input_fps:.1f} Hz | output {processing_fps:.1f} Hz | '
            f'worker capacity {worker_capacity_fps:.1f} FPS',
            f'Queue: dropped {dropped_delta} this interval, '
            f'{self.rgb_dropped_count} total | wait avg/p95 '
            f'{avg["queue_wait"]:.2f}/{p95["queue_wait"]:.2f} ms',
            f'Age: RGB avg/p95 {avg["message_age"]:.2f}/'
            f'{p95["message_age"]:.2f} ms | latest depth '
            f'{avg["depth_age"]:.2f}/{p95["depth_age"]:.2f} ms',
            '-------------------- stage avg/p95 --------------------',
            f'RGB conversion:  {avg["conversion"]:7.2f} / '
            f'{p95["conversion"]:7.2f} ms',
            f'Depth callback:  {avg["depth_conversion"]:7.2f} / '
            f'{p95["depth_conversion"]:7.2f} ms',
            f'Geometry:        {avg["geometry"]:7.2f} / '
            f'{p95["geometry"]:7.2f} ms',
            f'Depth resize:    {avg["depth_resize"]:7.2f} / '
            f'{p95["depth_resize"]:7.2f} ms',
            f'Mask extraction: {avg["mask_extract"]:7.2f} / '
            f'{p95["mask_extract"]:7.2f} ms',
            f'Gather/filter:   {avg["gather_filter"]:7.2f} / '
            f'{p95["gather_filter"]:7.2f} ms',
            f'Projection:      {avg["projection"]:7.2f} / '
            f'{p95["projection"]:7.2f} ms',
            f'Rasterization:   {avg["rasterize"]:7.2f} / '
            f'{p95["rasterize"]:7.2f} ms',
            f'Dilation:        {avg["dilate"]:7.2f} / '
            f'{p95["dilate"]:7.2f} ms',
            f'GPU staging:     {avg["gpu_staging"]:7.2f} / '
            f'{p95["gpu_staging"]:7.2f} ms',
            f'GPU upload:      {avg["gpu_upload"]:7.2f} / '
            f'{p95["gpu_upload"]:7.2f} ms',
            f'GPU kernels:     {avg["gpu_kernel"]:7.2f} / '
            f'{p95["gpu_kernel"]:7.2f} ms',
            f'GPU download:    {avg["gpu_download"]:7.2f} / '
            f'{p95["gpu_download"]:7.2f} ms',
            f'GPU pipeline:    {avg["gpu_pipeline"]:7.2f} / '
            f'{p95["gpu_pipeline"]:7.2f} ms',
            f'Publish:         {avg["publish"]:7.2f} / '
            f'{p95["publish"]:7.2f} ms',
            f'BEV pipeline:    {avg["bev_pipeline"]:7.2f} / '
            f'{p95["bev_pipeline"]:7.2f} ms',
            f'Total worker:    {avg["total"]:7.2f} / '
            f'{p95["total"]:7.2f} ms',
            '------------------------ points ------------------------',
            f'Sampled colored {avg_count["colored"]:.0f} -> valid depth '
            f'{avg_count["valid_depth"]:.0f} ({valid_ratio:.1f}%) -> '
            f'canvas {avg_count["projected"]:.0f} '
            f'({projected_ratio:.1f}%)',
            '===========================================================',
        ]
        self.get_logger().info('\n'.join(rows))

        self.last_report_time = report_time
        self.last_report_frame_count = self.frame_count
        self.last_report_rgb_count = self.rgb_received_count
        self.last_report_depth_count = self.depth_received_count
        self.last_report_dropped_count = self.rgb_dropped_count

    def destroy_node(self):
        """Stop the worker before ROS destroys its publisher handles."""
        self.is_running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        if not self.worker_thread.is_alive():
            self._release_gpu()
        elif self.cuda_context is not None:
            self.get_logger().warn(
                'BEV worker did not stop in time; CUDA cleanup deferred.')
        super().destroy_node()

    def _release_gpu(self):
        """Release CUDA allocations after the worker has left its context."""
        if self.cuda_context is None or self.cuda is None:
            return

        context_pushed = False
        try:
            self.cuda_context.push()
            context_pushed = True
            for allocation_name in (
                    'gpu_rgb_input', 'gpu_depth_input', 'gpu_bev_raw',
                    'gpu_bev_dilated', 'gpu_counts'):
                allocation = getattr(self, allocation_name, None)
                if allocation is not None:
                    allocation.free()
                    setattr(self, allocation_name, None)
            self.gpu_projection_kernel = None
            self.gpu_dilate_kernel = None
            self.gpu_events = None
            self.cuda_stream = None
            self.cuda_module = None
        except Exception as exc:
            self.get_logger().warn(f'CUDA cleanup warning: {exc}')
        finally:
            if context_pushed:
                self.cuda_context.pop()
            try:
                self.cuda_context.detach()
            except Exception:
                pass
            self.cuda_context = None


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
