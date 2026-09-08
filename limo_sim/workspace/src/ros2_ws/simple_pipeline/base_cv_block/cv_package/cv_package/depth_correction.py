#!/usr/bin/env python3
"""Publish cropped depth images and fill missing values from a planar LUT."""

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener


class DepthCorrection(Node):
    """Create once and use a ground-plane depth lookup table."""

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

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.publisher = self.create_publisher(
            Image, output_topic, output_qos)
        self.corrected_publisher = self.create_publisher(
            Image, corrected_topic, output_qos)
        self.depth_publisher = self.create_publisher(
            Image, depth_output_topic, output_qos)
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
        # again in the live processing path.
        self.depth_lut = lut
        valid_entries = int(np.count_nonzero(np.isfinite(lut)))
        self.get_logger().info(
            f'Offline LUT frozen: plane z={plane_height:.4f} m in '
            f'{self.plane_frame}, {valid_entries}/{lut.size} valid entries.')

    def colorize(self, depth):
        """Convert metric depth to a JET BGR image; invalid pixels stay black."""
        valid = np.isfinite(depth) & (depth > 0.0)
        safe_depth = np.where(valid, depth, self.min_depth_m)
        clipped = np.clip(safe_depth, self.min_depth_m, self.max_depth_m)
        normalized = (
            (clipped - self.min_depth_m)
            * (255.0 / (self.max_depth_m - self.min_depth_m))
        ).astype(np.uint8)
        colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        colorized[~valid] = 0
        return colorized

    def publish_image(self, publisher, image, header):
        """Publish one BGR visualization preserving the source header."""
        output_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        output_msg.header = header
        publisher.publish(output_msg)

    def publish_depth(self, depth, header):
        """Publish corrected metric depth as a 32-bit floating-point image."""
        output_msg = self.bridge.cv2_to_imgmsg(
            depth.astype(np.float32, copy=False), encoding='32FC1')
        output_msg.header = header
        self.depth_publisher.publish(output_msg)

    def depth_callback(self, msg):
        """Crop, resize, calibrate if needed, then publish both views."""
        try:
            depth = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough')
        except CvBridgeError as error:
            self.get_logger().error(f'Cannot convert depth image: {error}')
            return

        if depth.ndim != 2 or depth.shape[0] < 2:
            self.get_logger().warning(
                f'Invalid depth image shape: {depth.shape}',
                throttle_duration_sec=2.0)
            return

        # Astra/OpenNI uint16 depth is in millimetres; float depth is in metres.
        if depth.dtype == np.uint16 or msg.encoding in ('16UC1', 'mono16'):
            depth_m = depth.astype(np.float32) * 0.001
        else:
            depth_m = depth.astype(np.float32)

        lower_half = depth_m[depth_m.shape[0] // 2:, :]
        resized = cv2.resize(
            lower_half,
            (self.output_width, self.output_height),
            interpolation=cv2.INTER_NEAREST,
        )

        self.publish_image(
            self.publisher, self.colorize(resized), msg.header)
        self.update_offline_calibration(
            resized, msg, depth_m.shape[0], depth_m.shape[1])

        completed = resized.copy()
        if self.depth_lut is not None:
            missing = ~np.isfinite(completed) | (completed <= 0.0)
            fillable = missing & np.isfinite(self.depth_lut)
            completed[fillable] = self.depth_lut[fillable]

        self.publish_depth(completed, msg.header)
        self.publish_image(
            self.corrected_publisher, self.colorize(completed), msg.header)


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
