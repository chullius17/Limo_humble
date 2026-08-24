#!/usr/bin/env python3

import copy
import math
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.msg import ParticleCloud
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2


class CvAmclDebug(Node):
    """Diagnose CV likelihoods independently from AMCL's C++ fusion.

    The node builds a metric likelihood field from a static CV occupancy map.
    For every AMCL pose, it projects the current robot-frame CV point cloud
    into the map, samples the likelihood field, and publishes a new
    ParticleCloud containing raw, unnormalized CV-only likelihoods. Its output
    is intended for inspection and is not consumed by AMCL.
    """

    def __init__(self):
        super().__init__('cv_amcl_debug')
        self.bridge = CvBridge()

        # Input and output topic names.
        self.declare_parameter(
            'input_topic',
            '/limo/nav_map_package/online/maps/cv_map',
        )
        self.declare_parameter(
            'debug_topic',
            '/limo/nav_map_package/online/cv_amcl_debug/distance_field',
        )
        self.declare_parameter(
            'pointcloud_topic',
            '/limo/nav_map_package/online/cv_2_ptcld/points',
        )
        self.declare_parameter('particle_cloud_topic', '/particle_cloud')
        self.declare_parameter(
            'output_particle_cloud_topic',
            '/limo/nav_map_package/online/cv_amcl_debug/raw_particle_cloud',
        )
        # Map cells at or above this value are treated as obstacles.
        self.declare_parameter('occupied_threshold', 50)

        # The CV cloud must be expressed in the same robot frame represented
        # by each AMCL particle pose.
        self.declare_parameter('base_frame_id', 'base_link')

        # Likelihood-field parameters. They mirror the names and roles used
        # by the AMCL likelihood_field laser model.
        self.declare_parameter('z_hit', 0.5)
        self.declare_parameter('z_rand', 0.5)
        self.declare_parameter('sigma_hit', 0.2)
        self.declare_parameter('max_occ_dist', 2.0)
        self.declare_parameter('sensor_max_range', 10.0)
        self.declare_parameter('max_points', 600)
        self.declare_parameter('voxel_leaf_size', 0.05)
        self.declare_parameter('diagnostic_log_period_sec', 1.0)

        input_topic = str(self.get_parameter('input_topic').value)
        debug_topic = str(self.get_parameter('debug_topic').value)
        pointcloud_topic = str(self.get_parameter('pointcloud_topic').value)
        particle_cloud_topic = str(
            self.get_parameter('particle_cloud_topic').value
        )
        output_particle_cloud_topic = str(
            self.get_parameter('output_particle_cloud_topic').value
        )
        # Cached inputs and products. Scoring starts only after the map, CV
        # points, and AMCL particles have all been received at least once.
        self.latest_cv_cloud = None
        self.latest_particle_cloud = None
        self.cv_points = None
        self.distance_map_m = None
        self.map_info = None
        self.map_frame_id = None
        self.last_diagnostic_log_time = -math.inf

        # Static maps are latched: a subscriber joining after publication must
        # still receive the latest map.
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # The debug distance image is also latched for late RViz subscribers.
        debug_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.debug_pub = self.create_publisher(
            Image,
            debug_topic,
            debug_qos,
        )
        self.particle_cloud_pub = self.create_publisher(
            ParticleCloud,
            output_particle_cloud_topic,
            qos_profile_sensor_data,
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            input_topic,
            self.map_callback,
            map_qos,
        )
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            pointcloud_topic,
            self.cloud_callback,
            qos_profile_sensor_data,
        )
        self.particle_cloud_sub = self.create_subscription(
            ParticleCloud,
            particle_cloud_topic,
            self.particle_cloud_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f'Computing normalized distance from CV-map obstacles: '
            f'{input_topic} -> {debug_topic}; '
            f'CV cloud input: {pointcloud_topic}; '
            f'AMCL particles input: {particle_cloud_topic}; '
            f'CV-scored particles output: {output_particle_cloud_topic}'
        )

    def cloud_callback(self, msg: PointCloud2) -> None:
        """Cache CV points expressed in the robot base frame."""
        self.latest_cv_cloud = msg

        # A particle pose maps base_frame_id into the map. Accepting points in
        # another frame here would apply an incorrect rigid transformation.
        base_frame_id = str(self.get_parameter('base_frame_id').value)
        if msg.header.frame_id != base_frame_id:
            self.get_logger().warning(
                f'Ignoring CV cloud in frame {msg.header.frame_id!r}; '
                f'expected {base_frame_id!r}'
            )
            self.cv_points = None
            return

        # Only planar coordinates are required by the 2D occupancy map. NaN
        # points are discarded by the PointCloud2 iterator.
        points = [
            (float(point[0]), float(point[1]))
            for point in point_cloud2.read_points(
                msg,
                field_names=('x', 'y'),
                skip_nans=True,
            )
        ]
        raw_points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        voxel_leaf_size = float(
            self.get_parameter('voxel_leaf_size').value
        )
        if voxel_leaf_size <= 0.0:
            self.get_logger().error('voxel_leaf_size must be positive')
            self.cv_points = None
            return
        # Downsample once per incoming cloud, rather than once per particle.
        self.cv_points = self.voxel_grid(raw_points, voxel_leaf_size)

        # Re-evaluate the latest particle set immediately when perception
        # produces a fresher observation.
        if self.latest_particle_cloud is not None:
            self.publish_cv_scores(self.latest_particle_cloud)

    def particle_cloud_callback(self, msg: ParticleCloud) -> None:
        """Score AMCL particle poses against the CV likelihood field."""
        # Keeping the latest message also allows a later CV cloud callback to
        # trigger scoring without waiting for AMCL to publish again.
        self.latest_particle_cloud = msg
        self.publish_cv_scores(msg)

    @staticmethod
    def normalized_distance(
        occupancy: np.ndarray,
        occupied_threshold: int,
    ) -> np.ndarray:
        """Return a mono8 L2 distance image from occupied map cells."""
        # OpenCV returns, for every non-zero pixel, the Euclidean distance to
        # the closest zero pixel. Therefore obstacles must be encoded as zero.
        obstacles = occupancy >= occupied_threshold
        if not np.any(obstacles):
            return np.zeros(occupancy.shape, dtype=np.uint8)

        # Black pixels in the source PGM become occupied cells (value 100) in
        # the OccupancyGrid. OpenCV expects distance sources to be zero.
        distance_input = np.full(occupancy.shape, 255, dtype=np.uint8)
        distance_input[obstacles] = 0
        distances = cv2.distanceTransform(
            distance_input,
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )
        maximum = float(np.max(distances))
        if maximum <= 0.0:
            return np.zeros(occupancy.shape, dtype=np.uint8)
        return np.rint(distances * (255.0 / maximum)).astype(np.uint8)

    @staticmethod
    def metric_distance(
        occupancy: np.ndarray,
        occupied_threshold: int,
        resolution: float,
        max_occ_dist: float,
    ) -> np.ndarray:
        """Build the capped metric likelihood field used for CV scoring."""
        obstacles = occupancy >= occupied_threshold
        if not np.any(obstacles):
            return np.full(occupancy.shape, max_occ_dist, dtype=np.float32)

        distance_input = np.full(occupancy.shape, 255, dtype=np.uint8)
        distance_input[obstacles] = 0
        distances = cv2.distanceTransform(
            distance_input,
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )
        # OpenCV distances are measured in pixels. AMCL works in metres and
        # caps distant cells because they are equally poor obstacle matches.
        return np.minimum(distances * resolution, max_occ_dist)

    @staticmethod
    def voxel_grid(points: np.ndarray, leaf_size: float) -> np.ndarray:
        """Downsample XY points using PCL VoxelGrid centroid semantics.

        Adapted to NumPy from the PCL C++ VoxelGrid tutorial:
        https://pointclouds.org/documentation/tutorials/voxel_grid.html
        """
        if points is None or points.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float64)
        if leaf_size <= 0.0:
            raise ValueError('leaf_size must be positive')

        # Quantization assigns every point to an integer XY voxel. np.unique
        # returns the voxel id of each point through the inverse array.
        voxel_indices = np.floor(points / leaf_size).astype(np.int64)
        _, inverse, counts = np.unique(
            voxel_indices,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        # Accumulate all point coordinates per voxel and divide by occupancy to
        # obtain the same centroid representation used by PCL VoxelGrid.
        centroids = np.zeros((counts.size, 2), dtype=np.float64)
        np.add.at(centroids, inverse, points)
        centroids /= counts[:, np.newaxis]
        return centroids

    def publish_cv_scores(self, msg: ParticleCloud) -> None:
        """Publish particle poses with raw CV-only likelihoods."""
        # The computation needs a static likelihood field and a current CV
        # observation. Returning here is expected during node startup.
        if self.distance_map_m is None or self.map_info is None:
            return
        if self.cv_points is None or self.cv_points.size == 0:
            return
        if not msg.particles:
            return
        # Particle coordinates and the occupancy grid must share a frame; no
        # TF lookup is intentionally performed for particle poses.
        if msg.header.frame_id != self.map_frame_id:
            self.get_logger().warning(
                f'Cannot score particles in frame {msg.header.frame_id!r} '
                f'against CV map in frame {self.map_frame_id!r}'
            )
            return

        sigma_hit = float(self.get_parameter('sigma_hit').value)
        z_hit = float(self.get_parameter('z_hit').value)
        z_rand = float(self.get_parameter('z_rand').value)
        max_occ_dist = float(self.get_parameter('max_occ_dist').value)
        sensor_max_range = float(
            self.get_parameter('sensor_max_range').value
        )
        max_points = int(self.get_parameter('max_points').value)
        if (
            sigma_hit <= 0.0
            or z_hit < 0.0
            or z_rand < 0.0
            or max_occ_dist <= 0.0
            or sensor_max_range <= 0.0
            or max_points < 1
        ):
            self.get_logger().error(
                'Invalid CV likelihood parameters: sigma_hit, max_occ_dist, '
                'sensor_max_range and max_points must be positive; z_hit and '
                'z_rand must be non-negative'
            )
            return

        # Bound computation to particles x max_points. Voxelization has already
        # removed redundant neighbouring observations.
        # Random uniform subsampling removes spatial bias introduced by np.unique sorting
        points = self.cv_points
        if points.shape[0] > max_points:
            rng = np.random.default_rng()
            selected_indices = rng.choice(
                points.shape[0],
                size=max_points,
                replace=False,
            )
            points = points[selected_indices]

        if points.shape[0] == 0:
            return

        point_count = points.shape[0]

        # Precompute the inverse map-origin rotation used to convert world
        # coordinates into OccupancyGrid row and column indices.
        info = self.map_info
        resolution = float(info.resolution)
        origin = info.origin
        quaternion = origin.orientation
        origin_yaw = math.atan2(
            2.0 * (
                quaternion.w * quaternion.z
                + quaternion.x * quaternion.y
            ),
            1.0 - 2.0 * (
                quaternion.y * quaternion.y
                + quaternion.z * quaternion.z
            ),
        )
        map_cos = math.cos(origin_yaw)
        map_sin = math.sin(origin_yaw)
        height, width = self.distance_map_m.shape
        # Terms shared by all points and particles in this update.
        denominator = 2.0 * sigma_hit * sigma_hit
        random_likelihood = z_rand / sensor_max_range
        scores = np.empty(len(msg.particles), dtype=np.float64)

        for index, particle in enumerate(msg.particles):
            # Particle orientation represents the hypothetical base_link yaw
            # in the map frame.
            orientation = particle.pose.orientation
            yaw = math.atan2(
                2.0 * (
                    orientation.w * orientation.z
                    + orientation.x * orientation.y
                ),
                1.0 - 2.0 * (
                    orientation.y * orientation.y
                    + orientation.z * orientation.z
                ),
            )
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            # Apply the 2D rigid transform defined by this particle:
            # p_map = R(particle_yaw) * p_base + particle_translation.
            map_x = (
                particle.pose.position.x
                + cos_yaw * points[:, 0]
                - sin_yaw * points[:, 1]
            )
            map_y = (
                particle.pose.position.y
                + sin_yaw * points[:, 0]
                + cos_yaw * points[:, 1]
            )

            # Undo the map origin translation and rotation, then divide by map
            # resolution to obtain integer grid coordinates.
            delta_x = map_x - origin.position.x
            delta_y = map_y - origin.position.y
            columns = np.floor(
                (map_cos * delta_x + map_sin * delta_y) / resolution
            ).astype(np.int64)
            rows = np.floor(
                (-map_sin * delta_x + map_cos * delta_y) / resolution
            ).astype(np.int64)
            valid = (
                (columns >= 0)
                & (columns < width)
                & (rows >= 0)
                & (rows < height)
            )
            # Off-map observations receive AMCL's maximum-distance penalty;
            # valid observations sample the precomputed metric field.
            distances = np.full(point_count, max_occ_dist, dtype=np.float64)
            distances[valid] = self.distance_map_m[
                rows[valid], columns[valid]
            ]
            # Mixture of an obstacle-hit Gaussian and a uniform random term,
            # matching AMCL's likelihood_field measurement model.
            point_likelihoods = (
                z_hit * np.exp(-(distances * distances) / denominator)
                + random_likelihood
            )

            # Match AMCL's likelihood_field aggregation: p starts at one and
            # each observation contributes the cube of its likelihood.
            scores[index] = 1.0 + np.sum(point_likelihoods ** 3)

        # Preserve headers and poses, replacing only the particle weights.
        # Scores intentionally remain raw: they are neither normalized across
        # particles nor multiplied by the input AMCL weights. This lets AMCL
        # combine log(CV likelihood) with its laser likelihood before the
        # filter performs the final normalization and resampling steps.
        output = copy.deepcopy(msg)
        for particle, score in zip(output.particles, scores):
            particle.weight = float(score)
        self.log_particle_differences(msg, output, scores, point_count)
        self.particle_cloud_pub.publish(output)

    def log_particle_differences(
        self,
        input_cloud: ParticleCloud,
        output_cloud: ParticleCloud,
        cv_scores: np.ndarray,
        point_count: int,
    ) -> None:
        """Log pose integrity and the discriminative power of CV scores."""
        period = float(
            self.get_parameter('diagnostic_log_period_sec').value
        )
        now = time.monotonic()
        if period > 0.0 and now - self.last_diagnostic_log_time < period:
            return
        self.last_diagnostic_log_time = now

        if len(input_cloud.particles) != len(output_cloud.particles):
            self.get_logger().warning(
                'Cannot compare particle clouds with different sizes: '
                f'{len(input_cloud.particles)} input vs '
                f'{len(output_cloud.particles)} output'
            )
            return
        if not input_cloud.particles:
            return

        position_differences = []
        orientation_differences = []
        weight_differences = []

        # Compare particles by array index. The output is a deep copy of the
        # input, so every paired pose is expected to have exactly zero error.
        for input_particle, output_particle in zip(
            input_cloud.particles,
            output_cloud.particles,
        ):
            input_position = input_particle.pose.position
            output_position = output_particle.pose.position
            position_differences.append(math.sqrt(
                (output_position.x - input_position.x) ** 2
                + (output_position.y - input_position.y) ** 2
                + (output_position.z - input_position.z) ** 2
            ))

            input_orientation = input_particle.pose.orientation
            output_orientation = output_particle.pose.orientation
            input_quaternion = np.array([
                input_orientation.x,
                input_orientation.y,
                input_orientation.z,
                input_orientation.w,
            ], dtype=np.float64)
            output_quaternion = np.array([
                output_orientation.x,
                output_orientation.y,
                output_orientation.z,
                output_orientation.w,
            ], dtype=np.float64)
            input_norm = float(np.linalg.norm(input_quaternion))
            output_norm = float(np.linalg.norm(output_quaternion))
            if input_norm == 0.0 or output_norm == 0.0:
                orientation_differences.append(math.nan)
            else:
                # q and -q describe the same orientation, hence abs(dot).
                quaternion_dot = abs(float(np.dot(
                    input_quaternion / input_norm,
                    output_quaternion / output_norm,
                )))
                quaternion_dot = min(1.0, max(0.0, quaternion_dot))
                orientation_differences.append(
                    2.0 * math.acos(quaternion_dot)
                )

            weight_differences.append(abs(
                output_particle.weight - input_particle.weight
            ))

        position_differences = np.asarray(position_differences)
        orientation_differences = np.asarray(orientation_differences)
        weight_differences = np.asarray(weight_differences)

        # Normalize the raw CV-only scores only for diagnostics. The published
        # ParticleCloud intentionally retains the raw likelihoods so AMCL can
        # apply cv_weight_factor during log-domain fusion.
        score_min = float(np.min(cv_scores))
        score_max = float(np.max(cv_scores))
        score_p10, score_median, score_p90 = np.percentile(
            cv_scores,
            [10.0, 50.0, 90.0],
        )
        score_ratio = score_max / score_min if score_min > 0.0 else math.inf
        score_total = float(np.sum(cv_scores))
        if score_total > 0.0 and math.isfinite(score_total):
            normalized_scores = cv_scores / score_total
            squared_sum = float(np.sum(normalized_scores ** 2))
            effective_sample_size = (
                1.0 / squared_sum if squared_sum > 0.0 else 0.0
            )
        else:
            effective_sample_size = 0.0
        ess_fraction = effective_sample_size / len(cv_scores)

        self.get_logger().info(
            'CV particle diagnostics: '
            f'particles={len(input_cloud.particles)}, '
            f'points={point_count}, '
            f'score_min={score_min:.6g}, '
            f'score_p10={score_p10:.6g}, '
            f'score_median={score_median:.6g}, '
            f'score_p90={score_p90:.6g}, '
            f'score_max={score_max:.6g}, '
            f'score_ratio={score_ratio:.6g}, '
            f'ess={effective_sample_size:.1f}, '
            f'ess_fraction={ess_fraction:.3f}, '
            f'pose_position_mean={np.mean(position_differences):.6g} m, '
            f'pose_position_max={np.max(position_differences):.6g} m, '
            f'pose_angle_mean={np.nanmean(orientation_differences):.6g} rad, '
            f'pose_angle_max={np.nanmax(orientation_differences):.6g} rad, '
            f'weight_abs_mean={np.mean(weight_differences):.6g}, '
            f'weight_abs_max={np.max(weight_differences):.6g}'
        )

    def map_callback(self, msg: OccupancyGrid) -> None:
        """Build debug and metric distance fields from the static CV map."""
        threshold = int(self.get_parameter('occupied_threshold').value)
        if not 0 <= threshold <= 100:
            self.get_logger().error(
                'occupied_threshold must be between 0 and 100'
            )
            return

        width = int(msg.info.width)
        height = int(msg.info.height)
        if width <= 0 or height <= 0 or len(msg.data) != width * height:
            self.get_logger().error(
                f'Invalid CV map: {width}x{height}, {len(msg.data)} cells'
            )
            return

        try:
            occupancy = np.asarray(msg.data, dtype=np.int16).reshape(
                height,
                width,
            )
            # The mono8 image is only for visualization. Scoring uses the
            # separate metric field below, not normalized pixel intensities.
            debug_image = self.normalized_distance(occupancy, threshold)
            resolution = float(msg.info.resolution)
            max_occ_dist = float(
                self.get_parameter('max_occ_dist').value
            )
            if resolution <= 0.0 or max_occ_dist <= 0.0:
                self.get_logger().error(
                    'Map resolution and max_occ_dist must be positive'
                )
                return
            self.distance_map_m = self.metric_distance(
                occupancy,
                threshold,
                resolution,
                max_occ_dist,
            )
            # Retain the grid geometry required for map-to-cell conversion.
            self.map_info = copy.deepcopy(msg.info)
            self.map_frame_id = msg.header.frame_id
        except cv2.error as error:
            self.get_logger().error(
                f'Cannot compute CV distance transform: {error}'
            )
            return

        output = self.bridge.cv2_to_imgmsg(debug_image, encoding='mono8')
        output.header = msg.header
        self.debug_pub.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = CvAmclDebug()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
