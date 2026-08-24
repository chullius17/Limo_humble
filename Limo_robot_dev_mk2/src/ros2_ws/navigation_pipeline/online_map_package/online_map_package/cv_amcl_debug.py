#!/usr/bin/env python3

import copy
import math
import time
from functools import partial

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
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener


class CvAmclDebug(Node):
    """Reproduce and visualize AMCL's grid-SAD CV scoring path.

    The node discretizes the robot-local metric-BEV OccupancyGrid exactly like
    the C++ model. It renders the retained SAD cells and independently scores
    every AMCL particle against the static CV map. Its ParticleCloud output is
    diagnostic only and is never consumed by AMCL.
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
            'obstacle_grid_topic',
            '/limo/nav_map_package/online/metric_bev/'
            'cost_grid_binary_obstacles',
        )
        self.declare_parameter(
            'street_grid_topic',
            '/limo/nav_map_package/online/metric_bev/'
            'cost_grid_binary_street',
        )
        self.declare_parameter(
            'obstacle_grid_debug_topic',
            '/limo/nav_map_package/online/cv_amcl_debug/'
            'discretized_obstacles',
        )
        self.declare_parameter(
            'street_grid_debug_topic',
            '/limo/nav_map_package/online/cv_amcl_debug/'
            'discretized_street',
        )
        self.declare_parameter(
            'street_map_topic',
            '/limo/nav_map_package/online/maps/street_map',
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
        self.declare_parameter('map_frame_id', 'map')
        self.declare_parameter('odom_frame_id', 'odom')

        # Likelihood-field parameters. They mirror the names and roles used
        # by the AMCL likelihood_field laser model.
        self.declare_parameter('z_hit', 0.5)
        self.declare_parameter('z_rand', 0.5)
        self.declare_parameter('sigma_hit', 0.2)
        self.declare_parameter('distance_exponent', 3.0)
        self.declare_parameter('max_occ_dist', 2.0)
        self.declare_parameter('sensor_max_range', 10.0)
        self.declare_parameter('max_points', 200)
        self.declare_parameter('voxel_leaf_size', 0.05)
        self.declare_parameter('score_rate_hz', 1.0)
        self.declare_parameter('cv_weight_factor', 1.0)
        self.declare_parameter('cv_obstacle_weight_factor', 1.0)
        self.declare_parameter('cv_street_weight_factor', 1.0)
        self.declare_parameter('semantic_mismatch_penalty', 1.0)
        self.declare_parameter('sad_gain', 20.0)
        self.declare_parameter('sad_cell_size', 0.05)
        self.declare_parameter('sad_min_positive_mass', 5.0)
        self.declare_parameter('diagnostic_log_period_sec', 1.0)

        input_topic = str(self.get_parameter('input_topic').value)
        obstacle_grid_topic = str(
            self.get_parameter('obstacle_grid_topic').value
        )
        street_grid_topic = str(
            self.get_parameter('street_grid_topic').value
        )
        obstacle_grid_debug_topic = str(
            self.get_parameter('obstacle_grid_debug_topic').value
        )
        street_grid_debug_topic = str(
            self.get_parameter('street_grid_debug_topic').value
        )
        street_map_topic = str(
            self.get_parameter('street_map_topic').value
        )
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
        self.street_points = None
        self.distance_map_m = None
        self.static_cv_occupancy = None
        self.static_street_occupancy = None
        self.grid_templates = {}
        self.grid_template_frames = {}
        self.grid_mean_occupancy = {}
        self.discretized_grids = {}
        self.discretization_shapes = {}
        self.street_distance_map_m = None
        self.map_info = None
        self.street_map_info = None
        self.map_frame_id = None
        self.street_map_frame_id = None
        self.last_diagnostic_log_time = -math.inf
        self.previous_particle_statistics = None
        self.last_tf_diagnostic_warning_time = -math.inf

        # AMCL publishes map -> odom. Looking up odom in map provides the
        # translation and yaw correction introduced by localization.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Static maps are latched: a subscriber joining after publication must
        # still receive the latest map.
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Debug images are latched so RViz also receives the latest SAD
        # template when it starts after the mapping pipeline.
        debug_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.obstacle_grid_debug_pub = self.create_publisher(
            Image,
            obstacle_grid_debug_topic,
            debug_qos,
        )
        self.street_grid_debug_pub = self.create_publisher(
            Image,
            street_grid_debug_topic,
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
            partial(self.map_callback, street=False),
            map_qos,
        )
        self.obstacle_grid_sub = self.create_subscription(
            OccupancyGrid,
            obstacle_grid_topic,
            partial(self.local_grid_callback, grid_kind='obstacles'),
            map_qos,
        )
        self.street_grid_sub = self.create_subscription(
            OccupancyGrid,
            street_grid_topic,
            partial(self.local_grid_callback, grid_kind='street'),
            map_qos,
        )
        self.street_map_sub = self.create_subscription(
            OccupancyGrid,
            street_map_topic,
            partial(self.map_callback, street=True),
            map_qos,
        )
        self.particle_cloud_sub = self.create_subscription(
            ParticleCloud,
            particle_cloud_topic,
            self.particle_cloud_callback,
            qos_profile_sensor_data,
        )
        score_rate_hz = float(self.get_parameter('score_rate_hz').value)
        if score_rate_hz <= 0.0:
            raise ValueError('score_rate_hz must be greater than zero')
        self.score_timer = self.create_timer(
            1.0 / score_rate_hz,
            self.score_timer_callback,
        )

        self.get_logger().info(
            f'Discretizing binary CV grids to '
            f'{self.get_parameter("sad_cell_size").value:g} m: '
            f'obstacles {obstacle_grid_topic} -> '
            f'{obstacle_grid_debug_topic}; street {street_grid_topic} -> '
            f'{street_grid_debug_topic}; '
            f'static CV map: {input_topic}; '
            f'AMCL particles input: {particle_cloud_topic}; '
            f'CV-scored particles output: {output_particle_cloud_topic}; '
            f'score rate: {score_rate_hz:g} Hz'
        )

    def cloud_callback(self, msg: PointCloud2, street: bool) -> None:
        """Cache obstacle or street CV points in the robot base frame."""
        self.latest_cv_cloud = msg

        # A particle pose maps base_frame_id into the map. Accepting points in
        # another frame here would apply an incorrect rigid transformation.
        base_frame_id = str(self.get_parameter('base_frame_id').value)
        if msg.header.frame_id != base_frame_id:
            self.get_logger().warning(
                f'Ignoring CV cloud in frame {msg.header.frame_id!r}; '
                f'expected {base_frame_id!r}'
            )
            if street:
                self.street_points = None
            else:
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
            if street:
                self.street_points = None
            else:
                self.cv_points = None
            return
        # Downsample once per incoming cloud, rather than once per particle.
        voxel_points = self.voxel_grid(raw_points, voxel_leaf_size)
        if street:
            self.street_points = voxel_points
        else:
            self.cv_points = voxel_points

    def particle_cloud_callback(self, msg: ParticleCloud) -> None:
        """Cache the latest AMCL particle poses for timer-driven scoring."""
        self.latest_particle_cloud = msg

    def score_timer_callback(self) -> None:
        """Score the latest complete snapshot at the configured debug rate."""
        if self.latest_particle_cloud is not None:
            self.publish_sad_scores(self.latest_particle_cloud)

    def local_grid_callback(
        self,
        msg: OccupancyGrid,
        grid_kind: str,
    ) -> None:
        """Downsample one binary grid to occupied fractions per block."""
        width = int(msg.info.width)
        height = int(msg.info.height)
        resolution = float(msg.info.resolution)
        if (
            not msg.header.frame_id
            or width <= 0
            or height <= 0
            or resolution <= 0.0
            or len(msg.data) != width * height
        ):
            self.get_logger().warning('Ignoring malformed local CV grid')
            return

        cell_size = float(self.get_parameter('sad_cell_size').value)
        if cell_size <= 0.0:
            self.get_logger().error('sad_cell_size must be positive')
            return
        # Match C++ std::lround for positive values exactly.
        block_size = max(1, int(math.floor(cell_size / resolution + 0.5)))
        occupancy = np.asarray(msg.data, dtype=np.int16).reshape(
            height,
            width,
        )

        output_height = int(math.ceil(height / block_size))
        output_width = int(math.ceil(width / block_size))
        downsampled = np.full(
            (output_height, output_width),
            -1,
            dtype=np.int16,
        )
        threshold = int(self.get_parameter('occupied_threshold').value)
        origin = msg.info.origin
        origin_yaw = math.atan2(
            2.0 * (
                origin.orientation.w * origin.orientation.z
                + origin.orientation.x * origin.orientation.y
            ),
            1.0 - 2.0 * (
                origin.orientation.y * origin.orientation.y
                + origin.orientation.z * origin.orientation.z
            ),
        )
        origin_cos = math.cos(origin_yaw)
        origin_sin = math.sin(origin_yaw)
        template_cells = []
        for output_row, first_row in enumerate(range(0, height, block_size)):
            last_row = min(first_row + block_size, height)
            for output_column, first_column in enumerate(
                range(0, width, block_size)
            ):
                last_column = min(first_column + block_size, width)
                block = occupancy[
                    first_row:last_row,
                    first_column:last_column,
                ]
                known = block >= 0
                if not np.any(known):
                    continue
                occupancy_fraction = float(np.mean(block[known] >= threshold))
                downsampled[output_row, output_column] = int(round(
                    100.0 * occupancy_fraction
                ))
                local_x = 0.5 * (first_column + last_column) * resolution
                local_y = 0.5 * (first_row + last_row) * resolution
                source_x = (
                    origin.position.x
                    + origin_cos * local_x
                    - origin_sin * local_y
                )
                source_y = (
                    origin.position.y
                    + origin_sin * local_x
                    + origin_cos * local_y
                )
                template_cells.append((
                    source_x,
                    source_y,
                    occupancy_fraction,
                ))

        self.discretized_grids[grid_kind] = downsampled
        self.grid_templates[grid_kind] = np.asarray(
            template_cells,
            dtype=np.float64,
        ).reshape(-1, 3)
        self.grid_template_frames[grid_kind] = msg.header.frame_id
        self.grid_mean_occupancy[grid_kind] = (
            float(np.mean(self.grid_templates[grid_kind][:, 2]))
            if template_cells
            else math.nan
        )
        shape = (height, width, output_height, output_width, block_size)
        if self.discretization_shapes.get(grid_kind) != shape:
            known_values = downsampled[downsampled >= 0]
            occupied_fraction = (
                float(np.mean(known_values)) / 100.0
                if known_values.size
                else math.nan
            )
            self.get_logger().info(
                f'{grid_kind} grid discretized from {width}x{height} at '
                f'{resolution:g} m to {output_width}x{output_height} at '
                f'{resolution * block_size:g} m; '
                f'mean_occupancy={occupied_fraction:.3f}'
            )
            self.discretization_shapes[grid_kind] = shape

        # Image convention: occupied is black, free is white, fractional
        # occupancy is gray and unknown is fixed at mid-gray.
        debug_image = np.full(downsampled.shape, 127, dtype=np.uint8)
        known = downsampled >= 0
        debug_image[known] = np.rint(
            255.0 * (1.0 - downsampled[known] / 100.0)
        ).astype(np.uint8)
        debug_image = np.flipud(debug_image)
        output = self.bridge.cv2_to_imgmsg(debug_image, encoding='mono8')
        output.header = msg.header
        publisher = (
            self.obstacle_grid_debug_pub
            if grid_kind == 'obstacles'
            else self.street_grid_debug_pub
        )
        publisher.publish(output)

    def sad_template_in_base(self, grid_kind: str):
        """Transform one cached local grid template into base_link."""
        template = self.grid_templates.get(grid_kind)
        frame_id = self.grid_template_frames.get(grid_kind)
        if template is None or template.size == 0 or not frame_id:
            return None
        base_frame = str(self.get_parameter('base_frame_id').value)
        try:
            transform = self.tf_buffer.lookup_transform(
                base_frame,
                frame_id,
                Time(),
            )
        except TransformException as error:
            now = time.monotonic()
            if now - self.last_tf_diagnostic_warning_time >= 2.0:
                self.get_logger().warning(
                    f'Cannot transform SAD template into {base_frame}: {error}'
                )
                self.last_tf_diagnostic_warning_time = now
            return None

        translation = transform.transform.translation
        orientation = transform.transform.rotation
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
        source_x = template[:, 0]
        source_y = template[:, 1]
        return np.column_stack((
            translation.x + cos_yaw * source_x - sin_yaw * source_y,
            translation.y + sin_yaw * source_x + cos_yaw * source_y,
            template[:, 2],
        ))

    def publish_sad_scores(self, msg: ParticleCloud) -> None:
        """Publish the mean obstacle/street SAD likelihood per AMCL pose."""
        gain = float(self.get_parameter('sad_gain').value)
        factor = float(self.get_parameter('cv_weight_factor').value)
        obstacle_weight_factor = float(
            self.get_parameter('cv_obstacle_weight_factor').value
        )
        street_factor = float(
            self.get_parameter('cv_street_weight_factor').value
        )
        if (
            gain < 0.0
            or factor < 0.0
            or obstacle_weight_factor < 0.0
            or street_factor < 0.0
        ):
            self.get_logger().error(
                'SAD gain and CV weight factors must be non-negative'
            )
            return

        use_obstacles = obstacle_weight_factor > 0.0
        use_street = street_factor > 0.0
        if not use_obstacles and not use_street:
            return
        obstacle_template = (
            self.sad_template_in_base('obstacles')
            if use_obstacles
            else None
        )
        street_template = (
            self.sad_template_in_base('street') if use_street else None
        )
        if (
            (use_obstacles and obstacle_template is None)
            or (use_street and street_template is None)
            or (use_obstacles and self.static_cv_occupancy is None)
            or (use_street and self.static_street_occupancy is None)
            or (use_obstacles and self.map_info is None)
            or (use_street and self.street_map_info is None)
            or not msg.particles
        ):
            return
        if (
            (
                use_obstacles
                and msg.header.frame_id != self.map_frame_id
            )
            or (
                use_street
                and msg.header.frame_id != self.street_map_frame_id
            )
        ):
            self.get_logger().warning(
                f'Particle frame {msg.header.frame_id!r} does not match '
                f'semantic map frames {self.map_frame_id!r}, '
                f'{self.street_map_frame_id!r}'
            )
            return

        obstacle_sad = np.zeros(len(msg.particles), dtype=np.float64)
        street_sad = np.zeros(len(msg.particles), dtype=np.float64)
        for index, particle in enumerate(msg.particles):
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
            if use_obstacles:
                obstacle_sad[index] = self.particle_sad(
                    particle,
                    cos_yaw,
                    sin_yaw,
                    obstacle_template,
                    street=False,
                )
            if use_street:
                street_sad[index] = self.particle_sad(
                    particle,
                    cos_yaw,
                    sin_yaw,
                    street_template,
                    street=True,
                )

        obstacle_positive = (
            float(np.sum(obstacle_template[:, 2]))
            if use_obstacles
            else 0.0
        )
        obstacle_negative = (
            float(np.sum(1.0 - obstacle_template[:, 2]))
            if use_obstacles
            else 0.0
        )
        street_positive = (
            float(np.sum(street_template[:, 2])) if use_street else 0.0
        )
        street_negative = (
            float(np.sum(1.0 - street_template[:, 2]))
            if use_street
            else 0.0
        )
        min_positive_mass = float(
            self.get_parameter('sad_min_positive_mass').value
        )
        obstacle_active = (
            use_obstacles and obstacle_positive >= min_positive_mass
        )
        street_active = use_street and street_positive >= min_positive_mass
        obstacle_factor = (
            obstacle_weight_factor if obstacle_active else 0.0
        )
        active_street_factor = street_factor if street_active else 0.0
        semantic_factor_sum = obstacle_factor + active_street_factor
        if semantic_factor_sum <= 0.0:
            self.get_logger().warning(
                'Skipping CV debug SAD: foreground mass below '
                f'{min_positive_mass:.1f} cells '
                f'(obstacles={obstacle_positive:.1f}, '
                f'street={street_positive:.1f})'
            )
            return
        obstacle_mix = obstacle_factor / semantic_factor_sum
        street_mix = active_street_factor / semantic_factor_sum
        combined_sad = (
            obstacle_mix * obstacle_sad
            + street_mix * street_sad
        )
        raw_likelihood = np.exp(-gain * combined_sad)
        scores = np.power(raw_likelihood, factor)
        output = copy.deepcopy(msg)
        for particle, score in zip(output.particles, scores):
            particle.weight = float(score)
        self.log_sad_diagnostics(
            msg,
            output,
            obstacle_sad,
            street_sad,
            combined_sad,
            scores,
            obstacle_template.shape[0] if use_obstacles else 0,
            street_template.shape[0] if use_street else 0,
            obstacle_positive,
            obstacle_negative,
            street_positive,
            street_negative,
            min_positive_mass,
            obstacle_active,
            street_active,
            obstacle_weight_factor,
            street_factor,
            gain,
            factor,
        )
        self.particle_cloud_pub.publish(output)

    def particle_sad(
        self,
        particle,
        cos_yaw: float,
        sin_yaw: float,
        template: np.ndarray,
        street: bool,
    ) -> float:
        """Compute positive semantic mismatch for one map/template pair."""
        map_x = (
            particle.pose.position.x
            + cos_yaw * template[:, 0]
            - sin_yaw * template[:, 1]
        )
        map_y = (
            particle.pose.position.y
            + sin_yaw * template[:, 0]
            + cos_yaw * template[:, 1]
        )
        map_values, valid = self.sample_static_cv_map(map_x, map_y, street)
        local_positive = np.clip(template[:, 2], 0.0, 1.0)
        positive_mass = float(np.sum(local_positive))
        if positive_mass <= 0.0:
            return 0.0

        # Outside-map positive cells disagree maximally. Inside the map, a
        # local foreground cell is penalized only when its semantic map is 0.
        false_positive = np.ones(template.shape[0], dtype=np.float64)
        false_positive[valid] = 1.0 - map_values[valid]
        return float(
            np.sum(local_positive * false_positive) / positive_mass
        )

    def sample_static_cv_map(
        self,
        map_x: np.ndarray,
        map_y: np.ndarray,
        street: bool,
    ):
        """Sample one static binary semantic map and return its validity."""
        info = self.street_map_info if street else self.map_info
        occupancy = (
            self.static_street_occupancy
            if street
            else self.static_cv_occupancy
        )
        origin = info.origin
        quaternion = origin.orientation
        yaw = math.atan2(
            2.0 * (
                quaternion.w * quaternion.z
                + quaternion.x * quaternion.y
            ),
            1.0 - 2.0 * (
                quaternion.y * quaternion.y
                + quaternion.z * quaternion.z
            ),
        )
        map_cos = math.cos(yaw)
        map_sin = math.sin(yaw)
        delta_x = map_x - origin.position.x
        delta_y = map_y - origin.position.y
        columns = np.floor(
            (map_cos * delta_x + map_sin * delta_y) / info.resolution
        ).astype(np.int64)
        rows = np.floor(
            (-map_sin * delta_x + map_cos * delta_y) / info.resolution
        ).astype(np.int64)
        height, width = occupancy.shape
        valid = (
            (columns >= 0)
            & (columns < width)
            & (rows >= 0)
            & (rows < height)
        )
        values = np.zeros(map_x.shape, dtype=np.float64)
        values[valid] = occupancy[rows[valid], columns[valid]]
        return values, valid

    def log_sad_diagnostics(
        self,
        input_cloud: ParticleCloud,
        output_cloud: ParticleCloud,
        obstacle_sad: np.ndarray,
        street_sad: np.ndarray,
        combined_sad: np.ndarray,
        scores: np.ndarray,
        obstacle_cell_count: int,
        street_cell_count: int,
        obstacle_positive: float,
        obstacle_negative: float,
        street_positive: float,
        street_negative: float,
        min_positive_mass: float,
        obstacle_active: bool,
        street_active: bool,
        obstacle_weight_factor: float,
        street_factor: float,
        gain: float,
        factor: float,
    ) -> None:
        """Log positive-SAD spread, class mass, ESS and pose integrity."""
        period = float(self.get_parameter('diagnostic_log_period_sec').value)
        now = time.monotonic()
        if period > 0.0 and now - self.last_diagnostic_log_time < period:
            return
        self.last_diagnostic_log_time = now
        self.log_covariance_diagnostics(input_cloud)

        total = float(np.sum(scores))
        if total > 0.0 and math.isfinite(total):
            normalized = scores / total
            ess = 1.0 / float(np.sum(normalized * normalized))
        else:
            ess = 0.0
        sad_p10, sad_median, sad_p90 = np.percentile(
            combined_sad,
            [10.0, 50.0, 90.0],
        )
        obstacle_summary = np.percentile(obstacle_sad, [0.0, 50.0, 100.0])
        street_summary = np.percentile(street_sad, [0.0, 50.0, 100.0])
        combined_summary = np.percentile(combined_sad, [0.0, 50.0, 100.0])
        obstacle_best_index = int(np.argmin(obstacle_sad))
        street_best_index = int(np.argmin(street_sad))
        combined_best_index = int(np.argmin(combined_sad))

        def particle_pose(index: int):
            particle = input_cloud.particles[index]
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
            return particle.pose.position.x, particle.pose.position.y, yaw

        obstacle_best_pose = particle_pose(obstacle_best_index)
        street_best_pose = particle_pose(street_best_index)
        combined_best_pose = particle_pose(combined_best_index)
        obstacle_factor = (
            obstacle_weight_factor if obstacle_active else 0.0
        )
        active_street_factor = street_factor if street_active else 0.0
        semantic_factor_sum = obstacle_factor + active_street_factor
        obstacle_mix = obstacle_factor / semantic_factor_sum
        street_mix = active_street_factor / semantic_factor_sum
        weight_difference = np.asarray([
            abs(output.weight - source.weight)
            for source, output in zip(
                input_cloud.particles,
                output_cloud.particles,
            )
        ])
        self.get_logger().info(
            'CV grid positive-SAD diagnostics: '
            f'particles={len(input_cloud.particles)}, '
            f'cells=[{obstacle_cell_count}, {street_cell_count}], '
            f'class_mass=[obstacles=({obstacle_positive:.1f}, '
            f'{obstacle_negative:.1f}), street=({street_positive:.1f}, '
            f'{street_negative:.1f})], '
            f'active=[{int(obstacle_active)}, {int(street_active)}], '
            f'min_positive={min_positive_mass:.1f}, '
            f'factors=[{obstacle_weight_factor:.3f}, '
            f'{street_factor:.3f}], '
            f'mix=[{obstacle_mix:.3f}, {street_mix:.3f}], '
            f'gain={gain:.3f}, cv_factor={factor:.3f}, '
            f'local_occupancy=['
            f'obstacles={self.grid_mean_occupancy.get("obstacles", math.nan):.3f}, '
            f'street={self.grid_mean_occupancy.get("street", math.nan):.3f}], '
            f'obstacle_sad=['
            f'{obstacle_summary[0]:.6f}, {obstacle_summary[1]:.6f}, '
            f'{obstacle_summary[2]:.6f}], '
            f'street_sad=['
            f'{street_summary[0]:.6f}, {street_summary[1]:.6f}, '
            f'{street_summary[2]:.6f}], '
            f'combined_sad=['
            f'{combined_summary[0]:.6f}, {combined_summary[1]:.6f}, '
            f'{combined_summary[2]:.6f}], '
            f'sad_min={np.min(combined_sad):.6f}, '
            f'sad_p10={sad_p10:.6f}, '
            f'sad_median={sad_median:.6f}, sad_p90={sad_p90:.6f}, '
            f'sad_max={np.max(combined_sad):.6f}, '
            f'score_min={np.min(scores):.6g}, '
            f'score_max={np.max(scores):.6g}, '
            f'ess={ess:.1f}, ess_fraction={ess / len(scores):.3f}, '
            f'pose_position_max=0 m, pose_angle_max=0 rad, '
            f'weight_abs_mean={np.mean(weight_difference):.6g}, '
            f'weight_abs_max={np.max(weight_difference):.6g}'
        )
        self.get_logger().info(
            'CV grid positive-SAD best poses: '
            f'obstacles=({obstacle_best_pose[0]:.3f}, '
            f'{obstacle_best_pose[1]:.3f}, {obstacle_best_pose[2]:.3f}; '
            f'sad={obstacle_sad[obstacle_best_index]:.6f}), '
            f'street=({street_best_pose[0]:.3f}, '
            f'{street_best_pose[1]:.3f}, {street_best_pose[2]:.3f}; '
            f'sad={street_sad[street_best_index]:.6f}), '
            f'combined=({combined_best_pose[0]:.3f}, '
            f'{combined_best_pose[1]:.3f}, {combined_best_pose[2]:.3f}; '
            f'sad={combined_sad[combined_best_index]:.6f})'
        )

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

    @staticmethod
    def sample_distance_field(
        map_x: np.ndarray,
        map_y: np.ndarray,
        distance_map: np.ndarray,
        map_info,
        max_occ_dist: float,
    ) -> np.ndarray:
        """Sample one metric distance field at world-frame coordinates."""
        resolution = float(map_info.resolution)
        origin = map_info.origin
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
        delta_x = map_x - origin.position.x
        delta_y = map_y - origin.position.y
        columns = np.floor(
            (map_cos * delta_x + map_sin * delta_y) / resolution
        ).astype(np.int64)
        rows = np.floor(
            (-map_sin * delta_x + map_cos * delta_y) / resolution
        ).astype(np.int64)
        height, width = distance_map.shape
        valid = (
            (columns >= 0)
            & (columns < width)
            & (rows >= 0)
            & (rows < height)
        )
        distances = np.full(map_x.shape, max_occ_dist, dtype=np.float64)
        distances[valid] = distance_map[rows[valid], columns[valid]]
        return distances

    def score_semantic_template(
        self,
        msg: ParticleCloud,
        source_points: np.ndarray,
        expected_distance_map: np.ndarray,
        expected_map_info,
        opposite_distance_map: np.ndarray,
        opposite_map_info,
        max_points: int,
        sigma_hit: float,
        distance_exponent: float,
        z_hit: float,
        z_rand: float,
        max_occ_dist: float,
        sensor_max_range: float,
        mismatch_penalty: float,
    ):
        """Return soft semantic-template scores for one observed class."""
        points = source_points
        if points.shape[0] > max_points:
            # Match the deterministic, evenly spaced selection used by the
            # C++ model so repeated debug evaluations cannot jitter randomly.
            selected_indices = np.rint(np.linspace(
                0,
                points.shape[0] - 1,
                max_points,
            )).astype(np.int64)
            points = points[selected_indices]

        point_count = points.shape[0]
        random_likelihood = z_rand / sensor_max_range
        scores = np.empty(len(msg.particles), dtype=np.float64)
        mean_matches = np.empty(len(msg.particles), dtype=np.float64)
        mean_mismatches = np.empty(len(msg.particles), dtype=np.float64)

        for index, particle in enumerate(msg.particles):
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

            match_distances = self.sample_distance_field(
                map_x,
                map_y,
                expected_distance_map,
                expected_map_info,
                max_occ_dist,
            )
            mismatch_distances = self.sample_distance_field(
                map_x,
                map_y,
                opposite_distance_map,
                opposite_map_info,
                max_occ_dist,
            )
            match_strengths = np.exp(
                -0.5 * np.power(
                    match_distances / sigma_hit,
                    distance_exponent,
                )
            )
            mismatch_strengths = np.exp(
                -0.5 * np.power(
                    mismatch_distances / sigma_hit,
                    distance_exponent,
                )
            )
            point_likelihoods = (
                z_hit * match_strengths
                + random_likelihood
            )
            point_log_likelihoods = (
                np.log(np.maximum(point_likelihoods, 1.0e-12))
                - mismatch_penalty * mismatch_strengths
            )
            scores[index] = math.exp(float(np.mean(point_log_likelihoods)))
            mean_matches[index] = float(np.mean(match_strengths))
            mean_mismatches[index] = float(np.mean(mismatch_strengths))

        return scores, mean_matches, mean_mismatches, point_count

    def publish_cv_scores(self, msg: ParticleCloud) -> None:
        """Publish the unnormalized obstacle/street weighted likelihood product."""
        if (
            self.distance_map_m is None
            or self.street_distance_map_m is None
            or self.map_info is None
            or self.street_map_info is None
            or self.cv_points is None
            or self.street_points is None
            or self.cv_points.size == 0
            or self.street_points.size == 0
            or not msg.particles
        ):
            return
        if (
            msg.header.frame_id != self.map_frame_id
            or msg.header.frame_id != self.street_map_frame_id
        ):
            self.get_logger().warning(
                f'Particle frame {msg.header.frame_id!r} does not match '
                f'obstacle/street map frames {self.map_frame_id!r}, '
                f'{self.street_map_frame_id!r}'
            )
            return

        sigma_hit = float(self.get_parameter('sigma_hit').value)
        distance_exponent = float(
            self.get_parameter('distance_exponent').value
        )
        z_hit = float(self.get_parameter('z_hit').value)
        z_rand = float(self.get_parameter('z_rand').value)
        max_occ_dist = float(self.get_parameter('max_occ_dist').value)
        sensor_max_range = float(
            self.get_parameter('sensor_max_range').value
        )
        max_points = int(self.get_parameter('max_points').value)
        mismatch_penalty = float(
            self.get_parameter('semantic_mismatch_penalty').value
        )
        if (
            sigma_hit <= 0.0
            or distance_exponent <= 0.0
            or z_hit < 0.0
            or z_rand < 0.0
            or max_occ_dist <= 0.0
            or sensor_max_range <= 0.0
            or max_points < 1
            or mismatch_penalty < 0.0
        ):
            self.get_logger().error('Invalid CV likelihood parameters')
            return

        (
            obstacle_scores,
            obstacle_matches,
            obstacle_mismatches,
            obstacle_point_count,
        ) = self.score_semantic_template(
            msg,
            self.cv_points,
            self.distance_map_m,
            self.map_info,
            self.street_distance_map_m,
            self.street_map_info,
            max_points,
            sigma_hit,
            distance_exponent,
            z_hit,
            z_rand,
            max_occ_dist,
            sensor_max_range,
            mismatch_penalty,
        )
        (
            street_scores,
            street_matches,
            street_mismatches,
            street_point_count,
        ) = self.score_semantic_template(
            msg,
            self.street_points,
            self.street_distance_map_m,
            self.street_map_info,
            self.distance_map_m,
            self.map_info,
            max_points,
            sigma_hit,
            distance_exponent,
            z_hit,
            z_rand,
            max_occ_dist,
            sensor_max_range,
            mismatch_penalty,
        )

        obstacle_factor = float(
            self.get_parameter('cv_weight_factor').value
        )
        street_factor = float(
            self.get_parameter('cv_street_weight_factor').value
        )
        if obstacle_factor < 0.0 or street_factor < 0.0:
            self.get_logger().error('CV weight factors must be non-negative')
            return

        # This is the unnormalized CV-only equivalent of AMCL's separate
        # log-domain terms. No distribution-wide normalization is performed:
        # L_debug = L_obstacle^beta * L_street^gamma.
        weighted_obstacle_scores = np.power(
            obstacle_scores,
            obstacle_factor,
        )
        weighted_street_scores = np.power(
            street_scores,
            street_factor,
        )
        scores = weighted_obstacle_scores * weighted_street_scores
        output = copy.deepcopy(msg)
        for particle, score in zip(output.particles, scores):
            particle.weight = float(score)
        self.log_particle_differences(
            msg,
            output,
            scores,
            weighted_obstacle_scores,
            weighted_street_scores,
            obstacle_point_count,
            street_point_count,
            obstacle_factor,
            street_factor,
            mismatch_penalty,
            distance_exponent,
            obstacle_matches,
            obstacle_mismatches,
            street_matches,
            street_mismatches,
        )
        self.particle_cloud_pub.publish(output)

    @staticmethod
    def particle_cloud_statistics(cloud: ParticleCloud):
        """Return weighted planar mean and covariance for an AMCL cloud."""
        particles = cloud.particles
        if not particles:
            return None

        weights = np.asarray(
            [particle.weight for particle in particles],
            dtype=np.float64,
        )
        weights[~np.isfinite(weights) | (weights < 0.0)] = 0.0
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            weights.fill(1.0 / len(particles))
        else:
            weights /= weight_sum

        x = np.asarray(
            [particle.pose.position.x for particle in particles],
            dtype=np.float64,
        )
        y = np.asarray(
            [particle.pose.position.y for particle in particles],
            dtype=np.float64,
        )
        yaw = np.empty(len(particles), dtype=np.float64)
        for index, particle in enumerate(particles):
            orientation = particle.pose.orientation
            yaw[index] = math.atan2(
                2.0 * (
                    orientation.w * orientation.z
                    + orientation.x * orientation.y
                ),
                1.0 - 2.0 * (
                    orientation.y * orientation.y
                    + orientation.z * orientation.z
                ),
            )

        mean_x = float(np.sum(weights * x))
        mean_y = float(np.sum(weights * y))
        mean_yaw = math.atan2(
            float(np.sum(weights * np.sin(yaw))),
            float(np.sum(weights * np.cos(yaw))),
        )
        delta_x = x - mean_x
        delta_y = y - mean_y
        delta_yaw = np.arctan2(
            np.sin(yaw - mean_yaw),
            np.cos(yaw - mean_yaw),
        )
        covariance_xx = float(np.sum(weights * delta_x * delta_x))
        covariance_xy = float(np.sum(weights * delta_x * delta_y))
        covariance_yy = float(np.sum(weights * delta_y * delta_y))
        yaw_variance = float(np.sum(weights * delta_yaw * delta_yaw))
        stamp = cloud.header.stamp
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        return {
            'stamp_ns': stamp_ns,
            'mean_x': mean_x,
            'mean_y': mean_y,
            'mean_yaw': mean_yaw,
            'covariance_xx': covariance_xx,
            'covariance_xy': covariance_xy,
            'covariance_yy': covariance_yy,
            'position_covariance_trace': covariance_xx + covariance_yy,
            'yaw_variance': yaw_variance,
        }

    def log_covariance_diagnostics(self, cloud: ParticleCloud) -> None:
        """Log covariance growth relative to estimated Ackermann motion."""
        statistics = self.particle_cloud_statistics(cloud)
        if statistics is None:
            return

        delta_time = math.nan
        translation = math.nan
        rotation = math.nan
        position_trace_delta = math.nan
        position_trace_rate = math.nan
        position_trace_per_metre = math.nan
        yaw_variance_delta = math.nan
        previous = self.previous_particle_statistics
        if previous is not None:
            delta_time = (
                statistics['stamp_ns'] - previous['stamp_ns']
            ) * 1.0e-9
            if delta_time > 0.0:
                translation = math.hypot(
                    statistics['mean_x'] - previous['mean_x'],
                    statistics['mean_y'] - previous['mean_y'],
                )
                rotation = abs(math.atan2(
                    math.sin(
                        statistics['mean_yaw'] - previous['mean_yaw']
                    ),
                    math.cos(
                        statistics['mean_yaw'] - previous['mean_yaw']
                    ),
                ))
                position_trace_delta = (
                    statistics['position_covariance_trace']
                    - previous['position_covariance_trace']
                )
                position_trace_rate = position_trace_delta / delta_time
                if translation > 1.0e-6:
                    position_trace_per_metre = (
                        position_trace_delta / translation
                    )
                yaw_variance_delta = (
                    statistics['yaw_variance']
                    - previous['yaw_variance']
                )

        # Do not replace the temporal reference when the timer evaluates the
        # same ParticleCloud twice. This keeps the next delta tied to new AMCL
        # data rather than to a duplicate debug evaluation.
        if previous is None or statistics['stamp_ns'] != previous['stamp_ns']:
            self.previous_particle_statistics = statistics

        map_odom_distance, map_odom_yaw = self.map_to_odom_diagnostics()

        self.get_logger().info(
            'AMCL covariance diagnostics: '
            f'std_x={math.sqrt(statistics["covariance_xx"]):.4f} m, '
            f'std_y={math.sqrt(statistics["covariance_yy"]):.4f} m, '
            f'std_yaw={math.sqrt(statistics["yaw_variance"]):.4f} rad, '
            f'cov_xy={statistics["covariance_xy"]:.6g} m2, '
            f'position_cov_trace='
            f'{statistics["position_covariance_trace"]:.6g} m2, '
            f'dt={delta_time:.3f} s, '
            f'centroid_translation={translation:.4f} m, '
            f'centroid_rotation={rotation:.4f} rad, '
            f'position_cov_delta={position_trace_delta:.6g} m2, '
            f'position_cov_rate={position_trace_rate:.6g} m2/s, '
            f'position_cov_per_m={position_trace_per_metre:.6g} m, '
            f'yaw_variance_delta={yaw_variance_delta:.6g} rad2, '
            f'map_odom_distance={map_odom_distance:.4f} m, '
            f'map_odom_yaw={map_odom_yaw:.4f} rad'
        )

    def map_to_odom_diagnostics(self):
        """Return norm and yaw of the latest map <- odom transform."""
        map_frame = str(self.get_parameter('map_frame_id').value)
        odom_frame = str(self.get_parameter('odom_frame_id').value)
        try:
            transform = self.tf_buffer.lookup_transform(
                map_frame,
                odom_frame,
                Time(),
            )
        except TransformException as error:
            now = time.monotonic()
            if now - self.last_tf_diagnostic_warning_time >= 2.0:
                self.get_logger().warning(
                    f'Cannot inspect {map_frame} <- {odom_frame}: {error}'
                )
                self.last_tf_diagnostic_warning_time = now
            return math.nan, math.nan

        translation = transform.transform.translation
        orientation = transform.transform.rotation
        distance = math.hypot(translation.x, translation.y)
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
        return distance, yaw

    def log_particle_differences(
        self,
        input_cloud: ParticleCloud,
        output_cloud: ParticleCloud,
        cv_scores: np.ndarray,
        weighted_obstacle_scores: np.ndarray,
        weighted_street_scores: np.ndarray,
        obstacle_point_count: int,
        street_point_count: int,
        obstacle_factor: float,
        street_factor: float,
        mismatch_penalty: float,
        distance_exponent: float,
        obstacle_matches: np.ndarray,
        obstacle_mismatches: np.ndarray,
        street_matches: np.ndarray,
        street_mismatches: np.ndarray,
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

        self.log_covariance_diagnostics(input_cloud)

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
        # ParticleCloud intentionally retains the unnormalized likelihoods so
        # AMCL can reproduce the obstacle/street log-domain fusion.
        score_min = float(np.min(cv_scores))
        score_max = float(np.max(cv_scores))
        score_p10, score_median, score_p90 = np.percentile(
            cv_scores,
            [10.0, 50.0, 90.0],
        )
        score_ratio = score_max / score_min if score_min > 0.0 else math.inf

        def effective_sample_size(scores: np.ndarray) -> float:
            score_total = float(np.sum(scores))
            if score_total <= 0.0 or not math.isfinite(score_total):
                return 0.0
            normalized_scores = scores / score_total
            squared_sum = float(np.sum(normalized_scores ** 2))
            return 1.0 / squared_sum if squared_sum > 0.0 else 0.0

        obstacle_ess = effective_sample_size(weighted_obstacle_scores)
        street_ess = effective_sample_size(weighted_street_scores)
        combined_ess = effective_sample_size(cv_scores)
        particle_count = len(cv_scores)
        obstacle_ess_fraction = obstacle_ess / particle_count
        street_ess_fraction = street_ess / particle_count
        combined_ess_fraction = combined_ess / particle_count

        self.get_logger().info(
            'CV particle diagnostics: '
            f'particles={len(input_cloud.particles)}, '
            f'points_obstacles={obstacle_point_count}, '
            f'points_street={street_point_count}, '
            f'obstacle_factor={obstacle_factor:.3f}, '
            f'street_factor={street_factor:.3f}, '
            f'mismatch_penalty={mismatch_penalty:.3f}, '
            f'distance_exponent={distance_exponent:.3f}, '
            f'obstacle_match_mean={np.mean(obstacle_matches):.3f}, '
            f'obstacle_mismatch_mean={np.mean(obstacle_mismatches):.3f}, '
            f'street_match_mean={np.mean(street_matches):.3f}, '
            f'street_mismatch_mean={np.mean(street_mismatches):.3f}, '
            f'score_min={score_min:.6g}, '
            f'score_p10={score_p10:.6g}, '
            f'score_median={score_median:.6g}, '
            f'score_p90={score_p90:.6g}, '
            f'score_max={score_max:.6g}, '
            f'score_ratio={score_ratio:.6g}, '
            f'obstacle_ess_fraction={obstacle_ess_fraction:.3f}, '
            f'street_ess_fraction={street_ess_fraction:.3f}, '
            f'combined_ess={combined_ess:.1f}, '
            f'combined_ess_fraction={combined_ess_fraction:.3f}, '
            f'pose_position_mean={np.mean(position_differences):.6g} m, '
            f'pose_position_max={np.max(position_differences):.6g} m, '
            f'pose_angle_mean={np.nanmean(orientation_differences):.6g} rad, '
            f'pose_angle_max={np.nanmax(orientation_differences):.6g} rad, '
            f'weight_abs_mean={np.mean(weight_differences):.6g}, '
            f'weight_abs_max={np.max(weight_differences):.6g}'
        )

    def map_callback(self, msg: OccupancyGrid, street: bool) -> None:
        """Build a metric field for the obstacle or street static map."""
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
            resolution = float(msg.info.resolution)
            max_occ_dist = float(
                self.get_parameter('max_occ_dist').value
            )
            if resolution <= 0.0 or max_occ_dist <= 0.0:
                self.get_logger().error(
                    'Map resolution and max_occ_dist must be positive'
                )
                return
            distance_map = self.metric_distance(
                occupancy,
                threshold,
                resolution,
                max_occ_dist,
            )
            if street:
                self.street_distance_map_m = distance_map
                self.static_street_occupancy = (
                    occupancy >= threshold
                ).astype(np.float64)
                self.street_map_info = copy.deepcopy(msg.info)
                self.street_map_frame_id = msg.header.frame_id
            else:
                self.distance_map_m = distance_map
                self.static_cv_occupancy = (
                    occupancy >= threshold
                ).astype(np.float64)
                self.map_info = copy.deepcopy(msg.info)
                self.map_frame_id = msg.header.frame_id
        except cv2.error as error:
            self.get_logger().error(
                f'Cannot compute CV distance transform: {error}'
            )
            return


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
