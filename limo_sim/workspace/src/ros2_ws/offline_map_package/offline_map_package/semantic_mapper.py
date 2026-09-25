"""Direct PointCloud2 semantic mapping with optional Cartographer submap poses."""

from array import array
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, QoSProfile, ReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from offline_map_package.map_filters import median_saved_semantic
from offline_map_package.semantic_grid import (
    CLASS_NAMES, Geometry, SemanticGrid,
    read_class_cloud, transform_xy,
)


def planar_pose(position, rotation):
    """Validate a planar transform and return x, y and yaw."""
    values = [position.x, position.y, position.z,
              rotation.x, rotation.y, rotation.z, rotation.w]
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite transform')
    q = np.array(values[3:], dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        raise ValueError('Zero quaternion')
    x, y, z, w = q / norm
    if abs(2.0 * (w * y - z * x)) > 0.05 or abs(2.0 * (w * x + y * z)) > 0.05:
        raise ValueError('Semantic mapper requires planar frames (e.g. base_link -> map)')
    return (position.x, position.y,
            math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def stamp_ns(stamp):
    return int(stamp.sec) * 1000000000 + int(stamp.nanosec)


def map_pgm(values, mode):
    """Encode costs as a Nav2 raw-, scale- or trinary-mode binary PGM."""
    if values.ndim != 2:
        raise ValueError('Saved map must be a two-dimensional array')
    known = values >= 0
    if np.any(values < -1) or np.any(values[known] > 100):
        raise ValueError('Map costs must be between 0 and 100')
    if mode == 'raw':
        # Nav2 raw mode reads 0..100 directly; 255 represents unknown.
        pixels = np.where(known, values, 255).astype(np.uint8)
    elif mode == 'scale':
        # Scale grayscale PGM cannot distinguish unknown from a cost.
        if not np.all(known):
            raise ValueError('Scale PGM cannot encode unknown cells')
        pixels = np.rint(
            (100.0 - values.astype(np.float64)) * (255.0 / 100.0)
        ).astype(np.uint8)
    elif mode == 'trinary':
        pixels = np.full(values.shape, 205, dtype=np.uint8)
        pixels[known & (values <= 25)] = 254
        pixels[known & (values >= 65)] = 0
    else:
        raise ValueError('Map mode must be raw, scale or trinary')
    # OccupancyGrid starts at the lower-left; image rows start at the upper-left.
    pixels = np.flipud(pixels)
    header = 'P5\n# CREATOR: offline_map_package semantic_mapper\n{} {}\n255\n'.format(
        values.shape[1], values.shape[0])
    return header.encode('ascii') + pixels.tobytes()


def map_yaml(image_name, geometry, mode):
    """Create metadata accepted by the Foxy/Humble Nav2 map server."""
    if mode in ('raw', 'scale'):
        # Raw mode ignores thresholds. Scale mode uses the full [0, 1]
        # interval to reconstruct costs without the default threshold stretch.
        occupied_thresh, free_thresh = '1.0', '0.0'
    elif mode == 'trinary':
        # These thresholds separate the conventional 0/205/254 PGM pixels.
        occupied_thresh, free_thresh = '0.65', '0.196'
    else:
        raise ValueError('Map mode must be raw, scale or trinary')
    return (
        'image: {}\n'
        'mode: {}\n'
        'resolution: {:.17g}\n'
        'origin: [{:.17g}, {:.17g}, {:.17g}]\n'
        'negate: 0\n'
        'occupied_thresh: {}\n'
        'free_thresh: {}\n'
    ).format(image_name, mode, geometry.resolution, *geometry.origin,
             occupied_thresh, free_thresh)


def combine_semantic_and_laser(semantic, laser):
    """Overlay semantic evidence while preserving laser obstacles at cost 100."""
    if semantic.shape != laser.shape:
        raise ValueError('Semantic and laser map geometry differs')
    complete = laser.copy()
    semantic_known = semantic >= 0
    complete[semantic_known] = semantic[semantic_known]
    complete[laser == 100] = 100
    return complete


def cv_obstacle_map(semantic, laser):
    """Binarize CV evidence and fill unobserved cells only with laser free space."""
    if semantic.shape != laser.shape:
        raise ValueError('Semantic and laser map geometry differs')
    output = np.full(semantic.shape, -1, dtype=np.int8)
    output[(semantic >= 0) & (semantic < 40)] = 0
    output[(semantic >= 40) & (semantic <= 95)] = 100
    output[(semantic == -1) & (laser == 0)] = 0
    return output


class SemanticMapper(Node):
    """Fuse endpoint classes; expose cost grids and Nav2 map files."""

    def __init__(self):
        super().__init__('semantic_mapper')
        defaults = {
            'cloud_topic': '/limo/cv_package/visual_ptcld/points',
            'reference_map_topic': '/map', 'map_frame': 'map', 'odom_frame': 'odom',
            'pose_source': 'tf', 'submap_topic': '/submap_list', 'trajectory_id': 0,
            'resolution': 0.05, 'turquoise_cost': 60, 'white_cost': 30,
            'boardwalk_cost': 90, 'hit_log_odds': 0.85, 'miss_log_odds': 0.4,
            'log_odds_limit': 3.0, 'min_evidence': 0.5,
            'free_confirmations': 1,
            'max_cells': 2000000, 'max_output_cells': 4000000,
            'max_input_points': 100000, 'publish_rate_hz': 4.0,
            'tf_wait_sec': 0.5, 'submap_max_age_sec': 2.0,
            'save_directory': '', 'save_map_name': 'limo_map',
            'save_median_kernel': 1,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.config = {name: self.get_parameter(name).value for name in defaults}
        self.pose_source = self.config['pose_source']
        if self.pose_source not in ('tf', 'cartographer'):
            raise ValueError('pose_source must be tf or cartographer')
        for name in ('publish_rate_hz', 'tf_wait_sec', 'submap_max_age_sec'):
            if not math.isfinite(self.config[name]) or self.config[name] <= 0:
                raise ValueError(name + ' must be finite and positive')
        if self.config['max_input_points'] < 1:
            raise ValueError('max_input_points must be positive')
        median_kernel = self.config['save_median_kernel']
        if (isinstance(median_kernel, bool) or not isinstance(median_kernel, int)
                or median_kernel < 1
                or median_kernel % 2 == 0):
            raise ValueError('save_median_kernel must be a positive odd integer')
        if not self.config['map_frame'] or not self.config['odom_frame']:
            raise ValueError('Map and odom frames must be nonempty')
        self.grid = self.make_grid()
        self.laser_map = None
        self.reference_geometry = None
        self.last_geometry = None
        self.pending = None
        self.last_stamp = -1
        self.submaps = {}
        self.submap_stamp = None
        self.dirty = False
        self.map_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cloud_sub = self.create_subscription(
            PointCloud2, self.config['cloud_topic'], self.cloud_callback, sensor_qos)
        # SLAM supplies the complete laser layer, including geometry and unknowns.
        self.map_sub = self.create_subscription(
            OccupancyGrid, self.config['reference_map_topic'], self.map_callback, self.map_qos)
        if self.pose_source == 'cartographer':
            try:
                from cartographer_ros_msgs.msg import SubmapList
            except ImportError as error:
                raise RuntimeError(
                    'pose_source=cartographer requires cartographer_ros_msgs') from error
            self.submap_sub = self.create_subscription(
                SubmapList, self.config['submap_topic'], self.submaps_callback, sensor_qos)
        prefix = '/limo/map_package/offline'
        self.outputs = {
            name: self.create_publisher(OccupancyGrid, prefix + '/map/' + name + '_map',
                                        self.map_qos)
            for name in CLASS_NAMES}
        self.combined_pub = self.create_publisher(
            OccupancyGrid, prefix + '/map/combined_grid', self.map_qos)
        self.status_pub = self.create_publisher(
            String, prefix + '/map_saver/status', self.map_qos)
        self.reset_service = self.create_service(Trigger, prefix + '/reset_map', self.reset_map)
        self.save_service = self.create_service(
            Trigger, prefix + '/map_saver/save_map', self.save_map)
        self.retry_timer = self.create_timer(0.05, self.consume_pending)
        self.publish_timer = self.create_timer(
            1.0 / self.config['publish_rate_hz'], self.publish_maps)
        costs = ', '.join(name + '=' + str(int(cost))
                          for name, cost in zip(CLASS_NAMES, self.grid.costs))
        self.status('Ready: exterior_road=0; interior_road=0; laser=100 from '
                    + self.config['reference_map_topic'] + '; ' + costs
                    + '; pose_source=' + self.pose_source)
        if self.pose_source == 'tf':
            self.get_logger().info(
                'TF mode corrects classifications on revisits. Historical pose corrections '
                'require Cartographer submaps mode or a replay with corrected poses.')

    def make_grid(self):
        c = self.config
        grid = SemanticGrid(
            resolution=c['resolution'],
            costs=[c[name + '_cost'] for name in CLASS_NAMES],
            hit=c['hit_log_odds'], miss=c['miss_log_odds'],
            limit=c['log_odds_limit'], threshold=c['min_evidence'],
            max_cells=c['max_cells'], max_output_cells=c['max_output_cells'],
            free_confirmations=c['free_confirmations'])
        if self.pose_source == 'tf':
            grid.set_pose((0, 0), (0.0, 0.0, 0.0))
        return grid

    def status(self, message):
        self.status_pub.publish(String(data=message))
        self.get_logger().info(message)

    def cloud_callback(self, msg):
        stamp = stamp_ns(msg.header.stamp)
        if stamp <= 0 or not msg.header.frame_id:
            self.get_logger().warning('Cloud requires nonzero sensor stamp and frame_id',
                                      throttle_duration_sec=3.0)
            return
        # Do not count retries or delayed duplicate clouds as new observations.
        newest = max(self.last_stamp, stamp_ns(self.pending[0].header.stamp)
                     if self.pending else -1)
        if stamp <= newest:
            self.get_logger().warning(
                'Ignoring duplicate/out-of-order cloud; reset_map before replaying a bag',
                throttle_duration_sec=3.0)
            return
        if msg.width * msg.height > self.config['max_input_points']:
            self.get_logger().error(
                'Point cloud exceeds max_input_points', throttle_duration_sec=3.0)
            return
        self.pending = (msg, time.monotonic())
        self.consume_pending()

    def cloud_in_submap(self, msg):
        if self.pose_source == 'tf':
            transform = self.tf_buffer.lookup_transform(
                self.config['map_frame'], msg.header.frame_id,
                Time.from_msg(msg.header.stamp))
            pose = planar_pose(transform.transform.translation, transform.transform.rotation)
            return (0, 0), pose
        if not self.submaps or self.submap_stamp is None:
            raise ValueError('Waiting for Cartographer submap_list')
        age = abs(stamp_ns(msg.header.stamp) - stamp_ns(self.submap_stamp)) * 1e-9
        if age > self.config['submap_max_age_sec']:
            raise ValueError('Cartographer submap_list is too far from cloud timestamp')
        eligible = [key for key, frozen in self.submaps.items() if not frozen
                    and key[0] == self.config['trajectory_id']]
        if not eligible:
            raise ValueError('No non-frozen submaps for selected Cartographer trajectory')
        key = max(eligible, key=lambda item: item[1])
        # The global pose and map->odom transform must refer to the same epoch.
        # Sensor motion between the two timestamps is resolved in continuous odom.
        transform = self.tf_buffer.lookup_transform_full(
            self.config['map_frame'], Time.from_msg(self.submap_stamp),
            msg.header.frame_id, Time.from_msg(msg.header.stamp),
            self.config['odom_frame'])
        pose = planar_pose(transform.transform.translation, transform.transform.rotation)
        return key, pose

    def consume_pending(self):
        if self.pending is None:
            return
        msg, received = self.pending
        try:
            key, pose = self.cloud_in_submap(msg)
        except (TransformException, ValueError) as error:
            if time.monotonic() - received > self.config['tf_wait_sec']:
                self.pending = None
                self.get_logger().warning(
                    'Dropping cloud: ' + str(error), throttle_duration_sec=3.0)
            return
        self.pending = None
        try:
            xy, labels = read_class_cloud(msg)
            world = transform_xy(xy, pose)
            local = transform_xy(world, self.grid.poses[key], inverse=True)
            if self.grid.update(local, labels, key):
                self.dirty = True
            self.last_stamp = stamp_ns(msg.header.stamp)
        except (ValueError, MemoryError, OverflowError) as error:
            self.get_logger().error(str(error), throttle_duration_sec=3.0)

    def submaps_callback(self, msg):
        if msg.header.frame_id != self.config['map_frame']:
            self.get_logger().error(
                'SubmapList frame differs from map_frame', throttle_duration_sec=3.0)
            return
        if self.submap_stamp and stamp_ns(msg.header.stamp) < stamp_ns(self.submap_stamp):
            return
        try:
            poses = {(entry.trajectory_id, entry.submap_index):
                     planar_pose(entry.pose.position, entry.pose.orientation)
                     for entry in msg.submap}
        except ValueError as error:
            self.get_logger().error(str(error), throttle_duration_sec=3.0)
            return
        self.submaps = {(entry.trajectory_id, entry.submap_index): entry.is_frozen
                        for entry in msg.submap}
        for key, pose in poses.items():
            self.dirty = self.grid.set_pose(key, pose) or self.dirty
        visible = set(poses)
        if self.grid.visible != visible:
            self.grid.visible = visible
            self.dirty = True
        self.submap_stamp = msg.header.stamp

    def map_callback(self, msg):
        if msg.header.frame_id != self.config['map_frame']:
            self.get_logger().warning('Reference map frame differs from map_frame',
                                      throttle_duration_sec=3.0)
            return
        try:
            pose = planar_pose(msg.info.origin.position, msg.info.origin.orientation)
            resolution = float(msg.info.resolution)
            if (not math.isfinite(resolution) or resolution <= 0
                    or msg.info.width < 1 or msg.info.height < 1
                    or msg.info.width * msg.info.height > self.config['max_output_cells']):
                raise ValueError('Invalid or oversized reference map')
            geometry = Geometry(resolution, msg.info.width, msg.info.height, pose)
            values = np.asarray(msg.data, dtype=np.int16)
            if (values.size != geometry.width * geometry.height
                    or np.any(values < -1) or np.any(values > 100)):
                raise ValueError('Invalid laser map data length or occupancy values')
            laser = values.astype(np.int8).reshape(geometry.height, geometry.width)
        except ValueError as error:
            self.get_logger().warning(str(error), throttle_duration_sec=3.0)
            return
        # Replace atomically even when geometry is unchanged: SLAM can remove
        # or relocate obstacles after a correction. Never accumulate old hits.
        self.reference_geometry = geometry
        self.laser_map = laser
        self.dirty = True

    def save_geometry(self):
        """Use exactly the geometry of the latest accepted SLAM laser map."""
        if self.reference_geometry is None or self.laser_map is None:
            raise ValueError('Waiting for laser map on '
                             + self.config['reference_map_topic'])
        return self.reference_geometry

    def message(self, values, geometry, stamp):
        msg = OccupancyGrid()
        msg.header.frame_id = self.config['map_frame']
        msg.header.stamp = stamp
        msg.info.resolution = geometry.resolution
        msg.info.width, msg.info.height = geometry.width, geometry.height
        msg.info.origin.position.x = float(geometry.origin[0])
        msg.info.origin.position.y = float(geometry.origin[1])
        msg.info.origin.orientation.z = math.sin(geometry.origin[2] / 2.0)
        msg.info.origin.orientation.w = math.cos(geometry.origin[2] / 2.0)
        msg.data = array('b', values.ravel().tobytes())
        return msg

    def publish_maps(self):
        if not self.dirty or self.laser_map is None:
            return
        try:
            # Use one geometry for both sources so the live combined_grid is
            # the same laser/CV overlay that is written by save_map().
            geometry = self.save_geometry()
            rendered = self.grid.render(geometry)
        except (ValueError, MemoryError) as error:
            self.get_logger().error(str(error), throttle_duration_sec=3.0)
            return
        geometry, layers, semantic = rendered
        laser = self.laser_map
        combined = combine_semantic_and_laser(semantic, laser)
        self.last_geometry = geometry
        stamp = self.get_clock().now().to_msg()
        for name, layer in zip(CLASS_NAMES, layers):
            self.outputs[name].publish(self.message(layer, geometry, stamp))
        self.combined_pub.publish(self.message(combined, geometry, stamp))
        self.dirty = False

    def reset_map(self, _request, response):
        # Clear the retained maps, including the last latched view.
        geometry = self.reference_geometry or self.last_geometry
        self.grid = self.make_grid()
        self.submaps, self.submap_stamp = {}, None
        self.pending = None
        self.last_stamp = -1
        if geometry is not None:
            unknown = np.full((geometry.height, geometry.width), -1, dtype=np.int8)
            stamp = self.get_clock().now().to_msg()
            for publisher in self.outputs.values():
                publisher.publish(self.message(unknown, geometry, stamp))
        self.dirty = True
        # Reset only CV evidence; SLAM owns and resets its own laser map.
        self.publish_maps()
        response.success, response.message = True, 'CV evidence reset; SLAM laser map retained'
        self.status(response.message)
        return response

    def save_map(self, _request, response):
        temporary_files = []
        try:
            geometry = self.save_geometry()
            rendered = self.grid.render(geometry)
            if rendered is None:
                raise ValueError('Cannot render the complete map')
            _geometry, _layers, combined = rendered
            semantic = median_saved_semantic(combined, self.config['save_median_kernel'])
            laser = self.laser_map
            complete = combine_semantic_and_laser(semantic, laser)
            cv_obstacles = cv_obstacle_map(semantic, laser)
            if self.config['save_directory']:
                directory = Path(self.config['save_directory']).expanduser()
            else:
                root = next((p for p in Path(__file__).resolve().parents
                             if (p / 'src' / 'ros2_ws').is_dir()), None)
                if root is None:
                    raise ValueError('Set save_directory explicitly')
                directory = root / 'ros2_maps' / 'semantic'
            directory.mkdir(parents=True, exist_ok=True)
            map_name = str(self.config['save_map_name']).strip()
            if (not map_name or Path(map_name).name != map_name
                    or map_name in ('.', '..')
                    or map_name.endswith(('.pgm', '.yaml'))):
                raise ValueError('save_map_name must be a filename stem without a path or suffix')

            outputs = (
                (map_name + '_complete', complete, 'raw'),
                (map_name + '_laser', laser, 'raw'),
                (map_name + '_cv_obstacle', cv_obstacles, 'trinary'),
            )
            replacements = []
            for stem, values, mode in outputs:
                pgm_path = directory / (stem + '.pgm')
                yaml_path = directory / (stem + '.yaml')
                pgm_fd, pgm_temporary = tempfile.mkstemp(
                    prefix='.' + stem + '_', suffix='.pgm.partial', dir=str(directory))
                temporary_files.append(pgm_temporary)
                with os.fdopen(pgm_fd, 'wb') as stream:
                    stream.write(map_pgm(values, mode))
                replacements.append((pgm_temporary, pgm_path))

                yaml_fd, yaml_temporary = tempfile.mkstemp(
                    prefix='.' + stem + '_', suffix='.yaml.partial', dir=str(directory))
                temporary_files.append(yaml_temporary)
                with os.fdopen(yaml_fd, 'w', encoding='utf-8') as stream:
                    stream.write(map_yaml(pgm_path.name, geometry, mode))
                replacements.append((yaml_temporary, yaml_path))

            for temporary, destination in replacements:
                os.replace(temporary, destination)
                temporary_files.remove(temporary)
            response.success = True
            response.message = 'Saved complete, laser and CV obstacle maps in {}'.format(
                directory)
        except (OSError, ValueError, MemoryError) as error:
            response.success, response.message = False, str(error)
        finally:
            for temporary in temporary_files:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
        self.status(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SemanticMapper()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
