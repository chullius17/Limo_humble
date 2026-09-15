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
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from offline_map_package.semantic_grid import (
    CLASS_IDS, CLASS_NAMES, Geometry, SemanticGrid, read_class_cloud, transform_xy,
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


class SemanticMapper(Node):
    """Fuse endpoint classes; expose cost grids and exact numeric snapshots."""

    def __init__(self):
        super().__init__('semantic_mapper')
        defaults = {
            'cloud_topic': '/limo/cv_package/visual_ptcld/points',
            'reference_map_topic': '/map', 'map_frame': 'map', 'odom_frame': 'odom',
            'pose_source': 'tf', 'submap_topic': '/submap_list', 'trajectory_id': 0,
            'resolution': 0.05, 'turquoise_cost': 60, 'white_cost': 30,
            'boardwalk_cost': 90, 'hit_log_odds': 0.85, 'miss_log_odds': 0.4,
            'log_odds_limit': 3.0, 'min_evidence': 0.5,
            'max_cells': 2000000, 'max_output_cells': 4000000,
            'max_input_points': 100000, 'publish_rate_hz': 1.0,
            'tf_wait_sec': 0.5, 'submap_max_age_sec': 2.0,
            'save_directory': '',
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
        if not self.config['map_frame'] or not self.config['odom_frame']:
            raise ValueError('Map and odom frames must be nonempty')
        self.grid = self.make_grid()
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
        # Volatile subscription accepts both Cartographer and transient-local
        # SLAM publishers. It intentionally waits for a current map publication.
        self.map_sub = self.create_subscription(
            OccupancyGrid, self.config['reference_map_topic'], self.map_callback, sensor_qos)
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
        self.status('Ready: blue ignored; ' + costs + '; pose_source=' + self.pose_source)
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
            max_cells=c['max_cells'], max_output_cells=c['max_output_cells'])
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
        except ValueError as error:
            self.get_logger().warning(str(error), throttle_duration_sec=3.0)
            return
        if self.reference_geometry != geometry:
            self.reference_geometry = geometry
            self.dirty = True

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
        if not self.dirty:
            return
        try:
            rendered = self.grid.render(self.reference_geometry)
        except (ValueError, MemoryError) as error:
            self.get_logger().error(str(error), throttle_duration_sec=3.0)
            return
        if rendered is None:
            return
        geometry, layers, combined = rendered
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
        self.pending, self.last_stamp = None, -1
        if geometry is not None:
            unknown = np.full((geometry.height, geometry.width), -1, dtype=np.int8)
            stamp = self.get_clock().now().to_msg()
            for publisher in [self.combined_pub] + list(self.outputs.values()):
                publisher.publish(self.message(unknown, geometry, stamp))
        self.dirty = True
        response.success, response.message = True, 'Semantic evidence reset'
        self.status(response.message)
        return response

    def save_map(self, _request, response):
        temporary = None
        try:
            rendered = self.grid.render(self.reference_geometry)
            if rendered is None or not self.grid.sequence:
                raise ValueError('No semantic observations to save')
            geometry, layers, combined = rendered
            if self.config['save_directory']:
                directory = Path(self.config['save_directory']).expanduser()
            else:
                root = next((p for p in Path(__file__).resolve().parents
                             if (p / 'src' / 'ros2_ws').is_dir()), None)
                if root is None:
                    raise ValueError('Set save_directory explicitly')
                directory = root / 'ros2_maps' / 'semantic'
            directory.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(
                prefix='semantic_', suffix='.partial', dir=str(directory))
            with os.fdopen(fd, 'wb') as stream:
                # Preserve the actual costs and unknown values without image thresholds.
                np.savez_compressed(
                    stream, format_version=1, frame_id=self.config['map_frame'],
                    resolution=geometry.resolution, origin=geometry.origin,
                    class_ids=CLASS_IDS, costs=self.grid.costs,
                    turquoise=layers[0], white=layers[1], boardwalk=layers[2],
                    combined=combined, sensor_stamp_ns=self.last_stamp,
                    pose_source=self.pose_source)
            destination = str(Path(temporary).with_suffix('.npz'))
            os.rename(temporary, destination)
            temporary = None
            response.success, response.message = True, 'Saved semantic snapshot: ' + destination
        except (OSError, ValueError, MemoryError) as error:
            response.success, response.message = False, str(error)
        finally:
            if temporary is not None:
                os.unlink(temporary)
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
