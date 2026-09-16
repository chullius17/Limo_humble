"""Exercise ROS messages, timestamped TF and exact map snapshots."""

from array import array
from types import SimpleNamespace

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
from geometry_msgs.msg import Pose, TransformStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from std_srvs.srv import Trigger

from offline_map_package.semantic_mapper import (
    SemanticMapper, combine_semantic_and_laser, cv_obstacle_map,
)


@pytest.fixture
def node():
    rclpy.init(args=['--ros-args', '-p', 'resolution:=1.0'])
    mapper = SemanticMapper()
    try:
        yield mapper
    finally:
        mapper.destroy_node()
        rclpy.shutdown()


def transform(node, sec, x=0.0):
    msg = TransformStamped()
    msg.header.frame_id, msg.child_frame_id = 'map', 'base_link'
    msg.header.stamp.sec = sec
    msg.transform.translation.x = float(x)
    msg.transform.rotation.w = 1.0
    node.tf_buffer.set_transform(msg, 'test')


def cloud(sec=1, labels=(1, 2, 3, 4)):
    msg = PointCloud2()
    msg.header.frame_id, msg.header.stamp.sec = 'base_link', sec
    dtype = np.dtype({'names': ['x', 'y', 'z', 'class_id'],
                      'formats': ['<f4', '<f4', '<f4', 'u1'],
                      'offsets': [0, 4, 8, 12], 'itemsize': 16})
    points = np.zeros(len(labels), dtype=dtype)
    points['x'] = np.arange(len(labels)) + 0.1
    points['y'] = 0.1
    points['class_id'] = labels
    layout = [('x', 0, 7), ('y', 4, 7), ('z', 8, 7), ('class_id', 12, 2)]
    msg.fields = [PointField(name=name, offset=offset, datatype=datatype, count=1)
                  for name, offset, datatype in layout]
    msg.width, msg.height = len(labels), 1
    msg.point_step, msg.row_step = 16, 16 * msg.width
    msg.data = array('B', points.tobytes())
    return msg


def scan(sec=1, ranges=(1.0,), angle_min=0.0, angle_increment=1.0):
    msg = LaserScan()
    msg.header.frame_id, msg.header.stamp.sec = 'base_link', sec
    msg.angle_min, msg.angle_increment = angle_min, angle_increment
    msg.range_min, msg.range_max = 0.1, 10.0
    msg.ranges = list(ranges)
    return msg


def reference_map(values, resolution=1.0):
    values = np.asarray(values, dtype=np.int8)
    msg = OccupancyGrid()
    msg.header.frame_id = 'map'
    msg.info.resolution = resolution
    msg.info.width, msg.info.height = values.shape[1], values.shape[0]
    msg.info.origin.orientation.w = 1.0
    msg.data = array('b', values.ravel().tobytes())
    return msg


def read_pgm(path):
    with path.open('rb') as stream:
        assert stream.readline() == b'P5\n'
        assert stream.readline().startswith(b'# CREATOR:')
        width, height = (int(value) for value in stream.readline().split())
        assert stream.readline() == b'255\n'
        return np.frombuffer(stream.read(), dtype=np.uint8).reshape(height, width)


def read_scale_costs(path):
    """Decode the scale PGM written with YAML thresholds 0.0 and 1.0."""
    return np.rint((1.0 - read_pgm(path) / 255.0) * 100.0).astype(np.int8)


def read_trinary_costs(path):
    """Decode the known cells of the conventional trinary PGM."""
    pixels = read_pgm(path)
    costs = np.full(pixels.shape, -1, dtype=np.int8)
    costs[pixels == 254] = 0
    costs[pixels == 0] = 100
    return costs


def test_uses_sensor_tf_not_latest_and_deduplicates(node):
    transform(node, 1, 2.0)
    transform(node, 2, 10.0)
    msg = cloud()
    node.cloud_callback(msg)
    geometry, layers, combined = node.grid.render()
    assert geometry.origin[0] == 2.0  # blue at x=2 is observed road
    assert combined.tolist() == [[0, 60, 30, 90]]
    node.cloud_callback(msg)
    node.consume_pending()
    assert node.grid.sequence == 1
    output = node.message(combined, geometry, msg.header.stamp)
    assert list(output.data) == [0, 60, 30, 90]
    assert output.header.frame_id == 'map'
    node.publish_maps()
    assert not node.dirty


def test_missing_tf_is_retried_without_integration(node):
    node.cloud_callback(cloud())
    assert node.pending is not None
    assert node.grid.sequence == 0
    transform(node, 1)
    node.consume_pending()
    assert node.grid.sequence == 1
    assert node.pending is None


def test_scan_endpoints_are_accumulated_at_sensor_timestamp(node):
    transform(node, 1, 1.0)
    transform(node, 2, 10.0)
    node.scan_callback(scan(1, (1.1, 2.1), angle_increment=np.pi / 2))
    assert node.laser_grid.cells == {(2, 0), (1, 2)}
    node.scan_callback(scan(1, (5.0,)))
    assert node.laser_grid.cells == {(2, 0), (1, 2)}


def test_scan_waits_for_timestamped_tf(node):
    node.scan_callback(scan(1, (1.1,)))
    assert node.pending_scan is not None
    assert not node.laser_grid.cells
    transform(node, 1, 2.0)
    node.consume_pending_scan()
    assert node.pending_scan is None
    assert node.laser_grid.cells == {(3, 0)}


@pytest.mark.parametrize('road_class', [1, 5])
def test_road_cloud_clears_obstacle_and_saves_zero_cost(node, tmp_path, road_class):
    node.config['save_directory'] = str(tmp_path)
    node.config['save_median_kernel'] = 1
    node.map_callback(reference_map([[0, 0]]))
    node.laser_grid.update(np.array([[1.1, 0.1]]))
    for sec in range(1, 16):
        transform(node, sec)
        node.cloud_callback(cloud(sec, (4 if sec <= 10 else road_class,)))
    assert node.grid.render()[2].tolist() == [[0]]
    node.publish_maps()
    result = node.save_map(Trigger.Request(), Trigger.Response())
    assert result.success, result.message
    assert read_scale_costs(tmp_path / 'limo_map_complete.pgm').tolist() == [[0, 100]]
    assert read_trinary_costs(tmp_path / 'limo_map_laser.pgm').tolist() == [[0, 100]]
    assert read_trinary_costs(tmp_path / 'limo_map_cv_obstacle.pgm').tolist() == [[0, -1]]


def test_reset_and_save_exact_costs(node, tmp_path):
    node.map_callback(reference_map([[0, 0, 100, -1, 100]]))
    node.laser_grid.update(np.array([[2.1, 0.1], [4.1, 0.1]]))
    node.config['save_median_kernel'] = 1
    transform(node, 1)
    node.cloud_callback(cloud())
    node.config['save_directory'] = str(tmp_path)
    result = node.save_map(Trigger.Request(), Trigger.Response())
    assert result.success, result.message
    assert read_scale_costs(tmp_path / 'limo_map_complete.pgm').tolist() == [
        [0, 60, 100, 90, 100]]
    assert read_trinary_costs(tmp_path / 'limo_map_laser.pgm').tolist() == [
        [0, 0, 100, 0, 100]]
    assert read_trinary_costs(tmp_path / 'limo_map_cv_obstacle.pgm').tolist() == [
        [0, 100, -1, 100, -1]]
    metadata = (tmp_path / 'limo_map_complete.yaml').read_text(encoding='utf-8')
    assert 'image: limo_map_complete.pgm\n' in metadata
    assert 'mode: scale\n' in metadata
    assert 'occupied_thresh: 1.0\n' in metadata
    assert 'free_thresh: 0.0\n' in metadata
    assert 'resolution: 1\n' in metadata
    assert 'origin: [0, 0, 0]\n' in metadata
    laser_metadata = (tmp_path / 'limo_map_laser.yaml').read_text(encoding='utf-8')
    assert 'image: limo_map_laser.pgm\n' in laser_metadata
    assert 'mode: trinary\n' in laser_metadata
    assert 'occupied_thresh: 0.65\n' in laser_metadata
    assert 'free_thresh: 0.196\n' in laser_metadata
    cv_metadata = (tmp_path / 'limo_map_cv_obstacle.yaml').read_text(encoding='utf-8')
    assert 'image: limo_map_cv_obstacle.pgm\n' in cv_metadata
    assert 'mode: trinary\n' in cv_metadata
    assert 'occupied_thresh: 0.65\n' in cv_metadata
    assert 'free_thresh: 0.196\n' in cv_metadata
    assert not list(tmp_path.glob('*.npz'))
    node.publish_maps()
    node.reset_map(Trigger.Request(), Trigger.Response())
    assert node.grid.sequence == 0
    assert node.last_stamp == -1
    assert node.pending is None


def test_save_median_removes_isolated_black_without_changing_live_map(
        node, tmp_path):
    node.map_callback(reference_map(np.full((3, 3), -1)))
    node.laser_grid.update(np.array([[0.1, 0.1]]))
    node.grid.update(np.array([[1.1, 1.1]]), np.array([4]))
    node.config['save_directory'] = str(tmp_path)

    live_before = node.grid.render(node.reference_geometry)[2]
    assert live_before[1, 1] == 90
    result = node.save_map(Trigger.Request(), Trigger.Response())
    assert result.success, result.message

    saved = read_scale_costs(tmp_path / 'limo_map_complete.pgm')
    assert saved[1, 1] == 0
    assert (tmp_path / 'limo_map_complete.yaml').is_file()
    laser = read_trinary_costs(tmp_path / 'limo_map_laser.pgm')
    assert laser[2, 0] == 100  # PGM rows are vertically flipped.
    assert np.count_nonzero(laser == 100) == 1
    assert not list(tmp_path.glob('*.npz'))
    assert node.grid.render(node.reference_geometry)[2][1, 1] == 90


def test_complete_map_precedence():
    laser = np.array([[0, 0, 0, 100]], dtype=np.int8)
    semantic = np.array([[60, -1, 90, 30]], dtype=np.int8)
    complete = combine_semantic_and_laser(semantic, laser)
    np.testing.assert_array_equal(complete, [[60, 0, 90, 100]])


def test_cv_obstacle_cost_boundaries():
    complete = np.array([[-1, 0, 9, 10, 95, 96, 100]], dtype=np.int8)
    np.testing.assert_array_equal(
        cv_obstacle_map(complete), [[-1, 0, 0, 100, 100, -1, -1]])


def test_save_requires_laser_observations(node, tmp_path):
    transform(node, 1)
    node.cloud_callback(cloud(1, (2,)))
    node.config['save_directory'] = str(tmp_path)
    result = node.save_map(Trigger.Request(), Trigger.Response())
    assert not result.success
    assert 'laser observations' in result.message
    assert not list(tmp_path.glob('*'))


def test_save_before_semantic_observations_produces_three_maps(node, tmp_path):
    node.laser_grid.update(np.array([[0.1, 0.1], [2.1, 0.1]]))
    node.config['save_directory'] = str(tmp_path)
    result = node.save_map(Trigger.Request(), Trigger.Response())
    assert result.success, result.message
    expected = [[100, 0, 100]]
    assert read_scale_costs(tmp_path / 'limo_map_complete.pgm').tolist() == expected
    assert read_trinary_costs(tmp_path / 'limo_map_laser.pgm').tolist() == expected
    assert read_trinary_costs(tmp_path / 'limo_map_cv_obstacle.pgm').tolist() == [
        [-1, 0, -1]]


def test_cartographer_pose_updates_reposition_old_evidence(node):
    # Feed the documented SubmapList schema without requiring Cartographer
    # binaries. Integration against a running pose graph is a separate test.
    node.pose_source = 'cartographer'
    pose = Pose()
    pose.orientation.w = 1.0
    pose.position.x = 2.0
    entry = SimpleNamespace(trajectory_id=0, submap_index=4, is_frozen=False, pose=pose)
    msg = SimpleNamespace(header=SimpleNamespace(frame_id='map', stamp=cloud().header.stamp),
                          submap=[entry])
    node.submaps_callback(msg)
    node.grid.update(np.array([[0.1, 0.1]]), np.array([4]), (0, 4))
    assert node.grid.render()[0].origin[0] == 2.0
    pose.position.x = 6.0
    node.submaps_callback(msg)
    assert node.grid.render()[0].origin[0] == 6.0
    assert node.grid.render()[2].tolist() == [[90]]


def test_cartographer_insertion_uses_submap_epoch_and_sensor_motion(node):
    node.pose_source = 'cartographer'
    for sec, map_x, odom_x in [(1, 2.0, 0.0), (2, 10.0, 1.0)]:
        for parent, child, x in [('map', 'odom', map_x), ('odom', 'base_link', odom_x)]:
            tf = TransformStamped()
            tf.header.frame_id, tf.child_frame_id, tf.header.stamp.sec = parent, child, sec
            tf.transform.translation.x, tf.transform.rotation.w = x, 1.0
            node.tf_buffer.set_transform(tf, 'test')
    pose = Pose()
    pose.position.x, pose.orientation.w = 2.0, 1.0
    entry = SimpleNamespace(trajectory_id=0, submap_index=4, is_frozen=False, pose=pose)
    node.submaps_callback(SimpleNamespace(
        header=SimpleNamespace(frame_id='map', stamp=cloud(1).header.stamp), submap=[entry]))
    node.cloud_callback(cloud(2, (2,)))
    assert node.grid.sequence == 1
    assert node.grid.render()[0].origin[0] == 3.0
    # Applying the global correction afterwards moves the same observation once.
    pose.position.x = 10.0
    node.submaps_callback(SimpleNamespace(
        header=SimpleNamespace(frame_id='map', stamp=cloud(2).header.stamp), submap=[entry]))
    assert node.grid.render()[0].origin[0] == 11.0
