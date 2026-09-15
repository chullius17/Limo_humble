"""Exercise ROS messages, timestamped TF and exact map snapshots."""

from array import array
from types import SimpleNamespace

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
from geometry_msgs.msg import Pose, TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
from std_srvs.srv import Trigger

from offline_map_package.semantic_mapper import SemanticMapper
from offline_map_package.semantic_grid import Geometry


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


def read_pgm(path):
    with path.open('rb') as stream:
        assert stream.readline() == b'P5\n'
        assert stream.readline().startswith(b'# CREATOR:')
        width, height = (int(value) for value in stream.readline().split())
        assert stream.readline() == b'255\n'
        return np.frombuffer(stream.read(), dtype=np.uint8).reshape(height, width)


def test_uses_sensor_tf_not_latest_and_deduplicates(node):
    transform(node, 1, 2.0)
    transform(node, 2, 10.0)
    msg = cloud()
    node.cloud_callback(msg)
    geometry, layers, combined = node.grid.render()
    assert geometry.origin[0] == 3.0  # blue at x=2 is ignored
    assert combined.tolist() == [[60, 30, 90]]
    node.cloud_callback(msg)
    node.consume_pending()
    assert node.grid.sequence == 1
    output = node.message(combined, geometry, msg.header.stamp)
    assert list(output.data) == [60, 30, 90]
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


def test_reset_and_save_exact_costs(node, tmp_path):
    transform(node, 1)
    node.cloud_callback(cloud())
    node.config['save_directory'] = str(tmp_path)
    result = node.save_map(Trigger.Request(), Trigger.Response())
    assert result.success, result.message
    assert read_pgm(tmp_path / 'limo_map.pgm').tolist() == [[60, 30, 90]]
    metadata = (tmp_path / 'limo_map.yaml').read_text(encoding='utf-8')
    assert 'image: limo_map.pgm\n' in metadata
    assert 'mode: raw\n' in metadata
    assert 'resolution: 1\n' in metadata
    assert 'origin: [1, 0, 0]\n' in metadata
    assert not list(tmp_path.glob('*.npz'))
    node.publish_maps()
    node.reset_map(Trigger.Request(), Trigger.Response())
    assert node.grid.sequence == 0
    assert node.last_stamp == -1
    assert node.pending is None


def test_save_median_removes_isolated_black_without_changing_live_map(
        node, tmp_path):
    node.reference_geometry = Geometry(1.0, 3, 3, (0.0, 0.0, 0.0))
    node.grid.update(np.array([[1.1, 1.1]]), np.array([4]))
    node.config['save_directory'] = str(tmp_path)

    live_before = node.grid.render(node.reference_geometry)[2]
    assert live_before[1, 1] == 90
    result = node.save_map(Trigger.Request(), Trigger.Response())
    assert result.success, result.message

    saved = read_pgm(tmp_path / 'limo_map.pgm')
    assert saved[1, 1] == 255
    assert (tmp_path / 'limo_map.yaml').is_file()
    assert not list(tmp_path.glob('*.npz'))
    assert node.grid.render(node.reference_geometry)[2][1, 1] == 90


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
