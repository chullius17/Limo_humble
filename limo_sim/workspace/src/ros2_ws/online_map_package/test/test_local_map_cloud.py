"""Verify semantic-cloud ingestion and local OccupancyGrid generation."""

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
from geometry_msgs.msg import TransformStamped, Twist  # noqa: E402
from rclpy.time import Time  # noqa: E402
from sensor_msgs.msg import PointCloud2, PointField  # noqa: E402

from online_map_package.local_map_final import LocalMapFinal  # noqa: E402


INPUT_DTYPE = np.dtype({
    'names': ['x', 'y', 'z', 'class_id'],
    'formats': ['<f4', '<f4', '<f4', 'u1'],
    'offsets': [0, 4, 8, 12],
    'itemsize': 16,
})


@pytest.fixture
def node():
    rclpy.init()
    instance = LocalMapFinal()
    yield instance
    instance.destroy_node()
    rclpy.shutdown()


def add_pose(node, stamp, x):
    tf = TransformStamped()
    tf.header.frame_id = 'odom'
    tf.header.stamp = stamp
    tf.child_frame_id = 'base_link'
    tf.transform.translation.x = x
    tf.transform.rotation.w = 1.0
    node.tf_buffer.set_transform(tf, 'test')


def make_cloud(xy, classes, stamp, bigendian=False):
    dtype = INPUT_DTYPE.newbyteorder('>' if bigendian else '<')
    data = np.zeros(len(xy), dtype=dtype)
    if len(xy):
        data['x'] = np.asarray(xy)[:, 0]
        data['y'] = np.asarray(xy)[:, 1]
        data['class_id'] = classes
    msg = PointCloud2()
    msg.header.frame_id = 'base_link'
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = len(data)
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='class_id', offset=12,
                   datatype=PointField.UINT8, count=1),
    ]
    msg.is_bigendian = bigendian
    msg.point_step = dtype.itemsize
    msg.row_step = msg.width * msg.point_step
    msg.is_dense = True
    msg.data = data.tobytes()
    return msg


def grid_cell(node, grid, x, y):
    cell_x = int(np.floor(x / node.grid_resolution))
    cell_y = int(np.floor(
        (y - node.grid_origin_y) / node.grid_resolution))
    return int(grid[cell_y, cell_x])


def test_grid_geometry_matches_green_rectangle(node):
    assert node.cells_x == 125
    assert node.cells_y == 133
    assert node.grid_resolution == pytest.approx(0.02)
    assert node.grid_origin_y == pytest.approx(-1.33)


def test_red_trapezoid_keeps_front_edge_aligned_with_yellow(node):
    yellow = np.asarray(node.yellow_vertices)
    inner = np.asarray(node.inner_vertices)
    assert inner[0, 0] > yellow[0, 0]
    np.testing.assert_allclose(inner[1:3, 0], node.rectangle_length)
    assert np.all(np.abs(inner[1:3, 1]) < np.abs(yellow[1:3, 1]))


def test_commanded_speed_scales_decay_like_local_pointcloud(node):
    assert node._motion_ratio() == 0.0
    command = Twist()
    command.linear.x = 0.255
    node.cmd_vel_callback(command)
    assert node._motion_ratio() == pytest.approx(0.5)
    command.linear.x = 0.0
    command.angular.z = -1.0
    node.cmd_vel_callback(command)
    assert node._motion_ratio() == pytest.approx(1.0)
    node.cmd_vel_timeout_sec = 0.1
    assert node._motion_ratio(
        node.last_cmd_vel_time_ns + 100_000_001) == 0.0


def test_callback_keeps_live_cloud_and_admits_only_yellow_red_band(node):
    stamp = Time(seconds=10.0).to_msg()
    add_pose(node, stamp, 0.0)
    msg = make_cloud([[1.0, 0.0], [0.6, 0.1]], [2, 4], stamp)

    node.cloud_callback(msg)

    assert len(node.live_points_odom) == 2
    np.testing.assert_array_equal(node.live_classes, [2, 4])
    assert len(node.memory.points) == 1
    np.testing.assert_allclose(node.memory.points, [[0.6, 0.1]])
    np.testing.assert_array_equal(node.memory.classes, [4])


def test_live_and_memory_clouds_fuse_with_offline_costs_and_maximum(node):
    live_points = np.array([
        [0.60, 0.10],
        [1.00, 0.00],
        [-0.01, 0.00],
    ])
    live_classes = np.array([2, 2, 4])
    memory_points = np.array([
        [0.601, 0.101],
        [2.50, 0.00],
    ])
    memory_classes = np.array([4, 4])
    points = np.concatenate((live_points, memory_points))
    classes = np.concatenate((live_classes, memory_classes))

    grid = node._rasterize(points, classes)

    assert grid_cell(node, grid, 0.60, 0.10) == 90
    assert grid_cell(node, grid, 1.00, 0.00) == 60
    assert np.count_nonzero(grid) == 2


def test_occupancy_grid_message_has_expected_frame_geometry_and_data(node):
    costs = node._rasterize(np.array([[0.60, 0.10]]), np.array([4]))
    stamp = Time(seconds=10.0).to_msg()

    msg = node._make_grid(costs, stamp)

    assert msg.header.frame_id == 'base_link'
    assert msg.header.stamp == stamp
    assert msg.info.resolution == pytest.approx(0.02)
    assert msg.info.width == 125
    assert msg.info.height == 133
    assert msg.info.origin.position.x == pytest.approx(0.0)
    assert msg.info.origin.position.y == pytest.approx(-1.33)
    assert len(msg.data) == 125 * 133
    assert max(msg.data) == 90


def test_output_cloud_contains_only_points_contributing_to_grid(node):
    stamp = Time(seconds=10.0).to_msg()
    msg = node._make_cloud(
        np.array([[0.60, 0.10], [1.00, 0.00], [-0.01, 0.00],
                  [2.50, 0.00]]),
        np.array([2, 4, 4, 2]),
        stamp,
    )

    assert msg.header.frame_id == 'base_link'
    assert msg.header.stamp == stamp
    assert msg.height == 1
    assert msg.width == 2
    assert msg.point_step == INPUT_DTYPE.itemsize
    points = np.frombuffer(msg.data, dtype=INPUT_DTYPE)
    np.testing.assert_allclose(points['x'], [0.60, 1.00])
    np.testing.assert_allclose(points['y'], [0.10, 0.00])
    np.testing.assert_array_equal(points['class_id'], [2, 4])


@pytest.mark.parametrize('bigendian', [False, True])
def test_read_organized_padded_cloud_filters_classes(node, bigendian):
    stamp = Time(seconds=1).to_msg()
    msg = make_cloud(np.zeros((4, 2)), [2, 4, 6, 1], stamp, bigendian)
    dtype = INPUT_DTYPE.newbyteorder('>' if bigendian else '<')
    data = np.zeros(4, dtype=dtype)
    data['x'] = [0.6, 0.7, 0.8, 0.9]
    data['class_id'] = [2, 4, 6, 1]
    msg.width, msg.height, msg.row_step = 2, 2, 40
    msg.data = data[:2].tobytes() + bytes(8) + data[2:].tobytes() + bytes(8)

    xy, labels = node._read_cloud(msg)

    np.testing.assert_allclose(xy, [[0.6, 0], [0.7, 0]])
    np.testing.assert_array_equal(labels, [2, 4])


def test_missing_tf_does_not_change_live_cloud_or_memory(node):
    stamp = Time(seconds=10).to_msg()
    add_pose(node, stamp, 0.0)
    msg = make_cloud([[0.6, 0.0]], [2], stamp)
    node.cloud_callback(msg)
    saved_live = node.live_points_odom.copy()
    saved_memory = node.memory.points.copy()

    msg.header.stamp = Time(seconds=10.1).to_msg()
    node.cloud_callback(msg)

    np.testing.assert_array_equal(node.live_points_odom, saved_live)
    np.testing.assert_array_equal(node.memory.points, saved_memory)
    assert node.memory.last_observation_stamp == 10.0


def test_delayed_tf_recovers_cloud_without_another_sensor_frame(node, monkeypatch):
    monkeypatch.setattr(node, '_check_clock', lambda: Time(seconds=10.05))
    stamp = Time(seconds=10.0).to_msg()
    add_pose(node, Time(seconds=9.95).to_msg(), 1.0)
    node.cloud_callback(make_cloud([[0.6, 0.0]], [2], stamp))
    assert len(node.pending_clouds) == 1
    assert node.live_stamp_sec is None

    # The exact transform is interpolated when the next TF arrives.
    add_pose(node, Time(seconds=10.05).to_msg(), 1.2)
    monkeypatch.setattr(node, '_check_clock', lambda: Time(seconds=10.10))
    node.retry_pending_clouds()

    assert not node.pending_clouds
    np.testing.assert_allclose(node.memory.points, [[1.7, 0.0]])
    np.testing.assert_allclose(node.live_points_odom, [[1.7, 0.0]])
    assert node.live_stamp_sec == 10.0


def test_pending_frames_are_processed_in_timestamp_order(node, monkeypatch):
    monkeypatch.setattr(node, '_check_clock', lambda: Time(seconds=10.1))
    for seconds, label in [(10.05, 4), (10.0, 2), (10.0, 2)]:
        node.cloud_callback(make_cloud(
            [[0.6, 0.0]], [label], Time(seconds=seconds).to_msg()))
    assert len(node.pending_clouds) == 2  # Duplicate frame ignored.
    add_pose(node, Time(seconds=9.9).to_msg(), 0.0)
    add_pose(node, Time(seconds=10.1).to_msg(), 0.0)
    node.retry_pending_clouds()
    assert not node.pending_clouds
    assert set(node.memory.classes) == {2, 4}
    assert node.live_stamp_sec == pytest.approx(10.05)
    np.testing.assert_array_equal(node.live_classes, [4])


def test_expired_head_does_not_block_ready_newer_frame(node, monkeypatch):
    monkeypatch.setattr(node, '_check_clock', lambda: Time(seconds=10.0))
    node.cloud_callback(make_cloud(
        [[0.6, 0.0]], [2], Time(seconds=10.0).to_msg()))
    monkeypatch.setattr(node, '_check_clock', lambda: Time(seconds=10.1))
    stamp = Time(seconds=10.1).to_msg()
    add_pose(node, stamp, 0.0)
    node.cloud_callback(make_cloud([[0.6, 0.1]], [4], stamp))
    assert node.live_stamp_sec is None
    monkeypatch.setattr(node, '_check_clock', lambda: Time(seconds=10.21))
    node.retry_pending_clouds()
    assert not node.pending_clouds
    np.testing.assert_array_equal(node.memory.classes, [4])


def test_pending_queue_is_bounded_and_expires_without_tf(node, monkeypatch):
    node.max_pending_clouds = 2
    monkeypatch.setattr(node, '_check_clock', lambda: Time(seconds=10.1))
    for seconds in [10.0, 10.01, 10.02]:
        node.cloud_callback(make_cloud(
            [[0.6, 0.0]], [2], Time(seconds=seconds).to_msg()))
    assert len(node.pending_clouds) == 2
    assert node.pending_clouds[0][0].nanoseconds == 10_010_000_000
    monkeypatch.setattr(node, '_check_clock', lambda: Time(seconds=10.31))
    node.retry_pending_clouds()
    assert not node.pending_clouds
    assert node.live_stamp_sec is None
    assert len(node.memory.points) == 0


def test_older_frame_cannot_replace_newer_live_cloud(node):
    add_pose(node, Time(seconds=10.0).to_msg(), 0.0)
    add_pose(node, Time(seconds=10.1).to_msg(), 0.0)
    node.cloud_callback(make_cloud(
        [[0.6, 0.0]], [4], Time(seconds=10.1).to_msg()))
    node.cloud_callback(make_cloud(
        [[0.6, 0.1]], [2], Time(seconds=10.0).to_msg()))
    assert not node.pending_clouds
    assert node.live_stamp_sec == pytest.approx(10.1)
    np.testing.assert_array_equal(node.live_classes, [4])


def test_clock_reset_clears_pending_clouds(node):
    node.cloud_callback(make_cloud(
        [[0.6, 0.0]], [2], Time(seconds=10.0).to_msg()))
    assert node.pending_clouds
    node.last_clock_ns = node.get_clock().now().nanoseconds + 10_000_000_000
    node._check_clock()
    assert not node.pending_clouds
    assert node.live_stamp_sec is None


def test_waiting_for_tf_does_not_stop_memory_publication(node, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(node, '_check_clock', lambda: Time(seconds=10.0))
    add_pose(node, Time(seconds=10.0).to_msg(), 0.0)
    node.cloud_callback(make_cloud(
        [[0.6, 0.0]], [2], Time(seconds=10.0).to_msg()))
    monkeypatch.setattr(node, '_check_clock', lambda: Time(seconds=10.05))
    node.cloud_callback(make_cloud(
        [[0.6, 0.1]], [4], Time(seconds=10.05).to_msg()))
    grids, clouds = [], []
    monkeypatch.setattr(node, 'grid_pub', SimpleNamespace(publish=grids.append))
    monkeypatch.setattr(node, 'cloud_pub', SimpleNamespace(publish=clouds.append))
    node.publish_grid()
    assert len(node.pending_clouds) == 1
    assert len(grids) == len(clouds) == 1
    assert clouds[0].width == 1
    assert max(grids[0].data) == 60
