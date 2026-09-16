"""Verify class 4 serialization without starting ROS publishers or threads."""

from collections import deque
import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('rclpy')
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header

from cv_package.boardwalk import BOARDWALK_COUNTS, BOARDWALK_TIMINGS, BoardwalkClassifier
from cv_package.visual_ptcld import VisualPtcld
from cv_package.cloud_cpu import RayCache


def detector_stub(enabled=True):
    """Provide deterministic depth and TF to the real publishing method."""
    transform = TransformStamped()
    transform.transform.rotation.w = 1.0
    published = []
    detector = SimpleNamespace(
        sensor_lock=threading.Lock(),
        depth_image=np.ones((1, 4), dtype=np.float32),
        camera_intrinsics=(8.0, 8.0, 0.0, 0.0, 4, 1),
        cloud_min_depth=0.1,
        cloud_max_depth=2.0,
        input_crop_y_min=0.0,
        bev_frame='base_link',
        enable_boardwalk=enabled,
        enable_debug_publications=False,
        blue_radius_min=0.10,
        blue_radius_max=0.16,
        boardwalk_propagation_radius=0.10,
        pointcloud_voxel_size=0.02,
        ray_cache=RayCache(),
        boardwalk_classifier=BoardwalkClassifier(),
        tf_buffer=SimpleNamespace(lookup_transform=lambda *args: transform),
        quaternion_to_rotation=VisualPtcld.quaternion_to_rotation,
        pointcloud_pub=SimpleNamespace(publish=published.append),
        get_logger=lambda: SimpleNamespace(warning=lambda *args, **kwargs: None),
    )
    for name in ('LABEL_BLUE', 'LABEL_TURQUOISE', 'LABEL_BACKGROUND',
                 'LABEL_BOARDWALK', 'LABEL_INTERIOR_BLUE', 'CLOUD_DTYPE', 'CLOUD_FIELDS'):
        setattr(detector, name, getattr(VisualPtcld, name))
    detector.voxelize_bev_cloud = (
        lambda points, class_ids: VisualPtcld.voxelize_bev_cloud(
            detector, points, class_ids))
    return detector, published


def test_metric_voxelization_reduces_every_class_separately():
    detector = SimpleNamespace(
        LABEL_BLUE=VisualPtcld.LABEL_BLUE,
        pointcloud_voxel_size=0.02,
    )
    points = np.array([
        [0.001, 0.001],
        [0.002, 0.002],
        [0.011, 0.011],
        [0.019, 0.019],
        [0.012, 0.012],
        [-0.001, -0.001],
        [-0.019, -0.019],
    ], dtype=np.float32)
    labels = np.array([1, 1, 3, 3, 4, 2, 2], dtype=np.uint8)

    output, output_labels = VisualPtcld.voxelize_bev_cloud(
        detector, points, labels)

    np.testing.assert_array_equal(output_labels, [1, 3, 4, 2])
    np.testing.assert_allclose(output[0], [0.0015, 0.0015])
    np.testing.assert_allclose(output[1], [0.015, 0.015])
    np.testing.assert_allclose(output[2], [0.012, 0.012])
    np.testing.assert_allclose(output[3], [-0.010, -0.010])


@pytest.mark.parametrize('enabled', [True, False])
def test_projected_cloud_preserves_geometry_and_serializes_class_four(enabled):
    detector, published = detector_stub(enabled)
    header = Header(frame_id='camera')
    header.stamp.sec = 42
    result = VisualPtcld.publish_pointcloud(
        detector, np.array([[0, 0]]), np.array([[0, 3]]),
        np.array([[0, 1], [0, 2]]), 4, 1, header)
    assert len(published) == 1
    cloud = published[0]
    points = np.frombuffer(cloud.data, dtype=VisualPtcld.CLOUD_DTYPE)
    np.testing.assert_array_equal(
        points['class_id'], [1, 2, 4 if enabled else 3, 3])
    np.testing.assert_array_equal(points['x'], [0.0, 0.375, 0.125, 0.25])
    np.testing.assert_array_equal(points['y'], np.zeros(4))
    np.testing.assert_array_equal(points['z'], np.zeros(4))
    assert cloud.header.frame_id == 'base_link'
    assert cloud.header.stamp == header.stamp
    assert cloud.width == result[0] == 4
    assert cloud.point_step == 16
    assert cloud.row_step == len(cloud.data) == 64
    stats = result[-1]
    if enabled:
        assert stats['boardwalk_final_count'] == 1
        # Preparation includes classification; projection/serialization does not.
        assert result[1] >= result[3] + result[4] + result[5] + stats['boardwalk_total_ms']
    else:
        assert 'boardwalk_final_count' not in stats
    assert stats['published_blue_count'] == 1
    assert stats['published_turquoise_count'] == 1
    assert stats['published_background_count'] == (1 if enabled else 2)
    assert stats['published_boardwalk_count'] == (1 if enabled else 0)


def test_missing_depth_does_not_record_a_zero_classification_sample():
    detector, published = detector_stub()
    detector.depth_image = None
    empty = np.empty((0, 2), dtype=np.int32)
    result = VisualPtcld.publish_pointcloud(
        detector, empty, empty, empty, 4, 1, Header(frame_id='camera'))
    assert result[0] is None
    assert result[-1] == {}
    assert not published


def test_empty_cloud_is_published_with_zero_workload():
    detector, published = detector_stub()
    empty = np.empty((0, 2), dtype=np.int32)
    result = VisualPtcld.publish_pointcloud(
        detector, empty, empty, empty, 4, 1, Header(frame_id='camera'))
    assert published[0].width == 0
    assert result[-1]['boardwalk_final_count'] == 0
    assert result[-1]['boardwalk_blue_query_ms'] == 0


def test_telemetry_reports_distribution_counts_and_missing_current_sample():
    detector, _ = detector_stub()
    keys = BOARDWALK_TIMINGS + BOARDWALK_COUNTS
    detector.telemetry_stats = {key: deque([1.0, 2.0]) for key, _ in keys}
    current = {key: float('nan') for key, _ in keys}
    output = VisualPtcld.boardwalk_diagnostics(detector, current)
    assert 'Rolling window: 2 clouds' in output
    assert 'P95: 1.950 ms' in output
    assert 'Max: 2.000 ms' in output
    assert 'Current: n/a' in output
    assert 'First-pass seeds' in output
    assert 'Added by second pass' in output
    assert 'direct points' in output
    assert 'White -> blue query' in output
    assert 'White -> seed query' in output
    assert 'Blue filter: OpenCV CPU' in output


@pytest.mark.parametrize('debug_enabled', [False, True])
def test_image_flag_gates_all_images_but_always_publishes_cloud(debug_enabled):
    from turbojpeg import TurboJPEG

    detector, clouds = detector_stub()
    images = []
    detector.enable_debug_publications = debug_enabled
    detector.bridge = SimpleNamespace(
        imgmsg_to_cv2=lambda *args, **kwargs: np.array([[1, 3, 3, 2]], dtype=np.uint8))
    detector.blue_boundary_kernel = np.ones((7, 7), dtype=np.uint8)
    detector.roi_y_min = 0.0
    detector.roi_y_max = 1.0
    detector.point_voxel_size = 1
    detector.voxelize_points = lambda points, width: VisualPtcld.voxelize_points(
        detector, points, width)
    detector.publish_pointcloud = lambda *args, **kwargs: VisualPtcld.publish_pointcloud(
        detector, *args, **kwargs)
    timings = []
    detector.log_diagnostics = lambda width, height, stats: timings.append(stats)
    for name in ('debug_pub', 'lines_pub', 'blue_filter_debug_pub'):
        # Subscribers cannot override the explicit publication flag.
        setattr(detector, name, SimpleNamespace(
            get_subscription_count=lambda: 1, publish=images.append))
    detector.jpeg = TurboJPEG() if debug_enabled else None
    detector.debug_jpeg_quality = 85
    def encode(frame, header):
        assert debug_enabled, 'JPEG encoding must not run with debug disabled'
        return VisualPtcld.encode_debug_image(detector, frame, header)
    detector.encode_debug_image = encode
    header = Header(frame_id='camera')
    header.stamp.sec = 42

    VisualPtcld.process_image(detector, SimpleNamespace(header=header))

    assert len(clouds) == 1
    assert len(images) == (3 if debug_enabled else 0)
    points = np.frombuffer(clouds[0].data, dtype=VisualPtcld.CLOUD_DTYPE)
    np.testing.assert_array_equal(points['class_id'], [1, 2, 4, 3])
    assert timings[0]['boardwalk_final_count'] == 1
    assert timings[0]['worker_cpu_ms'] >= 0.0
    assert timings[0]['outside_worker_ms'] >= 0.0
    for key in ('step8_projection', 'step8_transform', 'step8_serialize'):
        assert np.isfinite(timings[0][key])
    for msg in images:
        assert msg.header == header
        assert msg.format == 'bgr8; jpeg compressed bgr8'
        assert detector.jpeg.decode(bytes(msg.data)).size > 0


def test_cloud_voxels_preserve_input_order_classes_and_nonfinite_points():
    detector, _ = detector_stub()
    points = np.array([
        [0.002, 0.003], [-0.002, -0.003], [0.003, 0.004],
        [0.004, 0.005], [-0.003, -0.004], [np.nan, 0.0],
        [0.002, 0.003], [0.002, 0.003],
    ], dtype=np.float32)
    labels = np.array([3, 2, 1, 3, 2, 3, 4, 1], dtype=np.uint8)
    original = points.copy()
    output, output_labels = detector.voxelize_bev_cloud(points, labels)
    np.testing.assert_array_equal(output_labels, [3, 2, 1, 3, 4])
    np.testing.assert_allclose(output[:3], [
        [0.003, 0.004], [-0.0025, -0.0035], [0.0025, 0.0035],
    ])
    np.testing.assert_array_equal(output[3:], points[[5, 6]])
    np.testing.assert_array_equal(points, original)


def test_interior_blue_does_not_seed_boardwalk_and_is_voxelized_after_bev():
    detector, published = detector_stub()
    detector.camera_intrinsics = (100.0, 100.0, 0.0, 0.0, 4, 1)
    detector.blue_radius_min = 0.005
    detector.blue_radius_max = 0.02
    empty = np.empty((0, 2), dtype=np.int32)
    result = VisualPtcld.publish_pointcloud(
        detector, empty, empty, np.array([[0, 2]]),
        4, 1, Header(frame_id='camera'),
        interior_blue_points=np.array([[0, 0], [0, 1]]))
    points = np.frombuffer(published[0].data, dtype=VisualPtcld.CLOUD_DTYPE)
    np.testing.assert_array_equal(points['class_id'], [3, 5])
    np.testing.assert_allclose(points['x'], [0.02, 0.005])
    assert result[-2] == 3  # Before metric voxelization.
    assert result[-1]['published_interior_blue_count'] == 1
    assert result[-1]['boardwalk_blue_count'] == 0
    assert result[-1]['boardwalk_final_count'] == 0


@pytest.mark.parametrize('pixel_voxel_size', [1, 5])
def test_interior_blue_mask_and_pixel_voxelization(pixel_voxel_size):
    detector, published = detector_stub()
    # White at column 9 leaves blue columns 0..5 outside its 7x7 dilation.
    labels = np.array([[1] * 9 + [3, 2, 0]], dtype=np.uint8)
    detector.bridge = SimpleNamespace(imgmsg_to_cv2=lambda *a, **kw: labels)
    detector.depth_image = np.ones(labels.shape, dtype=np.float32)
    detector.camera_intrinsics = (100.0, 100.0, 0.0, 0.0, 12, 1)
    detector.blue_boundary_kernel = np.ones((7, 7), dtype=np.uint8)
    detector.roi_y_min, detector.roi_y_max = 0.0, 1.0
    detector.point_voxel_size = pixel_voxel_size
    detector.voxelize_points = lambda points, width: VisualPtcld.voxelize_points(
        detector, points, width)
    before_projection = []

    def publish(*args, **kwargs):
        before_projection.append(kwargs['interior_blue_points'].copy())
        return VisualPtcld.publish_pointcloud(detector, *args, **kwargs)

    detector.publish_pointcloud = publish
    stats = []
    detector.log_diagnostics = lambda w, h, value: stats.append(value)
    VisualPtcld.process_image(
        detector, SimpleNamespace(header=Header(frame_id='camera')))
    expected_cols = [0, 1, 2, 3, 4, 5] if pixel_voxel_size == 1 else [2, 5]
    np.testing.assert_array_equal(before_projection[0][:, 1], expected_cols)
    points = np.frombuffer(published[0].data, dtype=VisualPtcld.CLOUD_DTYPE)
    road = points[points['class_id'] == 5]
    assert len(road) == (3 if pixel_voxel_size == 1 else 2)
    assert np.all(road['x'] < 0.06)
    assert stats[0]['boardwalk_blue_count'] == (3 if pixel_voxel_size == 1 else 1)


def test_interior_blue_requires_valid_depth():
    detector, published = detector_stub()
    detector.depth_image[:] = [np.nan, 0.0, 3.0, 1.0]
    empty = np.empty((0, 2), dtype=np.int32)
    VisualPtcld.publish_pointcloud(
        detector, empty, empty, empty, 4, 1, Header(frame_id='camera'),
        interior_blue_points=np.array([[0, 0], [0, 1], [0, 2], [0, 3]]))
    points = np.frombuffer(published[0].data, dtype=VisualPtcld.CLOUD_DTYPE)
    np.testing.assert_array_equal(points['class_id'], [5])
    np.testing.assert_allclose(points['x'], [0.375])


def test_boardwalk_uses_boundary_points_before_metric_voxelization():
    detector, published = detector_stub()
    detector.pointcloud_voxel_size = 0.2
    result = VisualPtcld.publish_pointcloud(
        detector, np.array([[0, 0], [0, 1]]),
        np.empty((0, 2), dtype=np.int32), np.array([[0, 2]]),
        4, 1, Header(frame_id='camera'),
        interior_blue_points=np.array([[0, 3]]))
    points = np.frombuffer(published[0].data, dtype=VisualPtcld.CLOUD_DTYPE)
    # White is 0.125 m from the original blue but 0.1875 m from its
    # voxel centroid: voxelizing before classification would miss this seed.
    np.testing.assert_array_equal(points['class_id'], [1, 4, 5])
    np.testing.assert_allclose(points['x'], [0.0625, 0.25, 0.375])
    assert result[-1]['boardwalk_blue_count'] == 2
    assert result[-1]['published_blue_count'] == 1
    assert result[-1]['boardwalk_final_count'] == 1
