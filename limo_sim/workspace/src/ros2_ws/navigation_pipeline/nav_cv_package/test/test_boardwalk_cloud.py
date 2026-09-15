"""Verify class 4 serialization without starting ROS publishers or threads."""

from collections import deque
import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('rclpy')
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header

from nav_cv_package.boardwalk import BOARDWALK_COUNTS, BOARDWALK_TIMINGS, BoardwalkClassifier
from nav_cv_package.visual_ptcld import VisualPtcld
from nav_cv_package.cloud_cpu import RayCache


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
        cloud_max_depth=5.0,
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
                 'LABEL_BOARDWALK', 'CLOUD_DTYPE', 'CLOUD_FIELDS'):
        setattr(detector, name, getattr(VisualPtcld, name))
    detector.voxelize_bev_cloud = (
        lambda points, class_ids: VisualPtcld.voxelize_bev_cloud(
            detector, points, class_ids))
    return detector, published


def test_metric_voxelization_keeps_blue_and_separates_other_classes():
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

    np.testing.assert_array_equal(output_labels, [1, 1, 3, 4, 2])
    np.testing.assert_array_equal(output[:2], points[:2])
    np.testing.assert_allclose(output[2], [0.015, 0.015])
    np.testing.assert_allclose(output[3], [0.012, 0.012])
    np.testing.assert_allclose(output[4], [-0.010, -0.010])


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
    np.testing.assert_array_equal(points['class_id'], [1, 2, 4 if enabled else 3, 3])
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
    detector.publish_pointcloud = lambda *args: VisualPtcld.publish_pointcloud(
        detector, *args)
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
    np.testing.assert_array_equal(output_labels, [3, 2, 1, 3, 4, 1])
    np.testing.assert_allclose(output[:2], [[0.003, 0.004], [-0.0025, -0.0035]])
    np.testing.assert_array_equal(output[2:], points[[2, 5, 6, 7]])
    np.testing.assert_array_equal(points, original)
