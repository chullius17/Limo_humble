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
from cv_package.simple_boundaries import CurbDetector


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
        blue_radius_min=0.10,
        blue_radius_max=0.16,
        boardwalk_propagation_radius=0.10,
        boardwalk_classifier=BoardwalkClassifier(),
        tf_buffer=SimpleNamespace(lookup_transform=lambda *args: transform),
        quaternion_to_rotation=CurbDetector.quaternion_to_rotation,
        pointcloud_pub=SimpleNamespace(publish=published.append),
        boardwalk_grid_debug_pub=SimpleNamespace(
            get_subscription_count=lambda: 0, publish=lambda msg: None),
        get_logger=lambda: SimpleNamespace(warning=lambda *args, **kwargs: None),
    )
    for name in ('LABEL_BLUE', 'LABEL_TURQUOISE', 'LABEL_BACKGROUND',
                 'LABEL_BOARDWALK', 'CLOUD_DTYPE', 'CLOUD_FIELDS'):
        setattr(detector, name, getattr(CurbDetector, name))
    return detector, published


@pytest.mark.parametrize('enabled', [True, False])
def test_projected_cloud_preserves_geometry_and_serializes_class_four(enabled):
    detector, published = detector_stub(enabled)
    header = Header(frame_id='camera')
    header.stamp.sec = 42
    result = CurbDetector.publish_pointcloud(
        detector, np.array([[0, 0]]), np.array([[0, 3]]),
        np.array([[0, 1], [0, 2]]), 4, 1, header)
    assert len(published) == 1
    cloud = published[0]
    points = np.frombuffer(cloud.data, dtype=CurbDetector.CLOUD_DTYPE)
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
        assert stats == {}


def test_missing_depth_does_not_record_a_zero_classification_sample():
    detector, published = detector_stub()
    detector.depth_image = None
    empty = np.empty((0, 2), dtype=np.int32)
    result = CurbDetector.publish_pointcloud(
        detector, empty, empty, empty, 4, 1, Header(frame_id='camera'))
    assert result[0] is None
    assert result[-1] == {}
    assert not published


def test_empty_cloud_is_published_with_zero_workload():
    detector, published = detector_stub()
    empty = np.empty((0, 2), dtype=np.int32)
    result = CurbDetector.publish_pointcloud(
        detector, empty, empty, empty, 4, 1, Header(frame_id='camera'))
    assert published[0].width == 0
    assert result[-1]['boardwalk_final_count'] == 0
    assert result[-1]['boardwalk_blue_dt_ms'] == 0


def test_telemetry_reports_distribution_counts_and_missing_current_sample():
    detector, _ = detector_stub()
    keys = BOARDWALK_TIMINGS + BOARDWALK_COUNTS
    detector.telemetry_stats = {key: deque([1.0, 2.0]) for key, _ in keys}
    current = {key: float('nan') for key, _ in keys}
    output = CurbDetector.boardwalk_diagnostics(detector, current)
    assert 'Rolling window: 2 clouds' in output
    assert 'P95: 1.950 ms' in output
    assert 'Max: 2.000 ms' in output
    assert 'Current: n/a' in output
    assert 'First-pass seeds' in output
    assert 'Added by second pass' in output
    assert 'Grid cells' in output
    assert 'Distance from blue' in output
    assert 'Distance from seeds' in output


def test_grid_debug_publishes_a_jpeg_with_the_bev_header_on_demand():
    from turbojpeg import TurboJPEG

    detector, clouds = detector_stub()
    images = []
    detector.boardwalk_grid_debug_pub.get_subscription_count = lambda: 1
    detector.boardwalk_grid_debug_pub.publish = images.append
    detector.jpeg = TurboJPEG()
    detector.debug_jpeg_quality = 85
    detector.encode_debug_image = lambda frame, header: CurbDetector.encode_debug_image(
        detector, frame, header)
    header = Header(frame_id='camera')
    header.stamp.sec = 42
    result = CurbDetector.publish_pointcloud(
        detector, np.array([[0, 0]]), np.array([[0, 3]]),
        np.array([[0, 1], [0, 2]]), 4, 1, header)
    assert len(images) == len(clouds) == 1
    assert images[0].header == clouds[0].header
    assert images[0].header.frame_id == 'base_link'
    assert images[0].format == 'bgr8; jpeg compressed bgr8'
    decoded = detector.jpeg.decode(bytes(images[0].data))
    assert decoded.shape == detector.boardwalk_classifier.debug_image.shape
    assert result[-1]['boardwalk_debug_publish_ms'] > 0
