"""Check adaptive segmentation, image compatibility and the live ROS pipeline."""

import queue
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

from cv_package.lane_detector_binary import BinaryLaneDetector


@pytest.fixture
def detector():
    rclpy.init(args=['--ros-args', '-p', 'enable_telemetry:=false'])
    node = None
    try:
        node = BinaryLaneDetector()
        # Exercise the real processing methods deterministically in most tests.
        node._stop.set()
        node.worker_thread.join()
        node.pub_thread.join()
        yield node
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def set_debug(node, mask, overlay, fallback=False):
    node.debug_mask_pub = SimpleNamespace(get_subscription_count=lambda: int(mask))
    node.debug_overlay_pub = SimpleNamespace(get_subscription_count=lambda: int(overlay))
    node.debug_fallback_overlay_pub = SimpleNamespace(
        get_subscription_count=lambda: int(fallback))


@pytest.mark.parametrize('shape', [(480, 640), (481, 641), (600, 800), (120, 160)])
@pytest.mark.parametrize('encoding', ['bgr8', 'rgb8', 'mono8'])
def test_labels_match_reference_with_padded_camera_rows(detector, shape, encoding):
    height, width = shape
    rng = np.random.default_rng(31)
    channels, gray_code, _ = detector._ENCODINGS[encoding]
    image_shape = shape + ((channels,) if channels > 1 else ())
    native = rng.integers(0, 256, image_shape, dtype=np.uint8)
    # Nonzero padding must not leak into the visible image.
    padded = np.full((height, width * channels + 7), 231, dtype=np.uint8)
    padded[:, :width * channels] = native.reshape(height, width * channels)
    msg = Image(height=height, width=width, encoding=encoding,
                step=padded.shape[1], data=padded.tobytes())
    view, code, bgr_code = detector._view_source(msg)
    assert np.shares_memory(view, np.frombuffer(msg.data, dtype=np.uint8))
    settings = dict(detector._settings, enable_variance_fallback=False)
    labels, mask, overlay, fallback = detector._process_frame(view, code, bgr_code, settings)

    small = cv2.resize(native[height // 2:], (320, 120), interpolation=cv2.INTER_AREA)
    band = small[12:120]
    gray = band if gray_code is None else cv2.cvtColor(band, gray_code)
    # Independent mean-filter reference for integer C=5 and an odd 31-pixel block.
    mean = cv2.boxFilter(gray, -1, (31, 31), borderType=cv2.BORDER_REPLICATE)
    road = gray.astype(np.int16) <= mean.astype(np.int16) - 5
    expected = np.zeros((120, 320), dtype=np.uint8)
    expected[12:] = np.where(road, 1, 3)
    np.testing.assert_array_equal(labels, expected)
    assert labels.dtype == np.uint8
    assert mask is None and overlay is None and fallback is None


@pytest.mark.parametrize('mask_enabled', [False, True])
@pytest.mark.parametrize('overlay_enabled', [False, True])
@pytest.mark.parametrize('fallback_enabled', [False, True])
def test_debug_colors_and_queued_buffers_are_independent(
        detector, mask_enabled, overlay_enabled, fallback_enabled):
    set_debug(detector, mask_enabled, overlay_enabled, fallback_enabled)
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    frame[260:470, 310:330] = 20
    labels, mask, overlay, fallback = detector._process_frame(
        frame, cv2.COLOR_BGR2GRAY, None, detector._settings)
    expected_mask = np.zeros((120, 320, 3), dtype=np.uint8)
    expected_mask[labels == 1] = (255, 0, 0)
    assert np.any(labels == 1)
    assert np.any(labels == 3)
    if mask_enabled:
        np.testing.assert_array_equal(mask, expected_mask)
    else:
        assert mask is None
    if overlay_enabled:
        background = cv2.resize(frame[240:], (320, 120), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(
            overlay, cv2.addWeighted(background, 0.7, expected_mask, 0.5, 0))
    else:
        assert overlay is None
    if fallback_enabled:
        expected_fallback = expected_mask.copy()
        expected_fallback[12:, :, 2] = detector._buf_low_variance
        background = cv2.resize(frame[240:], (320, 120), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(
            fallback, cv2.addWeighted(background, 0.7, expected_fallback, 0.5, 0))
    else:
        assert fallback is None
    images = (labels, mask, overlay, fallback)
    snapshots = [None if item is None else item.copy() for item in images]
    detector._process_frame(np.zeros_like(frame), cv2.COLOR_BGR2GRAY, None, detector._settings)
    for item, snapshot in zip(images, snapshots):
        if item is not None:
            np.testing.assert_array_equal(item, snapshot)


@pytest.mark.parametrize('name,value', [
    ('adaptive_block_size', 2), ('adaptive_block_size', 1),
    ('adaptive_c', float('nan')), ('adaptive_c', float('inf')),
    ('debug_jpeg_quality', 101),
    ('fallback_variance_threshold', -1.0),
    ('fallback_variance_threshold', float('nan')),
    ('fallback_variance_threshold', float('inf')),
    ('fallback_gray_threshold', -1), ('fallback_gray_threshold', 256),
])
def test_invalid_parameter_batch_does_not_change_settings(detector, name, value):
    before = detector._settings.copy()
    result = detector.set_parameters_atomically([
        Parameter('enable_telemetry', value=True), Parameter(name, value=value)])
    assert not result.successful
    assert detector._settings == before
    assert detector.get_parameter('enable_telemetry').value is False


def test_runtime_threshold_changes_affect_segmentation(detector):
    assert detector.set_parameters_atomically([
        Parameter('enable_variance_fallback', value=False)]).successful
    frame = np.full((240, 320, 3), 100, dtype=np.uint8)
    frame[130:, 150:170] = 80
    first = detector._process_frame(frame, cv2.COLOR_BGR2GRAY, None, detector._settings)[0]
    result = detector.set_parameters_atomically([
        Parameter('adaptive_block_size', value=3), Parameter('adaptive_c', value=-1.0)])
    assert result.successful
    second = detector._process_frame(frame, cv2.COLOR_BGR2GRAY, None, detector._settings)[0]
    assert np.count_nonzero(second == 1) > np.count_nonzero(first == 1)
    assert not detector.set_parameters_atomically([Parameter('roi_y_min', value=0.2)]).successful


def test_drop_oldest_queue():
    target = queue.Queue(maxsize=1)
    assert BinaryLaneDetector._put_latest(target, 'old') == 0
    assert BinaryLaneDetector._put_latest(target, 'new') == 1
    assert target.get_nowait() == 'new'


def test_live_ros_pipeline_preserves_headers_formats_and_telemetry():
    rclpy.init(args=['--ros-args', '-p', 'debug_probe_interval_frames:=1'])
    detector = BinaryLaneDetector()
    peer = Node('binary_detector_test_camera')
    executor = SingleThreadedExecutor()
    executor.add_node(detector)
    executor.add_node(peer)
    received = {name: [] for name in ('labels', 'mask', 'overlay', 'fallback')}
    peer.create_subscription(Image, detector.label_pub.topic_name, received['labels'].append, 1)
    peer.create_subscription(
        CompressedImage, detector.debug_mask_pub.topic_name, received['mask'].append, 1)
    peer.create_subscription(
        CompressedImage, detector.debug_overlay_pub.topic_name, received['overlay'].append, 1)
    peer.create_subscription(
        CompressedImage, detector.debug_fallback_overlay_pub.topic_name,
        received['fallback'].append, 1)
    publisher = peer.create_publisher(Image, '/rgb/image_raw', qos_profile_sensor_data)
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    frame[260:470, 310:330] = 20
    msg = detector.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
    msg.header.frame_id = 'camera_optical_frame'
    msg.header.stamp = peer.get_clock().now().to_msg()
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            publisher.publish(msg)
            executor.spin_once(timeout_sec=0.02)
            if all(received.values()) and detector.frame_counter >= 30:
                break
        assert all(received.values())
        label = received['labels'][-1]
        assert (label.height, label.width, label.encoding, label.step) == (120, 320, 'mono8', 320)
        assert set(np.frombuffer(label.data, np.uint8)) == {0, 1, 3}
        for messages in received.values():
            assert messages[-1].header == msg.header
        for name in ('mask', 'overlay', 'fallback'):
            compressed = received[name][-1]
            assert compressed.format == 'bgr8; jpeg compressed bgr8'
            decoded = cv2.imdecode(np.frombuffer(compressed.data, np.uint8), cv2.IMREAD_COLOR)
            assert decoded.shape == (120, 320, 3)
        assert detector.frame_counter >= 30
        assert all(detector.telemetry_stats.values())
        assert all(0 < value < 100 for value in detector.telemetry_stats['fallback_percent'])
        assert all(0 <= value < 10000 for values in detector.telemetry_stats.values()
                   for value in values)
    finally:
        executor.shutdown()
        detector.destroy_node()
        peer.destroy_node()
        rclpy.shutdown()
    assert not detector.worker_thread.is_alive()
    assert not detector.pub_thread.is_alive()


@pytest.mark.parametrize('block_size', [3, 31, 51])
@pytest.mark.parametrize('c', [5.0, 2.7, -2.7, 0.0, 1e10, -1e10])
def test_shared_mean_preserves_opencv_adaptive_result(detector, block_size, c):
    gray = np.random.default_rng(17).integers(0, 256, (108, 320), dtype=np.uint8)
    detector._resize_band(np.zeros((240, 320), dtype=np.uint8))
    settings = dict(detector._settings, adaptive_block_size=block_size,
                    adaptive_c=c, fallback_variance_threshold=0.0)
    detector._threshold_road(gray, settings)
    # OpenCV uses an integer C internally; use equivalent bounded extremes.
    expected = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        block_size, max(-256.0, min(256.0, c)))
    np.testing.assert_array_equal(detector._buf_road, expected)
    assert not np.any(detector._buf_low_variance)


@pytest.mark.parametrize('intensity,label', [(0, 1), (100, 1), (150, 1), (151, 3), (255, 3)])
def test_uniform_regions_use_absolute_threshold(detector, intensity, label):
    settings = dict(detector._settings, enable_telemetry=True)
    frame = np.full((240, 320), intensity, dtype=np.uint8)
    labels, _, _, _ = detector._process_frame(frame, None, cv2.COLOR_GRAY2BGR, settings)
    assert np.all(labels[:12] == 0)
    assert np.all(labels[12:] == label)
    np.testing.assert_allclose(detector._buf_variance, 0.0, atol=0.01)
    assert detector.telemetry_stats['fallback_percent'][-1] == 100.0


def test_local_fallback_matches_independent_variance_reference(detector):
    gray = np.full((108, 320), 70, dtype=np.uint8)
    gray[:, 160:] = 220
    # High-contrast texture, plus weak noise on a bright uniform region.
    gray[20:80, 100:140] = np.where(np.indices((60, 40)).sum(axis=0) % 2, 240, 40)
    gray[20:80, 200:240] += (np.indices((60, 40)).sum(axis=0) % 2).astype(np.uint8)
    padded = np.pad(gray.astype(np.float64), 1, mode='edge')
    neighborhoods = np.stack([padded[y:y + 108, x:x + 320]
                              for y in range(3) for x in range(3)])
    variance = np.var(neighborhoods, axis=0)
    fallback = variance < 25.0
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 5.0)
    expected = np.where(fallback, np.where(gray <= 150, 255, 0), adaptive)
    detector._resize_band(np.zeros((240, 320), dtype=np.uint8))
    settings = dict(detector._settings, adaptive_block_size=3)
    detector._threshold_road(gray, settings)
    np.testing.assert_allclose(detector._buf_variance, variance, atol=0.01)
    np.testing.assert_array_equal(detector._buf_low_variance != 0, fallback)
    np.testing.assert_array_equal(detector._buf_road, expected)
    assert fallback[0, 0] and fallback[-1, -1]
    assert not fallback[30, 120]
    assert detector._buf_road[0, 0] == 255
    assert detector._buf_road[0, -1] == 0
    assert detector._buf_road[30, 220] == 0


def test_fallback_parameters_apply_at_runtime_and_report_correct_units(detector):
    frame = np.full((240, 320), 100, dtype=np.uint8)

    def process():
        labels, _, _, _ = detector._process_frame(
            frame, None, cv2.COLOR_GRAY2BGR, detector._settings)
        return labels[12:]

    assert detector.set_parameters_atomically([
        Parameter('enable_telemetry', value=True)]).successful
    assert np.all(process() == 1)
    assert detector.set_parameters_atomically([
        Parameter('fallback_gray_threshold', value=80)]).successful
    assert np.all(process() == 3)
    assert detector.set_parameters_atomically([
        Parameter('fallback_gray_threshold', value=120),
        Parameter('fallback_variance_threshold', value=0.0)]).successful
    assert np.all(process() == 3)
    assert detector.telemetry_stats['fallback_percent'][-1] == 0.0
    assert detector.set_parameters_atomically([
        Parameter('fallback_variance_threshold', value=25.0)]).successful
    assert np.all(process() == 1)
    assert detector.set_parameters_atomically([
        Parameter('enable_variance_fallback', value=False)]).successful
    assert np.all(process() == 3)
    assert detector.telemetry_stats['fallback_percent'][-1] == 0.0
    reports = []
    detector.get_logger = lambda: SimpleNamespace(info=reports.append)
    detector._log_telemetry_report()
    assert 'fallback_percent: 60.00 %' in reports[-1]
    assert 'adaptive_threshold:' in reports[-1]


@pytest.mark.parametrize('encoding', ['bgr8', 'rgb8', 'mono8'])
def test_fallback_overlay_marks_red_blue_and_purple_and_handles_disabling(detector, encoding):
    set_debug(detector, True, True, True)
    gray = np.full((240, 320), 50, dtype=np.uint8)
    gray[:, 160:] = 200
    channels, gray_code, bgr_code = detector._ENCODINGS[encoding]
    frame = gray if channels == 1 else np.repeat(gray[:, :, None], channels, axis=2)
    labels, mask, overlay, fallback = detector._process_frame(
        frame, gray_code, bgr_code, detector._settings)
    background = cv2.cvtColor(gray[120:], cv2.COLOR_GRAY2BGR)
    # Samples: uniform road, uniform background, road at the contrast edge,
    # and the invalid area above the ROI. Values are BGR before blending.
    for y, x, color in (
        (60, 20, (255, 0, 255)),
        (60, 300, (0, 0, 255)),
        (60, 155, (255, 0, 0)),
        (0, 20, (0, 0, 0)),
    ):
        tint = np.array([[color]], dtype=np.uint8)
        expected = cv2.addWeighted(background[y:y + 1, x:x + 1], 0.7, tint, 0.5, 0)
        np.testing.assert_array_equal(fallback[y:y + 1, x:x + 1], expected)
    assert not np.any(mask[:, :, 2])
    assert not np.array_equal(fallback, overlay)
    assert labels[60, 20] == 1 and labels[60, 300] == 3

    # A runtime disable must not show the previous frame's cached red mask.
    result = detector.set_parameters_atomically([
        Parameter('enable_variance_fallback', value=False)])
    assert result.successful
    _, _, overlay, fallback = detector._process_frame(
        frame, gray_code, bgr_code, detector._settings)
    np.testing.assert_array_equal(fallback, overlay)
