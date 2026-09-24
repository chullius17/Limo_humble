"""Verify seeded growth, gradient barriers, debug images and ROS publication."""

from collections import deque
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

from cv_package.lane_detector_waterfall import WaterfallLaneDetector


@pytest.fixture
def detector():
    rclpy.init(args=['--ros-args', '-p', 'enable_telemetry:=false'])
    node = None
    try:
        node = WaterfallLaneDetector()
        node._stop.set()
        node.worker_thread.join()
        node.pub_thread.join()
        yield node
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def process_gray(detector, frame, **settings):
    return detector._process_frame(
        frame, None, cv2.COLOR_GRAY2BGR, dict(detector._settings, **settings))[0]


@pytest.mark.parametrize('gray,label', [(0, 1), (80, 1), (81, 3), (255, 3)])
def test_uniform_frames_and_seed_threshold_equality(detector, gray, label):
    labels = process_gray(detector, np.full((240, 320), gray, np.uint8))
    assert labels.shape == (120, 320) and labels.dtype == np.uint8
    assert np.all(labels[:12] == 0)
    assert np.all(labels[12:] == label)
    assert not np.any(detector._buf_barriers)


def test_growth_reaches_brighter_pixels_across_a_gentle_ramp(detector):
    frame = np.tile((30 + np.arange(320) // 2).astype(np.uint8), (240, 1))
    labels = process_gray(detector, frame, enable_telemetry=True)
    assert np.all(labels[12:] == 1)
    assert np.all(detector._buf_seeds[:, -1] == 0)
    assert not np.any(detector._buf_barriers)
    assert detector.telemetry_stats['seeded_components'][-1] == 1
    assert 0 < detector.telemetry_stats['seed_percent'][-1] < 100
    assert detector.telemetry_stats['road_percent'][-1] == 100


def test_strong_edge_stops_growth_and_dilation_excludes_seeds(detector):
    frame = np.full((240, 320), 40, np.uint8)
    frame[:, 160:] = 200
    labels = process_gray(detector, frame, barrier_dilation_iterations=0)
    assert np.all(labels[12:, :159] == 1)
    assert np.all(labels[12:, 159:] == 3)
    assert np.all(detector._buf_barriers[:, 159:161] == 255)
    assert not np.any(detector._buf_seeds[:, 159:])
    old_barriers = detector._buf_barriers.copy()
    labels = process_gray(detector, frame, barrier_dilation_iterations=1)
    assert np.all(detector._buf_barriers[:, 158:162] == 255)
    assert np.count_nonzero(detector._buf_barriers) > np.count_nonzero(old_barriers)
    assert not np.any(detector._buf_seeds[detector._buf_barriers != 0])
    assert np.all(labels[12:, 158:] == 3)


def test_seed_band_restricts_starts_without_restricting_growth(detector):
    frame = np.full((240, 320), 40, np.uint8)
    labels = process_gray(detector, frame, seed_y_min=0.8)
    assert not np.any(detector._buf_seeds[:86])
    assert np.all(labels[12:] == 1)
    frame[200:] = 200
    assert np.any(process_gray(detector, frame) == 1)
    assert not np.any(process_gray(detector, frame, seed_y_min=0.8) == 1)


def test_native_components_match_independent_flood_fill(detector):
    detector._resize_band(np.zeros((240, 320), np.uint8))
    rng = np.random.default_rng(7)
    free = rng.random((108, 320)) > 0.4
    seeds = (rng.random(free.shape) > 0.99) & free
    detector._buf_free[:] = free.astype(np.uint8) * 255
    detector._buf_seeds[:] = seeds.astype(np.uint8) * 255
    visited = seeds.copy()
    pending = deque(map(tuple, np.argwhere(seeds)))
    while pending:
        y, x = pending.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (0 <= ny < free.shape[0] and 0 <= nx < free.shape[1]
                    and free[ny, nx] and not visited[ny, nx]):
                visited[ny, nx] = True
                pending.append((ny, nx))
    assert detector._propagate_seeds() > 0
    np.testing.assert_array_equal(detector._buf_road != 0, visited)
    detector._buf_seeds.fill(0)
    assert detector._propagate_seeds() == 0
    assert not np.any(detector._buf_road)


def test_components_do_not_connect_diagonally(detector):
    detector._resize_band(np.zeros((240, 320), np.uint8))
    detector._buf_free.fill(0)
    detector._buf_seeds.fill(0)
    detector._buf_free[10, 10] = detector._buf_free[11, 11] = 255
    detector._buf_seeds[10, 10] = 255
    assert detector._propagate_seeds() == 1
    assert detector._buf_road[10, 10] == 255
    assert detector._buf_road[11, 11] == 0


@pytest.mark.parametrize('encoding', ['bgr8', 'rgb8', 'mono8'])
@pytest.mark.parametrize('shape', [(480, 640), (481, 641)])
def test_padded_ingress_and_geometry_match_full_crop_reference(detector, encoding, shape):
    height, width = shape
    channels, gray_code, _ = detector._ENCODINGS[encoding]
    native = np.random.default_rng(11).integers(
        0, 256, shape + ((channels,) if channels > 1 else ()), dtype=np.uint8)
    padded = np.full((height, width * channels + 7), 231, np.uint8)
    padded[:, :width * channels] = native.reshape(height, width * channels)
    msg = Image(height=height, width=width, encoding=encoding,
                step=padded.shape[1], data=padded.tobytes())
    view, code, bgr = detector._view_source(msg)
    assert np.shares_memory(view, np.frombuffer(msg.data, np.uint8))
    labels = detector._process_frame(view, code, bgr, detector._settings)[0]
    small = cv2.resize(native[height // 2:], (320, 120), interpolation=cv2.INTER_AREA)
    gray = small if gray_code is None else cv2.cvtColor(small, gray_code)
    reference_frame = np.zeros((240, 320), np.uint8)
    reference_frame[120:] = gray
    np.testing.assert_array_equal(labels, process_gray(detector, reference_frame))


@pytest.mark.parametrize('mask_on,overlay_on,seeds_on', [
    (False, False, False), (True, False, False), (False, True, False),
    (False, False, True), (True, True, True),
])
def test_debug_colors_subscriber_gating_and_frame_ownership(
        detector, mask_on, overlay_on, seeds_on):
    detector.debug_mask_pub = SimpleNamespace(get_subscription_count=lambda: int(mask_on))
    detector.debug_overlay_pub = SimpleNamespace(get_subscription_count=lambda: int(overlay_on))
    detector.debug_seeds_overlay_pub = SimpleNamespace(
        get_subscription_count=lambda: int(seeds_on))
    frame = np.full((240, 320, 3), 40, np.uint8)
    frame[:, 160:] = 200
    images = detector._process_frame(frame, cv2.COLOR_BGR2GRAY, None, detector._settings)
    labels, mask, overlay, seeds = images
    assert (mask is not None) == mask_on
    assert (overlay is not None) == overlay_on
    assert (seeds is not None) == seeds_on
    if mask_on:
        np.testing.assert_array_equal(mask[60, 10], [255, 0, 0])
        assert not np.any(mask[:12])
    if seeds_on:
        np.testing.assert_array_equal(seeds[60, 10], [0, 255, 0])
        np.testing.assert_array_equal(seeds[60, 159], [0, 0, 255])
        np.testing.assert_array_equal(seeds[60, 200], [200, 200, 200])
        np.testing.assert_array_equal(seeds[:12], frame[120:132])
    saved = [None if item is None else item.copy() for item in images]
    detector._process_frame(np.zeros_like(frame), cv2.COLOR_BGR2GRAY, None, detector._settings)
    for item, expected in zip(images, saved):
        if item is not None:
            np.testing.assert_array_equal(item, expected)


@pytest.mark.parametrize('name,value', [
    ('seed_max_gray', -1), ('seed_max_gray', 256),
    ('gradient_threshold', -1.0), ('gradient_threshold', float('nan')),
    ('gradient_threshold', float('inf')), ('seed_y_min', 1.0),
    ('seed_y_min', -0.1), ('barrier_dilation_iterations', -1),
    ('barrier_dilation_iterations', 6), ('debug_jpeg_quality', 101),
    ('seed_erosion_iterations', -1), ('seed_erosion_iterations', 6),
])
def test_invalid_parameter_batch_is_atomic(detector, name, value):
    before = detector._settings.copy()
    result = detector.set_parameters_atomically([
        Parameter('enable_telemetry', value=True), Parameter(name, value=value)])
    assert not result.successful
    assert detector._settings == before
    assert detector.get_parameter('enable_telemetry').value is False


def test_runtime_parameters_change_barriers_and_seed_selection(detector):
    frame = np.full((240, 320), 40, np.uint8)
    frame[:, 160:] = 200
    assert np.all(process_gray(detector, frame)[12:, 200:] == 3)
    assert detector.set_parameters_atomically([
        Parameter('gradient_threshold', value=255.0)]).successful
    assert np.all(process_gray(detector, frame)[12:] == 1)
    assert detector.set_parameters_atomically([
        Parameter('seed_max_gray', value=20)]).successful
    assert np.all(process_gray(detector, frame)[12:] == 3)


def test_live_ros_pipeline_preserves_headers_and_publishes_debug():
    rclpy.init(args=['--ros-args', '-p', 'debug_probe_interval_frames:=1'])
    detector = WaterfallLaneDetector()
    peer = Node('waterfall_test_camera')
    executor = SingleThreadedExecutor()
    executor.add_node(detector)
    executor.add_node(peer)
    received = {name: [] for name in ('labels', 'mask', 'overlay', 'seeds')}
    for name, publisher, message_type in (
        ('labels', detector.label_pub, Image),
        ('mask', detector.debug_mask_pub, CompressedImage),
        ('overlay', detector.debug_overlay_pub, CompressedImage),
        ('seeds', detector.debug_seeds_overlay_pub, CompressedImage),
    ):
        peer.create_subscription(message_type, publisher.topic_name, received[name].append, 1)
    publisher = peer.create_publisher(Image, '/rgb/image_raw', qos_profile_sensor_data)
    frame = np.full((240, 320, 3), 40, np.uint8)
    frame[:, 160:] = 200
    msg = detector.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
    msg.header.frame_id = 'camera_optical_frame'
    msg.header.stamp = peer.get_clock().now().to_msg()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            publisher.publish(msg)
            executor.spin_once(timeout_sec=0.02)
            if all(received.values()) and detector.frame_counter >= 30:
                break
        assert all(received.values()) and detector.frame_counter >= 30
        labels = received['labels'][-1]
        assert (labels.height, labels.width, labels.encoding) == (120, 320, 'mono8')
        assert set(np.frombuffer(labels.data, np.uint8)) == {0, 1, 3}
        for messages in received.values():
            assert messages[-1].header == msg.header
        for name in ('mask', 'overlay', 'seeds'):
            jpeg = received[name][-1]
            assert jpeg.format == 'bgr8; jpeg compressed bgr8'
            decoded = cv2.imdecode(np.frombuffer(jpeg.data, np.uint8), cv2.IMREAD_COLOR)
            assert decoded.shape == (120, 320, 3)
        assert all(detector.telemetry_stats.values())
        assert all(value == 1 for value in detector.telemetry_stats['seeded_components'])
    finally:
        executor.shutdown()
        detector.destroy_node()
        peer.destroy_node()
        rclpy.shutdown()
    assert not detector.worker_thread.is_alive() and not detector.pub_thread.is_alive()


@pytest.mark.parametrize('height,width', [(1, 1), (2, 2), (1, 20), (20, 1)])
def test_erosion_prevents_small_or_thin_seed_groups_from_flooding(detector, height, width):
    frame = np.full((240, 320), 100, np.uint8)
    frame[160:160 + height, 100:100 + width] = 40
    # Disable barriers to isolate a spurious seed's ability to flood the ROI.
    unfiltered = process_gray(detector, frame, gradient_threshold=255.0,
                              seed_erosion_iterations=0)
    assert np.all(unfiltered[12:] == 1)
    filtered = process_gray(detector, frame, gradient_threshold=255.0,
                            seed_erosion_iterations=1, enable_telemetry=True)
    assert not np.any(detector._buf_seeds)
    assert np.all(filtered[12:] == 3)
    assert detector.telemetry_stats['seed_removed_percent'][-1] == 100.0
    assert detector.telemetry_stats['seeded_components'][-1] == 0


def test_erosion_removes_small_group_touching_roi_corner(detector):
    frame = np.full((240, 320), 100, np.uint8)
    frame[132:134, :2] = 40
    labels = process_gray(detector, frame, gradient_threshold=255.0,
                          seed_erosion_iterations=1)
    assert not np.any(detector._buf_seeds)
    assert np.all(labels[12:] == 3)


def test_surviving_seed_core_grows_and_debug_shows_only_filtered_seeds(detector):
    detector.debug_seeds_overlay_pub = SimpleNamespace(get_subscription_count=lambda: 1)
    frame = np.full((240, 320), 100, np.uint8)
    frame[160:165, 100:105] = 40
    settings = dict(detector._settings, gradient_threshold=255.0,
                    seed_erosion_iterations=1, enable_telemetry=True)
    labels, _, _, debug = detector._process_frame(frame, None, cv2.COLOR_GRAY2BGR, settings)
    assert cv2.countNonZero(detector._buf_seeds) == 9
    assert np.all(labels[12:] == 1)
    # Frame row 160 is output row 40. Only the central 3x3 remains green.
    np.testing.assert_array_equal(debug[40, 100], [40, 40, 40])
    np.testing.assert_array_equal(debug[42, 102], [0, 255, 0])
    np.testing.assert_array_equal(debug[60, 200], [100, 100, 100])
    assert detector.telemetry_stats['seed_removed_percent'][-1] == 64.0
    assert detector.telemetry_stats['seed_percent'][-1] == pytest.approx(100 * 9 / (108 * 320))
    assert detector.telemetry_stats['seed_erosion'][-1] >= 0


def test_erosion_strength_can_change_at_runtime_and_disable(detector):
    frame = np.full((240, 320), 100, np.uint8)
    frame[160:165, 100:105] = 40
    for iterations, expected_seeds in ((2, 1), (3, 0), (0, 25)):
        assert detector.set_parameters_atomically([
            Parameter('seed_erosion_iterations', value=iterations)]).successful
        labels = process_gray(detector, frame, gradient_threshold=255.0,
                              enable_telemetry=True)
        assert cv2.countNonZero(detector._buf_seeds) == expected_seeds
        assert np.all(labels[12:] == (1 if expected_seeds else 3))
    assert detector.telemetry_stats['seed_removed_percent'][-1] == 0.0
    process_gray(detector, np.full_like(frame, 100), enable_telemetry=True)
    assert detector.telemetry_stats['seed_removed_percent'][-1] == 0.0
