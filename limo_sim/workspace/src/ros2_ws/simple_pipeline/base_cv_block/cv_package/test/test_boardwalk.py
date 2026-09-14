"""Check BEV distance transforms against direct distances between grid cells."""

import numpy as np
import pytest

from cv_package.boardwalk import (
    BOARDWALK_COUNTS, BOARDWALK_TIMINGS, BoardwalkClassifier,
)


def classify_boardwalk(points, labels, minimum, maximum, radius, resolution=0.01):
    return BoardwalkClassifier(resolution).classify(
        points, labels, minimum, maximum, radius)


def reference_labels(points, labels, minimum, maximum, radius, resolution=0.01):
    """Use pairwise cell distances, independently of OpenCV and grid cropping."""
    points = np.floor(points.astype(np.float64) / resolution)
    minimum, maximum, radius = (value / resolution for value in (minimum, maximum, radius))
    output = labels.copy()
    blue = points[labels == 1]
    white_indices = np.flatnonzero(labels == 3)
    white = points[white_indices]
    if not len(blue) or not len(white):
        return output
    distances = np.linalg.norm(white[:, None, :] - blue[None, :, :], axis=2).min(axis=1)
    eligible = distances <= maximum
    seeds = eligible & (distances > minimum)
    if not np.any(seeds):
        return output
    seed_distances = np.linalg.norm(
        white[:, None, :] - white[seeds][None, :, :], axis=2).min(axis=1)
    selected = seeds | (eligible & (seed_distances < radius))
    output[white_indices[selected]] = 4
    return output


@pytest.mark.parametrize('seed', range(12))
def test_matches_direct_distances(seed):
    """Match both raster distance passes against a direct geometric oracle."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(-1, 1, (250, 2))
    labels = rng.choice([0, 1, 2, 3, 4], len(points)).astype(np.uint8)
    original_points = points.copy()
    expected = reference_labels(points, labels, 0.12, 0.4, 0.15)
    original_labels = labels.copy()
    stats = classify_boardwalk(points, labels, 0.12, 0.4, 0.15)
    np.testing.assert_array_equal(labels, expected)
    np.testing.assert_array_equal(points, original_points)
    assert stats['boardwalk_final_count'] == np.count_nonzero(labels != original_labels)
    assert stats['boardwalk_final_count'] == (
        stats['boardwalk_seed_count'] + stats['boardwalk_propagated_count'])
    assert stats['boardwalk_white_count'] == (
        stats['boardwalk_eligible_count'] + stats['boardwalk_outside_count'])
    assert all(np.isfinite(stats[key]) and stats[key] >= 0
               for key, _ in BOARDWALK_TIMINGS + BOARDWALK_COUNTS)


def test_thresholds_and_nonrecursive_propagation():
    """Respect strict minimum/radius, inclusive maximum and fixed seeds."""
    # Binary-exact distances make equality checks independent of rounding.
    points = np.column_stack(([0, 0.5, 0.75, 1.0, 1.125, 0.375, 0.125], np.zeros(7)))
    labels = np.array([1, 3, 3, 3, 3, 3, 3], dtype=np.uint8)
    stats = classify_boardwalk(points, labels, 0.5, 1.0, 0.5, resolution=0.125)
    # 0.125 lies within 0.5 of a newly propagated point, but not a seed.
    # 1.125 lies near a seed but remains beyond the maximum blue distance.
    np.testing.assert_array_equal(labels, [1, 4, 4, 4, 3, 4, 3])
    assert stats['boardwalk_seed_count'] == 2
    assert stats['boardwalk_propagated_count'] == 2

    labels = np.array([1, 3, 3, 3, 3, 3, 3], dtype=np.uint8)
    classify_boardwalk(points, labels, 0.5, 1.0, 0.25, resolution=0.125)
    assert labels[1] == 3  # Exactly at the propagation radius is excluded.


@pytest.mark.parametrize('minimum,maximum,radius', [
    (0.5, 1.0, 0.0), (0.5, 0.5, 0.5), (0.0, 0.0, 0.0),
])
def test_zero_radius_and_empty_seed_band(minimum, maximum, radius):
    points = np.array([[0, 0], [0, 0], [0.5, 0], [0.75, 0]], dtype=float)
    labels = np.array([1, 3, 3, 3], dtype=np.uint8)
    expected = reference_labels(points, labels, minimum, maximum, radius)
    stats = classify_boardwalk(points, labels, minimum, maximum, radius)
    np.testing.assert_array_equal(labels, expected)
    assert stats['boardwalk_seed_dt_ms'] == 0


@pytest.mark.parametrize('labels', [[], [3, 3], [1, 2], [2, 4]])
def test_missing_sources_or_targets(labels):
    labels = np.array(labels, dtype=np.uint8)
    points = np.zeros((len(labels), 2))
    original = labels.copy()
    stats = classify_boardwalk(points, labels, 0.1, 0.16, 0.1)
    np.testing.assert_array_equal(labels, original)
    assert stats['boardwalk_final_count'] == 0
    assert stats['boardwalk_blue_dt_ms'] == 0
    assert stats['boardwalk_grid_cells'] == 0


def test_nonfinite_coordinates_are_not_grid_sources_or_targets():
    points = np.array([[np.nan, 0], [0, 0], [np.inf, 0], [0.125, 0]])
    labels = np.array([1, 1, 3, 3], dtype=np.uint8)
    stats = classify_boardwalk(points, labels, 0.1, 0.16, 0.1)
    np.testing.assert_array_equal(labels, [1, 1, 3, 4])
    assert stats['boardwalk_nonfinite_count'] == 2


def test_distances_use_both_bev_axes():
    points = np.array([[0, 0], [0.05, 0.05], [0.05, 1.0], [0.1, 0.1]])
    labels = np.array([1, 3, 3, 3], dtype=np.uint8)
    classify_boardwalk(points, labels, 0.1, 0.16, 0.1)
    np.testing.assert_array_equal(labels, [1, 4, 3, 4])


@pytest.mark.parametrize('shape', [(1, 20), (20, 1)])
def test_single_row_or_column_has_no_artificial_border_sources(shape):
    points = np.array([[0.0, 0.0], [0.0, 0.125], [0.0, 0.5]])
    if shape[0] == 1:
        points = points[:, ::-1]
    labels = np.array([1, 3, 3], dtype=np.uint8)
    stats = classify_boardwalk(points, labels, 0.1, 0.16, 0.1)
    np.testing.assert_array_equal(labels, [1, 4, 3])
    assert min(stats['boardwalk_grid_width'], stats['boardwalk_grid_height']) == 1


def test_multiple_points_and_classes_in_one_cell_keep_their_identities():
    points = np.array([[0, 0], [0.001, 0], [0.125, 0], [0.126, 0], [0.127, 0]])
    labels = np.array([1, 3, 3, 3, 2], dtype=np.uint8)
    stats = classify_boardwalk(points, labels, 0.1, 0.16, 0.1)
    np.testing.assert_array_equal(labels, [1, 3, 4, 4, 2])
    assert stats['boardwalk_seed_count'] == stats['boardwalk_final_count'] == 2


def test_reused_buffers_do_not_leak_sources_between_frames():
    classifier = BoardwalkClassifier()
    rng = np.random.RandomState(42)
    for size, scale in [(150, 1.0), (40, 0.3), (200, 1.5), (15, 0.2)]:
        points = rng.uniform(-scale, scale, (size, 2))
        labels = rng.choice([1, 3], size).astype(np.uint8)
        expected = reference_labels(points, labels, 0.1, 0.3, 0.12)
        classifier.classify(points, labels, 0.1, 0.3, 0.12)
        np.testing.assert_array_equal(labels, expected)


@pytest.mark.parametrize('far', [10.0, 1e300])
def test_grid_limit_preserves_cloud_and_reports_skipped_classification(far):
    points = np.array([[0, 0], [far, far]], dtype=float)
    labels = np.array([1, 3], dtype=np.uint8)
    stats = BoardwalkClassifier(max_cells=100).classify(points, labels, 0.1, 0.16, 0.1)
    np.testing.assert_array_equal(labels, [1, 3])
    assert stats['boardwalk_grid_skipped'] == 1
    assert stats['boardwalk_blue_dt_ms'] == 0


@pytest.mark.parametrize('resolution', [0.0, -0.01, np.nan, np.inf])
def test_invalid_grid_resolution_is_rejected(resolution):
    with pytest.raises(ValueError):
        BoardwalkClassifier(resolution=resolution)


@pytest.mark.parametrize('max_cells', [0, -10, 1.5])
def test_invalid_cell_limit_is_rejected(max_cells):
    with pytest.raises(ValueError):
        BoardwalkClassifier(max_cells=max_cells)


def test_debug_panels_show_the_same_cells_used_for_classification():
    classifier = BoardwalkClassifier()
    points = np.array([[0, 0], [0.125, 0], [0.075, 0], [0.4, 0]])
    labels = np.array([1, 3, 3, 3], dtype=np.uint8)
    expected = reference_labels(points, labels, 0.1, 0.16, 0.1)
    stats = classifier.classify(points, labels, 0.1, 0.16, 0.1, debug=True)
    np.testing.assert_array_equal(labels, expected)
    frame = classifier.debug_image
    assert frame.shape == (49, 900, 3)
    # The input panel contains both white observations. Only the seed is
    # magenta after DT1; the second observation is added after DT2.
    np.testing.assert_array_equal(frame[48, 12], [255, 255, 255])
    np.testing.assert_array_equal(frame[48, 300 + 12], [255, 0, 255])
    np.testing.assert_array_equal(frame[48, 300 + 7], [255, 255, 255])
    np.testing.assert_array_equal(frame[48, 600 + 7], [255, 0, 255])
    np.testing.assert_array_equal(frame[48, 600 + 40], [255, 255, 255])
    assert stats['boardwalk_debug_render_ms'] > 0


def test_debug_is_cleared_when_disabled_and_reports_an_unevaluated_grid():
    classifier = BoardwalkClassifier()
    points = np.array([[0, 0], [0.125, 0]])
    labels = np.array([1, 3], dtype=np.uint8)
    classifier.classify(points, labels.copy(), 0.1, 0.16, 0.1, debug=True)
    stats = classifier.classify(points, labels.copy(), 0.1, 0.16, 0.1)
    assert classifier.debug_image is None
    assert stats['boardwalk_debug_render_ms'] == 0
    classifier.classify(points, np.array([3, 3]), 0.1, 0.16, 0.1, debug=True)
    assert classifier.debug_image.shape == (80, 640, 3)


def test_auto_backend_falls_back_on_the_same_frame_without_retrying(monkeypatch):
    import sys
    from types import SimpleNamespace

    attempts = []

    def unavailable(max_cells):
        attempts.append(max_cells)
        raise RuntimeError('CUDA intentionally unavailable in this test')

    monkeypatch.setitem(sys.modules, 'cv_package.boardwalk_cuda',
                        SimpleNamespace(CudaBoardwalkGrid=unavailable))
    classifier = BoardwalkClassifier(backend='auto')
    points = np.array([[0, 0], [0.125, 0], [0.075, 0]])
    for _ in range(2):
        labels = np.array([1, 3, 3], dtype=np.uint8)
        expected = reference_labels(points, labels, 0.1, 0.16, 0.1)
        stats = classifier.classify(points, labels, 0.1, 0.16, 0.1)
        np.testing.assert_array_equal(labels, expected)
        assert stats['boardwalk_gpu_fallback'] == 1
        assert stats['boardwalk_gpu_used'] == 0
    assert len(attempts) == 1
    assert classifier.active_backend == 'cpu'
