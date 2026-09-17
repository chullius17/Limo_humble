"""Check metric boardwalk labels against a direct geometric reference."""

import numpy as np
import pytest
import cv2

from cv_package.boardwalk import (
    BOARDWALK_COUNTS, BOARDWALK_TIMINGS, classify_boardwalk,
)


def reference_labels(points, labels, minimum, maximum, radius):
    """Use all pairwise distances as a small-cloud correctness oracle."""
    output = labels.copy()
    blue = points[labels == 1]
    white_indices = np.flatnonzero(labels == 3)
    white = points[white_indices]
    if not len(blue) or not len(white):
        return output
    distances = np.linalg.norm(white[:, None, :] - blue[None, :, :], axis=2).min(axis=1)
    eligible = distances <= maximum
    output[white_indices[distances > maximum]] = 6
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
    """Match both distance passes without a raster or approximate queries."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(-1, 1, (250, 2))
    labels = rng.choice([0, 1, 2, 3, 4], len(points)).astype(np.uint8)
    original_points = points.copy()
    expected = reference_labels(points, labels, 0.12, 0.4, 0.15)
    original_labels = labels.copy()
    stats = classify_boardwalk(points, labels, 0.12, 0.4, 0.15)
    np.testing.assert_array_equal(labels, expected)
    np.testing.assert_array_equal(points, original_points)
    assert stats['boardwalk_final_count'] == np.count_nonzero(
        (original_labels == 3) & (labels == 4))
    assert stats['boardwalk_outside_count'] == np.count_nonzero(
        (original_labels == 3) & (labels == 6))
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
    stats = classify_boardwalk(points, labels, 0.5, 1.0, 0.5)
    # 0.125 lies within 0.5 of a newly propagated point, but not a seed.
    # 1.125 lies near a seed but remains beyond the maximum blue distance.
    np.testing.assert_array_equal(labels, [1, 4, 4, 4, 6, 4, 3])
    assert stats['boardwalk_seed_count'] == 2
    assert stats['boardwalk_propagated_count'] == 2

    labels = np.array([1, 3, 3, 3, 3, 3, 3], dtype=np.uint8)
    classify_boardwalk(points, labels, 0.5, 1.0, 0.25)
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
    assert stats['boardwalk_seed_build_ms'] == 0
    assert stats['boardwalk_seed_query_ms'] == 0


@pytest.mark.parametrize('labels', [[], [3, 3], [1, 2], [2, 4]])
def test_missing_sources_or_targets(labels):
    labels = np.array(labels, dtype=np.uint8)
    points = np.zeros((len(labels), 2))
    original = labels.copy()
    stats = classify_boardwalk(points, labels, 0.1, 0.16, 0.1)
    np.testing.assert_array_equal(labels, original)
    assert stats['boardwalk_final_count'] == 0
    assert stats['boardwalk_blue_build_ms'] == 0


def test_nonfinite_coordinates_are_not_tree_sources_or_targets():
    points = np.array([[np.nan, 0], [0, 0], [np.inf, 0], [0.125, 0]])
    labels = np.array([1, 1, 3, 3], dtype=np.uint8)
    stats = classify_boardwalk(points, labels, 0.1, 0.16, 0.1)
    np.testing.assert_array_equal(labels, [1, 1, 3, 4])
    assert stats['boardwalk_nonfinite_count'] == 2


def test_distances_use_both_bev_axes():
    points = np.array([[0, 0], [0.05, 0.05], [0.05, 1.0], [0.1, 0.1]])
    labels = np.array([1, 3, 3, 3], dtype=np.uint8)
    classify_boardwalk(points, labels, 0.1, 0.16, 0.1)
    np.testing.assert_array_equal(labels, [1, 4, 6, 4])


from cv_package.boardwalk import BoardwalkClassifier


def test_points_in_one_former_grid_cell_remain_distinct():
    points = np.array([[0.001, 0], [0.002, 0], [0.008, 0]])
    labels = np.array([1, 3, 3], dtype=np.uint8)
    classify_boardwalk(points, labels, 0.004, 0.009, 0.0)
    np.testing.assert_array_equal(labels, [1, 3, 4])


@pytest.mark.parametrize('shape', [(1, 1), (9, 13), (120, 320)])
def test_blue_filter_uses_the_seven_by_seven_white_neighborhood(shape):
    labels = np.zeros(shape, dtype=np.uint8)
    labels[-1, -1] = 3
    if shape[0] > 1:
        labels[max(0, shape[0] - 4), max(0, shape[1] - 4)] = 1
    kernel = np.ones((7, 7), dtype=np.uint8)
    result = BoardwalkClassifier().filter_blue(labels, kernel)
    expected = (labels == 1) & (
        cv2.dilate(
            (labels == 3).astype(np.uint8), kernel,
            borderType=cv2.BORDER_CONSTANT, borderValue=0) != 0)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize('thresholds', [(-1, 1, 1), (2, 1, 1), (0, 1, -1), (0, np.inf, 1)])
def test_invalid_thresholds(thresholds):
    with pytest.raises(ValueError):
        BoardwalkClassifier().classify(np.empty((0, 2)), np.empty(0), *thresholds)
