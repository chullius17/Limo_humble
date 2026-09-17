"""Classify observed soft-obstacle BEV points with two CPU neighbor passes."""

import time

import cv2
import numpy as np
from scipy.spatial import cKDTree


BOARDWALK_TIMINGS = (
    ('boardwalk_total_ms', 'Total'),
    ('boardwalk_prepare_ms', 'Point preparation'),
    ('boardwalk_blue_build_ms', 'Blue tree build'),
    ('boardwalk_blue_query_ms', 'Soft obstacle -> exterior road'),
    ('boardwalk_seed_build_ms', 'Seed tree build'),
    ('boardwalk_seed_query_ms', 'Soft obstacle -> seed query'),
    ('boardwalk_labels_ms', 'Point label update'),
)
BOARDWALK_COUNTS = (
    ('boardwalk_blue_count', 'Exterior road sources'),
    ('boardwalk_white_count', 'Soft obstacle input'),
    ('boardwalk_eligible_count', 'Soft obstacle within max'),
    ('boardwalk_seed_count', 'First-pass seeds'),
    ('boardwalk_propagated_count', 'Added by second pass'),
    ('boardwalk_final_count', 'Final boardwalk'),
    ('boardwalk_outside_count', 'Interior boardwalk (pass 1)'),
    ('boardwalk_nonfinite_count', 'Non-finite points ignored'),
)


class BoardwalkClassifier:
    """Apply the point-based rules from commit 4f7f16e using CPU cKDTree."""

    @staticmethod
    def filter_blue(labels, kernel, blue_label=1, white_label=3):
        """Keep exterior-road pixels near the soft-obstacle region."""
        # OpenCV performs the soft-obstacle dilation efficiently on CPU.
        # Intersecting it with the source mask preserves only the boundary.
        white_dilated = cv2.dilate(
            (labels == white_label).astype(np.uint8), kernel,
            borderType=cv2.BORDER_CONSTANT, borderValue=0)
        return (labels == blue_label) & (white_dilated != 0)

    def classify(
            self, points, class_ids, minimum, maximum, propagation_radius,
            blue_label=1, white_label=3, boardwalk_label=4,
            interior_boardwalk_label=6):
        """Relabel selected soft obstacles; preserve coordinates and order."""
        if (not np.isfinite([minimum, maximum, propagation_radius]).all()
                or not 0 <= minimum <= maximum or propagation_radius < 0):
            raise ValueError('Invalid boardwalk distance thresholds')

        started = time.perf_counter()
        stats = {key: 0.0 for key, _ in BOARDWALK_TIMINGS}
        stats.update({key: 0 for key, _ in BOARDWALK_COUNTS})

        stage = time.perf_counter()
        finite = np.isfinite(points).all(axis=1)
        blue_indices = np.flatnonzero((class_ids == blue_label) & finite)
        white_indices = np.flatnonzero((class_ids == white_label) & finite)
        # cKDTree expects contiguous float64 coordinates.
        blue = np.ascontiguousarray(points[blue_indices], dtype=np.float64)
        white = np.ascontiguousarray(points[white_indices], dtype=np.float64)
        stats['boardwalk_blue_count'] = len(blue)
        stats['boardwalk_white_count'] = len(white)
        stats['boardwalk_nonfinite_count'] = int(np.count_nonzero(~finite))
        stats['boardwalk_prepare_ms'] = (time.perf_counter() - stage) * 1000.0

        eligible = np.zeros(len(white), dtype=bool)
        outside = eligible.copy()
        seeds = eligible.copy()
        selected = eligible.copy()

        if len(blue) and len(white):
            stage = time.perf_counter()
            blue_tree = cKDTree(blue)
            stats['boardwalk_blue_build_ms'] = (
                time.perf_counter() - stage) * 1000.0
            stage = time.perf_counter()
            # The unbounded exact query preserves the inclusive maximum.
            distance_blue, _ = blue_tree.query(white, k=1, eps=0.0, p=2)
            stats['boardwalk_blue_query_ms'] = (
                time.perf_counter() - stage) * 1000.0

            eligible = distance_blue <= maximum
            outside = distance_blue > maximum
            seeds = eligible & (distance_blue > minimum)
            selected = seeds.copy()
            remaining = np.flatnonzero(eligible & ~seeds)
            if np.any(seeds) and len(remaining) and propagation_radius > 0:
                stage = time.perf_counter()
                seed_tree = cKDTree(white[seeds])
                stats['boardwalk_seed_build_ms'] = (
                    time.perf_counter() - stage) * 1000.0
                stage = time.perf_counter()
                distance_seed, _ = seed_tree.query(
                    white[remaining], k=1, eps=0.0, p=2,
                    distance_upper_bound=propagation_radius)
                stats['boardwalk_seed_query_ms'] = (
                    time.perf_counter() - stage) * 1000.0
                # Only first-pass seeds are sources: propagation is not recursive.
                selected[remaining[distance_seed < propagation_radius]] = True

        stats['boardwalk_eligible_count'] = int(np.count_nonzero(eligible))
        stats['boardwalk_seed_count'] = int(np.count_nonzero(seeds))
        stats['boardwalk_propagated_count'] = int(
            np.count_nonzero(selected & ~seeds))
        stats['boardwalk_final_count'] = int(np.count_nonzero(selected))
        stats['boardwalk_outside_count'] = int(np.count_nonzero(outside))

        stage = time.perf_counter()
        class_ids[white_indices[selected]] = boardwalk_label
        class_ids[white_indices[outside]] = interior_boardwalk_label
        stats['boardwalk_labels_ms'] = (
            time.perf_counter() - stage) * 1000.0

        stats['boardwalk_total_ms'] = (
            time.perf_counter() - started) * 1000.0
        return stats

    def close(self):
        """Keep worker shutdown uniform; the CPU classifier owns no resources."""
        pass


def classify_boardwalk(
        points, class_ids, minimum, maximum, propagation_radius,
        blue_label=1, white_label=3, boardwalk_label=4,
        interior_boardwalk_label=6):
    """Compatibility entry point for the point classifier."""
    return BoardwalkClassifier().classify(
        points, class_ids, minimum, maximum, propagation_radius,
        blue_label, white_label, boardwalk_label, interior_boardwalk_label)
