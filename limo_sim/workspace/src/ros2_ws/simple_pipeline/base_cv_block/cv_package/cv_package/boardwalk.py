"""Classify observed white BEV points with two nearest-neighbor passes."""

import time

import numpy as np
from scipy.spatial import cKDTree


BOARDWALK_TIMINGS = (
    ('boardwalk_total_ms', 'Total'),
    ('boardwalk_blue_build_ms', 'Blue tree build'),
    ('boardwalk_blue_query_ms', 'White -> blue query'),
    ('boardwalk_seed_build_ms', 'Seed tree build'),
    ('boardwalk_seed_query_ms', 'White -> seed query'),
)
BOARDWALK_COUNTS = (
    ('boardwalk_blue_count', 'Blue sources'),
    ('boardwalk_white_count', 'White input'),
    ('boardwalk_eligible_count', 'White within max distance'),
    ('boardwalk_seed_count', 'First-pass seeds'),
    ('boardwalk_propagated_count', 'Added by second pass'),
    ('boardwalk_final_count', 'Final boardwalk'),
    ('boardwalk_outside_count', 'White outside max distance'),
    ('boardwalk_nonfinite_count', 'Non-finite points ignored'),
)


def classify_boardwalk(
        points, class_ids, minimum, maximum, propagation_radius,
        blue_label=1, white_label=3, boardwalk_label=4):
    """Relabel white points in place and return per-cloud timings and counts.

    Points are metric BEV (x, y) coordinates. Thresholds must be finite and
    satisfy 0 <= minimum <= maximum and propagation_radius >= 0.
    Coordinates and all labels other than the selected white labels are kept.
    """
    started = time.perf_counter()
    stats = {key: 0.0 for key, _ in BOARDWALK_TIMINGS}
    stats.update({key: 0 for key, _ in BOARDWALK_COUNTS})

    def finished():
        stats['boardwalk_total_ms'] = (time.perf_counter() - started) * 1000.0
        return stats

    # Trees require finite coordinates. An invalid point must neither become
    # a distance source nor prevent the remaining cloud from being classified.
    finite = np.isfinite(points).all(axis=1)
    blue_indices = np.flatnonzero((class_ids == blue_label) & finite)
    white_indices = np.flatnonzero((class_ids == white_label) & finite)
    stats['boardwalk_blue_count'] = len(blue_indices)
    stats['boardwalk_white_count'] = len(white_indices)
    stats['boardwalk_nonfinite_count'] = int(np.count_nonzero(~finite))
    if not len(blue_indices) or not len(white_indices):
        stats['boardwalk_outside_count'] = len(white_indices)
        return finished()

    # Convert each subset once: cKDTree uses contiguous float64 coordinates.
    # Query whole batches on its default single worker, including on older
    # Jetson SciPy versions that do not accept the newer "workers" argument.
    blue_xy = np.ascontiguousarray(points[blue_indices], dtype=np.float64)
    white_xy = np.ascontiguousarray(points[white_indices], dtype=np.float64)
    stage = time.perf_counter()
    blue_tree = cKDTree(blue_xy)
    stats['boardwalk_blue_build_ms'] = (time.perf_counter() - stage) * 1000.0
    stage = time.perf_counter()
    # An unbounded exact query preserves the inclusive maximum even at zero
    # distance; bounded queries can exclude neighbors exactly on their bound.
    distance_blue, _ = blue_tree.query(white_xy, k=1, eps=0.0, p=2)
    stats['boardwalk_blue_query_ms'] = (time.perf_counter() - stage) * 1000.0

    eligible = distance_blue <= maximum
    seeds = eligible & (distance_blue > minimum)
    stats['boardwalk_eligible_count'] = int(np.count_nonzero(eligible))
    stats['boardwalk_outside_count'] = len(white_indices) - stats['boardwalk_eligible_count']
    stats['boardwalk_seed_count'] = int(np.count_nonzero(seeds))
    if not stats['boardwalk_seed_count']:
        return finished()

    selected = seeds.copy()
    remaining = np.flatnonzero(eligible & ~seeds)
    if len(remaining) and propagation_radius > 0.0:
        stage = time.perf_counter()
        seed_tree = cKDTree(white_xy[seeds])
        stats['boardwalk_seed_build_ms'] = (time.perf_counter() - stage) * 1000.0
        stage = time.perf_counter()
        # Only first-pass seeds are sources: newly selected points never
        # propagate recursively. The strict radius matches classes.py.
        distance_seed, _ = seed_tree.query(
            white_xy[remaining], k=1, eps=0.0, p=2,
            distance_upper_bound=propagation_radius)
        stats['boardwalk_seed_query_ms'] = (time.perf_counter() - stage) * 1000.0
        propagated = remaining[distance_seed < propagation_radius]
        selected[propagated] = True
        stats['boardwalk_propagated_count'] = len(propagated)

    # Keep observed geometry and white points outside the allowed band.
    # This stage introduces class 4; it does not delete cloud observations.
    class_ids[white_indices[selected]] = boardwalk_label
    stats['boardwalk_final_count'] = int(np.count_nonzero(selected))
    return finished()
