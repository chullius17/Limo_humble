"""Check save-time denoising without requiring a running ROS environment."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from offline_map_package.map_filters import median_saved_semantic


@pytest.mark.parametrize('cost', [30, 60, 90])
def test_median_removes_isolated_costs_without_mutating_snapshot(cost):
    source = np.zeros((7, 7), dtype=np.int8)
    source[3, 3] = cost
    source.flags.writeable = False
    filtered = median_saved_semantic(source, 3)
    assert not filtered.any()
    assert source[3, 3] == cost
    assert not np.shares_memory(filtered, source)
    assert filtered.dtype == source.dtype


def test_median_fills_small_free_holes_but_preserves_unknown_cells():
    source = np.full((9, 9), 90, dtype=np.int8)
    source[2, 2] = 0
    source[6, 6] = -1
    filtered = median_saved_semantic(source, 3)
    assert filtered[2, 2] == 90
    assert filtered[6, 6] == -1
    assert np.count_nonzero(filtered == 90) == 80


def test_isolated_observation_among_unknowns_returns_to_unknown():
    source = np.full((5, 5), -1, dtype=np.int8)
    source[2, 2] = 60
    assert np.all(median_saved_semantic(source, 3) == -1)


@pytest.mark.parametrize('shape', [(1, 1), (1, 9), (9, 1), (260, 9)])
@pytest.mark.parametrize('kernel', [3, 5])
def test_edges_and_row_blocks_match_independent_median(shape, kernel):
    source = np.random.default_rng(17).choice(
        np.array([-1, 0, 30, 60, 90], dtype=np.int8), size=shape)
    padded = np.pad(source, kernel // 2, mode='edge')
    expected = source.copy()
    for y, x in np.ndindex(shape):
        if source[y, x] >= 0:
            expected[y, x] = np.median(padded[y:y + kernel, x:x + kernel])
    np.testing.assert_array_equal(median_saved_semantic(source, kernel), expected)


def test_disabled_filter_returns_an_independent_exact_copy():
    source = np.array([[-1, 0, 30, 60, 90]], dtype=np.int8)
    filtered = median_saved_semantic(source, 1)
    np.testing.assert_array_equal(filtered, source)
    assert not np.shares_memory(filtered, source)
    assert median_saved_semantic(np.empty((0, 0), dtype=np.int8), 3).shape == (0, 0)


@pytest.mark.parametrize('kernel', [0, -1, 2, 4, 3.0, True])
def test_invalid_kernels_are_rejected(kernel):
    with pytest.raises(ValueError, match='save_median_kernel'):
        median_saved_semantic(np.zeros((3, 3), dtype=np.int8), kernel)


@pytest.mark.parametrize('profile,kernel', [('real', 3), ('sim', 1)])
def test_only_real_profile_enables_save_filter(profile, kernel):
    path = Path(__file__).parents[1] / 'config' / ('mapping_' + profile + '.yaml')
    settings = yaml.safe_load(path.read_text())
    assert settings['semantic_mapper']['save_median_kernel'] == kernel
