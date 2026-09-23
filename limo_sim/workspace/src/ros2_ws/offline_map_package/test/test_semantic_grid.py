"""Regression tests for semantic costs, correction and submap reprojection."""

from array import array
from types import SimpleNamespace

import numpy as np
import pytest

from offline_map_package.semantic_grid import (
    Geometry, LaserEndpointGrid, SemanticGrid, read_class_cloud,
)


def mapper():
    grid = SemanticGrid(resolution=1.0)
    grid.set_pose((0, 0), (0.0, 0.0, 0.0))
    return grid


def test_laser_endpoint_grid_accumulates_and_rasterizes_cost_100():
    grid = LaserEndpointGrid(resolution=1.0, max_cells=4)
    assert grid.update(np.array([[0.1, 0.1], [2.1, 1.1], [2.9, 1.9]]))
    assert not grid.update(np.array([[0.2, 0.2]]))
    geometry = grid.geometry()
    assert geometry == Geometry(1.0, 3, 2, (0.0, 0.0, 0.0))
    assert grid.render(geometry).tolist() == [[100, 0, 0], [0, 0, 100]]


def test_costs_and_blue_road():
    grid = mapper()
    grid.update(np.array([[0.1, 0.1], [1.1, 0.1], [2.1, 0.1], [3.1, 0.1]]),
                np.array([1, 2, 3, 4]))
    _, layers, combined = grid.render(Geometry(1.0, 4, 1, (0, 0, 0)))
    assert combined.tolist() == [[0, 60, 30, 90]]
    assert layers[:, 0].tolist() == [[0, 60, 0, 0], [0, 0, 30, 0], [0, 0, 0, 90]]


def test_repeated_white_corrects_saturated_boardwalk():
    grid = mapper()
    xy = np.array([[0.1, 0.1]])
    for _ in range(30):
        grid.update(xy, np.array([4]))
    assert grid.render()[2][0, 0] == 90
    grid.update(xy, np.array([3]))
    assert grid.render()[2][0, 0] == 90
    for _ in range(15):
        grid.update(xy, np.array([3]))
    assert grid.render()[2][0, 0] == 30
    assert grid.render()[1][2, 0, 0] == 0


@pytest.mark.parametrize('obstacle_class', [2, 3, 4])
@pytest.mark.parametrize('road_class', [1, 5])
def test_blue_clears_observed_cell_and_obstacles_can_return(obstacle_class, road_class):
    grid = mapper()
    xy = np.array([[0.1, 0.1], [1.1, 0.1]])
    for _ in range(30):
        grid.update(xy, np.full(2, obstacle_class))
    old_cost = grid.render()[2][0, 0]
    grid.update(xy[:1], np.array([road_class]))
    assert grid.render()[2].tolist() == [[old_cost, old_cost]]
    for _ in range(4):
        grid.update(xy[:1], np.array([road_class]))
    _, layers, combined = grid.render()
    assert combined.tolist() == [[0, old_cost]]
    assert layers[:, 0, 0].tolist() == [0, 0, 0]
    for _ in range(30):
        grid.update(xy[:1], np.array([road_class]))
    for _ in range(5):
        grid.update(xy[:1], np.array([obstacle_class]))
    assert grid.render()[2][0, 0] == old_cost


def test_both_blue_classes_are_road_distinct_from_unknown():
    grid = mapper()
    grid.update(np.array([[0.1, 0.1], [1.1, 0.1]]), np.array([5, 1]))
    _, layers, combined = grid.render(Geometry(1.0, 3, 1, (0, 0, 0)))
    assert combined.tolist() == [[0, 0, -1]]
    assert layers[:, 0].tolist() == [[0, 0, -1]] * 3


def test_interior_boardwalk_contributes_to_boardwalk_evidence():
    interior, boardwalk = mapper(), mapper()
    xy = np.array([[0.1, 0.1]])
    interior.update(xy, np.array([6]))
    boardwalk.update(xy, np.array([4]))
    assert interior.render()[2].tolist() == [[90]]
    for one, canonical in zip(interior.tiles.values(), boardwalk.tiles.values()):
        np.testing.assert_allclose(one[0], canonical[0])


@pytest.mark.parametrize('labels', [[1, 5], [1, 5, 4]])
def test_blue_labels_share_evidence_without_mutating_input(labels):
    mixed, canonical = mapper(), mapper()
    labels = np.array(labels, dtype=np.uint8)
    original = labels.copy()
    xy = np.full((len(labels), 2), 0.1)
    mixed.update(xy, labels)
    canonical.update(xy, np.where(labels == 1, 5, labels))
    np.testing.assert_array_equal(labels, original)
    for one, many in zip(mixed.tiles.values(), canonical.tiles.values()):
        np.testing.assert_allclose(one[0], many[0])
    if len(labels) == 2:
        single = mapper()
        single.update(xy[:1], np.array([5]))
        for one, many in zip(single.tiles.values(), mixed.tiles.values()):
            np.testing.assert_allclose(one[0], many[0])
        assert mixed.render()[2].tolist() == [[0]]


def test_mixed_road_and_obstacle_evidence_has_no_density_bias():
    single, dense = mapper(), mapper()
    labels = np.array([4, 5])
    single.update(np.full((2, 2), 0.1), labels)
    dense.update(np.full((2000, 2), 0.1), np.tile(labels, 1000))
    for one, many in zip(single.tiles.values(), dense.tiles.values()):
        np.testing.assert_allclose(one[0], many[0])
    assert single.render()[2].tolist() == [[-1]]


def test_density_does_not_multiply_confidence():
    single, dense = mapper(), mapper()
    single.update(np.array([[0.1, 0.1]]), np.array([2]))
    dense.update(np.full((10000, 2), 0.1), np.full(10000, 2))
    for one, many in zip(single.tiles.values(), dense.tiles.values()):
        np.testing.assert_allclose(one[0], many[0])


def test_blue_clears_only_observed_endpoint_without_free_rays():
    grid = mapper()
    grid.update(np.array([[4.1, 0.1]]), np.array([2]))
    for _ in range(20):
        assert grid.update(np.array([[4.1, 0.1]]), np.array([1]))
    _, _, combined = grid.render(Geometry(1.0, 5, 1, (0, 0, 0)))
    assert combined.tolist() == [[-1, -1, -1, -1, 0]]


def test_negative_coordinates_use_floor():
    grid = mapper()
    grid.update(np.array([[-0.1, -0.1], [0.1, 0.1]]), np.array([2, 3]))
    geometry, _, combined = grid.render()
    assert geometry.origin == (-1.0, -1.0, 0.0)
    assert combined.tolist() == [[60, -1], [-1, 30]]


def test_submaps_move_independently_without_ghosts():
    grid = mapper()
    grid.set_pose((0, 1), (2, 0, 0))
    grid.update(np.array([[0.1, 0.1]]), np.array([2]), (0, 0))
    grid.update(np.array([[0.1, 0.1]]), np.array([4]), (0, 1))
    geometry = Geometry(1.0, 5, 2, (0, 0, 0))
    assert grid.render(geometry)[2][0].tolist() == [60, -1, 90, -1, -1]
    grid.set_pose((0, 0), (2, 0, np.pi / 2))
    grid.set_pose((0, 1), (4, 1, 0))
    assert grid.render(geometry)[2].tolist() == [
        [-1, 60, -1, -1, -1], [-1, -1, -1, -1, 90]]


def test_newer_overlapping_submap_can_replace_higher_cost():
    grid = mapper()
    grid.set_pose((0, 1), (0, 0, 0))
    grid.update(np.array([[0.1, 0.1]]), np.array([4]), (0, 0))
    grid.update(np.array([[0.1, 0.1]]), np.array([3]), (0, 1))
    assert grid.render()[2].tolist() == [[30]]
    grid.visible = {(0, 0)}
    assert grid.render()[2].tolist() == [[90]]


def test_rotated_reference_grid_and_crop():
    grid = mapper()
    grid.update(np.array([[-0.5, 0.5], [5.5, 0.5]]), np.array([2, 3]))
    geometry = Geometry(1.0, 1, 1, (0, 0, np.pi / 2))
    assert grid.render(geometry)[2].tolist() == [[60]]


def test_equal_class_evidence_stays_unknown():
    grid = mapper()
    for _ in range(20):
        grid.update(np.array([[0.1, 0.1], [0.2, 0.2]]), np.array([2, 4]))
    assert grid.render()[2].tolist() == [[-1]]


def test_memory_limit_rejects_entire_update():
    grid = SemanticGrid(resolution=1.0, max_cells=1024)
    grid.set_pose((0, 0), (0, 0, 0))
    grid.update(np.array([[0.1, 0.1]]), np.array([2]))
    with pytest.raises(MemoryError):
        grid.update(np.array([[0.1, 0.1], [40, 0]]), np.array([4, 4]))
    assert grid.render()[2].tolist() == [[60]]
    assert grid.sequence == 1


def cloud(endian='<', width=2, height=2):
    layout = [('x', 4, 7), ('y', 8, 7), ('z', 12, 7), ('class_id', 0, 2)]
    fields = [SimpleNamespace(name=name, offset=offset, datatype=datatype, count=1)
              for name, offset, datatype in layout]
    data = array('B', bytes(height * (width * 20 + 8)))
    dtype = np.dtype({'names': ['x', 'y', 'z', 'class_id'],
                      'formats': [endian + 'f4'] * 3 + ['u1'],
                      'offsets': [4, 8, 12, 0], 'itemsize': 20})
    points = np.ndarray((height, width), dtype=dtype, buffer=data,
                        strides=(width * 20 + 8, 20))
    points['x'], points['y'], points['z'], points['class_id'] = 1, 2, 0, 3
    return SimpleNamespace(fields=fields, data=data, width=width, height=height,
                           row_step=width * 20 + 8, point_step=20,
                           is_bigendian=endian == '>'), points


@pytest.mark.parametrize('endian', ['<', '>'])
def test_cloud_layout_and_invalid_points(endian):
    msg, points = cloud(endian)
    points['x'][0, 0] = np.nan
    points['class_id'][1, 0] = 1
    points['class_id'][1, 1] = 6
    xy, labels = read_class_cloud(msg)
    assert xy.tolist() == [[1, 2], [1, 2], [1, 2]]
    assert labels.tolist() == [3, 1, 6]


def test_cloud_rejects_bad_schema_and_truncated_buffer():
    msg, _ = cloud()
    msg.fields[-1].datatype = 7
    with pytest.raises(ValueError, match='field'):
        read_class_cloud(msg)
    msg, _ = cloud()
    msg.data = msg.data[:-1]
    with pytest.raises(ValueError, match='Truncated'):
        read_class_cloud(msg)


@pytest.mark.parametrize('old_class,new_class,cost', [
    (4, 3, 30), (4, 1, 0), (4, 5, 0), (3, 4, 90), (5, 4, 90),
])
def test_real_profile_revises_saturated_cells_in_two_observations(old_class, new_class, cost):
    from pathlib import Path
    import yaml

    profile = yaml.safe_load((Path(__file__).parents[1] / 'config' /
                              'mapping_real.yaml').read_text())['semantic_mapper']
    grid = SemanticGrid(
        hit=profile['hit_log_odds'], miss=profile['miss_log_odds'],
        limit=profile['log_odds_limit'], threshold=profile['min_evidence'])
    grid.set_pose((0, 0), (0.0, 0.0, 0.0))
    xy = np.array([[0.01, 0.01], [0.11, 0.01]])
    for _ in range(30):
        grid.update(xy, np.full(2, old_class))
    old_map = grid.render()[2].copy()
    grid.update(xy[:1], np.array([new_class]))
    np.testing.assert_array_equal(grid.render()[2], old_map)
    grid.update(xy[:1], np.array([new_class]))
    expected = old_map.copy()
    expected[0, 0] = cost
    np.testing.assert_array_equal(grid.render()[2], expected)
