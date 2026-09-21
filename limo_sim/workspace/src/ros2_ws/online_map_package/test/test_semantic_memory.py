"""Exercise band admission, motion compensation and deletion."""

import math

import numpy as np

from online_map_package.semantic_memory import SemanticMemory, transform_xy


POSE = (0.0, 0.0, 0.0)


def memory(**kwargs):
    return SemanticMemory(2.5, 2.66, 1.95, 0.6, 0.2, **kwargs)


def test_observations_spawn_immediately_only_in_yellow_red_band():
    state = memory()
    xy = np.array([
        [0.60, 0.0],   # Longitudinal band near the robot.
        [1.00, 0.45],  # Lateral band between sloping sides.
        [1.00, 0.0],   # Green trapezoid.
        [0.20, 0.0],   # Outside yellow trapezoid.
        [0.60, 0.1],   # Unsupported class.
    ])
    state.observe(xy, np.array([2, 4, 2, 4, 6]), POSE, 1.0)
    np.testing.assert_allclose(state.points, [[0.6, 0.0], [1.0, 0.45]])
    np.testing.assert_array_equal(state.classes, [2, 4])
    np.testing.assert_allclose(state.confidences, [1.0, 1.0])
    np.testing.assert_allclose(state.last_update, [1.0, 1.0])


def test_red_and_yellow_geometry_have_expected_membership():
    state = memory()
    points = np.array([
        [0.55, 0.0], [0.60, 0.0], [0.75, 0.0],
        [1.0, 0.0], [2.30, 0.0], [2.40, 0.0], [2.50, 0.0],
    ])
    np.testing.assert_array_equal(
        state.inside_trapezoid(points),
        [True, True, True, True, True, True, True])
    np.testing.assert_array_equal(
        state.inside_inner_trapezoid(points),
        [False, False, True, True, True, True, True])


def test_translation_rotation_and_inverse_reproject_every_time():
    points = np.array([[0.2, 0.3], [1.2, -0.4]])
    pose = (1.1, -0.7, math.pi / 2)
    odom = transform_xy(points, pose)
    np.testing.assert_allclose(odom, [[0.8, -0.5], [1.5, 0.5]])
    np.testing.assert_allclose(transform_xy(odom, pose, True), points)

    state = memory()
    state.observe(np.array([[0.6, 0.0]]), np.array([2]), POSE, 1.0)
    xy, classes, _ = state.prune((0.1, 0.0, 0.0), 1.1)
    np.testing.assert_allclose(xy, [[0.5, 0.0]])
    np.testing.assert_array_equal(classes, [2])
    rotated = (0.0, 0.0, math.pi / 4)
    xy, _, _ = state.prune(rotated, 1.2)
    coordinate = 0.6 / math.sqrt(2)
    np.testing.assert_allclose(xy, [[coordinate, -coordinate]])


def spawned(**kwargs):
    state = memory(**kwargs)
    state.observe(np.array([[0.6, 0.0]]), np.array([2]), POSE, 1.0)
    return state


def test_confidence_uses_standard_decay_in_red_and_higher_decay_in_yellow():
    state = spawned()
    _, _, confidence = state.prune(POSE, 2.0)
    np.testing.assert_allclose(confidence, [math.exp(-0.3)])
    _, _, confidence = state.prune((0.1, 0.0, 0.0), 3.0)
    np.testing.assert_allclose(confidence, [math.exp(-0.4)])
    state.prune((0.1, 0.0, 0.0), 14.0)
    assert len(state.points) == 0


def test_motion_ratio_scales_decay_and_stationary_points_do_not_decay():
    state = spawned()
    _, _, confidence = state.prune(POSE, 2.0, motion_ratio=0.0)
    np.testing.assert_allclose(confidence, [1.0])
    _, _, confidence = state.prune(POSE, 4.0, motion_ratio=0.5)
    np.testing.assert_allclose(confidence, [math.exp(-0.3)])


def test_reentry_into_red_and_leaving_rectangle_delete_points():
    state = spawned()
    state.prune((-0.4, 0.0, 0.0), 1.2)
    assert len(state.points) == 0
    state = spawned()
    state.prune((1.0, 0.0, 0.0), 1.2)
    assert len(state.points) == 0


def test_limit_and_voxels_preserve_class_identity():
    state = memory(maximum_points=17, voxel_size=0.001)
    xy = np.column_stack((np.full(500, 0.6), np.linspace(-0.2, 0.2, 500)))
    labels = np.tile([2, 4], 250)
    state.observe(xy, labels, POSE, 1.0)
    assert len(state.points) == 17
    assert set(state.classes) == {2, 4}

    state = memory()
    state.observe(np.array([[0.6, 0.0]] * 3), np.array([2, 2, 4]), POSE, 1.0)
    assert len(state.points) == 2
    assert set(state.classes) == {2, 4}


def test_limit_discards_lowest_confidence_points_first():
    state = memory(maximum_points=3, voxel_size=0.001)
    state.points = np.column_stack((
        np.linspace(0.6, 1.0, 5), np.linspace(0.3, 0.5, 5)))
    state.classes = np.array([2, 4, 2, 4, 2], dtype=np.uint8)
    state.confidences = np.array([0.1, 0.9, 0.4, 0.8, 0.7])
    state.last_update = np.arange(5, dtype=np.float64)

    state._reduce()

    assert len(state.points) == 3
    np.testing.assert_allclose(
        np.sort(state.confidences), [0.7, 0.8, 0.9])


def test_new_observation_refreshes_voxel_but_stale_frame_is_ignored():
    state = spawned()
    state.observe(np.array([[0.6, 0.0]]), np.array([2]), POSE, 2.0)
    assert len(state.points) == 1
    np.testing.assert_allclose(state.confidences, [1.0])
    np.testing.assert_allclose(state.last_update, [2.0])
    state.observe(np.array([[0.65, 0.0]]), np.array([4]), POSE, 1.5)
    assert len(state.points) == 1
    np.testing.assert_array_equal(state.classes, [2])
    state.reset()
    assert len(state.points) == 0
    assert state.last_observation_stamp is None
