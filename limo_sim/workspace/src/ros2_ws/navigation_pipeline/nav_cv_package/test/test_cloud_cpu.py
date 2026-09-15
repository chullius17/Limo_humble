"""Compare optimized geometry with the previous float32 projection/row keys."""

import numpy as np
import pytest

from nav_cv_package.cloud_cpu import RayCache, transform_xy, voxel_groups


def reference_projection(rows, cols, z, intrinsics, width, height, crop, r, t):
    fx, fy, cx, cy, iw, ih = intrinsics
    start = int(ih * crop)
    u = (cols.astype(np.float32) + 0.5) * iw / width - 0.5
    v = start + (rows.astype(np.float32) + 0.5) * (ih - start) / height - 0.5
    camera = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    result = camera.astype(np.float32) @ r[:2, :].T
    result += np.asarray(t, dtype=np.float32)
    return result


@pytest.mark.parametrize('crop', [0.0, 0.1, 0.5, 0.73])
@pytest.mark.parametrize('shape', [(320, 120), (640, 480), (1, 1)])
def test_cached_projection_matches_reference(crop, shape):
    rng = np.random.RandomState(42)
    width, height = shape
    intrinsics = (513.3, 514.2, 319.4, 241.2, 640, 480)
    rows = rng.randint(height, size=2700)
    cols = rng.randint(width, size=2700)
    z = rng.uniform(0.1, 5.0, size=2700).astype(np.float32)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation = rotation.astype(np.float32)
    translation = (0.24, -1.23)
    ray_x, ray_y = RayCache().get(intrinsics, width, height, crop)
    actual = transform_xy(ray_x[cols] * z, ray_y[rows] * z, z,
                          rotation, translation)
    expected = reference_projection(rows, cols, z, intrinsics,
                                    width, height, crop, rotation, translation)
    # Reordering float32 multiplication/division can change the last bits.
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    assert actual.dtype == np.float32


def test_ray_cache_reuses_and_invalidates_all_inputs():
    cache = RayCache()
    base = (500.0, 501.0, 319.0, 241.0, 640, 480)
    previous = cache.get(base, 320, 120, 0.5)
    assert cache.get(base, 320, 120, 0.5)[0] is previous[0]
    cases = [(base, 321, 120, 0.5), (base, 321, 121, 0.5),
             (base, 321, 121, 0.25)]
    for i in range(6):
        changed = list(base)
        changed[i] += 1
        cases.append((tuple(changed), 320, 120, 0.5))
    for args in cases:
        current = cache.get(*args)
        assert current[0] is not previous[0]
        fresh = RayCache().get(*args)
        np.testing.assert_array_equal(current[0], fresh[0])
        np.testing.assert_array_equal(current[1], fresh[1])
        previous = current


@pytest.mark.parametrize('size', [0.01, 0.02, 0.1])
def test_integer_voxel_keys_match_row_keys(size):
    rng = np.random.RandomState(17)
    points = rng.uniform(-2.0, 2.0, (3000, 2)).astype(np.float32)
    points[100:200] = points[:100]
    labels = rng.randint(1, 5, len(points)).astype(np.uint8)
    labels[100:200] = labels[:100]
    cells = np.floor(points.astype(np.float64) / size).astype(np.int64)
    _, expected_first, expected_inverse = np.unique(
        np.column_stack((labels, cells)), axis=0,
        return_index=True, return_inverse=True)
    first, inverse = voxel_groups(points, labels, size)
    np.testing.assert_array_equal(first, expected_first)
    np.testing.assert_array_equal(inverse, expected_inverse)


@pytest.mark.parametrize('extent', [1e8, 1e20])
def test_sparse_large_voxel_ranges_do_not_overflow(extent):
    points = np.array([[-extent, -extent], [extent, extent],
                       [-extent, -extent], [extent, extent]])
    labels = np.array([2, 3, 2, 4], dtype=np.uint8)
    first, inverse = voxel_groups(points, labels, 0.01)
    assert len(first) == 3
    assert inverse[0] == inverse[2]
    assert inverse[1] != inverse[3]


def test_empty_direct_transform():
    empty = np.empty(0, dtype=np.float32)
    assert transform_xy(empty, empty, empty, np.eye(3), (0, 0)).shape == (0, 2)
