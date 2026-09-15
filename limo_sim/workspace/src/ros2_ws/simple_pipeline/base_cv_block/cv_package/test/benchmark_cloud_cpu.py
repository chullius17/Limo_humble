"""Compare CPU kernels on synthetic data, without subscribing to ROS topics.

Run from the cv_package directory with its source on PYTHONPATH:
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python3 test/benchmark_cloud_cpu.py
Requires the test dependencies; results are not end-to-end ROS timings.
"""

import time

import numpy as np

from cv_package.cloud_cpu import RayCache, transform_xy, voxel_groups
from test_cloud_cpu import reference_projection


def measure(function):
    for _ in range(10):
        function()
    samples = []
    for _ in range(100):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000.0)
    return np.median(samples), np.percentile(samples, 95)


def main():
    rng = np.random.RandomState(42)
    points = rng.uniform(-0.5, 0.5, (1700, 2)).astype(np.float32)
    labels = rng.randint(2, 5, len(points)).astype(np.uint8)

    def old_groups():
        cells = np.floor(points.astype(np.float64) / 0.02).astype(np.int64)
        _, first, inverse = np.unique(
            np.column_stack((labels.astype(np.int64), cells)), axis=0,
            return_index=True, return_inverse=True)
        return first, inverse

    def new_groups():
        return voxel_groups(points, labels, 0.02)

    for before, after in zip(old_groups(), new_groups()):
        np.testing.assert_array_equal(before, after)
    rows = rng.randint(120, size=2700)
    cols = rng.randint(320, size=2700)
    depth = rng.uniform(0.1, 5.0, 2700).astype(np.float32)
    intrinsics = (513.3, 514.2, 319.4, 241.2, 640, 480)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation = rotation.astype(np.float32)
    translation = (0.24, -1.23)
    cache = RayCache()

    def projected():
        x, y = cache.get(intrinsics, 320, 120, 0.5)
        return transform_xy(x[cols] * depth, y[rows] * depth, depth,
                            rotation, translation)

    def reference():
        return reference_projection(rows, cols, depth, intrinsics,
                                    320, 120, 0.5, rotation, translation)

    np.testing.assert_allclose(projected(), reference(), rtol=2e-6, atol=2e-6)
    print('Synthetic: 2700 projected points; 1700 non-blue voxel inputs')
    for title, function in [('voxel grouping before', old_groups),
                            ('voxel grouping after', new_groups),
                            ('projection before', reference),
                            ('projection after (warm cache)', projected)]:
        median, p95 = measure(function)
        print('{}: median={:.3f} ms p95={:.3f} ms'.format(title, median, p95))


if __name__ == '__main__':
    main()
