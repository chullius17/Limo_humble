"""Compare real CUDA classification with OpenCV on the same BEV lattice."""

import numpy as np
import pytest

from cv_package.boardwalk import BoardwalkClassifier


@pytest.fixture(scope='module')
def gpu():
    cuda = pytest.importorskip('pycuda.driver')
    cuda.init()
    if cuda.Device.count() == 0:
        pytest.skip('No CUDA device')
    classifier = BoardwalkClassifier(backend='cuda')
    yield classifier
    classifier.close()


def compare(gpu, points, labels, minimum=0.1, maximum=0.16, radius=0.1, debug=False):
    cpu = BoardwalkClassifier(resolution=gpu.resolution)
    expected = labels.copy()
    cpu_stats = cpu.classify(points, expected, minimum, maximum, radius, debug=debug)
    actual = labels.copy()
    gpu_stats = gpu.classify(points, actual, minimum, maximum, radius, debug=debug)
    np.testing.assert_array_equal(actual, expected)
    for key in ('boardwalk_eligible_count', 'boardwalk_seed_count',
                'boardwalk_propagated_count', 'boardwalk_final_count',
                'boardwalk_outside_count'):
        assert gpu_stats[key] == cpu_stats[key]
    assert gpu_stats['boardwalk_gpu_used'] == 1
    assert gpu.gpu_error is None
    if debug:
        np.testing.assert_array_equal(gpu.debug_image, cpu.debug_image)


@pytest.mark.parametrize('seed', range(16))
def test_cuda_matches_opencv_across_frames_and_grid_sizes(gpu, seed):
    rng = np.random.RandomState(seed)
    size = 80 + seed * 55
    points = rng.uniform(-1.5, 1.5, (size, 2))
    labels = rng.choice([0, 1, 2, 3, 4], size).astype(np.uint8)
    compare(gpu, points, labels, debug=(seed == 0))


@pytest.mark.parametrize('thresholds', [
    (0.5, 1.0, 0.5), (0.5, 1.0, 0.25), (0.5, 1.0, 0.0),
    (0.5, 0.5, 0.5), (0.0, 0.0, 0.0), (0.1, 0.16, 0.1),
])
def test_cuda_threshold_edges_and_no_recursive_propagation(gpu, thresholds):
    points = np.column_stack(([0, 0.5, 0.75, 1.0, 1.125, 0.375, 0.125], np.zeros(7)))
    labels = np.array([1, 3, 3, 3, 3, 3, 3], dtype=np.uint8)
    compare(gpu, points, labels, *thresholds)


@pytest.mark.parametrize('vertical', [True, False])
def test_cuda_thin_grids_and_duplicate_cell_observations(gpu, vertical):
    points = np.array([[0, 0], [0.001, 0], [0.125, 0], [0.126, 0], [0.4, 0]])
    if vertical:
        points = points[:, ::-1]
    labels = np.array([1, 3, 3, 3, 3], dtype=np.uint8)
    compare(gpu, points, labels)


def test_cuda_context_is_balanced_when_used_by_a_worker_thread(gpu):
    from concurrent.futures import ThreadPoolExecutor
    import pycuda.driver as cuda

    def classify():
        assert cuda.Context.get_current() is None
        compare(gpu, np.array([[0, 0], [0.125, 0]]), np.array([1, 3], dtype=np.uint8))
        assert cuda.Context.get_current() is None

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(classify).result()


def test_cuda_radius_limit_is_explicit():
    classifier = BoardwalkClassifier(backend='cuda')
    try:
        with pytest.raises(ValueError, match='radius exceeds'):
            classifier.classify(np.array([[0, 0], [3, 3]]),
                                np.array([1, 3], dtype=np.uint8), 0.1, 2.0, 0.1)
        with pytest.raises(RuntimeError, match='previously failed'):
            classifier.classify(np.array([[0, 0], [0.125, 0]]),
                                np.array([1, 3], dtype=np.uint8), 0.1, 0.16, 0.1)
    finally:
        classifier.close()
