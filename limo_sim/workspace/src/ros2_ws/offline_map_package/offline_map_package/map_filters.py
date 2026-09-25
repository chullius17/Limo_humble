"""Filters for saved semantic snapshots, independent of ROS."""

import numpy as np


def median_saved_semantic(semantic, kernel):
    """Filter observed costs without changing the input or filling unknown cells.

    Unknown neighbors participate as -1, so isolated observations can revert
    to unknown, but never-observed cells cannot become free or occupied.
    Laser data must be overlaid afterwards, without filtering it.
    """
    if (isinstance(kernel, bool) or not isinstance(kernel, int)
            or kernel < 1 or kernel % 2 == 0):
        raise ValueError('save_median_kernel must be a positive odd integer')
    if semantic.ndim != 2:
        raise ValueError('Semantic map must be a two-dimensional array')
    filtered = semantic.copy()
    if kernel == 1 or not semantic.size:
        return filtered
    radius = kernel // 2
    padded = np.pad(semantic, radius, mode='edge')
    height, width = semantic.shape
    # Limit temporary storage on the Jetson instead of stacking a full map
    # for every kernel offset. Each block still reads the original snapshot.
    for start in range(0, height, 128):
        stop = min(start + 128, height)
        neighborhoods = np.stack([
            padded[start + y:stop + y, x:x + width]
            for y in range(kernel) for x in range(kernel)
        ])
        middle = kernel * kernel // 2
        neighborhoods.partition(middle, axis=0)
        filtered[start:stop] = neighborhoods[middle]
    filtered[semantic < 0] = -1
    return filtered
