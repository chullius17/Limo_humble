"""CPU geometry helpers, independent of ROS for numerical verification."""

import numpy as np


class RayCache:
    """Cache normalized pixel rays until calibration, crop or shape changes."""

    def __init__(self):
        self.key = None

    def get(self, intrinsics, width, height, crop_y_min):
        key = (tuple(intrinsics), width, height, crop_y_min)
        if key != self.key:
            fx, fy, cx, cy, info_width, info_height = intrinsics
            crop_start = int(info_height * crop_y_min)
            cols = ((np.arange(width, dtype=np.float32) + 0.5)
                    * info_width / width - 0.5)
            rows = (crop_start + (np.arange(height, dtype=np.float32) + 0.5)
                    * (info_height - crop_start) / height - 0.5)
            self.x = (cols - cx) / fx
            self.y = (rows - cy) / fy
            self.key = key
        return self.x, self.y


def transform_xy(x, y, z, rotation, translation):
    """Transform directly into BEV XY, without an Nx3 matrix or BLAS call."""
    result = np.empty((len(z), 2), dtype=np.float32)
    for axis in range(2):
        result[:, axis] = (
            x * rotation[axis, 0] + y * rotation[axis, 1]
            + z * rotation[axis, 2] + np.float32(translation[axis]))
    return result


def voxel_groups(points, labels, size):
    """Return first indices and group IDs using collision-free integer keys.

    Translate negative cells to zero and pack class/X/Y into int64. Sparse
    extents never allocate a dense grid. Use row keys for extreme ranges that
    cannot be packed, preventing integer overflow or accidental class merging.
    """
    cells = np.floor(points.astype(np.float64) / size)
    if np.isfinite(cells).all() and np.all(np.abs(cells) < 2**62):
        cells = cells.astype(np.int64)
        minimum = cells.min(axis=0)
        maximum = cells.max(axis=0)
        span_x = int(maximum[0]) - int(minimum[0]) + 1
        span_y = int(maximum[1]) - int(minimum[1]) + 1
        if (int(labels.max()) + 1) * span_x * span_y <= 2**63:
            cells -= minimum
            keys = ((labels.astype(np.int64) * span_x + cells[:, 0])
                    * span_y + cells[:, 1])
            _, first, inverse = np.unique(
                keys, return_index=True, return_inverse=True)
            return first, inverse
    keys = np.column_stack((labels, cells))
    _, first, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True)
    return first, inverse
