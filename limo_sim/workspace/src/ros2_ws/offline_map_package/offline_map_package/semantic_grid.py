"""Sparse, revisable semantic evidence; independent of ROS and image libraries."""

from dataclasses import dataclass
import math

import numpy as np


CLASS_IDS = np.array([2, 3, 4], dtype=np.uint8)
CLASS_NAMES = ('turquoise', 'white', 'boardwalk')
DEFAULT_COSTS = (60, 30, 90)
TILE_SIZE = 32


def transform_xy(points, pose, inverse=False):
    """Apply an SE(2) pose to an N x 2 array of metric points."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    if inverse:
        return (points - (x, y)) @ rotation
    return points @ rotation.T + (x, y)


def read_class_cloud(msg):
    """Read XYZ/class_id with offsets, row padding and endian from PointCloud2.

    Only the fields used by visual_ptcld are required. Blue, invalid labels,
    nonfinite points and empty clouds contribute no evidence, including misses.
    """
    fields = {field.name: field for field in msg.fields}
    required = ('x', 'y', 'z', 'class_id')
    if any(name not in fields for name in required):
        raise ValueError('PointCloud2 requires x, y, z and class_id fields')
    endian = '>' if msg.is_bigendian else '<'
    formats = []
    for name in required:
        field = fields[name]
        expected = 2 if name == 'class_id' else 7  # UINT8 / FLOAT32
        size = 1 if name == 'class_id' else 4
        if (field.datatype != expected or field.count != 1
                or field.offset < 0 or field.offset + size > msg.point_step):
            raise ValueError('Invalid PointCloud2 field: ' + name)
        formats.append('u1' if name == 'class_id' else endian + 'f4')
    if msg.row_step < msg.width * msg.point_step:
        raise ValueError('PointCloud2 row_step is too small')
    if len(msg.data) < msg.height * msg.row_step:
        raise ValueError('Truncated PointCloud2 buffer')
    if not msg.width or not msg.height:
        return np.empty((0, 2)), np.empty(0, dtype=np.uint8)
    dtype = np.dtype({
        'names': required, 'formats': formats,
        'offsets': [fields[name].offset for name in required],
        'itemsize': msg.point_step,
    })
    points = np.ndarray(
        (msg.height, msg.width), dtype=dtype, buffer=msg.data,
        strides=(msg.row_step, msg.point_step))
    valid = (np.isin(points['class_id'], CLASS_IDS)
             & np.isfinite(points['x']) & np.isfinite(points['y'])
             & np.isfinite(points['z']))
    return (np.column_stack((points['x'][valid], points['y'][valid])),
            points['class_id'][valid])


@dataclass(frozen=True)
class Geometry:
    """Shared OccupancyGrid geometry, including a potentially rotated origin."""

    resolution: float
    width: int
    height: int
    origin: tuple

    def indices(self, points):
        local = transform_xy(points, self.origin, inverse=True)
        cells = np.floor(local / self.resolution).astype(np.int64)
        valid = ((cells[:, 0] >= 0) & (cells[:, 0] < self.width)
                 & (cells[:, 1] >= 0) & (cells[:, 1] < self.height))
        return cells[:, 1] * self.width + cells[:, 0], valid


class SemanticGrid:
    """Sparse tiles in local submap frames, projected only for publication.

    Each cell receives at most one unit of evidence per cloud, split between
    observed classes. A class observation is a miss for incompatible classes
    *at that endpoint*, never along a ray or throughout the camera FOV.
    Costs remain fixed; evidence selects the class rather than scaling its cost.
    """

    def __init__(self, resolution=0.05, costs=DEFAULT_COSTS, hit=0.85,
                 miss=0.4, limit=3.0, threshold=0.5,
                 max_cells=2000000, max_output_cells=4000000):
        values = [resolution, hit, miss, limit, threshold]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Resolution and evidence parameters must be positive')
        if threshold > limit:
            raise ValueError('Evidence threshold must not exceed saturation')
        if (len(costs) != 3 or any(int(c) != c or c < 1 or c > 100 for c in costs)):
            raise ValueError('Three integer semantic costs in [1, 100] required')
        if max_cells < TILE_SIZE ** 2 or max_output_cells < 1:
            raise ValueError('Invalid map memory limits')
        self.resolution = resolution
        self.costs = np.asarray(costs, dtype=np.int8)
        self.hit, self.miss, self.limit, self.threshold = hit, miss, limit, threshold
        self.max_cells, self.max_output_cells = max_cells, max_output_cells
        self.tiles = {}
        self.poses = {}
        self.visible = None
        self.sequence = 0
        self.allocated_cells = 0

    def set_pose(self, key, pose):
        """Move an existing semantic submap without resampling its evidence."""
        if len(pose) != 3 or not np.isfinite(pose).all():
            raise ValueError('Invalid submap pose')
        changed = self.poses.get(key) != tuple(pose)
        self.poses[key] = tuple(pose)
        return changed

    def update(self, points, labels, key=(0, 0)):
        """Integrate one cloud in the selected submap's local coordinates."""
        if key not in self.poses:
            raise ValueError('Submap pose not available')
        valid = np.isfinite(points).all(axis=1) & np.isin(labels, CLASS_IDS)
        points, labels = points[valid], labels[valid]
        if not len(points):
            return False
        cells = np.floor(points / self.resolution).astype(np.int64)
        cells, inverse = np.unique(cells, axis=0, return_inverse=True)
        counts = np.stack([
            np.bincount(inverse, weights=labels == label, minlength=len(cells))
            for label in CLASS_IDS], axis=1)
        fractions = counts / counts.sum(axis=1, keepdims=True)
        updates = self.hit * fractions - self.miss * (1.0 - fractions)
        tile_xy, tile_inverse = np.unique(
            cells // TILE_SIZE, axis=0, return_inverse=True)
        keys = [(key, int(x), int(y)) for x, y in tile_xy]
        needed = sum(tile_key not in self.tiles for tile_key in keys)
        if self.allocated_cells + needed * TILE_SIZE ** 2 > self.max_cells:
            raise MemoryError('Semantic tile limit reached; save/reset or increase max_cells')
        if self.sequence >= np.iinfo(np.uint32).max:
            raise OverflowError('Observation sequence exhausted; save and reset')
        self.sequence += 1
        for index, tile_key in enumerate(keys):
            if tile_key not in self.tiles:
                self.tiles[tile_key] = (
                    np.zeros((TILE_SIZE ** 2, 3), dtype=np.float32),
                    np.zeros(TILE_SIZE ** 2, dtype=np.uint32))
                self.allocated_cells += TILE_SIZE ** 2
            scores, seen = self.tiles[tile_key]
            selected = tile_inverse == index
            local = cells[selected] % TILE_SIZE
            flat = local[:, 1] * TILE_SIZE + local[:, 0]
            scores[flat] = np.clip(scores[flat] + updates[selected], -self.limit, self.limit)
            seen[flat] = self.sequence
        return True

    def render(self, geometry=None):
        """Rasterize submaps into three cost layers plus a semantic composite.

        At overlapping submap cells the latest observation wins. Thus an old
        submap cannot indefinitely resurrect evidence contradicted on a revisit.
        Within one observation, the more confident cell wins resampling ties.
        """
        positions, evidence, sequences = [], [], []
        for (key, tx, ty), (scores, seen) in self.tiles.items():
            if self.visible is not None and key not in self.visible:
                continue
            indices = np.flatnonzero(seen)
            if not len(indices):
                continue
            points = np.column_stack((
                tx * TILE_SIZE + indices % TILE_SIZE + 0.5,
                ty * TILE_SIZE + indices // TILE_SIZE + 0.5)) * self.resolution
            positions.append(transform_xy(points, self.poses[key]))
            evidence.append(scores[indices])
            sequences.append(seen[indices])
        if not positions:
            if geometry is None:
                return None
            world = np.empty((0, 2))
            scores, seen = np.empty((0, 3)), np.empty(0, dtype=np.uint32)
        else:
            world = np.concatenate(positions)
            scores = np.concatenate(evidence)
            seen = np.concatenate(sequences)
        if geometry is None:
            low = np.floor(world.min(axis=0) / self.resolution)
            high = np.floor(world.max(axis=0) / self.resolution)
            width, height = (high - low + 1).astype(int)
            geometry = Geometry(self.resolution, int(width), int(height),
                                (low[0] * self.resolution, low[1] * self.resolution, 0.0))
        if (not math.isfinite(geometry.resolution) or geometry.resolution <= 0
                or not np.isfinite(geometry.origin).all()
                or geometry.width < 1 or geometry.height < 1):
            raise ValueError('Invalid reference map geometry')
        size = geometry.width * geometry.height
        if size > self.max_output_cells:
            raise MemoryError('Output grid limit reached; use a larger resolution or smaller map')
        layers = np.full((3, size), -1, dtype=np.int8)
        combined = np.full(size, -1, dtype=np.int8)
        indices, valid = geometry.indices(world)
        indices, scores, seen = indices[valid], scores[valid], seen[valid]
        if len(indices):
            confidence = scores.max(axis=1)
            order = np.lexsort((confidence, seen, indices))
            ordered_indices = indices[order]
            last = np.r_[ordered_indices[1:] != ordered_indices[:-1], True]
            winners = order[last]
            classes = scores[winners].argmax(axis=1)
            best = scores[winners].max(axis=1)
            # Equal class scores remain unknown instead of favoring a label by order.
            unique_best = (scores[winners] == best[:, None]).sum(axis=1) == 1
            certain = (best >= self.threshold) & unique_best
            target, classes = indices[winners][certain], classes[certain]
            layers[:, target] = 0
            layers[classes, target] = self.costs[classes]
            combined[target] = self.costs[classes]
        shape = (geometry.height, geometry.width)
        return geometry, layers.reshape((3,) + shape), combined.reshape(shape)
