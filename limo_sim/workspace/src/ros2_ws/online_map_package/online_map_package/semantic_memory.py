"""Maintain semantic points observed in the band between two trapezoids."""

import math

import numpy as np


def transform_xy(points, pose, inverse=False):
    """Apply a planar odom-from-base pose, or its inverse."""
    x, y, yaw = pose
    rotation = np.array([
        [math.cos(yaw), -math.sin(yaw)],
        [math.sin(yaw), math.cos(yaw)],
    ])
    if inverse:
        return (points - [x, y]) @ rotation
    return points @ rotation.T + [x, y]


class SemanticMemory:
    """Store selected observations in odom and reproject them into base."""

    def __init__(self, length, width, height, near_width, inner_inset=0.2,
                 maximum_points=300,
                 minimum_confidence=0.3, confidence_decay_per_sec=0.1,
                 yellow_decay_multiplier=3.0, voxel_size=0.03):
        self.length = length
        self.width = width
        self.height = height
        self.near_width = near_width
        self.inner_inset = inner_inset
        self.maximum_points = maximum_points
        self.minimum_confidence = minimum_confidence
        self.decay = confidence_decay_per_sec
        self.yellow_decay_multiplier = yellow_decay_multiplier
        self.voxel_size = voxel_size
        self.reset()

    def reset(self):
        """Discard the persistent cloud after a simulation clock reset."""
        self.points = np.empty((0, 2), dtype=np.float64)
        self.classes = np.empty(0, dtype=np.uint8)
        self.confidences = np.empty(0, dtype=np.float64)
        self.last_update = np.empty(0, dtype=np.float64)
        self.last_observation_stamp = None

    def inside_rectangle(self, xy):
        return ((xy[:, 0] >= 0.0) & (xy[:, 0] <= self.length)
                & (np.abs(xy[:, 1]) <= self.width / 2.0))

    def inside_trapezoid(self, xy):
        distance = xy[:, 0] - (self.length - self.height)
        width = self.near_width + distance / self.height * (
            self.width - self.near_width)
        return ((distance >= 0.0) & (distance <= self.height)
                & (np.abs(xy[:, 1]) <= width / 2.0 + 1e-9))

    def inside_inner_trapezoid(self, xy):
        """Test the inset trapezoid whose front edge stays on the yellow one."""
        yellow_near_half = self.near_width / 2.0
        slope = (self.width / 2.0 - yellow_near_half) / self.height
        lateral_shift = self.inner_inset * math.hypot(1.0, slope)
        near_half = (
            yellow_near_half + slope * self.inner_inset - lateral_shift)
        far_half = self.width / 2.0 - lateral_shift
        inner_height = self.height - self.inner_inset
        distance = xy[:, 0] - (
            self.length - self.height + self.inner_inset)
        width = 2.0 * (
            near_half + distance / inner_height * (far_half - near_half))
        return ((distance >= 0.0) & (distance <= inner_height)
                & (np.abs(xy[:, 1]) <= width / 2.0 + 1e-9))

    def observe(self, xy, classes, pose, stamp):
        """Immediately admit class 2/4 points in the yellow-red band."""
        if (self.last_observation_stamp is not None
                and stamp <= self.last_observation_stamp):
            return
        self.last_observation_stamp = stamp
        selected = (np.isin(classes, [2, 4])
                    & np.isfinite(xy).all(axis=1)
                    & self.inside_trapezoid(xy)
                    & ~self.inside_inner_trapezoid(xy))
        selected_count = int(np.count_nonzero(selected))
        self.points = np.concatenate((
            self.points, transform_xy(xy[selected], pose)))
        self.classes = np.concatenate((
            self.classes, classes[selected].astype(np.uint8)))
        self.confidences = np.concatenate((
            self.confidences, np.ones(selected_count)))
        self.last_update = np.concatenate((
            self.last_update, np.full(selected_count, stamp)))
        self._reduce()

    def _keep(self, indices):
        self.points = self.points[indices]
        self.classes = self.classes[indices]
        self.confidences = self.confidences[indices]
        self.last_update = self.last_update[indices]

    def _reduce(self):
        if not len(self.points):
            return
        # Deduplicate per class in odom; the newest evidence wins each voxel.
        keys = np.column_stack((
            np.floor(self.points / self.voxel_size).astype(np.int64),
            self.classes,
        ))
        newest = np.argsort(-self.last_update, kind='stable')
        _, first = np.unique(keys[newest], axis=0, return_index=True)
        self._keep(newest[first])
        if len(self.points) > self.maximum_points:
            # Always discard lower-confidence evidence first. If the cutoff
            # contains ties (normally fresh observations at confidence 1),
            # sample those spatially instead of keeping an arbitrary prefix.
            cutoff = np.sort(self.confidences)[-self.maximum_points]
            preferred = np.flatnonzero(self.confidences > cutoff)
            tied = np.flatnonzero(self.confidences == cutoff)
            needed = self.maximum_points - len(preferred)
            tie_order = np.lexsort((
                self.points[tied, 1],
                self.points[tied, 0],
                self.classes[tied],
            ))
            positions = np.linspace(
                0, len(tie_order) - 1, needed, dtype=int)
            self._keep(np.concatenate((preferred, tied[tie_order[positions]])))

    def prune(self, pose, stamp, motion_ratio=1.0):
        """Reproject survivors and delete expired, red or out-of-bounds points."""
        xy = transform_xy(self.points, pose, inverse=True)
        elapsed = np.maximum(0.0, stamp - self.last_update)
        decay_factor = np.where(
            self.inside_trapezoid(xy), self.yellow_decay_multiplier, 1.0)
        motion_ratio = float(np.clip(motion_ratio, 0.0, 1.0))
        self.confidences *= np.exp(
            -self.decay * motion_ratio * decay_factor * elapsed)
        self.last_update = np.maximum(self.last_update, stamp)
        keep = (self.inside_rectangle(xy) & ~self.inside_inner_trapezoid(xy)
                & (self.confidences >= self.minimum_confidence))
        self._keep(keep)
        return xy[keep], self.classes.copy(), self.confidences.copy()
