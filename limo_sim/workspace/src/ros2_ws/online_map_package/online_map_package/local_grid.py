# Copyright 2026 Giulio Cataldo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build the inflated semantic OccupancyGrid used by local_map_final."""

import array
import math

from nav_msgs.msg import OccupancyGrid
import numpy as np


class LocalGrid:
    """Rasterize semantic points and serialize the local OccupancyGrid."""

    def __init__(self, length, width, resolution, inflation_radius,
                 yellow_line_cost, soft_obstacle_cost, boardwalk_cost):
        self.length = float(length)
        self.metric_width = float(width)
        self.resolution = float(resolution)
        self.inflation_radius = float(inflation_radius)
        self.width = int(round(self.length / self.resolution))
        self.height = int(round(self.metric_width / self.resolution))
        if (not math.isclose(
                self.width * self.resolution, self.length, abs_tol=1e-9)
                or not math.isclose(
                    self.height * self.resolution,
                    self.metric_width, abs_tol=1e-9)):
            raise ValueError(
                'grid_resolution must divide rectangle dimensions exactly')
        self.origin_y = -0.5 * self.metric_width
        self.class_costs = np.array([
            0, 0, yellow_line_cost, soft_obstacle_cost,
            boardwalk_cost, 0, boardwalk_cost,
        ], dtype=np.int16)
        radius_cells = int(math.ceil(
            self.inflation_radius / self.resolution))
        self.inflation_offsets = [
            (dy, dx)
            for dy in range(-radius_cells, radius_cells + 1)
            for dx in range(-radius_cells, radius_cells + 1)
            if math.hypot(dx, dy) * self.resolution
            <= self.inflation_radius + 1e-12
        ]

    def rasterize(self, points, classes):
        """Rasterize all supported classes, retaining the maximum per cell."""
        grid = np.zeros((self.height, self.width), dtype=np.int16)
        if not len(points):
            return grid
        valid = (
            np.isfinite(points).all(axis=1)
            & np.isin(classes, [1, 2, 3, 4, 5, 6])
            & (points[:, 0] >= 0.0)
            & (points[:, 0] < self.length)
            & (points[:, 1] >= self.origin_y)
            & (points[:, 1] < -self.origin_y)
        )
        if not np.any(valid):
            return grid
        selected = points[valid]
        selected_classes = classes[valid]
        cell_x = np.floor(
            selected[:, 0] / self.resolution).astype(np.int64)
        cell_y = np.floor(
            (selected[:, 1] - self.origin_y) /
            self.resolution).astype(np.int64)
        flat_indices = cell_y * self.width + cell_x
        np.maximum.at(
            grid.ravel(), flat_indices, self.class_costs[selected_classes])
        return self._inflate(grid)

    def _inflate(self, grid):
        """Dilate nonzero semantic costs within the configured radius."""
        if self.inflation_radius <= 0.0 or not np.any(grid):
            return grid
        inflated = grid.copy()
        for dy, dx in self.inflation_offsets:
            if dx == 0 and dy == 0:
                continue
            source_y0 = max(0, -dy)
            source_y1 = min(self.height, self.height - dy)
            source_x0 = max(0, -dx)
            source_x1 = min(self.width, self.width - dx)
            target_y0, target_y1 = source_y0 + dy, source_y1 + dy
            target_x0, target_x1 = source_x0 + dx, source_x1 + dx
            np.maximum(
                inflated[target_y0:target_y1, target_x0:target_x1],
                grid[source_y0:source_y1, source_x0:source_x1],
                out=inflated[target_y0:target_y1, target_x0:target_x1],
            )
        return inflated

    def make_message(self, costs, stamp, frame_id):
        """Serialize a rasterized cost array as nav_msgs/OccupancyGrid."""
        grid = OccupancyGrid()
        grid.header.stamp = stamp
        grid.header.frame_id = frame_id
        grid.info.map_load_time = stamp
        grid.info.resolution = self.resolution
        grid.info.width = self.width
        grid.info.height = self.height
        grid.info.origin.position.x = 0.0
        grid.info.origin.position.y = self.origin_y
        grid.info.origin.orientation.w = 1.0
        grid.data = array.array(
            'b', costs.astype(np.int8, copy=False).ravel().tobytes())
        return grid
