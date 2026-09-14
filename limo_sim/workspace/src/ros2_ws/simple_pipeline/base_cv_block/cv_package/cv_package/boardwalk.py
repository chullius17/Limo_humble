"""Classify observed white BEV points on a reusable metric image grid."""

import time

import cv2
import numpy as np


BOARDWALK_TIMINGS = (
    ('boardwalk_total_ms', 'Total'),
    ('boardwalk_raster_ms', 'Grid preparation'),
    ('boardwalk_blue_dt_ms', 'Distance from blue'),
    ('boardwalk_first_pass_ms', 'First-pass selection'),
    ('boardwalk_seed_dt_ms', 'Distance from seeds'),
    ('boardwalk_second_pass_ms', 'Second-pass selection'),
    ('boardwalk_labels_ms', 'Point label update'),
    ('boardwalk_gpu_init_ms', 'CUDA initialization (cold)'),
    ('boardwalk_gpu_upload_ms', 'CUDA upload'),
    ('boardwalk_gpu_download_ms', 'CUDA result download'),
    ('boardwalk_debug_render_ms', 'Grid debug rendering'),
    ('boardwalk_debug_publish_ms', 'Grid JPEG + publish (extra)'),
)
BOARDWALK_COUNTS = (
    ('boardwalk_blue_count', 'Blue sources'),
    ('boardwalk_white_count', 'White input'),
    ('boardwalk_eligible_count', 'White within max distance'),
    ('boardwalk_seed_count', 'First-pass seeds'),
    ('boardwalk_propagated_count', 'Added by second pass'),
    ('boardwalk_final_count', 'Final boardwalk'),
    ('boardwalk_outside_count', 'White outside max distance'),
    ('boardwalk_nonfinite_count', 'Non-finite points ignored'),
    ('boardwalk_grid_width', 'Grid columns'),
    ('boardwalk_grid_height', 'Grid rows'),
    ('boardwalk_grid_cells', 'Grid cells'),
    ('boardwalk_grid_skipped', 'Grid limit exceeded'),
    ('boardwalk_gpu_used', 'CUDA backend used'),
    ('boardwalk_gpu_fallback', 'CPU fallback after CUDA error'),
)


class BoardwalkClassifier:
    """Run two Euclidean distance transforms on a cropped BEV lattice.

    One instance belongs to the node's processing worker. Buffers are reused
    across frames, so an instance must not be called concurrently.
    """

    def __init__(self, resolution=0.01, max_cells=1000000, backend='cpu'):
        if not np.isfinite(resolution) or resolution <= 0.0:
            raise ValueError('Boardwalk grid resolution must be finite and positive')
        if max_cells <= 0 or int(max_cells) != max_cells:
            raise ValueError('Boardwalk grid cell limit must be a positive integer')
        if backend not in ('cpu', 'cuda', 'auto'):
            raise ValueError('Boardwalk backend must be cpu, cuda or auto')
        self.backend = backend
        self.active_backend = 'cpu' if backend == 'cpu' else 'pending'
        self.gpu_error = None
        self._cuda = None
        self.resolution = float(resolution)
        self.max_cells = int(max_cells)
        self._mask = np.empty(0, dtype=np.uint8)
        self._distance = np.empty(0, dtype=np.float32)
        self.debug_image = None

    def classify(
            self, points, class_ids, minimum, maximum, propagation_radius,
            blue_label=1, white_label=3, boardwalk_label=4, debug=False):
        """Relabel selected white points in place; return timings and counts.

        Input coordinates and thresholds are in meters. Thresholds must be
        finite, with 0 <= minimum <= maximum and propagation_radius >= 0.
        Distances refer to raster cells; original coordinates are never snapped
        to the grid. White points beyond the maximum blue distance remain white.
        """
        if self.backend == 'cuda' and self.gpu_error is not None:
            raise RuntimeError(f'CUDA backend previously failed: {self.gpu_error}')
        started = time.perf_counter()
        stats = {key: 0.0 for key, _ in BOARDWALK_TIMINGS}
        stats.update({key: 0 for key, _ in BOARDWALK_COUNTS})
        # Drop the previous frame, including when debug is disabled or the
        # current grid cannot be evaluated. Never publish an old grid as new.
        self.debug_image = None
        debug_geometry = None
        seeds = selected = None

        def finished():
            stats['boardwalk_gpu_fallback'] = int(
                self.backend == 'auto' and self.gpu_error is not None)
            if debug:
                rendering_started = time.perf_counter()
                self.debug_image = self._render_debug_grid(
                    debug_geometry, seeds, selected, stats)
                stats['boardwalk_debug_render_ms'] = (
                    time.perf_counter() - rendering_started) * 1000.0
            stats['boardwalk_total_ms'] = (time.perf_counter() - started) * 1000.0
            return stats

        stage = time.perf_counter()
        finite = np.isfinite(points).all(axis=1)
        blue_indices = np.flatnonzero((class_ids == blue_label) & finite)
        white_indices = np.flatnonzero((class_ids == white_label) & finite)
        stats['boardwalk_blue_count'] = len(blue_indices)
        stats['boardwalk_white_count'] = len(white_indices)
        stats['boardwalk_nonfinite_count'] = int(np.count_nonzero(~finite))
        if not len(blue_indices) or not len(white_indices):
            stats['boardwalk_outside_count'] = len(white_indices)
            stats['boardwalk_raster_ms'] = (time.perf_counter() - stage) * 1000.0
            return finished()

        # Anchor cells to the BEV frame, rather than the cloud minimum, so a
        # changing bounding box does not move the lattice between frames.
        # Other classes never affect the grid extent or act as distance sources.
        indices = np.concatenate((blue_indices, white_indices))
        with np.errstate(over='ignore', invalid='ignore'):
            cells = np.floor(points[indices].astype(np.float64) / self.resolution)
            cells -= cells.min(axis=0)
            extent = cells.max(axis=0) + 1.0

        # Bound allocation and EDT work before converting indices to integers.
        # An oversized cloud keeps its original labels and is reported by the
        # caller; never silently coarsen the grid or crop away distance sources.
        if (not np.isfinite(extent).all()
                or np.any(extent > self.max_cells)
                or extent[0] * extent[1] > self.max_cells):
            stats['boardwalk_grid_skipped'] = 1
            stats['boardwalk_raster_ms'] = (time.perf_counter() - stage) * 1000.0
            return finished()

        width, height = (int(value) for value in extent)
        size = width * height
        stats['boardwalk_grid_width'] = width
        stats['boardwalk_grid_height'] = height
        stats['boardwalk_grid_cells'] = size
        cells = cells.astype(np.intp)
        blue_cells = cells[:len(blue_indices)]
        white_cells = cells[len(blue_indices):]
        white_x, white_y = white_cells.T
        if debug:
            debug_geometry = (width, height, blue_cells, white_cells)

        if self.backend != 'cpu' and self.gpu_error is None:
            stats['boardwalk_raster_ms'] = (time.perf_counter() - stage) * 1000.0
            try:
                # Initialize in the processing thread, not the ROS main thread.
                # The backend balances context push/pop on every invocation.
                if self._cuda is None:
                    initializing = time.perf_counter()
                    try:
                        from cv_package.boardwalk_cuda import CudaBoardwalkGrid
                        self._cuda = CudaBoardwalkGrid(self.max_cells)
                    finally:
                        stats['boardwalk_gpu_init_ms'] = (
                            time.perf_counter() - initializing) * 1000.0
                eligible, seeds, selected, gpu_times = self._cuda.classify(
                    cells, len(blue_indices), width, height,
                    minimum / self.resolution, maximum / self.resolution,
                    propagation_radius / self.resolution)
            except Exception as error:
                self.gpu_error = str(error)
                try:
                    self.close()
                except Exception:
                    pass  # Preserve the original CUDA failure for diagnostics.
                if self.backend == 'cuda':
                    raise
                # In auto mode classify this very frame on CPU. Failed CUDA
                # work has not modified any host labels. Do not retry per frame.
                self.active_backend = 'cpu'
                stage = time.perf_counter()
            else:
                self.active_backend = 'cuda'
                stats['boardwalk_gpu_used'] = 1
                for key, elapsed in gpu_times.items():
                    stats[key] += elapsed
                stats['boardwalk_eligible_count'] = int(np.count_nonzero(eligible))
                stats['boardwalk_outside_count'] = (
                    len(white_indices) - stats['boardwalk_eligible_count'])
                stats['boardwalk_seed_count'] = int(np.count_nonzero(seeds))
                stats['boardwalk_propagated_count'] = int(np.count_nonzero(selected & ~seeds))
                stage = time.perf_counter()
                class_ids[white_indices[selected]] = boardwalk_label
                stats['boardwalk_final_count'] = int(np.count_nonzero(selected))
                stats['boardwalk_labels_ms'] = (time.perf_counter() - stage) * 1000.0
                return finished()

        if self._mask.size < size:
            capacity = min(self.max_cells, max(size, 2 * self._mask.size))
            self._mask = np.empty(capacity, dtype=np.uint8)
            self._distance = np.empty(capacity, dtype=np.float32)
        mask = self._mask[:size].reshape(height, width)
        distance = self._distance[:size].reshape(height, width)

        # Only observed blue cells are zero. Empty cells and image edges must
        # not become artificial sources. A white and blue observation may share
        # a cell: white remains a query target and its blue distance is zero.
        mask.fill(255)
        mask[blue_cells[:, 1], blue_cells[:, 0]] = 0
        stats['boardwalk_raster_ms'] += (time.perf_counter() - stage) * 1000.0
        stage = time.perf_counter()
        cv2.distanceTransform(
            mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, dst=distance)
        stats['boardwalk_blue_dt_ms'] = (time.perf_counter() - stage) * 1000.0

        stage = time.perf_counter()
        # Sampling only observed white cells avoids inventing points in holes.
        # Compare in pixel units to keep the EDT image reusable and avoid a
        # full-grid multiplication by the metric resolution each pass.
        distance_blue = distance[white_y, white_x].astype(np.float64)
        eligible = distance_blue <= maximum / self.resolution
        seeds = eligible & (distance_blue > minimum / self.resolution)
        stats['boardwalk_eligible_count'] = int(np.count_nonzero(eligible))
        stats['boardwalk_outside_count'] = (
            len(white_indices) - stats['boardwalk_eligible_count'])
        stats['boardwalk_seed_count'] = int(np.count_nonzero(seeds))
        stats['boardwalk_first_pass_ms'] = (time.perf_counter() - stage) * 1000.0
        if not stats['boardwalk_seed_count']:
            return finished()

        selected = seeds.copy()
        remaining = np.flatnonzero(eligible & ~seeds)
        if len(remaining) and propagation_radius > 0.0:
            stage = time.perf_counter()
            # Reuse the buffers, clearing every old blue source. Only seeds
            # from the first pass can propagate; newly selected whites cannot.
            mask.fill(255)
            mask[white_y[seeds], white_x[seeds]] = 0
            cv2.distanceTransform(
                mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, dst=distance)
            stats['boardwalk_seed_dt_ms'] = (time.perf_counter() - stage) * 1000.0
            stage = time.perf_counter()
            distance_seed = distance[
                white_y[remaining], white_x[remaining]].astype(np.float64)
            propagated = remaining[
                distance_seed < propagation_radius / self.resolution]
            selected[propagated] = True
            stats['boardwalk_propagated_count'] = len(propagated)
            stats['boardwalk_second_pass_ms'] = (time.perf_counter() - stage) * 1000.0

        stage = time.perf_counter()
        # Map the grid decision back to every original white point, including
        # multiple observations in one cell. Preserve geometry and point order.
        class_ids[white_indices[selected]] = boardwalk_label
        stats['boardwalk_final_count'] = int(np.count_nonzero(selected))
        stats['boardwalk_labels_ms'] = (time.perf_counter() - stage) * 1000.0
        return finished()

    def close(self):
        """Release CUDA resources once no classification call is in progress."""
        if self._cuda is not None:
            self._cuda.close()
            self._cuda = None

    def _render_debug_grid(self, geometry, seeds, selected, stats):
        """Show the actual EDT lattice and both classification passes in BGR.

        One displayed pixel is one grid cell. Panels keep the EDT indexing:
        columns increase with BEV x, rows increase with BEV y. Unknown cells
        are black, observations white/blue, and boardwalk selections magenta.
        """
        if geometry is None:
            frame = np.zeros((80, 640, 3), dtype=np.uint8)
            reason = ('Grid cell limit exceeded' if stats['boardwalk_grid_skipped']
                      else 'No blue/white pair: grid not evaluated')
            cv2.putText(frame, reason, (8, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)
            return frame

        width, height, blue_cells, white_cells = geometry
        title_height = 48
        panel_width = max(width, 300)
        frame = np.zeros((height + title_height, panel_width * 3, 3), dtype=np.uint8)
        white_x, white_y = white_cells.T
        titles = ('Input grid', 'DT1: boardwalk seeds', 'DT2: final boardwalk')
        for index, (title, selection) in enumerate(zip(titles, (None, seeds, selected))):
            offset = index * panel_width
            panel = frame[title_height:, offset:offset + width]
            panel[white_y, white_x] = (255, 255, 255)
            if selection is not None:
                panel[white_y[selection], white_x[selection]] = (255, 0, 255)
            # Blue remains visible where multiple observations share a cell.
            panel[blue_cells[:, 1], blue_cells[:, 0]] = (255, 0, 0)
            cv2.putText(frame, title, (offset + 4, 17), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f'{width}x{height} | {self.resolution:g} m/cell',
                        (offset + 4, 36), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return frame
