"""Bounded Euclidean distance transforms for the Jetson CUDA backend."""

import math
import os
import shutil

import numpy as np


CUDA_SOURCE = r'''
#include <math_constants.h>

// Cell flags preserve overlapping observations without color precedence.
// Each scatter uses atomicOr because several cloud points may share a cell.
enum { BLUE = 1, WHITE = 2, ELIGIBLE = 4, SEED = 8, SELECTED = 16 };

extern "C" __global__ void rasterize(
    const int *cells, int *flags, int blue_count, int point_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < point_count)
        atomicOr(flags + cells[i], i < blue_count ? BLUE : WHITE);
}

// Horizontal squared distance to the nearest source in this row. The search
// stops at the first source on either side, or at the configured radius.
extern "C" __global__ void horizontal_distance(
    const int *flags, float *horizontal, const int *seed_count,
    int width, int height, int radius, int source)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height || (source == SEED && *seed_count == 0)) return;
    if (flags[i] & source) { horizontal[i] = 0.0f; return; }
    int x = i % width;
    float best = CUDART_INF_F;
    for (int dx = 1; dx <= radius; ++dx) {
        if ((x >= dx && (flags[i - dx] & source)) ||
            (x + dx < width && (flags[i + dx] & source))) {
            best = float(dx * dx);
            break;
        }
    }
    horizontal[i] = best;
}

// Minimize horizontal_distance(x, source_y) + (y - source_y)^2.
// This separable transform is exact within the requested radius. Values
// farther away are irrelevant to classification and may remain infinity.
// Only observed white cells need the final vertical result.
extern "C" __global__ void vertical_distance(
    const int *flags, const float *horizontal, float *distance,
    const int *seed_count, int width, int height, int radius, int source)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height || !(flags[i] & WHITE) ||
        (source == SEED && *seed_count == 0)) return;
    int y = i / width;
    float best = horizontal[i];
    for (int dy = 1; dy <= radius; ++dy) {
        float dy2 = float(dy * dy);
        if (dy2 >= best) break;
        if (y >= dy) best = fminf(best, horizontal[i - dy * width] + dy2);
        if (y + dy < height)
            best = fminf(best, horizontal[i + dy * width] + dy2);
    }
    distance[i] = sqrtf(best);
}

extern "C" __global__ void select_seeds(
    int *flags, const float *distance, int *seed_count, int size,
    double minimum, double maximum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size || !(flags[i] & WHITE)) return;
    // Match OpenCV's float32 distance image, then compare to double thresholds.
    double d = double(distance[i]);
    if (d <= maximum) {
        flags[i] |= ELIGIBLE;
        if (d > minimum) {
            flags[i] |= SEED | SELECTED;
            atomicAdd(seed_count, 1);
        }
    }
}

extern "C" __global__ void propagate(
    int *flags, const float *distance, const int *seed_count,
    int size, double radius)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size || *seed_count == 0) return;
    // Only first-pass seeds are distance sources. No iterative flood fill.
    if ((flags[i] & ELIGIBLE) && !(flags[i] & SEED) &&
        double(distance[i]) < radius) flags[i] |= SELECTED;
}

extern "C" __global__ void gather_white(
    const int *cells, const int *flags, unsigned char *output,
    int blue_count, int white_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < white_count) output[i] = (unsigned char)flags[cells[blue_count + i]];
}
'''


class CudaBoardwalkGrid:
    """Keep the raster and both distance passes on one retained CUDA context.

    Only sparse cell indices are uploaded and one flag byte per white point
    is downloaded. Allocation, compiled kernels and timing events are reused.
    """

    MAX_RADIUS = 128
    STAGES = (
        'boardwalk_gpu_upload_ms', 'boardwalk_raster_ms',
        'boardwalk_blue_dt_ms', 'boardwalk_first_pass_ms',
        'boardwalk_seed_dt_ms', 'boardwalk_second_pass_ms',
        'boardwalk_gpu_download_ms',
    )

    def __init__(self, max_cells):
        # Import lazily: CPU operation does not require a working CUDA runtime.
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule

        nvcc = shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
        if not os.path.isfile(nvcc):
            raise RuntimeError('CUDA compiler nvcc was not found')
        cuda.init()
        self.cuda = cuda
        self.context = cuda.Device(0).retain_primary_context()
        self.max_cells = max_cells
        self.grid_capacity = self.point_capacity = 0
        self.buffers = {}
        self.events = {}
        self.functions = {}
        self.module = None
        try:
            self.context.push()
            try:
                # No fast-math: retain float32 sqrt behavior at threshold edges.
                self.module = SourceModule(
                    CUDA_SOURCE, nvcc=nvcc, options=['--fmad=false'], no_extern_c=True)
                for name in ('rasterize', 'horizontal_distance', 'vertical_distance',
                             'select_seeds', 'propagate', 'gather_white'):
                    self.functions[name] = self.module.get_function(name)
                self.events = {key: (cuda.Event(), cuda.Event()) for key in self.STAGES}
                self.buffers['seed_count'] = cuda.mem_alloc(4)
            finally:
                cuda.Context.pop()
        except Exception:
            self.close()
            raise

    def _reserve(self, size, point_count):
        """Grow device storage geometrically while respecting the grid limit."""
        if size > self.grid_capacity:
            capacity = min(self.max_cells, max(size, 2 * self.grid_capacity))
            for name in ('flags', 'horizontal', 'distance'):
                if name in self.buffers:
                    self.buffers.pop(name).free()
                self.buffers[name] = self.cuda.mem_alloc(4 * capacity)
            self.grid_capacity = capacity
        if point_count > self.point_capacity:
            capacity = max(point_count, 2 * self.point_capacity)
            for name, stride in (('cells', 4), ('output', 1)):
                if name in self.buffers:
                    self.buffers.pop(name).free()
                self.buffers[name] = self.cuda.mem_alloc(stride * capacity)
            self.point_capacity = capacity

    def classify(self, cells, blue_count, width, height, minimum, maximum, radius):
        """Return per-white eligibility, seed and selection flags plus timings.

        Thresholds are in cells. Bound searches to keep extreme resolutions
        from turning a bounded CUDA stencil into unbounded work on the Nano.
        """
        blue_radius = int(min(math.ceil(maximum), max(width, height) - 1))
        seed_radius = int(min(math.ceil(radius), max(width, height) - 1))
        if max(blue_radius, seed_radius) > self.MAX_RADIUS:
            raise ValueError('CUDA distance radius exceeds 128 cells; use the CPU backend')

        cuda = self.cuda
        size = width * height
        point_count = len(cells)
        white_count = point_count - blue_count
        host_cells = np.ascontiguousarray(cells[:, 1] * width + cells[:, 0], dtype=np.int32)
        output = np.empty(white_count, dtype=np.uint8)
        block = (128, 1, 1)
        grid = ((size + 127) // 128, 1, 1)
        stages_used = []

        def start(key):
            stages_used.append(key)
            self.events[key][0].record()

        def stop(key):
            self.events[key][1].record()

        self.context.push()
        try:
            self._reserve(size, point_count)
            b = self.buffers
            f = self.functions
            start('boardwalk_gpu_upload_ms')
            cuda.memcpy_htod(b['cells'], host_cells)
            stop('boardwalk_gpu_upload_ms')

            start('boardwalk_raster_ms')
            cuda.memset_d32(b['flags'], 0, size)
            cuda.memset_d32(b['seed_count'], 0, 1)
            f['rasterize'](b['cells'], b['flags'], np.int32(blue_count),
                           np.int32(point_count), block=block,
                           grid=((point_count + 127) // 128, 1, 1))
            stop('boardwalk_raster_ms')

            def transform(search_radius, source):
                f['horizontal_distance'](
                    b['flags'], b['horizontal'], b['seed_count'],
                    np.int32(width), np.int32(height),
                    np.int32(min(search_radius, width - 1)), np.int32(source),
                    block=block, grid=grid)
                f['vertical_distance'](
                    b['flags'], b['horizontal'], b['distance'], b['seed_count'],
                    np.int32(width), np.int32(height),
                    np.int32(min(search_radius, height - 1)), np.int32(source),
                    block=block, grid=grid)

            start('boardwalk_blue_dt_ms')
            transform(blue_radius, 1)
            stop('boardwalk_blue_dt_ms')
            start('boardwalk_first_pass_ms')
            f['select_seeds'](b['flags'], b['distance'], b['seed_count'],
                               np.int32(size), np.float64(minimum), np.float64(maximum),
                               block=block, grid=grid)
            stop('boardwalk_first_pass_ms')

            if radius > 0.0:
                start('boardwalk_seed_dt_ms')
                transform(seed_radius, 8)
                stop('boardwalk_seed_dt_ms')
                start('boardwalk_second_pass_ms')
                f['propagate'](b['flags'], b['distance'], b['seed_count'],
                               np.int32(size), np.float64(radius), block=block, grid=grid)
                stop('boardwalk_second_pass_ms')

            start('boardwalk_gpu_download_ms')
            f['gather_white'](b['cells'], b['flags'], b['output'],
                               np.int32(blue_count), np.int32(white_count),
                               block=block, grid=((white_count + 127) // 128, 1, 1))
            cuda.memcpy_dtoh(output, b['output'])
            stop('boardwalk_gpu_download_ms')
            self.events['boardwalk_gpu_download_ms'][1].synchronize()
            timings = {key: self.events[key][0].time_till(self.events[key][1])
                       for key in stages_used}
        finally:
            # Every call balances its push, including failures. No autoinit
            # context is left on the worker's stack or on the ROS main thread.
            cuda.Context.pop()
        return (output & 4) != 0, (output & 8) != 0, (output & 16) != 0, timings

    def close(self):
        """Release device resources after the processing worker has stopped."""
        if self.context is None:
            return
        self.context.push()
        try:
            for allocation in self.buffers.values():
                allocation.free()
            self.buffers.clear()
            self.functions.clear()
            self.events.clear()
            self.module = None
        finally:
            self.cuda.Context.pop()
            self.context.detach()
            self.context = None
