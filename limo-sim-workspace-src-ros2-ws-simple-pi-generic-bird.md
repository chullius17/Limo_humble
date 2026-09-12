# Diagnosis: `curb_detector` Step 8a cost, and proposed fix

## Diagnosis

### Symptom
`curb_detector` (`simple_boundaries.py`)'s profiling output shows `[Step 8a: Cloud Preparation]` costing 20-95ms per frame on the real LIMO, dominating `Total Latency` and capping the worker at 15-40 FPS instead of tracking the ~30 Hz camera input.

### Investigation trail
1. **First hypothesis (logs `boundaries.txt`/`boundaries_1.txt`): chronic TF failure.** Every cycle logged `Waiting for TF camera_color_optical_frame -> base_link: ... extrapolation into the future`. Root-caused to `ros2_astra_camera`'s `ob_camera_node.cpp`: the driver computes `camera_link -> camera_*_optical_frame` once from calibration (genuinely rigid) but re-broadcasts it on the *dynamic* `/tf` topic every 100ms (`tf_publish_rate` default 10 Hz) instead of publishing it once on `/tf_static`. Against ~30Hz images, most lookups landed past the last 10Hz sample.
   - **Fix applied:** `tf_publish_rate: 0.0` added to the `dabai_u3.launch.xml` include in `limo_real.launch.py` (already committed to the launch file). This makes the driver's `publishStaticTransforms()` take the one-shot `/tf_static` path (`ob_camera_node.cpp:477-489`) instead of spawning `publishDynamicTransforms()`.
   - **Result (`boundaries_2.txt`, 1320 frames):** zero TF warnings — fix confirmed working. But `Step 8a` was **unchanged** (~28-33ms avg). This falsified "TF failure was the dominant cost."
2. **Second hypothesis: which sub-phase of `publish_pointcloud()` actually costs the time?** Added three sub-timers (`8a.1 Lock/Read`, `8a.2 TF Lookup`, `8a.3 Projection/Serialize`) around the existing code in `publish_pointcloud()` — no logic change, purely instrumentation (already committed).
   - **Result (`boundaries_3.txt`):** `8a.1` ≈ 0.01ms, `8a.2` ≈ 0.5ms, **`8a.3` ≈ 17-24ms — over 95% of Step 8a.** `8a.3` is pure vectorized NumPy (pixel→camera→BEV projection, structured-array fill, `PointCloud2` field assembly, `.tobytes()`) over only ~1700-1900 points — content that should cost microseconds, not tens of milliseconds. A 1000x+ gap between "work done" and "time taken" is the signature of a thread waiting for the CPU/GIL, not computing.
3. **Confirmed directly with `top -H -p <curb_detector pid>` on the robot (live, PID 24138, 11 threads):**
   - Main/executor thread (24138): **33-38% of a core continuously** — this is the thread running `depth_callback`/`fallback_depth_callback`/`camera_info_callback`.
   - Worker thread (25940, runs `_processing_worker` → `publish_pointcloud`, i.e. all of Step 8a incl. `8a.3`): only **15-27% of a core** — visibly starved relative to the main thread despite doing the heavier compute.
   - **System-wide `load average: 7.62, 5.36, 3.18` on a 4-core Jetson Nano**, with `astra_camera_node` at 51.7% CPU, `depth_correction` at 46.2%, `curb_detector`'s three active threads summing to ~67%, plus `lane_detector`/`ydlidar_ros2_driver`. Demand comfortably exceeds 4 cores.

### Conclusion — two stacked causes, not one
1. **In-process GIL contention**: `curb_detector`'s worker thread executes `8a.3` as many small sequential NumPy calls, each a GIL release/reacquire point; the main executor thread in the same process is kept busy nearly continuously by per-frame image callbacks, so the GIL round-robin repeatedly favors the main thread between the worker's small ops.
2. **System-wide CPU oversubscription**: load average 7.6 vs. 4 physical cores means the OS scheduler is forced to time-slice regardless of GIL behavior — a perfectly GIL-friendly rewrite still has a ceiling imposed by not enough physical CPU for everything running concurrently on this Nano.

## Proposition (fix)

Address (1) first — it's fully contained in `simple_boundaries.py`, cheap, and reduces this node's own contribution to (2) as a side effect; then reassess whether (2) still limits things.

### A. Shrink the main executor thread's per-frame footprint
`convert_depth()`/`fallback_depth_callback()` in `simple_boundaries.py` still run full-frame `imgmsg_to_cv2` + `astype(float32)*0.001` (and, on the fallback path, a `cv2.resize`) synchronously on the executor thread for every incoming depth message — competing for the GIL with the worker thread on every single frame, not just during fallback. Apply the same crop-before-convert / zero-copy `np.frombuffer` ingress pattern already validated and shipped in `depth_correction.py::_view_source` and `simple_detector.py::_view_source` to `convert_depth()`, so the executor thread does less synchronous NumPy work per callback and holds the GIL for shorter, less frequent stretches.

### B. Reduce GIL exposure inside `8a.3`
Fuse the sequence of small NumPy calls in `publish_pointcloud()`'s math section into fewer, larger calls so the worker thread has fewer GIL-release points to be preempted at:
- Combine the `camera_points`/`bev_points` computation (currently `column_stack` → `astype` → matmul → separate `+=`) using preallocated output buffers and in-place ops (`np.multiply(..., out=...)`, `cv2.gemm`/`@` into a preallocated array) — mirrors the buffer-reuse pattern already used in `depth_correction.py` this session.
- Write directly into the `cloud_points` structured array's fields as they're computed rather than building `camera_points`/`bev_points` as separate intermediate arrays.
- This won't reduce total CPU work much, but reduces the number of scheduling-vulnerable boundaries, which is what `top -H` shows is actually being lost.

### C. Address system-wide oversubscription (separate follow-up, larger scope)
Not part of this immediate fix, but flagged: `astra_camera_node` (51.7%) and `depth_correction` (46.2%) are the two heaviest processes on the Nano besides `curb_detector` itself. Worth a follow-up pass (out of scope here) to check whether `astra_camera_node`'s CPU can be trimmed (it publishes point clouds/IR/multiple streams that may not all be consumed — check `enable_point_cloud`, `enable_ir`, `color_depth_synchronization` in `dabai_u3.launch.xml`'s launch args in `limo_real.launch.py`), independent of the TF-rate fix already applied.

## Files touched (for A + B)

- `limo_sim/workspace/src/ros2_ws/simple_pipeline/base_cv_block/cv_package/cv_package/simple_boundaries.py`:
  - `convert_depth()` (currently `:285-297`) and `fallback_depth_callback()` (`:318-353`) — zero-copy ingress, matching `depth_correction.py::_view_source`'s pattern.
  - `publish_pointcloud()`'s `8a.3` math section (`:645-674` post-instrumentation) — fused/in-place array ops.

No topic, message, or parameter contract changes; bit-identical output verification (same harness pattern used for `depth_correction.py`/`simple_detector.py` this session) applies before/after.

## Verification

1. Bit-identical harness (same approach as `depth_correction.py`/`simple_detector.py`): assert `convert_depth()`'s zero-copy path and the fused `8a.3` math produce identical output to the current implementation across uint16/float32, padded/unpadded, NaN/inf cases.
2. `colcon build --packages-select cv_package --symlink-install && colcon test --packages-select cv_package` in the container — no new lint findings.
3. Real-robot re-run with the existing `8a.1`/`8a.2`/`8a.3` sub-timers: confirm `8a.3` drops toward the sub-ms range its actual computational content implies, and re-check `top -H` on `curb_detector`'s PID to confirm the worker thread's CPU share rises relative to the main thread's.
4. Re-check system load average — expect some reduction from (A) alone since the main thread does less synchronous work per frame, but do not expect it to fully resolve until (C) is addressed separately.

## Rollback

A and B are independent, revertable changes confined to `simple_boundaries.py`; no other files change as part of this fix.
