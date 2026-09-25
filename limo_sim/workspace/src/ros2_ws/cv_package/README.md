# Computer vision profiles

Start CV explicitly, independently of mapping:

```bash
# Physical robot, after custom_start/limo_real.launch.py starts the sensors:
ros2 launch cv_package cv_real.launch.py
# Simulation:
ros2 launch cv_package cv_sim.launch.py
```


`cv_real.launch.py` (and the default `cv.launch.py`) runs the waterfall detector;
`cv_sim.launch.py` keeps the HSV color detector. `launch.lane_detector` selects
`waterfall` or `color`, and the `lane_detector` YAML section supplies its parameters.
The real profile contains seed/gradient/morphology parameters rather than HSV
thresholds. The launch keeps the detector node name `/lane_node` and remaps
`lane_waterfall_labels/raw` to the common `/limo/cv_package/detection/lane_labels/raw`
topic consumed by `visual_ptcld`. Debug topics retain their `lane_waterfall_*`
names, including `/limo/cv_package/detection/lane_waterfall_seeds_overlay/compressed`
for red barriers and green seeds. `desktop_cv.launch.py` remains a viewer only.
For runtime tuning in the launched pipeline use, for example:

```bash
ros2 param set /lane_node gradient_threshold 15.0
```

Mapping and `user_package` continue to use
`/limo/cv_package/visual_ptcld/points`; their real profiles select `cv_real.yaml`
when `start_cv:=true`. CV stays opt-in there to avoid duplicating an already
running `cv_real`. The selected CV profile is independent of `use_sim_time`.

`cv_real.yaml` enables `visual_ptcld.road_boardwalk_only`. The outgoing cloud
`/limo/cv_package/visual_ptcld/points` contains only these semantic classes:

| Class ID | Meaning | Mapping cost |
| --- | --- | --- |
| 1 | Border road | 0 |
| 5 | Interior road | 0 |
| 4 | Boardwalk | 90 |
| 6 | Interior boardwalk | 90 |

Yellow-line points (2) and unclassified background points (3) are excluded.
They are never promoted to road or boardwalk. Filtering happens after metric
boardwalk recognition, before cloud voxelization and publication. The intermediate
lane-label image still contains background candidates needed to recognize
boardwalk and yellow labels needed to avoid treating yellow pixels as road.
Those intermediate labels are not map observations.

`road_boardwalk_only` is a startup parameter and requires `enable_boardwalk: true`.
It defaults to false, so the simulation CV retains all existing classes and rules.
Offline and online mapping profiles no longer start CV automatically. Launch
CV separately; the mapping launch's explicit `start_cv:=true` override remains
available. Restart CV after changing its profile.

## Adaptive binary lane detector (manual)

`lane_detector_binary` is an optional alternative to `lane_detector`; no launch
or YAML profile starts it. After building and sourcing the workspace:

```bash
ros2 run cv_package lane_detector_binary --ros-args \
  -p rgb_topic:=/rgb/image_raw \
  -p adaptive_block_size:=31 -p adaptive_c:=5.0
```

The ROS node is `/binary_lane_detector`. Input is `sensor_msgs/Image`, with the
same lower-half crop, 320x120 output and vertical ROI as the color detector
(`roi_y_min=0.1`, `roi_y_max=1.0`, relative to that crop). The resized ROI is
converted to grayscale, then processed with OpenCV `ADAPTIVE_THRESH_MEAN_C`
and `THRESH_BINARY_INV`. Road is the inverse-threshold foreground (including
equality with the threshold, following OpenCV's rounding rules). The mean is
computed within the ROI, replicating its borders.

By default, each pixel uses a local variance fallback. The same square kernel
computes `mean(I)` and `mean(I^2)` in float32; population variance is
`max(mean(I^2) - mean(I)^2, 0)`. Where variance is strictly below
`fallback_variance_threshold`, road is instead `I <= fallback_gray_threshold`.
Elsewhere the shared mean is reused for the adaptive comparison, preserving
OpenCV's mean rounding and inverse-threshold C convention. Thus a uniformly
dark region can become road while a uniformly bright region remains background,
even when high-contrast lines are present elsewhere in the ROI.

The new runtime parameters are:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `enable_variance_fallback` | `true` | Set false for the original OpenCV adaptive threshold |
| `fallback_variance_threshold` | `25.0` | Nonnegative finite variance, squared gray levels (25 corresponds to standard deviation 5); 0 selects no fallback pixels |
| `fallback_gray_threshold` | `150` | Absolute road threshold, integer in [0, 255] |

These defaults are starting values to tune on camera images. Local moments and
masks use reusable buffers; no Python loops over pixels or neighborhoods are
needed. Disabling the fallback skips the local variance calculation entirely.

| Output topic (under `/limo/cv_package/detection/`) | Format |
| --- | --- |
| `lane_binary_labels/raw` | `Image`, `mono8`: 0 outside ROI, 1 road, 3 background |
| `lane_binary_masks/compressed` | `CompressedImage`, JPEG: blue road on black |
| `lane_binary_overlay/compressed` | `CompressedImage`, JPEG: blue road over camera image |
| `lane_binary_fallback_overlay/compressed` | `CompressedImage`, JPEG: road/fallback overlay over camera image |

The dedicated fallback overlay shows blue for road classified by the adaptive
threshold, red for fallback regions classified as background, and purple where
road and fallback overlap. The blue road mask and red fallback mask are combined
before blending over the camera crop (camera weight 0.7, mask weight 0.5, as in
the existing overlay). No tint is added outside the ROI. When fallback is
disabled, this topic shows the normal blue road overlay. Like the other debug
topics, it is composed and JPEG-encoded only when subscribers are detected;
all requested overlays share the resized camera background.

All outputs preserve the input header. Label 2 is never emitted. To replace the
classic detector for existing consumers, stop that detector and remap the binary
label output at startup:

```bash
ros2 run cv_package lane_detector_binary --ros-args \
  -r limo/cv_package/detection/lane_binary_labels/raw:=/limo/cv_package/detection/lane_labels/raw
```

`adaptive_block_size` (odd integer >= 3, pixels in the resized ROI), `adaptive_c`
(finite double), `debug_jpeg_quality` (1-100, default 85) and `enable_telemetry`
(default true) can be changed at runtime, for example:

```bash
ros2 param set /binary_lane_detector adaptive_block_size 51
ros2 param set /binary_lane_detector adaptive_c 7.0
ros2 param set /binary_lane_detector fallback_variance_threshold 25.0
ros2 param set /binary_lane_detector fallback_gray_threshold 150
ros2 param set /binary_lane_detector enable_variance_fallback false
```

`rgb_topic`, ROI limits, `opencv_num_threads` (default 1) and
`debug_probe_interval_frames` (default 30) are startup-only parameters. The node
uses zero-copy BGR/RGB/mono ingress (including padded rows), reusable processing
buffers, single-frame drop-oldest queues and a separate JPEG publisher worker.
Debug is computed only when subscribers are detected. Telemetry logs averages
over 30 frames: input/processing rates, queue drops, transport/final message age,
and processing/encoding stage times. `fallback_percent` reports the mean percentage
of ROI pixels using the absolute threshold, including zero while disabled; the
`adaptive_threshold` timing includes the local variance and fallback work. Measure throughput on the Nano with the
intended camera and debug subscribers; host timings are not Nano benchmarks.

## Waterfall lane detector

`cv_real.launch.py` selects `lane_detector_waterfall` through `cv_real.yaml`.
For a standalone detector without depth correction or point-cloud generation,
build and source the workspace, then run:

```bash
ros2 run cv_package lane_detector_waterfall --ros-args \
  -p rgb_topic:=/rgb/image_raw \
  -p seed_max_gray:=80 -p gradient_threshold:=20.0
```

The node `/waterfall_lane_detector` uses the same lower-half crop, 320x120 output,
ROI parameters, camera encodings, QoS and input headers as the other detectors.
On the grayscale ROI it computes Sobel 3x3 with scale 1/8 and replicated borders.
The gradient magnitude is `abs(Gx) + abs(Gy)` without a square root. The scale
makes an axis-aligned ramp of one gray level per pixel produce gradient 1.
Pixels with gradient strictly above `gradient_threshold` are barriers. An optional
3x3 morphological closing bridges small holes and discontinuities without leaving
the barriers thicker (one iteration by default; set
`barrier_closing_iterations=0` to disable).

Seeds are pixels with `gray <= seed_max_gray`, outside the barriers and inside
the allowed seed band. Optional 3x3 square erosion removes small or thin seed
groups before propagation (disabled by default). When enabled, surviving seed cores
start growth; the traversable mask is unchanged, so they can still reach the
whole region. Erosion uses zero padding outside the ROI. This is a thickness
filter, not an exact connected-component area cutoff.

Native OpenCV 4-connected component labeling selects
all traversable regions containing at least one seed. Thus brighter pixels can
become road when connected to a seed without crossing a barrier. Barrier pixels
and unseeded regions remain background. No seeds means no road. This is seeded
region growth on a barrier mask, not OpenCV watershed.

| Runtime parameter | Default | Meaning |
| --- | --- | --- |
| `seed_max_gray` | `80` | Maximum seed intensity, integer [0, 255] |
| `gradient_threshold` | `15.0` | Barrier threshold in scaled Sobel L1 units, finite [0, 255] |
| `seed_y_min` | `0.0` | First seed row as a fraction of the ROI height, [0, 1); e.g. 0.7 restricts seeds to the bottom 30%, without limiting growth |
| `barrier_closing_iterations` | `1` | Number of 3x3 barrier closings, integer [0, 5]; 0 disables gap filling |
| `seed_erosion_iterations` | `0` | Number of 3x3 seed erosions, integer [0, 5]; 0 disables filtering |
| `debug_jpeg_quality` | `85` | JPEG quality [1, 100] |
| `enable_telemetry` | `true` | Enable timing and segmentation statistics |

`rgb_topic`, `roi_y_min`, `roi_y_max`, `opencv_num_threads` (default 1) and
`debug_probe_interval_frames` (default 30) are startup-only, as in the binary node.
The seed and gradient defaults are initial tuning values. Incomplete barriers
can let a region spread through a gap; dark objects can create their own seeds.
Closing and restricting the seed band can help, but should be tuned on camera data.

| Output topic (under `/limo/cv_package/detection/`) | Format |
| --- | --- |
| `lane_waterfall_labels/raw` | `Image`, `mono8`: 0 outside ROI, 1 road, 3 background |
| `lane_waterfall_masks/compressed` | JPEG: blue road mask on black |
| `lane_waterfall_overlay/compressed` | JPEG: blue road overlay on camera image |
| `lane_waterfall_seeds_overlay/compressed` | JPEG: red barriers and green seeds on camera image |

The dedicated seed overlay displays the actual barriers after closing and only
surviving seeds after erosion. Red and green replace those pixels for clear visibility; all
other pixels show the original camera crop, including outside the ROI. The blue
road overlay is available separately. Debug images are produced only when their
topics have subscribers, share a single resized camera background, and retain
the input header. JPEG encoding runs in the publisher worker.

Buffers are reused, queues retain only the latest frame, and seed propagation
uses native connected components and a vectorized component lookup rather than
Python pixel loops. Telemetry reports grayscale/resize, gradient and growth
timings, publishing/latency, drops and rates, seed/barrier/road percentages, and
the number of seeded components. `seed_erosion` measures erosion time (also
included in `region_growing`); `seed_removed_percent` is the percentage of admitted
seed pixels removed by erosion, or zero if there were no seeds. `seed_percent`
measures the surviving seeds as a percentage of the ROI. For example:

```bash
ros2 param set /waterfall_lane_detector seed_erosion_iterations 2
```

Nano performance must be measured on the device.

To replace the classic detector, stop it and add this remap to the command:

```bash
-r limo/cv_package/detection/lane_waterfall_labels/raw:=/limo/cv_package/detection/lane_labels/raw
```
