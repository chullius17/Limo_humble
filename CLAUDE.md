# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Autonomous-driving stack for the AgileX LIMO (Ackermann conversion), targeting **ROS 2 Foxy** (Ubuntu 20.04) both in Gazebo 11 simulation and on the real robot (Jetson Nano). Despite the repo name `Limo_humble` and the `foxy` branch, everything under `limo_sim/` is Foxy; the Humble workspace was the reference port source (see `limo_sim/workspace/reports/report_ros2_foxy_gazebo.txt`).

The colcon workspace is `limo_sim/workspace` (mounted as `/workspace` inside the dev container).

## Development environment

All build/run happens inside the `limo_sim` Docker image (ROS 2 Foxy desktop + gazebo_ros_pkgs + robot_localization + nav2 map_server/lifecycle_manager + OpenCV + `onnxruntime==1.10.0`):

```bash
cd limo_sim/ros2_foxy_dev
./build.sh                # build the limo_sim image
./start_limo_docker.sh    # create-or-start the persistent limo_sim container (recreates it if the image changed)
./run.sh                  # alternative: throwaway --rm container
docker exec -it limo_sim bash   # extra terminal in the running container
xhost +local:docker             # if RViz/Gazebo cannot open a window
```

## Build and run

```bash
cd /workspace
source /opt/ros/foxy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Build a subset when touching drivers (both docs in `limo_sim/*.md` use this form):

```bash
colcon build --packages-select limo_base custom_start --symlink-install --cmake-clean-cache
```

`limo_sim/workspace/src/.gitignore` tracks `COLCON_IGNORE` markers for `ydlidar_ros2_driver` and `ros2_astra_camera`; those hardware driver packages are skipped in sim-only builds and must be un-ignored for real-robot builds.

Bring-up:

```bash
ros2 launch custom_start limo_circuit.launch.py   # Gazebo circuit world + Ackermann robot + EKF
ros2 launch custom_start limo_real.launch.py      # real robot: limo_base (motion_mode:=1), YDLidar, DaBai U3 camera, EKF
ros2 launch cv_package cv.launch.py               # classic-CV perception (or: ai_cv_package ai_cv.launch.py)
ros2 launch map_package map.launch.py             # costmap + per-color mappers + display + saver
ros2 launch user_package limo_app.launch.py       # RViz + map server + planning + controller + GUI
ros2 launch user_package ai_limo_app.launch.py    # same, wired to the ai_* packages
```

Tests are the stock ament linters only (`test_flake8.py`, `test_pep257.py`, `test_copyright.py` in each Python package):

```bash
colcon test --packages-select cv_package
colcon test-result --verbose
python3 -m pytest src/ros2_ws/simple_pipeline/base_cv_block/cv_package/test/test_flake8.py   # single test
```

## Architecture

Two parallel perception pipelines feed one shared planning/control/UI stack. They are **duplicated code**, not a shared library: every module in `ai_block/` is an `ai_`-prefixed near-copy of its `base_cv_block/` counterpart, with `ai_`-prefixed packages, nodes and topics. A change in one usually needs mirroring in the other.

`limo_sim/workspace/src/ros2_ws/simple_pipeline/`:

- `base_cv_block/` — classic OpenCV pipeline: `cv_package` (color lane detection → curb/boundary extraction + depth correction → BEV → color classification), `map_package` (costmap + one `mapper` node per color channel + display + PGM/YAML saver), `simple_mapping` (newer single-node `metric_bev` + temporally filtered `mapper` alternative to `map_package`), `traj_package` (route networks, combinator, A*), `limo_rviz` (static TFs, nav2 map_server, RViz).
- `ai_block/` — same chain driven by a YOLO segmentation model: `ai_cv_package` (ONNX Runtime `lane_detector` loading `best.onnx` from the package share), `ai_map_package`, `ai_traj_package`, `ai_limo_rviz`.
- `limo_controller` — `controller_trajectory`: pure-pursuit-style `FollowSequencePlan` action server publishing `/cmd_vel` (helper in `limo_controller/utils/pursuit_pt.py`).
- `user_package` — `user_server` (CLI/service front end), `gui` (PyQt5), `mission_visualizer` (renders map + goals + paths to an image topic), `control_viz` (plots mission telemetry to JPGs).
- `limo_interfaces` — `GetSequencePlan.srv`, `MissionCommand.srv`, `GenerateControlPlot.srv`, `Mission.action`, `FollowSequencePlan.action`.

Data flow (base pipeline; AI pipeline is the same with `ai_` names):

```
/rgb/image_raw + /depth_camera/depth/image_raw
  → cv_package: lane labels → boundaries (PointCloud2) → BEV → classification/output/raw
  → map_package: costmap → mapper (per color, in `odom`) → map_display / map_saver
  → traj_package: routes_builder (CENTER_ROAD + OPEN masks) → route_combinator → astar_server (/plan_sequence_path)
  → traj_package/coordinator: Mission action server → FollowSequencePlan → limo_controller → /cmd_vel
  → user_package: gui / user_server issue goals, mission_visualizer + control_viz render results
```

Node-to-node topics are namespaced `/limo/<package>/<node>/<signal>`; mission control and health signals live under `/limo/mission/*` (`enable`, `pause`, `goals`, `paths`, `state`, `diagnostics`, `health/astar`, `health/controller`).

Artifacts are written relative to a project root discovered at runtime by walking parents for a `src/ros2_ws` directory (`find_project_root` in `map_saver.py`, `limo_rviz.launch.py`, `control_viz.py`) — this is why the workspace layout must keep `src/ros2_ws` in place:

- maps: `limo_sim/workspace/ros2_maps/base_pipeline/{base_cv,ai_cv}/limo_map.{pgm,yaml}` (also what `limo_rviz`'s map_server loads)
- control plots: `control_logs/` under the same `{base_cv,ai_cv}` split, selected by the `ai_mode` launch argument threaded through `limo_app.launch.py` → `user.launch.py` → `control_viz`.

Vendored upstream sources live outside `ros2_ws`: `limo_ros2/` (AgileX base driver, `limo_car` Ackermann URDF/Gazebo, `limo_bringup`), `ydlidar_ros2_driver/` + `YDLidar-SDK/`, `ros2_astra_camera/`, plus `custom_start/` (worlds, models, EKF config, bring-up launches).

## Robot-specific gotchas (documented in limo_sim/*.md, in Italian)

- **Ackermann**: the chassis firmware always reports `motion_mode: 2` (Mecanum). `limo_driver` takes a `motion_mode` parameter override (`-1` auto, `0` diff, `1` Ackermann, `2` Mecanum); `limo_real.launch.py` forces `1`. Verify with `ros2 param get /limo_base motion_mode` and the log line `Command motion mode override: 1` — not with `/limo_status`. The AgileX phone app must be disconnected or `control_mode` leaves ROS.
- **DaBai U3 camera**: `astra_camera` needs the Orbbec ARM64 OpenNI2 runtime installed also as `libOpenNI2.so.0`, otherwise the node silently loads `/usr/lib/libOpenNI2.so.0`. Check with `ldd install/astra_camera/lib/astra_camera/astra_camera_node | grep OpenNI`. Use the `dabai_u3.launch.xml` profile with depth `640x400`, and install the udev rules from `ros2_astra_camera/astra_camera/scripts/install.sh`.
- A Gazebo server from another ROS distro on port 11345 causes `Address already in use` / `Conflicting gazebo versions` — kill it before launching.

## Conventions

- Code comments and docstrings are written in English; the standalone reports/docs under `limo_sim/` are in Italian.
- `discarded_scripts/` directories hold superseded implementations that are intentionally kept and not built (no entry points); do not treat them as live code.
- Tunables are ROS parameters declared in the node and set in the launch file, not constants edited in place.
