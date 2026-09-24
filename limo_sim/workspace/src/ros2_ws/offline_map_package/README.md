# Semantic mapping from PointCloud2

'map.launch.py' consumes '/limo/cv_package/visual_ptcld/points' directly: 'x,y,z'
FLOAT32 fields, a 'class_id' UINT8 field, metric coordinates in 'base_link', and
the sensor timestamp. It also consumes the complete SLAM Toolbox laser layer on
'/map': each message replaces the preceding one, including free and unknown
cells. The mapper does not accumulate '/scan' endpoints; scans are consumed by
SLAM Toolbox.

| class_id | Class | Cost |
| --- | --- | --- |
| 1 | exterior road | 0 |
| 2 | yellow lines | 60 |
| 3 | soft obstacle | 30 |
| 4 | boardwalk | 90 |
| 5 | interior road | 0 |
| 6 | interior boardwalk | 90 |

'visual_ptcld' separates 'interior road' as 'original_road & ~exterior_road'
before applying the same ROI. The boundary is the intersection of the road with
the 'soft obstacle' dilation. Interior-road points are voxelized in the image
with 'point_voxel_size' (5x5 pixels in the launch), projected into the BEV with
valid depth, and excluded from boardwalk KD-tree queries. After classification,
each class is voxelized separately in the metric grid using
'pointcloud_voxel_size_m' (2 cm in the launch) and published in the cloud.
During the first KD-tree pass, soft-obstacle points beyond 'blue_radius_max_m'
become interior boardwalk ('class_id=6'). The mapper retains this input ID and
accumulates it in the boardwalk layer with the same cost.

The node uses no images, OpenCV, or BEV projections. It reads the cloud buffer
with NumPy, transforms points at the sensor timestamp, and updates sparse tiles.
Dense maps are generated up to 4 Hz when data, poses, or the laser map change.
Parameters are in 'config/mapping_sim.yaml' and 'config/mapping_real.yaml'.
Each file contains 'launch', 'slam_toolbox', 'semantic_mapper', and
'map_save_gui' sections. They are profiles read by the main launch, not files
to pass directly to ROS with '--params-file'.

## Evidence and correction

Separate log-odds scores, independent from costs, are retained for every cell
and class. An observation increases its own class and reduces incompatible
classes in that cell; scores are saturated so later observations can correct
errors. Multiple points in one cell share one update per cloud. The class with
the greatest evidence above threshold determines the exact cost; confidence
does not multiply 0, 30, 60, or 90. Ties and insufficient confidence produce
unknown.

Exterior and interior road feed one road score, with one normalized update per
cell and cloud. Repeated road observations reduce obstacle evidence in the same
cell. When road prevails, the combined map and all three layers become 0; a
later obstacle observation can prevail again. Points represent classified
surfaces: cells are not cleared along 2D rays or over the full field of view.
Invalid or never-observed points provide no negative evidence. An old isolated
classification remains until it is observed again or moved by pose correction;
no indiscriminate time decay is applied.

## Running on Foxy

Inside the 'limo_sim' container, after building and sourcing
'/workspace/install/setup.bash':

~~~bash
ros2 launch offline_map_package map_sim.launch.py
~~~

The wrapper selects 'mapping_sim.yaml' and includes 'map.launch.py'. It starts
SLAM Toolbox, semantic mapping, RViz, and the save-map GUI with simulated time.
The simulation profile includes computer vision; 'visual_ptcld' contains
exterior-road points with 'class_id=1'.

With SLAM already running:

~~~bash
ros2 launch offline_map_package map.launch.py start_slam:=false
~~~

Add 'mode:=backend' for replay or headless use. On the physical robot, with
sensors and EKF already active:

~~~bash
ros2 launch offline_map_package map_real.launch.py
~~~

This uses 'mapping_real.yaml' and forces 'mode:=backend': SLAM and the mapper
run with real time and 'base_link', without a robot-side window. All shipped
mapping profiles (real and simulation) have 'start_cv: false'. Start CV separately
with 'ros2 launch cv_package cv_real.launch.py' on the robot, or
'ros2 launch cv_package cv_sim.launch.py' in simulation. Explicit 'start_cv:=true'
is still available when desired. On a PC, start only RViz and Save Map, connected
to robot topics and service:

~~~bash
ros2 launch offline_map_package desktop_offline.launch.py
~~~

Saved files remain on the LIMO. The container needs PC display and robot-network
access. Use the same 'ROS_DOMAIN_ID' on both machines, normally '0', and
'ROS_LOCALHOST_ONLY=0'.

Before the first launch, update this package in both workspaces:

~~~bash
source /opt/ros/foxy/setup.bash
colcon build --symlink-install --packages-select offline_map_package
source install/setup.bash
~~~

The workspace mounted in the PC Docker container can differ from this checkout;
make sure it contains the updated launch files. Press 'Ctrl-C' in the respective
terminals to stop mapping and windows. 'map.launch.py' without arguments keeps
the simulation profile. To use a custom file:

~~~bash
ros2 launch offline_map_package map.launch.py config_file:=/path/to/mapping.yaml
~~~

The CLI overrides 'start_slam', 'start_mapper', 'start_rviz', 'start_gui',
'use_sim_time', 'rviz_config', 'fixed_frame', 'pose_source', 'trajectory_id',
'resolution', and 'save_directory' remain available. An empty value uses YAML;
'resolution' overrides both SLAM and mapper resolution, while 'use_sim_time'
overrides every node. 'mode:=backend' disables GUIs; 'mode:=desktop' disables
SLAM and mapper. Relative RViz paths refer to 'limo_rviz/config'.

The node publishes no TF. A missing TF is awaited for 'tf_wait_sec', then the
cloud is discarded; it does not fall back to the latest pose. Duplicate or
out-of-order timestamps are ignored. Call 'reset_map' before rewinding a bag.
At save time, 'save_median_kernel: 3' removes isolated black boardwalk cells
with a 3x3 median; the live map is unchanged.

Outputs have exactly the geometry of the latest valid '/map'. CV points outside
that view remain in tiles and reappear if the map expands. Updates with unchanged
geometry still replace contents: obstacles cleared by SLAM disappear from the
combined map and the next saved map. Before the first '/map', CV is accumulated
but publishing and saving wait. 'reset_map' clears only CV evidence; the laser
layer remains from SLAM.

## Cartographer and loop closure

'pose_source:=tf' accumulates in the 'map' frame and corrects classes in
re-observed cells, but does not retroactively correct historic CV geometry after
loop closure. The laser layer follows the SLAM-updated map. Applying one
'map -> odom' transform to history cannot resolve differing corrections along
trajectory segments.

With external Cartographer and 'cartographer_ros_msgs' installed:

~~~bash
ros2 launch offline_map_package map.launch.py start_slam:=false pose_source:=cartographer
~~~

Cartographer must provide '/submap_list', TF, and '/map' from its occupancy-grid
node. Configure 'trajectory_id', 'map_frame', 'odom_frame', and 'submap_topic'
for the installation; 'odom_frame' must be continuous. The official message
publishes submap poses and IDs: [Cartographer API](https://google-cartographer-ros.readthedocs.io/en/latest/ros_api.html).

The mapper associates observations with the most recent non-frozen submap of
the selected trajectory and retains evidence in local coordinates. It resolves
cloud and submap poses at the same global-frame instant using
'lookup_transform_full'. Updated submap poses rebuild the global view from local
tiles; former positions are not imprinted. In overlaps, the most recently
observed cell wins. This semantic layer does not alter Cartographer scan matching
or use costs as lidar measurements. Final export should wait for poses after
the final optimization.

## Topics and saving

Prefix: '/limo/map_package/offline/map/'.

- 'turquoise_map' (yellow lines), 'white_map' (soft obstacle), and
  'boardwalk_map': selected-class cost, 0 in differently classified cells, and
  -1 in unknown or uncertain cells.
- 'combined_grid': live laser-plus-CV map; laser obstacles at 100 take
  precedence, then observed semantic costs. Other cells retain the laser value,
  including -1 when unknown.

All are 'nav_msgs/OccupancyGrid' with reliable/transient-local QoS. Their
content is a **semantic cost**, not physical occupancy probability. The former
binary 'cv_map' and 'street_map' topics are not produced.

~~~bash
ros2 service call /limo/map_package/offline/map_saver/save_map std_srvs/srv/Trigger '{}'
ros2 service call /limo/map_package/offline/reset_map std_srvs/srv/Trigger '{}'
~~~

Saving requires a valid laser map on 'reference_map_topic' (default '/map') and
creates three pairs in '/workspace/ros2_maps/semantic' (or 'save_directory'):

- 'limo_map_laser.pgm/.yaml': a raw copy of the SLAM laser layer, preserving
  occupancy values and unknown cells.
- 'limo_map_cv_obstacle.pgm/.yaml': a binary 'trinary' CV-layer map after the
  save filter. Costs 0..39 are free, 40..95 occupied, and other values unknown.
  Unobserved CV cells (-1) become free only if laser is exactly 0.
- 'limo_map_complete.pgm/.yaml': raw layer fusion. Priority is laser at 100,
  observed CV class, then laser state.

Change the 'limo_map' basename with 'save_map_name'. Raw PGM stores values
0..100 directly and 255 for unknown, so Nav2 reconstructs exact costs and
unknown cells. YAML filenames are unchanged and online launches load them
normally. A later save replaces all three pairs; no '.npz' file is produced.

## Verification and resources

~~~bash
colcon build --symlink-install --packages-select offline_map_package
python3 -m pytest src/ros2_ws/offline_map_package/test/test_semantic_grid.py src/ros2_ws/offline_map_package/test/test_semantic_mapper.py
python3 -m pytest src/ros2_ws/offline_map_package/test/test_mapping_launch.py
~~~

'max_cells' limits allocated tiles (about 20 bytes per cell plus overhead);
'max_output_cells' limits the dense view. Exceeding a limit reports an error
without clearing the map. Actual efficiency must be measured on Jetson Nano with
the real camera stream.

### Real-camera responsiveness

`map_real.launch.py` uses `mapping_real.yaml`, whose `log_odds_limit: 1.0`
reduces evidence memory for low camera frame rates. With the configured hit,
miss and threshold, a saturated cell changes class after two consecutive
pure-class observations instead of five. One contradictory observation retains
its previous class. Mixed-class cells can require more observations. This trades
some temporal stability for responsiveness; it does not increase camera FPS or
fill cells without observations. The simulation profile keeps its original tuning.

`cv_real.yaml` now selects the waterfall detector, with seed/gradient thresholds
and optional seed erosion/barrier dilation. Its label output is remapped to the
common topic consumed by `visual_ptcld`, preserving the semantic cloud topic.
The simulation profile retains the HSV detector. `launch.cv_config` in the mapping
profile selects `cv_real.yaml` or `cv_sim.yaml` independently of the clock;
`cv_config:=/path/to/profile.yaml` can override it when `start_cv:=true`.


The real CV profile preserves border road (1), interior road (5), boardwalk (4),
and interior boardwalk (6). Road variants map to cost 0; boardwalk variants to 90. Yellow-line
and remaining background points are discarded, not relabeled. The semantic
mapper still accepts the full class set for the unchanged simulation CV profile.


For slow manual mapping, `mapping_real.yaml` uses `minimum_travel_distance: 0.05`
metres and `minimum_travel_heading: 0.05` radians (about 3 degrees). SLAM publishes
its grid every 0.5 seconds, while the semantic mapper publishes at 4 Hz. These
settings increase accepted scan density and map publication work compared with
the 0.5 m / 0.5 rad SLAM defaults and the previous 2 s publication interval.
Restart the mapping backend to apply them; restarting starts a new mapping session.
They do not change camera FPS or the semantic evidence thresholds.
