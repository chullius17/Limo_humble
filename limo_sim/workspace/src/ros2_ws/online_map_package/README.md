# AMCL localization with laser and semantic point clouds

The 'config/mapping_sim.yaml' and 'config/mapping_real.yaml' profiles configure
startup, map server, and AMCL. 'online_map.launch.py' contains shared logic and
uses the optimized CV pipeline and three maps exported by the offline mapper in
'ros2_maps/semantic':

| File | Topic | Use |
| --- | --- | --- |
| 'limo_map_laser.yaml' | '/limo/map_package/online/maps/laser_map' | AMCL laser comparison |
| 'limo_map_cv_obstacle.yaml' | '/limo/map_package/online/maps/cv_obstacle' | AMCL CV comparison |
| 'limo_map_complete.yaml' | '/map' | Complete source for planner and global costmap |

'traj_package' can consume the '/map' published by the online profile. Its
'BorderFollowLayer' builds '/global_costmap/costmap', displayed in RViz as
**Global Costmap (Inflation)** with the 'costmap' color scheme and alpha 0.35,
as in the 'humble-navigation' branch.

On Foxy, 'always_send_full_costmap: true' avoids an RViz
'OccupancyGridUpdate' bug: the first row is repeated over every row, creating
stripes after the first correct map. The complete costmap is published at 1 Hz;
the palette does not cause the issue.

The '/limo/cv_package/visual_ptcld/points' cloud contains 'x,y,z' FLOAT32 and
'class_id' UINT8. AMCL selects **yellow lines=2, boardwalk=4, interior
boardwalk=6** as obstacles and **exterior road=1, interior road=5** as road.
**Soft obstacle=3**, unknown, and all other classes do not vote. No image or
local BEV grid enters AMCL.

For every laser update:

1. AMCL updates particles with the laser model using only the laser map.
2. It selects the closest cloud timestamp within 'cv_sync_tolerance'.
3. It transforms that cloud into the base frame at the laser timestamp, using
   'odom' as fixed frame and without depending on AMCL's estimated global pose.
4. After class filtering, it aggregates road and obstacle points separately in
   **XY** voxels of 'cv_voxel_size' (default 0.075 m). Each group contributes
   its centroid with weight 1. The 2 cm 'visual_ptcld' reduction remains the
   first stage. A road and obstacle sharing a voxel retain distinct votes.
5. It transforms this set with **every candidate pose** and computes the
   fraction of mismatching votes: obstacles on occupied cells and road on cells
   whose corresponding CV-map cost is exactly 0. Mismatch is normalized over
   all road plus obstacle votes, with no group-specific additional weights.
6. It applies the Humble-reference formula and normalizes before resampling:
   'w_final ∝ w_laser^laser_weight_factor × exp(-cv_weight_factor × cv_sad_gain × mismatch)'.

Mismatch keeps Humble's positive-SAD rule: free, unknown, and out-of-map cells
disagree with an observed obstacle. The negative-SAD component uses only
explicitly observed road: any cell other than cost 0, unknown, or out of map
disagrees. Missing points do not prove free space. No separate street map is
used, and soft obstacles do not participate.

A missing, temporally distant, malformed, TF-unavailable cloud, or fewer than
'cv_min_points' voxels (default 5), prevents the CV update. As in Humble's
online launch, 'cv_sync_tolerance' is 0.20 s. A cloud may be reused within that
limit **only if the lidar has no valid returns or 'laser_weight_factor' is
zero**. With valid lidar and a positive weight, every CV frame is used only
once. Reuse never renews the original-cloud time limit; TF odometry compensates
motion and configured CV gain is retained.

Current profiles use 'laser_weight_factor: 1.0' and 'cv_weight_factor: 1.0'.
A zero laser weight skips the laser model even if CV is absent. '/scan' is still
required to trigger updates and assess validity. 'cv_enabled:=false' or
'cv_weight_factor:=0.0' disables CV fusion. CV parameters are read at node
configuration, so restart after changing them.

Before fusion, 'cv_quality_gate_enabled: true' evaluates the CV-only
probability on current particle poses. A nearly uniform comparison is rejected
when KL divergence from uniform is below 'cv_min_information: 0.02' nats.
CV hypotheses are also rejected if XY spread exceeds
'cv_max_position_stddev: 0.5' m on the most dispersed axis or circular yaw
spread exceeds 'cv_max_yaw_stddev: 0.5' rad. Road remains active. Rejection
happens before any weight change, retaining a laser update when active. The
'CV update rejected' log reports the reason and measures. These values describe
CV ambiguity over available particles, not sensor covariance; validate the
initial thresholds on the circuit.

## Launching

After building 'nav2_amcl', 'limo_rviz', 'cv_package', and
'online_map_package', and sourcing the workspace:

~~~bash
cd /workspace
colcon build --packages-select nav2_amcl limo_rviz cv_package \
  online_map_package --symlink-install
source install/setup.bash
ros2 launch online_map_package online_map_sim.launch.py
~~~

With sensors, odometry, and CV already active on the LIMO:

~~~bash
ros2 launch online_map_package online_map_real.launch.py
~~~

On a PC, to open only RViz connected to LIMO topics:

~~~bash
ros2 launch online_map_package desktop_online.launch.py
~~~

Temporary simulation map or voxel overrides:

~~~bash
ros2 launch online_map_package online_map_sim.launch.py \
  map_directory:=/workspace/ros2_maps/semantic map_name:=limo_map \
  cv_voxel_size:=0.10 cv_min_points:=5.0
~~~

Change persistent values in the two YAML profiles; command-line arguments are
for temporary trials. The real profile does not restart CV or open windows.
'desktop_online.launch.py' uses that profile in desktop mode and launches RViz
only. Planner and controller are started independently;
`user_package/limo_app_sim.launch.py` and
`user_package/limo_app_real.launch.py` orchestrate all three packages for the
complete simulation and physical-robot stacks, respectively.

AMCL publishes 'map -> odom'; provide an initial pose through RViz or AMCL's
global-localization service. Do not start SLAM concurrently if it publishes the
same TF. The launch starts localization, map server, CV, and optional RViz. The
temporal semantic pipeline directly uses 'local_ctrl_map', with 'local_grid' as
the costmap support module and 'semantic_memory' as temporal memory.

'limo_controller' combines this grid with laser data: 'ObstacleLayer' marks and
clears live obstacles from '/scan', while 'StaticLayer' inserts graduated costs
from '/limo/map_package/online/local_costmap'. The final 'InflationLayer' works
on the combined result. The MPC controller evaluates local trajectories with
the full footprint; 'limo_dwb_critics' rejects in-place rotations, lateral
motion, and unrealizable curvature. LIMO Ackermann parameters are: footprint
0.322 x 0.220 m, wheelbase 0.20 m, minimum turning radius 0.462 m, maximum
speed 0.50 m/s, and acceleration limits 1.3 m/s² and 3.4 rad/s².

The MPC controller publishes '/cmd_vel_autonomy', while
'teleop_twist_keyboard' publishes '/cmd_vel_teleop'. The internal
'limo_controller' mux gives teleoperation priority 100 and autonomy priority
10, then publishes the selected command on '/cmd_vel'. After 0.5 s without new
key presses, teleoperation expires and autonomy resumes without interrupting
the 'FollowPath' action.

'local_ctrl_map' is started by the online profile and publishes the working
limits on '/limo/map_package/online/local_ctrl_map/markers' in 'base_link': a
persistent green 2.50 x 2.66 m rectangle and a yellow ROI trapezoid, 1.95 m
high and 0.60 to 2.66 m wide. Its large base has the rectangle width and meets
the front edge, 2.50 m from 'base_link'. **Local Map Regions** is already
enabled in 'online_map.rviz'. In desktop mode the node stays on the backend and
RViz displays the topic received from the LIMO.

A third red contour shows the inner trapezoid: its near edge and oblique sides
are inset by 20 cm, perpendicular to each edge
('inner_trapezoid_inset: 0.20'). Its wide base remains aligned with the yellow
trapezoid and green rectangle front edge. It is a visual 'base_link' reference;
point memory uses the yellow trapezoid. The three contour vertices and messages
are precomputed at initialization; publishing only updates timestamps.

'local_ctrl_map' fuses the live CV cloud and reprojected point memory, then
publishes '/limo/map_package/online/local_costmap' directly as a
'nav_msgs/OccupancyGrid'. The grid matches the green rectangle: 2.50 x 2.66 m,
origin '(0, -1.33)' in 'base_link', and exactly 125 x 133 cells at 2 cm
resolution. All live classes 1--6 contribute: road (1/5) costs 0, yellow line
(2) 60, soft obstacle (3) 30, and boardwalk (4/6) 90. Where multiple points
fall in one cell, the greatest cost is retained. The source grid is not
pre-inflated ('inflation_radius: 0.0'); Nav2's local-costmap 'InflationLayer'
is the only inflation after laser fusion. Confidence controls persistence, not
the fixed class costs.

At 10 Hz, all points used to build the grid are also published as
'sensor_msgs/PointCloud2' on
'/limo/map_package/online/local_ctrl_map/points': all recent-live points and
classes, plus memory for classes 2/4 reprojected into the current frame and
capped at 300 entries. The cloud is in 'base_link' and has 'x', 'y', 'z', and
'class_id' fields. It is available for debugging; PointCloud2 displays exist in
the online RViz configuration but are disabled by default.

At every CV frame, all classes are retained as the live source for 0.50 s. In
parallel, points **inside the yellow trapezoid and outside the red one** are
immediately transformed to 'odom' at the cloud timestamp and added to memory.
If the TF is not yet available, the frame waits in a queue capped at
'max_pending_clouds: 10'. The node retries every 20 ms, without blocking other
callbacks, for at most 'tf_wait_timeout_sec: 0.20' ROS seconds from receipt.
Frames are processed in timestamp order; expired or over-capacity frames are
discarded with a warning. A clock reset also empties the queue.

Persistent points are reprojected at 10 Hz even without a new CV frame. They
are removed when entering the red trapezoid, leaving the rectangle, or falling
below 'minimum_confidence' (0.30). They start at confidence 1. Decay is
integrated at each reprojection from their region and current '/cmd_vel'. The
larger linear/angular speed ratio scales decay from zero to one: below 0.01 m/s
and 0.02 rad/s it does not decay; it reaches maximum at 0.50 m/s or 1.00 rad/s.
Outside yellow but inside green, maximum decay is
'confidence_decay_per_sec' (0.10/s); inside yellow and outside red, it is
multiplied by 'yellow_decay_multiplier: 3.0'. In red the point is removed
immediately. At threshold 0.30 and maximum ratio, an unobserved point lasts
about 12 s in the green-only region and 4 s in the yellow region; it does not
expire while stationary. 'cmd_vel_timeout_sec: 0.0' keeps the last command with
no timeout. Confidence describes memory, not classifier certainty.

A separate 3 cm class-wise voxel filter avoids duplicates. The
'maximum_points: 300' limit is shared by both persistent classes: lowest
confidence points are removed first; ties are distributed spatially. Configure
parameters in the 'local_ctrl_map' profile section; override the cap with
'local_map_maximum_points:=300' and restart to apply it. RViz displays the
source grid in **Local Semantic Costmap**, before laser fusion and Nav2
inflation. The controller launch no longer starts a separate converter because
this grid is already ready for the local costmap.

The 'CV cloud fusion' logs show time difference, input points, voxels,
particles, and the effective voxel-times-particle comparison count.

## Verification

'nav2_amcl/test/test_cv_cloud.cpp' verifies class selection and fusion,
voxelization, layout/endian handling, transforms, and weights.
'test_cv_sync.cpp' verifies temporal compensation and behavior without usable
data. Python tests in 'test/test_localization_launch.py' verify maps, topics,
and launch parameters. Tune and assess convergence with real sensors or a
rosbag.
