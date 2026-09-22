# LIMO sampled Ackermann MPC on Nav2 Foxy

`limo_dwb_critics::AckermannMPCController` is a Nav2 controller derived from
`dwb_core::DWBLocalPlanner`. It reuses DWB plan transformation, costmap locking,
critics, lifecycle and trajectory visualization, and replaces the command search
with finite-sample nonlinear shooting MPC. It does **not** implement MPPI's
soft-weighted update or a continuous constrained optimization solver.

Each cycle:

1. Read the measured pose. Initialize speed and bicycle steering from odometry
   on a fresh start, then update the actuator-state estimate according to
   `MPC.feedback`.
2. Shift the previous winning target sequence by the elapsed control steps.
3. Sample piecewise target velocity/steering sequences around that solution,
   plus broad exploration, constant-curvature seeds and a braking sequence.
   Perturbations repeat across cycles to reduce sampling-induced command noise;
   their nominal sequence still shifts and changes with every solution.
4. Roll out the constrained bicycle model over the full horizon. Score these
   trajectories with DWB critics plus normalized acceleration/steering effort.
5. Publish **only the first reachable command** of the winning sequence. Repeat
   from the measured pose at the next control cycle.

The default budget is 768 sequences, 50 steps of 0.05 s, and five control
segments. This is a finite search, not a guarantee of a globally optimal or
recursively feasible solution. If every candidate is invalid, the controller
raises DWB's normal no-legal-trajectories exception; it never reuses a stale
command as a fallback. A new path, deactivation, failed search, or a control gap
longer than three periods discards the warm start.

The control period/watchdog uses a monotonic wall clock, matching Foxy's
controller loop. Repeated Gazebo `/clock` stamps (for example 10 Hz clock
publication with 20 Hz control) do not erase the command ramp. Backward ROS
clock jumps still reset the prediction.

## Model and parameters

`FollowPath.MPC.model_dt` must equal `1 / controller_frequency`. The prediction
horizon is `MPC.time_steps * MPC.model_dt`; `sim_time`, `vx_samples` and
`vtheta_samples` apply only to the standard DWB fallback. Parameters are read at
configuration time; reconfigure the controller after changing them.

`MPC.feedback: OPEN_LOOP` continues the velocity/steering ramp from the previous
issued command, like the command-state convention of Humble's OPEN_LOOP
velocity smoother. This avoids a startup lock where each cycle commands only
`v_odom + acceleration * dt` while actuator lag/friction keeps odometry near
zero. Position feedback, collision checks and the progress checker remain
active. The rollout assumes commanded speed/steering are tracked; it does not
estimate actuator lag. `CLOSED_LOOP` instead starts each rollout at measured
velocity and is appropriate when low-level tracking and odometry are reliable.
Commands are never retained across failed searches, new paths or stale cycles.

The model enforces the minimum radius, longitudinal acceleration/deceleration,
steering rate and yaw-rate limits at **every predicted step**. The smaller of
`acc_lim_theta` and `-decel_lim_theta` is used in both angular directions.
Reversing first brakes to zero. Observed overspeed is reduced at the configured
rates instead of being clipped instantaneously.

`AckermannKinematics.min_turning_radius` is shared by the model and the existing
critic. The critic remains a final check on the command sent to Nav2.
`MPC.max_steering_rate` limits the equivalent bicycle angle, not the inner wheel
angle. The YAML uses 0.5 rad/s (0.025 rad per 20 Hz command), a tuning estimate,
not a measured LIMO
actuator specification. `MPC.wheelbase: 0.20` and `MPC.rear_axle_to_base: 0.10`
match the Ackermann URDF; set the latter to zero if the navigation base frame
is actually located on the rear axle.

### Gazebo model and command conversion

`limo_app.launch.py profile:=sim` passes `robot_model:=sim` to the controller
launch. This overrides the bicycle wheelbase to 0.24 m and rear-axle offset to
0.12 m, matching `limo_car/gazebo/ackermann.xacro`. Its minimum radius is 0.55 m,
leaving margin for the wheel collision-center offsets when enforcing the
simulated inner steering joint's 30-degree limit.
`robot_model:=real` preserves the physical/custom geometry from the YAML.
When launching `control.launch.py` directly for Gazebo, specify
`robot_model:=sim` explicitly; `use_sim_time` alone does not select geometry.

Foxy's `gazebo_ros_ackermann_drive` interprets `Twist.angular.z` as an equivalent
steering angle and multiplies it by the sign of longitudinal speed internally.
It is **not** the same command interface as Nav2 or the physical LIMO driver.
The simulation launch starts `limo_car/gazebo_twist_adapter.py`, converting
standard `/cmd_vel` to `/cmd_vel_gazebo` using
`angular.z = atan(wheelbase * yaw_rate / abs(speed))`, with steering saturation
and zero-speed handling. A wall-clock watchdog stops the plugin after 0.5 s
without commands. The physical robot continues receiving standard `/cmd_vel`.

Source: [Gazebo Foxy Ackermann drive implementation](https://github.com/ros-simulation/gazebo_ros_pkgs/blob/foxy/gazebo_plugins/src/gazebo_ros_ackermann_drive.cpp).
The simulator must be restarted/robot respawned after updating its command
topic; restarting only the navigation app cannot change an already spawned
Gazebo plugin. Do not publish directly to `/cmd_vel_gazebo` using standard
yaw-rate commands.

The model assumes rear-axle longitudinal speed and no slip. It integrates the
rear-axle arc and translates the predicted pose to `base_link`. Output `Twist`
contains longitudinal speed and yaw rate, with `linear.y = 0`, matching the
driver command interface. It does not use observed lateral velocity. Verify
odometry and actuator response against this convention: the current LIMO driver
computes Ackermann odometry using additional steering projections, so agreement
with the physical robot is **not established by these software tests**.
Below `MPC.steering_feedback_min_velocity` (0.05 m/s), yaw-rate/velocity is too
noisy to estimate steering: retain the previous command's steering, or assume
centered wheels on a fresh start, as requested by the driver's zero-speed command. This
does not model steering feedback, actuator lag, wheel slip or moving obstacles.

`MPC.acceleration_weight` and `MPC.steering_weight` penalize squared normalized
input increments, averaged over the horizon. They are additional regularizers,
not a translation of MPPI's `gamma` or `temperature`.

`MPC.steering_command_weight: 0.2` additionally penalizes the squared first
steering increment normalized by `max_steering_rate * model_dt`.
`MPC.steering_rate_change_weight: 0.1` penalizes its difference from the previous
issued steering increment, in the same normalized units. These immediate
costs are **not divided by the horizon length**, so increasing the prediction
horizon does not dilute command continuity. Steering-rate history resets with
the warm start. Both are soft costs: collision rejection can still force a
change or braking, and no command is filtered after trajectory evaluation.
The YAML reduces local steering sampling deviation to 0.10 rad while retaining
full-range seeds and broad exploration. The MPPI-derived critic weights remain
unchanged.

## Cost correspondence with Humble MPPI

The reference is `origin/humble-navigation` at
`Limo_robot_dev_mk2/src/ros2_ws/navigation_pipeline/nav_limo_controller/config/mppi_control_params.yaml`.
The two new DWB critics deliberately replace raw DWB map-grid costs so the
Humble weights have comparable units. All cost powers are fixed at 1, as in
that reference profile.

| Humble MPPI term | Foxy configuration | Behavior |
| --- | --- | --- |
| CostCritic, weight 3 | `MppiObstacle.scale: 3` | Mean center cost / 254; critical cost 300; ordinary repulsion off within 0.5 m of the real goal |
| GoalCritic, weight 5 | `MppiPath.GoalCritic` | Mean distance in meters, enabled within 1 m of the goal |
| GoalAngleCritic, weight 3 | `MppiPath.GoalAngleCritic` | Mean wrapped yaw error in radians, within 0.5 m |
| PathAlignCritic, weight 10 | `MppiPath.PathAlignCritic` | Mean path-pose error, with orientations; off within 0.5 m or if local path occupancy exceeds 15% |
| PathFollowCritic, weight 5 | `MppiPath.PathFollowCritic` | Terminal distance to a metric lookahead target; off within 1 m of the goal |
| PathAngleCritic, weight 2 | `MppiPath.PathAngleCritic` | Terminal heading toward the lookahead target, enabled for initial errors above 1 rad and outside 0.5 m; forward/reverse symmetric |
| ConstraintCritic / collision_cost | Model constraints and exceptions | Infeasible/colliding candidates are rejected, not assigned a finite penalty |

This is an adaptation, not a port of MPPI critics. DWB has no batch-wide
furthest-trajectory index: the MPPI point offsets are replaced by
`lookahead_distance: 1.25` m. Alignment uses monotone, distance-limited nearest
path poses. Goal-distance gates use the actual goal rather than a clipped local
path endpoint. Every predicted pose is collision checked; there is no
`trajectory_point_step` skipping. Both footprint contour and center are
checked, while footprint interiors are not exhaustively rasterized.

Inflation now matches Humble (`0.30` m, scaling factor `8.0`), retaining this
branch's physical footprint and semantic lethal threshold 86. The finite set
of predictions uses discrete collision checks, not continuous swept-volume
checking or a terminal invariant-set constraint. Equivalent weights do not
guarantee identical behavior; simulation and hardware tuning remain necessary.

## Build and test

Inside the Foxy workspace:

```bash
source /opt/ros/foxy/setup.bash
colcon build --symlink-install --packages-select limo_dwb_critics limo_controller
source install/setup.bash
colcon test --packages-select limo_dwb_critics --event-handlers console_direct+
colcon test-result --verbose
```

The existing `limo_controller` launch loads `config/dwb_params.yaml`, which now
selects this plugin. For a baseline comparison, set `FollowPath.plugin` back to
`dwb_core::DWBLocalPlanner`; the original generator and its parameters remain.

Before physical operation, validate closed-loop simulation on a straight path,
tight curve, slalom, obstruction and goal approach. Measure lateral error,
clearance, steering increments and controller execution time. At 20 Hz the
whole control cycle must fit within 50 ms; a model-only benchmark does not
include costmap or footprint scoring. Enable `publish_evaluation` temporarily
for DWB candidate diagnostics (`MpcEffort` is the additional cost).
