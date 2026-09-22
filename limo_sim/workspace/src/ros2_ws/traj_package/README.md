# Trajectory planning

`traj_package` contains the Nav2 SMAC planner and `rviz_goal_bridge`, which
receives RViz goals on `/goal_pose`, searches for a feasible solution, and
publishes the path. It can be launched without `limo_controller`:

```bash
ros2 launch traj_package trajectory.launch.py
```

It requires the robot map and transforms. Pass a custom configuration with
`planner_params_file:=/path/to/parameters.yaml`.

## Outputs

| Topic | Type | Contents |
| --- | --- | --- |
| `/limo/planning/path` | `nav_msgs/Path` | Most recent validated path |
| `/limo/planning/status` | `std_msgs/String` | Planning state and failures |

Both topics use reliable, transient-local QoS with depth 1. A consumer started
after planning can therefore receive the most recent path with the same QoS. A
new request first publishes an empty path to invalidate the preceding one; only
a successful search publishes a new non-empty path.

## Execution

The package exposes no control services and does not send `FollowPath` goals.
The former `enable_control` and `auto_start_control` flags were removed.

Execution belongs to `limo_controller`: its `path_executor` node receives the
path and handles START, pause, resume, and cancellation through
`/limo/control/set_active` and `/limo/control/set_enabled`. A new path
interrupts the active one and requires a new START. The GUI also displays the
planning state.

`user_package/limo_app_sim.launch.py` and `limo_app_real.launch.py` compose
mapping, planning, and control for the two runtime profiles.
