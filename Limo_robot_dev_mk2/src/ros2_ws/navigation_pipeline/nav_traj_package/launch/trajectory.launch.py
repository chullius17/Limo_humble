"""Launch the LIMO Nav2 global planner with SMAC Hybrid-A*."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    """Create the ROS 2 Humble global-planner launch description."""
    package_share = get_package_share_directory('nav_traj_package')
    default_params_file = os.path.join(
        package_share,
        'config',
        'smac_hybrid_params.yaml',
    )

    params_file = LaunchConfiguration('params_file')
    map_topic = LaunchConfiguration('map_topic')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')

    # Rewriting the placeholder in the YAML keeps one configuration usable on
    # both the real LIMO and Gazebo without relying on launch substitutions
    # inside the parameter file itself.
    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key='',
        param_rewrites={
            'use_sim_time': use_sim_time,
            (
                'global_costmap.global_costmap.ros__parameters.'
                'static_layer.map_topic'
            ): map_topic,
        },
        convert_types=True,
    )

    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[configured_params],
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_global_planner',
        output='screen',
        parameters=[{
            'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
            'autostart': ParameterValue(autostart, value_type=bool),
            'node_names': ['planner_server'],
        }],
    )

    # RViz publishes selected poses on /goal_pose, while planner_server exposes
    # an action. The bridge is included to simplify graphical goal selection.
    rviz_goal_bridge = Node(
        package='nav_traj_package',
        executable='rviz_goal_bridge',
        name='rviz_goal_bridge',
        output='screen',
        parameters=[{
            'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
            'planner_id': 'GridBased',
            'costmap_topic': '/global_costmap/costmap',
            'adjusted_goal_topic': '/adjusted_goal_pose',
            # Search only a small neighborhood and cap SMAC requests to keep
            # interactive goal correction sustainable on the Jetson Nano.
            'enable_goal_adjustment': True,
            'position_search_radius': 0.30,
            'position_search_step': 0.05,
            'angle_search_step_deg': 22.5,
            'max_planning_attempts': 16,
            # Match the physical, unpadded footprint used by SMAC.
            'footprint_length': 0.32,
            'footprint_width': 0.20,
            # OccupancyGrid value 99 represents Nav2's inscribed cost.
            'collision_cost_threshold': 99,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params_file,
            description='Absolute path to the SMAC and global-costmap parameters.',
        ),
        DeclareLaunchArgument(
            'map_topic',
            default_value=(
                '/limo/nav_map_package/online/nav_map/combined_grid'
            ),
            description='OccupancyGrid used by the global planner.',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use the Gazebo/rosbag clock instead of the system clock.',
        ),
        DeclareLaunchArgument(
            'autostart',
            default_value='true',
            description='Configure and activate the Nav2 planner automatically.',
        ),
        planner_server,
        lifecycle_manager,
        rviz_goal_bridge,
    ])
