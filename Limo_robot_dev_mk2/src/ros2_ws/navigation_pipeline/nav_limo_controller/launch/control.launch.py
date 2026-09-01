"""Launch the Nav2 MPPI controller configured for the physical LIMO."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    """Create the controller, velocity smoother and lifecycle manager."""
    package_share = get_package_share_directory('nav_limo_controller')
    default_params_file = os.path.join(
        package_share,
        'config',
        'mppi_control_params.yaml',
    )

    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    log_level = LaunchConfiguration('log_level')

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key='',
        param_rewrites={'use_sim_time': use_sim_time},
        convert_types=True,
    )

    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[configured_params],
        arguments=['--ros-args', '--log-level', log_level],
        remappings=[('cmd_vel', 'cmd_vel_nav')],
    )

    velocity_smoother = Node(
        package='nav2_velocity_smoother',
        executable='velocity_smoother',
        name='velocity_smoother',
        output='screen',
        parameters=[configured_params],
        arguments=['--ros-args', '--log-level', log_level],
        remappings=[
            ('cmd_vel', 'cmd_vel_nav'),
            ('cmd_vel_smoothed', 'cmd_vel'),
        ],
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_limo_controller',
        output='screen',
        parameters=[{
            'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
            'autostart': ParameterValue(autostart, value_type=bool),
            'node_names': ['controller_server', 'velocity_smoother'],
        }],
        arguments=['--ros-args', '--log-level', log_level],
    )

    local_costmap_converter = Node(
        package='nav_limo_controller',
        executable='local_costmap',
        name='local_costmap_converter',
        output='screen',
        parameters=[{
            'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
            'input_topic': '/limo/nav_map_package/online/local_ptcld',
            'output_topic': '/limo/nav_map_package/online/local_costmap',
            'output_frame': 'base_link',
            'resolution': 0.05,
            'length': 2.46,
            'width': 2.66,
            'minimum_cost': 1.0,
            'minimum_confidence': 0.30,
            'scale_cost_by_confidence': False,
            'statistics_window_cycles': 30,
        }],
    )

    control_gui = Node(
        package='nav_limo_controller',
        executable='control_gui',
        name='control_gui',
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params_file,
            description='Absolute path to the LIMO MPPI parameter file.',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use Gazebo/rosbag time instead of the robot clock.',
        ),
        DeclareLaunchArgument(
            'autostart',
            default_value='true',
            description='Configure and activate controller nodes automatically.',
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='ROS log level for controller processes.',
        ),
        local_costmap_converter,
        controller_server,
        velocity_smoother,
        lifecycle_manager,
        control_gui,
    ])
