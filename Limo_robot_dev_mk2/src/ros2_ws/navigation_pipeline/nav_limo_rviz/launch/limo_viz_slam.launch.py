"""Launch RViz with the LIMO mapping displays preconfigured."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    nav_limo_rviz_share = get_package_share_directory('nav_limo_rviz')
    default_config = os.path.join(
        nav_limo_rviz_share, 'config', 'base_mapping.rviz'
    )
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use the clock published by Gazebo.',
        ),
        DeclareLaunchArgument(
            'rviz_config', default_value=default_config,
            description='Absolute path of the RViz configuration file.',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            parameters=[{'use_sim_time': use_sim_time}],
        ),
    ])
