"""Launch RViz with the LIMO mapping displays preconfigured."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    custom_start_share = get_package_share_directory('custom_start')
    default_config = os.path.join(
        custom_start_share, 'config', 'limo_mapping.rviz'
    )

    return LaunchDescription([
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
        ),
    ])
