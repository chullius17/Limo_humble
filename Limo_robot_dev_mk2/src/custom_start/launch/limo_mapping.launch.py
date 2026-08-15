"""UTILITY: Launch SLAM Toolbox with the frame and lidar settings of the simulated LIMO."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    custom_start_share = get_package_share_directory('custom_start')
    slam_toolbox_share = get_package_share_directory('slam_toolbox')

    params_file = os.path.join(
        custom_start_share, 'config', 'slam_toolbox.yaml'
    )
    use_sim_time = LaunchConfiguration('use_sim_time')

    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_toolbox_share, 'launch', 'online_async_launch.py')
        ),
        launch_arguments={
            'slam_params_file': params_file,
            'use_sim_time': use_sim_time,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use Gazebo simulation time.',
        ),
        slam_toolbox,
    ])
