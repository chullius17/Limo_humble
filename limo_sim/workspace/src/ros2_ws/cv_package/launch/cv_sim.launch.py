"""Start the CV pipeline and RViz with the sim profile."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    share = get_package_share_directory('cv_package')
    return LaunchDescription([IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'cv.launch.py')),
        launch_arguments={
            'config_file': os.path.join(share, 'config', 'cv_sim.yaml'),
            'start_rviz': 'true',
        }.items(),
    )])
