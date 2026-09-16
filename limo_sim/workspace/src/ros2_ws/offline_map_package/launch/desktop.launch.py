"""Start RViz and Map Saver in the PC container for the physical LIMO."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    share = get_package_share_directory('offline_map_package')
    return LaunchDescription([IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'map.launch.py')),
        launch_arguments={
            'config_file': os.path.join(share, 'config', 'mapping_real.yaml'),
            'mode': 'desktop',
        }.items(),
    )])
