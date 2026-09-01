"""Launch the complete online navigation application for the LIMO."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _include(package_name, launch_file, launch_arguments=None):
    """Create an include action for a launch file installed by a package."""
    package_share = get_package_share_directory(package_name)
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(package_share, 'launch', launch_file)
        ),
        launch_arguments=(launch_arguments or {}).items(),
    )


def generate_launch_description():
    """Start online mapping and the ordered navigation control stack."""
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')

    online_map_launch = _include(
        'online_map_package',
        'online_map.launch.py',
    )
    control_launch = _include(
        'nav_limo_controller',
        'control.launch.py',
        {
            'use_sim_time': use_sim_time,
            'autostart': autostart,
        },
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use Gazebo/rosbag time for navigation nodes.',
        ),
        DeclareLaunchArgument(
            'autostart',
            default_value='true',
            description='Configure and activate Nav2 nodes automatically.',
        ),
        online_map_launch,
        control_launch,
    ])
