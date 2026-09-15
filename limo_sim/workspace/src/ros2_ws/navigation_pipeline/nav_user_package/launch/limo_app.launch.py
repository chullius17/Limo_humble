"""Launch the complete online navigation application for the LIMO."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


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
    """Start each navigation subsystem with its own launch defaults."""
    trajectory_params = os.path.join(
        get_package_share_directory('nav_traj_package'),
        'config',
        'smac_hybrid_params.yaml',
    )
    controller_params = os.path.join(
        get_package_share_directory('nav_limo_controller'),
        'config',
        'mppi_control_params.yaml',
    )

    online_map_launch = _include(
        'online_map_package',
        'online_map.launch.py',
    )
    trajectory_launch = _include(
        'nav_traj_package',
        'trajectory.launch.py',
        {
            # Both child launches expose an argument named params_file.
            # Set it explicitly so they cannot reuse each other's launch
            # configuration.
            'params_file': trajectory_params,
            'map_topic': (
                '/limo/nav_map_package/online/nav_map/combined_grid'
            ),
            'use_sim_time': 'true',
            'autostart': 'true',
        },
    )
    control_launch = _include(
        'nav_limo_controller',
        'control.launch.py',
        {
            'params_file': controller_params,
            'use_sim_time': 'true',
            'autostart': 'true',
        },
    )

    return LaunchDescription([
        online_map_launch,
        trajectory_launch,
        control_launch,
    ])
