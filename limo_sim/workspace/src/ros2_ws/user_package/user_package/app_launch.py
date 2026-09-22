"""Compose mapping, planning, and control for a LIMO application profile."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _include(package_name, launch_file, arguments=None):
    """Include one subsystem without duplicating its internal parameters."""
    package_share = get_package_share_directory(package_name)
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(package_share, 'launch', launch_file)
        ),
        launch_arguments=(arguments or {}).items(),
    )


def _boolean(value):
    """Parse a strict ROS launch boolean."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    raise ValueError('Expected true or false, got {!r}'.format(value))


def _optional_boolean(context, name, default):
    value = LaunchConfiguration(name).perform(context)
    return default if value == '' else _boolean(value)


def _launch_app(context, profile):
    """Resolve overrides and compose the selected application's subsystems."""
    if profile not in ('sim', 'real'):
        raise ValueError('profile must be sim or real')

    simulation = profile == 'sim'
    use_sim_time = str(simulation).lower()
    start_gui = _optional_boolean(
        context, 'start_control_gui', simulation)

    map_topic = LaunchConfiguration('map_topic').perform(context)
    return [
        _include(
            'online_map_package',
            'online_map_{}.launch.py'.format(profile),
        ),
        _include(
            'traj_package',
            'trajectory.launch.py',
            {
                'map_topic': map_topic,
                'use_sim_time': use_sim_time,
                'autostart': 'true',
            },
        ),
        _include(
            'limo_controller',
            'control.launch.py',
            {
                'robot_model': profile,
                'use_sim_time': use_sim_time,
                'autostart': 'true',
                'start_gui': str(start_gui).lower(),
            },
        ),
    ]


def generate_app_launch_description(profile):
    """Create the complete application launch for ``sim`` or ``real``."""
    if profile not in ('sim', 'real'):
        raise ValueError('profile must be sim or real')

    return LaunchDescription([
        DeclareLaunchArgument(
            'map_topic', default_value='/map',
            description='Global OccupancyGrid consumed by traj_package.'),
        DeclareLaunchArgument(
            'start_control_gui', default_value='',
            description='Override the profile default for the control GUI.'),
        OpaqueFunction(function=_launch_app, args=[profile]),
    ])
