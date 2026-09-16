"""Launch mapping or its desktop clients from one YAML profile."""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _boolean(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    raise ValueError('Expected true or false, got {!r}'.format(value))


def _parameters(values):
    # Preserve strings (especially an empty save_directory) on Foxy as well.
    return {name: ParameterValue(value, value_type=str)
            if isinstance(value, str) else value
            for name, value in values.items()}


def _launch_mapping(context):
    config_file = os.path.expanduser(LaunchConfiguration('config_file').perform(context))
    with open(config_file, encoding='utf-8') as stream:
        profile = yaml.safe_load(stream)
    for section in ('launch', 'slam_toolbox', 'semantic_mapper', 'map_save_gui'):
        if not isinstance(profile, dict) or not isinstance(profile.get(section), dict):
            raise ValueError('{}: missing YAML mapping {!r}'.format(config_file, section))

    settings = dict(profile['launch'])
    for name in ('use_sim_time', 'start_slam', 'start_mapper', 'start_rviz', 'start_gui'):
        override = LaunchConfiguration(name).perform(context)
        settings[name] = _boolean(override if override else settings[name])
    for name in ('rviz_config', 'fixed_frame'):
        override = LaunchConfiguration(name).perform(context)
        if override:
            settings[name] = override

    mode = LaunchConfiguration('mode').perform(context)
    if mode == 'desktop':
        settings.update(start_slam=False, start_mapper=False)
        # A robot profile disables local windows; desktop mode enables them
        # unless the caller explicitly disables an individual GUI.
        for name in ('start_rviz', 'start_gui'):
            override = LaunchConfiguration(name).perform(context)
            settings[name] = _boolean(override) if override else True
    elif mode == 'backend':
        settings.update(start_rviz=False, start_gui=False)
    elif mode != 'profile':
        raise ValueError('mode must be profile, backend or desktop')

    clock = {'use_sim_time': settings['use_sim_time']}
    slam = dict(profile['slam_toolbox'])
    mapper = dict(profile['semantic_mapper'])
    for name, value_type in (('pose_source', str), ('trajectory_id', int),
                             ('resolution', float), ('save_directory', str)):
        override = LaunchConfiguration(name).perform(context)
        if override:
            mapper[name] = value_type(override)
            if name == 'resolution':
                slam[name] = mapper[name]

    nodes = []
    if settings['start_slam']:
        # Avoid the different Foxy/Humble argument names in online_async_launch.
        nodes.append(Node(
            package='slam_toolbox', executable='async_slam_toolbox_node',
            name='slam_toolbox', output='screen',
            parameters=[_parameters(slam), clock]))
    if settings['start_mapper']:
        nodes.append(Node(
            package='offline_map_package', executable='semantic_mapper',
            name='semantic_mapper', output='screen',
            additional_env={'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'},
            parameters=[_parameters(mapper), clock]))
    if settings['start_gui']:
        nodes.append(Node(
            package='offline_map_package', executable='map_save_gui',
            name='map_save_gui', output='screen',
            parameters=[_parameters(profile['map_save_gui']), clock]))
    if settings['start_rviz']:
        rviz_config = os.path.expanduser(settings['rviz_config'])
        if not os.path.isabs(rviz_config):
            rviz_config = os.path.join(
                get_package_share_directory('limo_rviz'), 'config', rviz_config)
        nodes.append(Node(
            package='rviz2', executable='rviz2', name='rviz2', output='screen',
            arguments=['-d', rviz_config, '-f', settings['fixed_frame']],
            parameters=[clock]))
    return nodes


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory('offline_map_package'), 'config', 'mapping_sim.yaml')
    overrides = (
        'use_sim_time', 'start_slam', 'start_mapper', 'start_rviz', 'start_gui',
        'rviz_config', 'fixed_frame', 'pose_source', 'trajectory_id',
        'resolution', 'save_directory',
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file', default_value=default_config,
            description='Mapping profile YAML (launch settings and node parameters).'),
        DeclareLaunchArgument(
            'mode', default_value='profile',
            description='profile: use YAML; backend: no GUI; desktop: GUI clients only.'),
        *[DeclareLaunchArgument(
            name, default_value='', description='Override YAML; empty uses the profile.')
          for name in overrides],
        OpaqueFunction(function=_launch_mapping),
    ])
