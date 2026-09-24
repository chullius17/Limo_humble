"""Launch online localization or its desktop client from one YAML profile."""

import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


AMCL_PARAMETER_TYPES = {
    'base_frame_id': str,
    'odom_frame_id': str,
    'global_frame_id': str,
    'scan_topic': str,
    'tf_broadcast': bool,
    'cv_enabled': bool,
    'cv_cloud_topic': str,
    'cv_buffer_size': int,
    'cv_sync_tolerance': float,
    'cv_voxel_size': float,
    'cv_min_points': float,
    'cv_min_non_road_voxels': int,
    'cv_occupied_threshold': int,
    'laser_weight_factor': float,
    'cv_weight_factor': float,
    'cv_sad_gain': float,
    'cv_quality_gate_enabled': bool,
    'cv_min_information': float,
    'cv_max_position_stddev': float,
    'cv_max_yaw_stddev': float,
    'max_particles': int,
    'min_particles': int,
    'workload_logging_enabled': bool,
    'alpha1': float,
    'alpha2': float,
    'alpha3': float,
    'alpha4': float,
}


def _boolean(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    raise ValueError('Expected true or false, got {!r}'.format(value))


def _project_root(start):
    for candidate in [start] + list(start.parents):
        if (candidate / 'src' / 'ros2_ws').is_dir():
            return candidate
    return None


def _profile(context):
    config_file = os.path.expanduser(
        LaunchConfiguration('config_file').perform(context))
    with open(config_file, encoding='utf-8') as stream:
        profile = yaml.safe_load(stream)
    for section in ('launch', 'map_servers', 'amcl'):
        if not isinstance(profile, dict) or not isinstance(
                profile.get(section), dict):
            raise ValueError(
                '{}: missing YAML mapping {!r}'.format(config_file, section))
    return profile


def _override(context, name, current, value_type):
    value = LaunchConfiguration(name).perform(context)
    selected = current if value == '' else value
    if value_type is bool:
        return _boolean(selected)
    return value_type(selected)


def _launch_online(context):
    profile = _profile(context)
    settings = dict(profile['launch'])
    for name in (
            'use_sim_time', 'start_cv', 'start_maps', 'start_amcl',
            'start_local_ctrl_map', 'start_rviz'):
        settings[name] = _override(context, name, settings[name], bool)
    for name in ('rviz_config', 'fixed_frame'):
        settings[name] = _override(context, name, settings[name], str)

    mode = LaunchConfiguration('mode').perform(context)
    if mode == 'desktop':
        settings.update(
            start_cv=False,
            start_maps=False,
            start_amcl=False,
            start_local_ctrl_map=False,
        )
        override = LaunchConfiguration('start_rviz').perform(context)
        settings['start_rviz'] = _boolean(override) if override else True
    elif mode == 'backend':
        settings['start_rviz'] = False
    elif mode != 'profile':
        raise ValueError('mode must be profile, backend or desktop')

    settings['cv_config'] = _override(
        context, 'cv_config', settings.get(
            'cv_config', 'cv_sim.yaml' if settings['use_sim_time'] else 'cv_real.yaml'), str)

    maps = dict(profile['map_servers'])
    maps['directory'] = _override(
        context, 'map_directory', maps.get('directory', ''), str)
    maps['name'] = _override(context, 'map_name', maps['name'], str)
    map_directory = os.path.expanduser(maps['directory'])
    if settings['start_maps']:
        if not map_directory:
            root = _project_root(Path(__file__).resolve())
            if root is None:
                raise RuntimeError('Cannot locate the LIMO workspace')
            map_directory = str(root / 'ros2_maps' / 'semantic')
        elif not os.path.isabs(map_directory):
            root = _project_root(Path(__file__).resolve())
            if root is None:
                raise RuntimeError(
                    'A relative map directory requires a LIMO workspace')
            map_directory = str(root / map_directory)

    amcl = dict(profile['amcl'])
    for name, value_type in AMCL_PARAMETER_TYPES.items():
        if name not in amcl:
            raise ValueError('AMCL profile is missing {!r}'.format(name))
        amcl[name] = _override(context, name, amcl[name], value_type)

    clock = {'use_sim_time': settings['use_sim_time']}
    map_specs = (
        ('complete_map_server', '_complete.yaml', maps['complete_topic']),
        ('laser_map_server', '_laser.yaml', maps['laser_topic']),
        ('cv_map_server', '_cv_obstacle.yaml', maps['cv_obstacle_topic']),
    )
    actions = []
    if settings['start_local_ctrl_map']:
        local_map = dict(profile.get('local_ctrl_map', {}))
        local_map['maximum_points'] = _override(
            context, 'local_map_maximum_points',
            local_map.get('maximum_points', 300), int)
        actions.append(Node(
            package='online_map_package', executable='local_ctrl_map',
            name='local_ctrl_map', output='screen', parameters=[{
                **local_map,
                **clock,
                'base_frame': amcl['base_frame_id'],
                'odometry_frame': amcl['odom_frame_id'],
            }]))
    if settings['start_cv']:
        cv_share = get_package_share_directory('cv_package')
        cv_profile = os.path.expanduser(settings['cv_config'])
        actions.append(GroupAction(actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                cv_share, 'launch', 'cv.launch.py')),
            launch_arguments={
                'config_file': os.path.join(cv_share, 'config', cv_profile),
                'mode': 'backend',
                'use_sim_time': str(settings['use_sim_time']).lower(),
                'visual_ptcld_enable_telemetry': 'false',
            }.items(),
        )]))
    if settings['start_maps']:
        for name, suffix, topic in map_specs:
            actions.append(Node(
                package='nav2_map_server', executable='map_server',
                name=name, output='screen', parameters=[{
                    'yaml_filename': os.path.join(
                        map_directory, maps['name'] + suffix),
                    'frame_id': maps['frame_id'],
                    **clock,
                }], remappings=[
                    ('map', topic),
                    ('map_metadata', topic + '_metadata'),
                ]))
        actions.append(Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_online_maps', output='screen',
            parameters=[{
                **clock,
                'autostart': True,
                'node_names': [name for name, _, _ in map_specs],
            }]))
    if settings['start_amcl']:
        arguments = {
            name: str(value).lower() if isinstance(value, bool) else str(value)
            for name, value in amcl.items()
        }
        arguments.update({
            'use_sim_time': str(settings['use_sim_time']).lower(),
            'map_topic': maps['laser_topic'],
            'cv_map_topic': maps['cv_obstacle_topic'],
        })
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('limo_rviz'),
                'launch', 'amcl.launch.py')),
            launch_arguments=arguments.items(),
        ))
    if settings['start_rviz']:
        rviz_config = os.path.expanduser(settings['rviz_config'])
        if not os.path.isabs(rviz_config):
            rviz_config = os.path.join(
                get_package_share_directory('limo_rviz'),
                'config', rviz_config)
        actions.append(Node(
            package='rviz2', executable='rviz2', name='rviz2',
            output='screen',
            arguments=['-d', rviz_config, '-f', settings['fixed_frame']],
            parameters=[clock]))
    return actions


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory('online_map_package'),
        'config', 'mapping_real.yaml')
    launch_overrides = (
        'use_sim_time', 'start_cv', 'start_maps', 'start_amcl',
        'start_local_ctrl_map', 'start_rviz',
        'rviz_config', 'fixed_frame', 'cv_config', 'map_directory', 'map_name',
        'local_map_maximum_points',
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file', default_value=default_config,
            description=(
                'Online localization profile YAML containing launch, map '
                'server and AMCL settings.')),
        DeclareLaunchArgument(
            'mode', default_value='profile',
            description=(
                'profile: use YAML; backend: no RViz; desktop: RViz only.')),
        *[DeclareLaunchArgument(
            name, default_value='',
            description='Override YAML; empty uses the profile.')
          for name in launch_overrides + tuple(AMCL_PARAMETER_TYPES)],
        OpaqueFunction(function=_launch_online),
    ])
