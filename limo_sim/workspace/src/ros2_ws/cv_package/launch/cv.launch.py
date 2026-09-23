"""Launch the CV pipeline from a simulation or real robot profile."""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _boolean(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    raise ValueError('Expected true or false, got {!r}'.format(value))


def _launch_cv(context):
    config_file = os.path.expanduser(LaunchConfiguration('config_file').perform(context))
    with open(config_file, encoding='utf-8') as stream:
        profile = yaml.safe_load(stream)
    for section in ('launch', 'lane_detector', 'depth_correction', 'visual_ptcld'):
        if not isinstance(profile, dict) or not isinstance(profile.get(section), dict):
            raise ValueError('{}: missing YAML mapping {!r}'.format(config_file, section))

    settings = profile['launch']
    use_sim_time = settings['use_sim_time']
    start_rviz = settings['start_rviz']
    override = LaunchConfiguration('use_sim_time').perform(context)
    if override:
        use_sim_time = _boolean(override)
    override = LaunchConfiguration('start_rviz').perform(context)
    if override:
        start_rviz = _boolean(override)

    lane_params = dict(profile['lane_detector'], use_sim_time=use_sim_time)
    depth_params = dict(profile['depth_correction'], use_sim_time=use_sim_time)
    cloud_params = dict(profile['visual_ptcld'], use_sim_time=use_sim_time)
    override = LaunchConfiguration('visual_ptcld_enable_telemetry').perform(context)
    if override:
        cloud_params['enable_telemetry'] = _boolean(override)

    lane_node = Node(
        package='cv_package', executable='lane_detector',
        name='lane_node', output='screen', emulate_tty=True,
        parameters=[lane_params],
    )
    depth_node = Node(
        package='cv_package', executable='depth_correction',
        name='depth_correction', output='screen', emulate_tty=True,
        parameters=[depth_params],
    )
    cloud_node = Node(
        package='cv_package', executable='visual_ptcld',
        name='visual_ptcld', output='screen', emulate_tty=True,
        additional_env={
            'OPENBLAS_NUM_THREADS': '1',
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'BLIS_NUM_THREADS': '1',
        },
        parameters=[cloud_params],
    )
    nodes = [
        lane_node,
        depth_node,
        RegisterEventHandler(OnProcessStart(
            target_action=lane_node, on_start=[cloud_node],
        )),
    ]
    if start_rviz:
        config_dir = os.path.join(get_package_share_directory('cv_package'), 'config')
        nodes.append(Node(
            package='rviz2', executable='rviz2', name='cv_rviz',
            output='screen',
            arguments=['-d', os.path.join(config_dir, settings['rviz_config'])],
            parameters=[{'use_sim_time': use_sim_time}],
        ))
    return nodes


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory('cv_package'), 'config', 'cv_real.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('config_file', default_value=default_config),
        DeclareLaunchArgument('use_sim_time', default_value='',
                              description='Override the profile clock.'),
        DeclareLaunchArgument('start_rviz', default_value='',
                              description='Override the profile RViz setting.'),
        DeclareLaunchArgument(
            'visual_ptcld_enable_telemetry', default_value='',
            description='Override the visual point cloud telemetry setting.'),
        OpaqueFunction(function=_launch_cv),
    ])
