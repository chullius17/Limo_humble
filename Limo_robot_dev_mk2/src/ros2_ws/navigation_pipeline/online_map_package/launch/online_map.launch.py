import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def find_project_root(start: Path):
    """Find the LIMO project root from source or install paths."""
    for candidate in [start] + list(start.parents):
        if (candidate / 'src' / 'ros2_ws').is_dir():
            return candidate
    return None


def generate_launch_description():
    laser_weight_factor_default = '0.0'
    cv_weight_factor_default = '1.0'
    cv_sync_tolerance_default = '0.20'

    nav_limo_rviz_share = get_package_share_directory('nav_limo_rviz')
    cost_threshold = LaunchConfiguration('cost_threshold')
    laser_weight_factor = LaunchConfiguration('laser_weight_factor')
    cv_weight_factor = LaunchConfiguration('cv_weight_factor')
    cv_sync_tolerance = LaunchConfiguration('cv_sync_tolerance')
    project_root = find_project_root(Path(__file__).resolve())
    if project_root is None:
        raise RuntimeError('Cannot locate the LIMO project root')
    map_directory = project_root / 'ros2_maps' / 'nav_pipeline'

    limo_rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                nav_limo_rviz_share,
                'launch',
                'limo_rviz.launch.py',
            )
        ),
        launch_arguments={
            'start_map_server': 'false',
            'publish_static_map_to_odom': 'false',
        }.items(),
    )

    amcl = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                nav_limo_rviz_share,
                'launch',
                'amcl.launch.py',
            )
        ),
        launch_arguments={
            'use_sim_time': 'true',
            'base_frame_id': 'base_link',
            # Keep the two AMCL sensor models independent: the laser model
            # scores /scan only against the static laser occupancy map.
            'map_topic': (
                '/limo/nav_map_package/online/maps/laser_map'
            ),
            # The CV model scores the robot-relative CV cloud only against
            # the static CV occupancy map.
            'cv_map_topic': (
                '/limo/nav_map_package/online/maps/cv_map'
            ),
            # AMCL is the only map -> odom publisher in the online pipeline.
            'tf_broadcast': 'true',
            'laser_weight_factor': laser_weight_factor,
            'cv_weight_factor': cv_weight_factor,
            'cv_sync_tolerance': cv_sync_tolerance,
        }.items(),
    )

    map_specs = (
        ('combined_map_server', 'limo_map_combined.yaml', '/map'),
        (
            'laser_map_server',
            'limo_map_laser.yaml',
            '/limo/nav_map_package/online/maps/laser_map',
        ),
        (
            'cv_map_server',
            'limo_map_cv.yaml',
            '/limo/nav_map_package/online/maps/cv_map',
        ),
    )
    map_servers = [
        Node(
            package='nav2_map_server',
            executable='map_server',
            name=node_name,
            output='screen',
            parameters=[{
                'yaml_filename': str(map_directory / yaml_name),
                'frame_id': 'map',
                'use_sim_time': True,
            }],
            remappings=[
                ('map', map_topic),
                ('map_metadata', f'{map_topic}_metadata'),
            ],
        )
        for node_name, yaml_name, map_topic in map_specs
    ]
    map_lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_online_maps',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': [spec[0] for spec in map_specs],
        }],
    )

    online_metric_bev = Node(
        package='online_map_package',
        executable='online_metric_bev',
        name='online_metric_bev',
        output='screen',
        parameters=[{
            'enable_telemetry': False,
        }],
    )

    cv_amcl_debug = Node(
        package='online_map_package',
        executable='cv_amcl_debug',
        name='cv_amcl_debug',
        output='screen',
        parameters=[{
            'input_topic': (
                '/limo/nav_map_package/online/maps/cv_map'
            ),
            'debug_topic': (
                '/limo/nav_map_package/online/cv_amcl_debug/distance_field'
            ),
            'pointcloud_topic': (
                '/limo/nav_map_package/online/cv_2_ptcld/points'
            ),
            'particle_cloud_topic': '/particle_cloud',
            'output_particle_cloud_topic': (
                '/limo/nav_map_package/online/cv_amcl_debug/'
                'raw_particle_cloud'
            ),
            'voxel_leaf_size': 0.05,
            'max_points': 600,
        }],
    )

    cv_2_ptcld = Node(
        package='online_map_package',
        executable='cv_2_ptcld',
        name='cv_2_ptcld',
        output='screen',
        parameters=[{
            'cost_threshold': ParameterValue(
                cost_threshold,
                value_type=float,
            ),
        }],
    )

    nav_map = Node(
        package='online_map_package',
        executable='online_nav_map',
        name='online_nav_map',
        output='screen',
        parameters=[{
            'global_frame': 'map',
            'static_map_topic': '/map',
            'cv_grid_topic': (
                '/limo/nav_map_package/online/metric_bev/'
                'cost_grid_combined'
            ),
            'scan_topic': '/scan',
            'output_topic': (
                '/limo/nav_map_package/online/nav_map/combined_grid'
            ),
            'publish_rate_hz': 10.0,
            'lidar_cost': 100,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'cost_threshold',
            default_value='40.0',
            description='Publish only cells with a cost above this value.',
        ),
        DeclareLaunchArgument(
            'laser_weight_factor',
            default_value=laser_weight_factor_default,
            description='Exponent applied to the normalized laser weight.',
        ),
        DeclareLaunchArgument(
            'cv_weight_factor',
            default_value=cv_weight_factor_default,
            description='Exponent applied to the raw CV likelihood.',
        ),
        DeclareLaunchArgument(
            'cv_sync_tolerance',
            default_value=cv_sync_tolerance_default,
            description='Maximum laser-to-CV timestamp error in seconds.',
        ),
        limo_rviz,
        amcl,
        *map_servers,
        map_lifecycle_manager,
        online_metric_bev,
        cv_amcl_debug,
        cv_2_ptcld,
        nav_map,
    ])
