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
    cv_obstacle_weight_factor_default = '1.0'
    cv_street_weight_factor_default = '1.0'
    cv_sad_gain_default = '20.0'
    cv_sad_cell_size_default = '0.05'
    cv_sad_min_positive_mass_default = '5.0'
    cv_sync_tolerance_default = '0.20'
    alpha1_default = '0.2'
    alpha2_default = '0.2'
    alpha3_default = '0.2'
    alpha4_default = '0.2'

    nav_limo_rviz_share = get_package_share_directory('nav_limo_rviz')
    cost_threshold = LaunchConfiguration('cost_threshold')
    laser_weight_factor = LaunchConfiguration('laser_weight_factor')
    cv_weight_factor = LaunchConfiguration('cv_weight_factor')
    cv_obstacle_weight_factor = LaunchConfiguration(
        'cv_obstacle_weight_factor'
    )
    cv_street_weight_factor = LaunchConfiguration(
        'cv_street_weight_factor'
    )
    cv_sad_gain = LaunchConfiguration('cv_sad_gain')
    cv_sad_cell_size = LaunchConfiguration('cv_sad_cell_size')
    cv_sad_min_positive_mass = LaunchConfiguration(
        'cv_sad_min_positive_mass'
    )
    cv_sync_tolerance = LaunchConfiguration('cv_sync_tolerance')
    alpha1 = LaunchConfiguration('alpha1')
    alpha2 = LaunchConfiguration('alpha2')
    alpha3 = LaunchConfiguration('alpha3')
    alpha4 = LaunchConfiguration('alpha4')
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
            # The CV model evaluates obstacle and street evidence against
            # their respective static occupancy maps.
            'cv_map_topic': (
                '/limo/nav_map_package/online/maps/cv_map'
            ),
            'cv_obstacle_grid_topic': (
                '/limo/nav_map_package/online/metric_bev/'
                'cost_grid_binary_obstacles'
            ),
            'cv_street_map_topic': (
                '/limo/nav_map_package/online/maps/street_map'
            ),
            'cv_street_grid_topic': (
                '/limo/nav_map_package/online/metric_bev/'
                'cost_grid_binary_street'
            ),
            # AMCL is the only map -> odom publisher in the online pipeline.
            'tf_broadcast': 'true',
            'laser_weight_factor': laser_weight_factor,
            'cv_weight_factor': cv_weight_factor,
            'cv_obstacle_weight_factor': cv_obstacle_weight_factor,
            'cv_street_weight_factor': cv_street_weight_factor,
            'cv_sad_gain': cv_sad_gain,
            'cv_sad_cell_size': cv_sad_cell_size,
            'cv_sad_min_positive_mass': cv_sad_min_positive_mass,
            'cv_sync_tolerance': cv_sync_tolerance,
            'alpha1': alpha1,
            'alpha2': alpha2,
            'alpha3': alpha3,
            'alpha4': alpha4,
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
        (
            'street_map_server',
            'limo_map_street.yaml',
            '/limo/nav_map_package/online/maps/street_map',
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
            'binary_threshold': ParameterValue(
                cost_threshold,
                value_type=float,
            ),
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
            'obstacle_grid_topic': (
                '/limo/nav_map_package/online/metric_bev/'
                'cost_grid_binary_obstacles'
            ),
            'street_grid_topic': (
                '/limo/nav_map_package/online/metric_bev/'
                'cost_grid_binary_street'
            ),
            'obstacle_grid_debug_topic': (
                '/limo/nav_map_package/online/cv_amcl_debug/'
                'discretized_obstacles'
            ),
            'street_grid_debug_topic': (
                '/limo/nav_map_package/online/cv_amcl_debug/'
                'discretized_street'
            ),
            'street_map_topic': (
                '/limo/nav_map_package/online/maps/street_map'
            ),
            'particle_cloud_topic': '/particle_cloud',
            'output_particle_cloud_topic': (
                '/limo/nav_map_package/online/cv_amcl_debug/'
                'raw_particle_cloud'
            ),
            'voxel_leaf_size': 0.05,
            # Debug scoring is deliberately lighter than AMCL's 600+600-point
            # production path so visualization cannot starve sensor callbacks.
            'max_points': 200,
            'score_rate_hz': 1.0,
            'cv_weight_factor': ParameterValue(
                cv_weight_factor,
                value_type=float,
            ),
            'cv_obstacle_weight_factor': ParameterValue(
                cv_obstacle_weight_factor,
                value_type=float,
            ),
            'cv_street_weight_factor': ParameterValue(
                cv_street_weight_factor,
                value_type=float,
            ),
            'sad_gain': ParameterValue(cv_sad_gain, value_type=float),
            'sad_cell_size': ParameterValue(
                cv_sad_cell_size,
                value_type=float,
            ),
            'sad_min_positive_mass': ParameterValue(
                cv_sad_min_positive_mass,
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
            description=(
                'Threshold used by obstacle and street binary CV grids.'
            ),
        ),
        DeclareLaunchArgument(
            'laser_weight_factor',
            default_value=laser_weight_factor_default,
            description='Exponent applied to the normalized laser weight.',
        ),
        DeclareLaunchArgument(
            'cv_weight_factor',
            default_value=cv_weight_factor_default,
            description='Exponent applied to the CV SAD likelihood.',
        ),
        DeclareLaunchArgument(
            'cv_obstacle_weight_factor',
            default_value=cv_obstacle_weight_factor_default,
            description=(
                'Relative obstacle evidence factor; zero disables obstacle SAD.'
            ),
        ),
        DeclareLaunchArgument(
            'cv_street_weight_factor',
            default_value=cv_street_weight_factor_default,
            description=(
                'Relative street evidence factor; zero disables street SAD.'
            ),
        ),
        DeclareLaunchArgument(
            'cv_sad_gain',
            default_value=cv_sad_gain_default,
            description='Gain converting normalized SAD into likelihood.',
        ),
        DeclareLaunchArgument(
            'cv_sad_cell_size',
            default_value=cv_sad_cell_size_default,
            description='Regular SAD template sampling size in metres.',
        ),
        DeclareLaunchArgument(
            'cv_sad_min_positive_mass',
            default_value=cv_sad_min_positive_mass_default,
            description=(
                'Minimum foreground mass required for a CV class to vote.'
            ),
        ),
        DeclareLaunchArgument(
            'cv_sync_tolerance',
            default_value=cv_sync_tolerance_default,
            description='Maximum laser-to-CV timestamp error in seconds.',
        ),
        DeclareLaunchArgument(
            'alpha1',
            default_value=alpha1_default,
            description='Rotation noise caused by Ackermann rotation.',
        ),
        DeclareLaunchArgument(
            'alpha2',
            default_value=alpha2_default,
            description='Rotation/steering noise caused by translation.',
        ),
        DeclareLaunchArgument(
            'alpha3',
            default_value=alpha3_default,
            description='Translation noise caused by translation.',
        ),
        DeclareLaunchArgument(
            'alpha4',
            default_value=alpha4_default,
            description='Translation noise caused by Ackermann rotation.',
        ),
        limo_rviz,
        amcl,
        *map_servers,
        map_lifecycle_manager,
        online_metric_bev,
        cv_amcl_debug,
        nav_map,
    ])
