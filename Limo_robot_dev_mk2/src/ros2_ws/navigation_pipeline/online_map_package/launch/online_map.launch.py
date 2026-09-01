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
    laser_weight_factor_default = '1.0'
    cv_weight_factor_default = '1.0'
    cv_obstacle_weight_factor_default = '1.0'
    cv_street_weight_factor_default = '1.0'
    cv_sad_gain_default = '20.0'
    cv_sad_cell_size_default = '0.075'
    cv_sad_min_cell_occupancy_default = '0.1'
    cv_sad_min_positive_mass_default = '5.0'
    max_particles_default = '800'
    min_particles_default = '300'
    workload_logging_enabled_default = 'true'
    cv_sync_tolerance_default = '0.20'
    alpha1_default = '0.2'
    alpha2_default = '0.2'
    alpha3_default = '0.2'
    alpha4_default = '0.2'

    nav_limo_rviz_share = get_package_share_directory('nav_limo_rviz')
    nav_cv_share = get_package_share_directory('nav_cv_package')
    cost_threshold = LaunchConfiguration('cost_threshold')
    classification_blue_distance_threshold_px = LaunchConfiguration(
        'classification_blue_distance_threshold_px'
    )
    classification_magenta_distance_threshold_px = LaunchConfiguration(
        'classification_magenta_distance_threshold_px'
    )
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
    cv_sad_min_cell_occupancy = LaunchConfiguration(
        'cv_sad_min_cell_occupancy'
    )
    cv_sad_min_positive_mass = LaunchConfiguration(
        'cv_sad_min_positive_mass'
    )
    max_particles = LaunchConfiguration('max_particles')
    min_particles = LaunchConfiguration('min_particles')
    workload_logging_enabled = LaunchConfiguration(
        'workload_logging_enabled'
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

    cv_pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav_cv_share, 'launch', 'cv.launch.py')
        ),
        launch_arguments={
            'classification_blue_distance_threshold_px': (
                classification_blue_distance_threshold_px
            ),
            'classification_magenta_distance_threshold_px': (
                classification_magenta_distance_threshold_px
            ),
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
            'cv_sad_min_cell_occupancy': cv_sad_min_cell_occupancy,
            'cv_sad_min_positive_mass': cv_sad_min_positive_mass,
            'max_particles': max_particles,
            'min_particles': min_particles,
            'workload_logging_enabled': workload_logging_enabled,
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
            'obstacle_grid_topic': (
                '/limo/nav_map_package/online/metric_bev/'
                'cost_grid_binary_obstacles'
            ),
            'street_grid_topic': (
                '/limo/nav_map_package/online/metric_bev/'
                'cost_grid_binary_street'
            ),
            'obstacle_grid_subsampled_topic': (
                '/limo/nav_map_package/online/cv_amcl_debug/'
                'subsampled_obstacles'
            ),
            'street_grid_subsampled_topic': (
                '/limo/nav_map_package/online/cv_amcl_debug/'
                'subsampled_street'
            ),
            'sad_cell_size': ParameterValue(
                cv_sad_cell_size,
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

    cv_pointcloud = Node(
        package='online_map_package',
        executable='cv_2_ptcld',
        name='cv_2_ptcld',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'input_topic': (
                '/limo/nav_map_package/online/metric_bev/'
                'cost_grid_combined'
            ),
            'output_topic': '/limo/nav_map_package/online/cv_cloud',
            'output_frame': 'base_link',
            'voxel_size': 0.03,
            'source_block_size': 5,
            'minimum_cells_per_block': 5,
            'minimum_cost': 30.0,
            'high_cost_threshold': 95.0,
            'high_cost_downsampling_factor': 2,
            'maximum_points': 2000,
            'statistics_window_cycles': 30,
        }],
    )

    local_ptcld = Node(
        package='online_map_package',
        executable='local_ptcld',
        name='local_ptcld',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'input_topic': '/limo/nav_map_package/online/cv_cloud',
            'output_topic': (
                '/limo/nav_map_package/online/local_ptcld'
            ),
            'base_frame': 'base_link',
            'odometry_frame': 'odom',
            'bounding_box_topic': (
                '/limo/nav_map_package/online/cloud_bounding_box'
            ),
            'roi_marker_topic': (
                '/limo/nav_map_package/online/cloud_roi'
            ),
            'roi_frame': 'online_metric_bev_origin_combined',
            'roi_trapezoid_height': 1.85,
            'roi_near_base_width': 0.60,
            'roi_far_base_width': 2.65,
            'bounding_box_length': 2.46,
            'bounding_box_width': 2.66,
            'cmd_vel_topic': '/cmd_vel',
            'maximum_decay_per_cycle': 0.02,
            'linear_speed_at_max_decay': 0.50,
            'angular_speed_at_max_decay': 1.00,
            'linear_stationary_threshold': 0.01,
            'angular_stationary_threshold': 0.02,
            # teleop_twist_keyboard publishes on key events, not continuously.
            'cmd_vel_timeout_sec': 0.0,
            'tf_lookup_timeout_sec': 0.0,
            'maximum_tf_age_sec': 0.03,
            'minimum_confidence': 0.30,
            'maximum_points': 2000,
            'point_statistics_window_cycles': 30,
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
            'classification_blue_distance_threshold_px',
            default_value='10.0',
            description='CV classification distance from blue in pixels.',
        ),
        DeclareLaunchArgument(
            'classification_magenta_distance_threshold_px',
            default_value='10.0',
            description='CV classification distance from magenta in pixels.',
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
                'Relative obstacle evidence factor; zero disables '
                'obstacle SAD.'
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
            'cv_sad_min_cell_occupancy',
            default_value=cv_sad_min_cell_occupancy_default,
            description=(
                'Minimum occupied fraction retained in a local SAD cell.'
            ),
        ),
        DeclareLaunchArgument(
            'cv_sad_min_positive_mass',
            default_value=cv_sad_min_positive_mass_default,
            description=(
                'Minimum foreground mass required for a CV class to vote.'
            ),
        ),
        DeclareLaunchArgument(
            'max_particles',
            default_value=max_particles_default,
            description=(
                'Maximum AMCL particle count, limited for Jetson Nano.'
            ),
        ),
        DeclareLaunchArgument(
            'min_particles',
            default_value=min_particles_default,
            description=(
                'Minimum AMCL particle count used by adaptive sampling.'
            ),
        ),
        DeclareLaunchArgument(
            'workload_logging_enabled',
            default_value=workload_logging_enabled_default,
            description='Publish throttled AMCL and CV workload counters.',
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
        cv_pipeline,
        amcl,
        *map_servers,
        map_lifecycle_manager,
        online_metric_bev,
        cv_amcl_debug,
        nav_map,
        cv_pointcloud,
        local_ptcld,
    ])
