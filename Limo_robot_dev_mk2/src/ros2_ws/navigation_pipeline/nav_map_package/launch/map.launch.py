import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.event_handlers import OnExecutionComplete, OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    custom_start_share = get_package_share_directory('custom_start')
    offline_mode = LaunchConfiguration('offline_mode')
    cost_threshold = LaunchConfiguration('cost_threshold')
    nav_rviz_config = os.path.join(
        custom_start_share,
        'config',
        'nav_mapping.rviz',
    )

    filtering_roi = {
        'roi_x_min_m': 0.0,
        'roi_x_max_m': 1.85,
        'roi_width_near_m': 0.60,
        'roi_width_far_m': 2.65,
    }

    limo_mapping = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                custom_start_share,
                'launch',
                'limo_mapping.launch.py',
            )
        )
    )
    limo_visualization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                custom_start_share,
                'launch',
                'limo_viz_slam.launch.py',
            )
        ),
        launch_arguments={
            'rviz_config': nav_rviz_config,
        }.items(),
    )

    metric_bev = Node(
        package='nav_map_package',
        executable='metric_bev',
        name='metric_bev',
        output='screen',
        parameters=[{'enable_telemetry': False}],
    )

    cv_2_ptcld = Node(
        package='nav_map_package',
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

    filtering_turquoise = Node(
        package='nav_map_package',
        executable='filtering',
        name='filtering_turquoise',
        output='screen',
        parameters=[{'color': 'TURQUOISE', **filtering_roi}],
    )
    filtering_white = Node(
        package='nav_map_package',
        executable='filtering',
        name='filtering_white',
        output='screen',
        parameters=[{'color': 'WHITE', **filtering_roi}],
    )
    filtering_magenta = Node(
        package='nav_map_package',
        executable='filtering',
        name='filtering_magenta',
        output='screen',
        parameters=[{'color': 'MAGENTA', **filtering_roi}],
    )

    cv_map_display = Node(
        package='nav_map_package',
        executable='cv_map_display',
        name='cv_map_display',
        output='screen',
        parameters=[{'enable_telemetry': False}],
    )

    nav_map = Node(
        package='nav_map_package',
        executable='nav_map',
        name='nav_map',
        output='screen',
        parameters=[{
            'global_frame': 'map',
            'static_map_topic': '/map',
            'offline_mode': ParameterValue(offline_mode, value_type=bool),
            'offline_cv_grid_topic': (
                '/limo/nav_map_package/cv_map_display/'
                'cv_map_occupancy_grid'
            ),
            'online_cv_grid_topic': (
                '/limo/nav_map_package/metric_bev/online/cost_grid_combined'
            ),
            'scan_topic': '/scan',
            'output_topic': '/limo/nav_map_package/nav_map/combined_grid',
            'publish_rate_hz': 10.0,
            'lidar_cost': 100,
        }],
    )

    nav2_map_saver = Node(
        package='nav2_map_server',
        executable='map_saver_server',
        name='map_saver',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'save_map_timeout': 30.0,
            'free_thresh_default': 0.25,
            'occupied_thresh_default': 0.65,
            'map_subscribe_transient_local': True,
        }],
    )
    map_saver_lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_nav_map_saver',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': ['map_saver'],
        }],
    )
    saver_node = Node(
        package='nav_map_package',
        executable='map_saver',
        name='nav_map_saver',
        output='screen',
        parameters=[{
            'map_topic': '/limo/nav_map_package/nav_map/combined_grid',
            'request_service': '/limo/nav_map_package/map_saver/save_map',
            'nav2_service': '/map_saver/save_map',
            'map_name': 'limo_map',
            'map_mode': 'scale',
        }],
    )

    map_save_gui = Node(
        package='nav_map_package',
        executable='map_save_gui',
        name='map_save_gui',
        output='screen',
        parameters=[{
            'save_service': '/limo/nav_map_package/map_saver/save_map',
        }],
    )

    mapping_nodes = [
        metric_bev,
        cv_2_ptcld,
        filtering_turquoise,
        filtering_white,
        filtering_magenta,
        cv_map_display,
        nav_map,
        nav2_map_saver,
        map_saver_lifecycle,
        saver_node,
        map_save_gui,
    ]

    start_mapping_nodes = RegisterEventHandler(
        OnExecutionComplete(
            target_action=limo_mapping,
            on_completion=mapping_nodes,
        )
    )
    start_visualization = RegisterEventHandler(
        OnProcessStart(
            target_action=map_save_gui,
            on_start=[limo_visualization],
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'offline_mode',
            default_value='true',
            description=(
                'Use the offline CV map display grid instead of the online '
                'MetricBEV combined grid.'
            ),
        ),
        DeclareLaunchArgument(
            'cost_threshold',
            default_value='40.0',
            description='Publish only cells with a cost above this value.',
        ),
        start_mapping_nodes,
        start_visualization,
        limo_mapping,
    ])
