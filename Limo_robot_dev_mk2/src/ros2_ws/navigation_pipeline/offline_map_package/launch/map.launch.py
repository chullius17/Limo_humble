"""Launch the complete offline mapping pipeline."""

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
    rviz_share = get_package_share_directory('nav_limo_rviz')
    cv_share = get_package_share_directory('nav_cv_package')
    cost_threshold = LaunchConfiguration('cost_threshold')
    filtering_roi = {
        'roi_x_min_m': 0.0,
        'roi_x_max_m': 1.85,
        'roi_width_near_m': 0.60,
        'roi_width_far_m': 2.65,
    }

    limo_mapping = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(rviz_share, 'launch', 'limo_mapping.launch.py')
        )
    )
    cv_pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(cv_share, 'launch', 'cv.launch.py')
        )
    )
    visualization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(rviz_share, 'launch', 'limo_viz_slam.launch.py')
        ),
        launch_arguments={
            'rviz_config': os.path.join(
                rviz_share,
                'config',
                'offline_nav_map.rviz',
            ),
        }.items(),
    )

    offline_metric_bev = Node(
        package='offline_map_package',
        executable='offline_metric_bev',
        name='offline_metric_bev',
        output='screen',
        parameters=[{'enable_telemetry': False}],
    )
    filtering_nodes = [
        Node(
            package='offline_map_package',
            executable='filtering',
            name=f'filtering_{color.lower()}',
            output='screen',
            parameters=[{'color': color, **filtering_roi}],
        )
        for color in ('TURQUOISE', 'WHITE', 'MAGENTA')
    ]
    cv_map_display = Node(
        package='offline_map_package',
        executable='cv_map_display',
        name='cv_map_display',
        output='screen',
        parameters=[{'enable_telemetry': False}],
    )
    nav_map = Node(
        package='offline_map_package',
        executable='nav_map',
        name='nav_map',
        output='screen',
        parameters=[{
            'global_frame': 'map',
            'static_map_topic': '/map',
            'cv_grid_topic': (
                '/limo/nav_map_package/offline/cv_map_display/'
                'cv_map_occupancy_grid'
            ),
            'scan_topic': '/scan',
            'output_topic': (
                '/limo/nav_map_package/offline/nav_map/combined_grid'
            ),
            'laser_map_topic': (
                '/limo/nav_map_package/offline/nav_map/laser_map'
            ),
            'cv_map_topic': '/limo/nav_map_package/offline/nav_map/cv_map',
            'street_map_topic': (
                '/limo/nav_map_package/offline/nav_map/street_map'
            ),
            'street_cost_min': 0.0,
            'street_cost_max': 20.0,
            'cv_cost_threshold': ParameterValue(
                cost_threshold,
                value_type=float,
            ),
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
        package='offline_map_package',
        executable='map_saver',
        name='nav_map_saver',
        output='screen',
        parameters=[{
            'combined_map_topic': (
                '/limo/nav_map_package/offline/nav_map/combined_grid'
            ),
            'laser_map_topic': (
                '/limo/nav_map_package/offline/nav_map/laser_map'
            ),
            'cv_map_topic': (
                '/limo/nav_map_package/offline/nav_map/cv_map'
            ),
            'street_map_topic': (
                '/limo/nav_map_package/offline/nav_map/street_map'
            ),
            'request_service': (
                '/limo/nav_map_package/offline/map_saver/save_map'
            ),
            'nav2_service': '/map_saver/save_map',
            'combined_map_name': 'limo_map_combined',
            'laser_map_name': 'limo_map_laser',
            'cv_map_name': 'limo_map_cv',
            'street_map_name': 'limo_map_street',
            'combined_map_mode': 'scale',
            'laser_map_mode': 'trinary',
            'cv_map_mode': 'trinary',
            'street_map_mode': 'trinary',
        }],
    )
    map_save_gui = Node(
        package='offline_map_package',
        executable='map_save_gui',
        name='map_save_gui',
        output='screen',
        parameters=[{
            'save_service': (
                '/limo/nav_map_package/offline/map_saver/save_map'
            ),
        }],
    )

    mapping_nodes = [
        offline_metric_bev,
        *filtering_nodes,
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
            on_start=[visualization],
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'cost_threshold',
            default_value='40.0',
            description='Publish only CV cells with a cost above this value.',
        ),
        start_mapping_nodes,
        start_visualization,
        cv_pipeline,
        limo_mapping,
    ])
