import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    nav_limo_rviz_share = get_package_share_directory('nav_limo_rviz')

    limo_rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                nav_limo_rviz_share,
                'launch',
                'limo_rviz.launch.py',
            )
        )
    )

    metric_bev = Node(
        package='nav_map_package',
        executable='metric_bev',
        name='metric_bev',
        output='screen',
        parameters=[{
            'enable_telemetry': False,
        }],
    )

    nav_map = Node(
        package='nav_map_package',
        executable='nav_map',
        name='nav_map',
        output='screen',
        parameters=[{
            'global_frame': 'map',
            'static_map_topic': '/map',
            'offline_mode': False,
            'online_cv_grid_topic': (
                '/limo/nav_map_package/metric_bev/online/'
                'cost_grid_combined'
            ),
            'scan_topic': '/scan',
            'output_topic': '/limo/nav_map_package/nav_map/combined_grid',
            'publish_rate_hz': 10.0,
            'lidar_cost': 100,
        }],
    )

    return LaunchDescription([
        limo_rviz,
        metric_bev,
        nav_map,
    ])
