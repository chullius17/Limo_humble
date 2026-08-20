import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    nav_limo_rviz_share = get_package_share_directory('nav_limo_rviz')
    cost_threshold = LaunchConfiguration('cost_threshold')

    limo_rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                nav_limo_rviz_share,
                'launch',
                'limo_rviz.launch.py',
            )
        )
    )

    online_metric_bev = Node(
        package='nav_map_package',
        executable='online_metric_bev',
        name='online_metric_bev',
        output='screen',
        parameters=[{
            'enable_telemetry': False,
        }],
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

    laser_cv_fusion = Node(
        package='nav_map_package',
        executable='laser_cv_fusion',
        name='laser_cv_fusion',
        output='screen',
    )

    nav_map = Node(
        package='nav_map_package',
        executable='nav_map',
        name='online_nav_map',
        output='screen',
        parameters=[{
            'global_frame': 'map',
            'static_map_topic': '/map',
            'offline_mode': False,
            'online_cv_grid_topic': (
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
        limo_rviz,
        online_metric_bev,
        cv_2_ptcld,
        laser_cv_fusion,
        nav_map,
    ])
