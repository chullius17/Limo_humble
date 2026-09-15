"""Build semantic cost layers directly from visual_ptcld's metric point cloud."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    offline_share = get_package_share_directory('offline_map_package')
    rviz_share = get_package_share_directory('limo_rviz')
    cv_share = get_package_share_directory('cv_package')
    use_sim_time = LaunchConfiguration('use_sim_time')

    def include(share, filename, enabled, arguments=None):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, 'launch', filename)),
            condition=IfCondition(LaunchConfiguration(enabled)),
            launch_arguments=(arguments or {}).items())

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('start_cv', default_value='true'),
        DeclareLaunchArgument(
            'start_slam', default_value='true',
            description='Start existing SLAM Toolbox; set false with external Cartographer.'),
        DeclareLaunchArgument('start_rviz', default_value='true'),
        DeclareLaunchArgument('start_gui', default_value='true'),
        DeclareLaunchArgument('pose_source', default_value='tf',
                              description='tf, or cartographer with external submap_list.'),
        DeclareLaunchArgument('trajectory_id', default_value='0'),
        DeclareLaunchArgument('resolution', default_value='0.05'),
        DeclareLaunchArgument('save_directory', default_value=''),
        include(cv_share, 'cv.launch.py', 'start_cv'),
        include(rviz_share, 'limo_mapping.launch.py', 'start_slam',
                {'use_sim_time': use_sim_time}),
        Node(
            package='offline_map_package', executable='semantic_mapper',
            name='semantic_mapper', output='screen',
            additional_env={'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'},
            parameters=[
                os.path.join(offline_share, 'config', 'semantic_mapping.yaml'),
                {'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
                 'pose_source': LaunchConfiguration('pose_source'),
                 'trajectory_id': ParameterValue(
                     LaunchConfiguration('trajectory_id'), value_type=int),
                 'resolution': ParameterValue(LaunchConfiguration('resolution'), value_type=float),
                 'save_directory': ParameterValue(
                     LaunchConfiguration('save_directory'), value_type=str)}]),
        Node(
            package='offline_map_package', executable='map_save_gui',
            name='map_save_gui', output='screen',
            condition=IfCondition(LaunchConfiguration('start_gui')),
            parameters=[{'use_sim_time': ParameterValue(use_sim_time, value_type=bool)}]),
        include(rviz_share, 'limo_viz_slam.launch.py', 'start_rviz',
                {'use_sim_time': use_sim_time,
                 'rviz_config': os.path.join(rviz_share, 'config', 'offline_map.rviz')}),
    ])
