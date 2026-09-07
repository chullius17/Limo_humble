"""Start the physical LIMO robot with wheel odometry and EKF fusion."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import (
    AnyLaunchDescriptionSource,
    PythonLaunchDescriptionSource,
)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    custom_start_share = get_package_share_directory('custom_start')
    limo_base_share = get_package_share_directory('limo_base')

    port_name = LaunchConfiguration('port_name')
    use_lidar = LaunchConfiguration('use_lidar')
    use_camera = LaunchConfiguration('use_camera')
    camera_x = LaunchConfiguration('camera_x')
    camera_y = LaunchConfiguration('camera_y')
    camera_z = LaunchConfiguration('camera_z')
    camera_roll = LaunchConfiguration('camera_roll')
    camera_pitch = LaunchConfiguration('camera_pitch')
    camera_yaw = LaunchConfiguration('camera_yaw')
    open_rviz = LaunchConfiguration('open_rviz')

    limo_base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(limo_base_share, 'launch', 'limo_base.launch.py')
        ),
        launch_arguments={
            'port_name': port_name,
            'odom_frame': 'odom',
            'base_frame': 'base_link',
            'odom_topic_name': 'odom',
            'imu_topic_name': '/limo/imu',
            # The EKF is the only publisher of odom -> base_link.
            'pub_odom_tf': 'false',
        }.items(),
    )

    imu_transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_to_imu',
        arguments=[
            '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
            'base_link', 'imu_link',
        ],
    )

    lidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(limo_base_share, 'launch', 'open_ydlidar_launch.py')
        ),
        condition=IfCondition(use_lidar),
    )

    camera_mount_transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_to_camera',
        condition=IfCondition(use_camera),
        arguments=[
            camera_x, camera_y, camera_z,
            # Foxy uses the legacy positional order: yaw, pitch, roll.
            camera_yaw, camera_pitch, camera_roll,
            'base_link', 'camera_link',
        ],
    )

    camera = GroupAction(
        condition=IfCondition(use_camera),
        actions=[
            SetRemap(
                src='/camera/color/image_raw',
                dst='/rgb/image_raw',
            ),
            SetRemap(
                src='/camera/color/camera_info',
                dst='/rgb/camera_info',
            ),
            SetRemap(
                src='/camera/depth/image_raw',
                dst='/depth_camera/depth/image_raw',
            ),
            SetRemap(
                src='/camera/depth/camera_info',
                dst='/depth_camera/depth/camera_info',
            ),
            SetRemap(
                src='/camera/depth/points',
                dst='/depth/points',
            ),
            IncludeLaunchDescription(
                AnyLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare('astra_camera'),
                        'launch',
                        'dabai_u3.launch.xml',
                    ])
                ),
                launch_arguments={
                    'camera_name': 'camera',
                    'color_width': '640',
                    'color_height': '480',
                    'color_fps': '20',
                    'depth_width': '640',
                    'depth_height': '400',
                    'depth_fps': '20',
                    'enable_point_cloud': 'true',
                }.items(),
            ),
        ],
    )

    ekf = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            os.path.join(custom_start_share, 'config', 'ekf.yaml'),
            {
                'use_sim_time': False,
            },
        ],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        condition=IfCondition(open_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'port_name', default_value='ttyTHS1',
            description='Serial device name used by the physical LIMO.',
        ),
        DeclareLaunchArgument(
            'use_lidar', default_value='true',
            description='Start the physical YDLidar driver.',
        ),
        DeclareLaunchArgument(
            'use_camera', default_value='true',
            description='Start the physical Astra depth camera.',
        ),
        DeclareLaunchArgument(
            'camera_x', default_value='0.10',
            description='Camera X offset from base_link in metres.',
        ),
        DeclareLaunchArgument(
            'camera_y', default_value='0.0',
            description='Camera Y offset from base_link in metres.',
        ),
        DeclareLaunchArgument(
            'camera_z', default_value='0.065',
            description='Camera Z offset from base_link in metres.',
        ),
        DeclareLaunchArgument(
            'camera_roll', default_value='0.0',
            description='Camera roll relative to base_link in radians.',
        ),
        DeclareLaunchArgument(
            'camera_pitch', default_value='0.0',
            description='Camera pitch relative to base_link in radians.',
        ),
        DeclareLaunchArgument(
            'camera_yaw', default_value='0.0',
            description='Camera yaw relative to base_link in radians.',
        ),
        DeclareLaunchArgument(
            'open_rviz', default_value='false',
            description='Start RViz.',
        ),
        limo_base,
        imu_transform,
        lidar,
        camera_mount_transform,
        camera,
        ekf,
        rviz,
    ])
