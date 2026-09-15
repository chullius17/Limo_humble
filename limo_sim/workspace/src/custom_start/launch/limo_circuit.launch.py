"""Launch Gazebo with the custom LIMO circuit and the Ackermann LIMO robot."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    custom_start_share = get_package_share_directory('custom_start')
    limo_car_share = get_package_share_directory('limo_car')

    default_world = os.path.join(
        custom_start_share, 'worlds', 'limo_circuit_world.world'
    )
    ekf_config = os.path.join(custom_start_share, 'config', 'ekf.yaml')
    model_path = os.path.join(custom_start_share, 'models')

    world = LaunchConfiguration('world')
    gui = LaunchConfiguration('gui')
    use_sim_time = LaunchConfiguration('use_sim_time')
    camera_x = LaunchConfiguration('camera_x')
    camera_y = LaunchConfiguration('camera_y')
    camera_z = LaunchConfiguration('camera_z')
    camera_roll = LaunchConfiguration('camera_roll')
    camera_pitch = LaunchConfiguration('camera_pitch')
    camera_yaw = LaunchConfiguration('camera_yaw')

    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(limo_car_share, 'launch', 'ackermann.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    # Reproduce the physical-camera TF chain used by limo_real.launch.py.
    camera_mount_transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='sim_base_link_to_camera',
        arguments=[
            camera_x, camera_y, camera_z,
            # Foxy positional order: yaw, pitch, roll.
            camera_yaw, camera_pitch, camera_roll,
            'base_link', 'camera_link',
        ],
    )

    depth_optical_transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='sim_camera_to_depth_optical',
        arguments=[
            '0.0', '0.0', '0.0',
            # Same optical convention used by the physical Astra driver:
            # roll=-pi/2, pitch=0, yaw=-pi/2.
            '-1.57079632679', '0.0', '-1.57079632679',
            'camera_link', 'depth_camera_frame_optical',
        ],
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            ekf_config,
            {'use_sim_time': use_sim_time},
        ],
    )

    gazebo_server = ExecuteProcess(
        cmd=[
            'gzserver', '--verbose', world,
            '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so',
        ],
        output='screen',
    )
    gazebo_client = ExecuteProcess(
        cmd=['gzclient'],
        condition=IfCondition(gui),
        output='screen',
    )

    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'limo',
            '-x', '0.0', '-y', '0.0', '-z', '0.30', '-Y', '0.0',
        ],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world', default_value=default_world,
            description='Absolute path of the Gazebo world containing the circuit.',
        ),
        DeclareLaunchArgument(
            'gui', default_value='true',
            description='Start the Gazebo graphical client.',
        ),
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use the clock published by Gazebo.',
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
        SetEnvironmentVariable(
            'GAZEBO_MODEL_PATH',
            [model_path, ':', EnvironmentVariable('GAZEBO_MODEL_PATH', default_value='')],
        ),
        robot_state_publisher,
        camera_mount_transform,
        depth_optical_transform,
        gazebo_server,
        gazebo_client,
        spawn_robot,
        ekf_node,
    ])
