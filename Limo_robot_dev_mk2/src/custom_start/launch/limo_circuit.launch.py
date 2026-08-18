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
    model_path = os.path.join(custom_start_share, 'models')

    world = LaunchConfiguration('world')
    gui = LaunchConfiguration('gui')

    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(limo_car_share, 'launch', 'ackermann.launch.py')
        ),
        launch_arguments={'use_sim_time': 'true'}.items(),
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
        SetEnvironmentVariable(
            'GAZEBO_MODEL_PATH',
            [model_path, ':', EnvironmentVariable('GAZEBO_MODEL_PATH', default_value='')],
        ),
        robot_state_publisher,
        gazebo_server,
        gazebo_client,
        spawn_robot,
    ])
