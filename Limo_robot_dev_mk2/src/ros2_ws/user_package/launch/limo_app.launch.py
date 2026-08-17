from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    use_controller = LaunchConfiguration('use_controller')

    use_controller_arg = DeclareLaunchArgument(
        'use_controller',
        default_value='true',
        description='Start and connect the trajectory controller'
    )

    trajectory_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('traj_package'),
                'launch',
                'trajectory.launch.py'
            )
        ),
        launch_arguments={'use_controller': use_controller}.items()
    )

    control_node = Node(
        package='limo_controller', 
        executable='controller',
        name='controller',
        output='screen',
        condition=IfCondition(use_controller)
    )

    user_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('user_package'),
                'launch',
                'user.launch.py'
            )
        )
    )

    return LaunchDescription([
        use_controller_arg,
        trajectory_launch,
        control_node,
        user_launch
    ])
