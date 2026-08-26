from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    use_controller = LaunchConfiguration('use_controller')
    use_sim_time = LaunchConfiguration('use_sim_time')
    ai_mode = LaunchConfiguration('ai_mode')

    use_controller_arg = DeclareLaunchArgument(
        'use_controller',
        default_value='true',
        description='Start and connect the trajectory controller'
    )

    ai_mode_arg = DeclareLaunchArgument(
        'ai_mode',
        default_value='false',
        description='Use AI-specific control-log storage'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use Gazebo/rosbag time instead of the robot clock'
    )

    trajectory_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('nav_traj_package'),
                'launch',
                'trajectory.launch.py'
            )
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    controller_params_file = os.path.join(
        get_package_share_directory('nav_limo_controller'),
        'config',
        'mppi_control_params.yaml'
    )
    control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('nav_limo_controller'),
                'launch',
                'control.launch.py'
            )
        ),
        launch_arguments={
            'params_file': controller_params_file,
            'use_sim_time': use_sim_time,
        }.items(),
        condition=IfCondition(use_controller)
    )

    user_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('nav_user_package'),
                'launch',
                'user.launch.py'
            )
        ),
        launch_arguments={'ai_mode': ai_mode}.items()
    )

    return LaunchDescription([
        use_controller_arg,
        use_sim_time_arg,
        ai_mode_arg,
        trajectory_launch,
        control_launch,
        user_launch
    ])
