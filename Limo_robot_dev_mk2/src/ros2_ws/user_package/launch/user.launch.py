from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    ai_mode = LaunchConfiguration('ai_mode')

    ai_mode_arg = DeclareLaunchArgument(
        'ai_mode',
        default_value='false',
        description='Store control plots in ai_control_logs when enabled'
    )

    visualizer = Node(
        package='user_package',
        executable='visualizer',
        name='visualizer',
        output='screen'
    )

    ctrl_viz = Node(
        package='user_package',
        executable='ctrl_viz',
        name='ctrl_viz',
        output='screen',
        parameters=[{'ai_mode': ai_mode}],
    )

    user_srv = Node(
        package='user_package',
        executable='user',
        name='user_server',
        output='screen'
    )

    gui = Node(
        package='user_package',
        executable='gui',
        name='gui',
        output='screen'
    )

    return LaunchDescription([
        ai_mode_arg,
        visualizer,
        ctrl_viz,
        user_srv,
        gui
    ])
