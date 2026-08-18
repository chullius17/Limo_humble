from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    use_controller = LaunchConfiguration('use_controller')

    use_controller_arg = DeclareLaunchArgument(
        'use_controller',
        default_value='true',
        description='Send planned paths to the controller action server'
    )

    center_lanes_network = Node(
        package='nav_traj_package',
        executable='routes',
        name='center_lanes_network',
        output='screen',
        parameters=[{
            'flag': 'CENTER_ROAD'
        }]
    )

    open_spaces_network = Node(
        package='nav_traj_package',
        executable='routes',
        name='open_spaces_network',
        output='screen',
        parameters=[{
            'flag': 'OPEN'
        }]
    )

    network_combination = Node(
        package='nav_traj_package',
        executable='route_combinator',
        name='network_combination',
        output='screen',
    )

    astar = Node(
        package='nav_traj_package',
        executable='astar',
        name='astar_server',
        output='screen',
    )

    coordinator = Node(
        package='nav_traj_package',
        executable='coordinator',
        name='mission_coordinator',
        output='screen',
        parameters=[{'use_controller': use_controller}],
    )

    return LaunchDescription([
        use_controller_arg,
        center_lanes_network,
        open_spaces_network,
        network_combination,
        astar,
        coordinator
    ])
