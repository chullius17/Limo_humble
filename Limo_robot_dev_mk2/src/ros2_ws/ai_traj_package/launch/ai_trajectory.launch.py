from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    center_lanes_network = Node(
        package='ai_traj_package',
        executable='ai_routes',
        name='ai_center_lanes_network',
        output='screen',
        parameters=[{
            'flag': 'CENTER_ROAD'
        }]
    )

    open_spaces_network = Node(
        package='ai_traj_package',
        executable='ai_routes',
        name='ai_open_spaces_network',
        output='screen',
        parameters=[{
            'flag': 'OPEN'
        }]
    )

    network_combination = Node(
        package='ai_traj_package',
        executable='ai_route_combinator',
        name='ai_network_combination',
        output='screen',
    )

    astar = Node(
        package='ai_traj_package',
        executable='ai_astar',
        name='ai_astar_server',
        output='screen',
    )

    ai_coordinator = Node(
        package='ai_traj_package',
        executable='ai_coordinator',
        name='ai_mission_coordinator',
        output='screen',
    )

    return LaunchDescription([
        center_lanes_network,
        open_spaces_network,
        network_combination,
        astar,
        # ai_coordinator
    ])
