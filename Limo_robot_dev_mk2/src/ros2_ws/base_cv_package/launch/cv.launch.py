from launch import LaunchDescription
from launch.actions import  RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node

def generate_launch_description():
    bev_node = Node(
        package='base_cv_package',
        executable='bev_node',
        name='bev_node',
        output='screen',
        emulate_tty=True,
    )

    lane_node = Node(
        package='base_cv_package',
        executable='lane_detector',
        name='lane_node',
        output='screen',
        emulate_tty=True,
    )

    boundary_node = Node( 
        package='base_cv_package',
        executable='boundaries',
        name='boundary_node',
        output='screen',
        emulate_tty=True,
    )

    boundary_trigger = RegisterEventHandler(
        OnProcessStart(
            target_action=lane_node,
            on_start=[boundary_node]
        )
    )

    bev_trigger = RegisterEventHandler(
        OnProcessStart(
            target_action=boundary_node,
            on_start=[bev_node]
        )
    )

    return LaunchDescription([
        lane_node,
        boundary_trigger,
        bev_trigger
    ])
