"""Launch the combined metric BEV and temporally filtered mapper."""

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    rviz_config = PathJoinSubstitution([
        FindPackageShare('limo_rviz'),
        'config',
        'mapping.rviz',
    ])

    metric_bev = Node(
        package='simple_mapping',
        executable='metric_bev',
        name='metric_bev',
        output='screen',
        emulate_tty=True,
    )

    mapper = Node(
        package='simple_mapping',
        executable='mapper',
        name='simple_mapper',
        output='screen',
        emulate_tty=True,
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='mapping_rviz',
        output='screen',
        arguments=['-d', rviz_config],
    )

    return LaunchDescription([
        metric_bev,
        mapper,
        rviz,
    ])
