from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    mapper_roi = {
        'roi_x_min_m': 0.0,
        'roi_x_max_m': 1.85,
        'roi_width_near_m': 0.60,
        'roi_width_far_m': 2.65,
    }

    costmap = Node(
        package='map_package',
        executable='costmap',
        name='costmap_boardwalk',
        output='screen',
        parameters=[{
            'enable_telemetry': False,
        }]
    )

    mapper_turquoise = Node(
        package='map_package',
        executable='mapper',
        name='mapper_solid',
        output='screen',
        parameters=[{
            'color': 'TURQUOISE',
            **mapper_roi,
        }]
    )

    mapper_white = Node(
        package='map_package',
        executable='mapper',
        name='mapper_white',
        output='screen',
        parameters=[{
            'color': 'WHITE',
            **mapper_roi,
        }]
    )

    display_node = Node(
        package='map_package',
        executable='map_display',
        name='map_displayer',
        output='screen',
        parameters=[{
            'enable_telemetry': False,
        }]
    )

    saver_node = Node(
        package='map_package',
        executable='map_save',
        name='map_saver',
        output='screen'
    )
    
    return LaunchDescription([
        costmap,
        mapper_turquoise,
        mapper_white,
        display_node,
        saver_node
    ])
