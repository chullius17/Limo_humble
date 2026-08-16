from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    mapper_roi = {
        'roi_x_min_m': 0.0,
        'roi_x_max_m': 1.85,
        'roi_width_near_m': 0.62,
        'roi_width_far_m': 2.60,
    }

    costmap = Node(
        package='ai_map_package',
        executable='costmap',
        name='ai_costmap',
        output='screen'
    )

    mapper_mgn = Node(
        package='ai_map_package',
        executable='mapper',
        name='mapper_magenta',
        output='screen',
        parameters=[{
            'color': 'MAGENTA',
            **mapper_roi,
        }]
    )

    mapper_red = Node(
        package='ai_map_package',
        executable='mapper',
        name='mapper_red',
        output='screen',
        parameters=[{
            'color': 'RED',
            **mapper_roi,
        }]
    )

    mapper_grn = Node(
        package='ai_map_package',
        executable='mapper',
        name='mapper_green',
        output='screen',
        parameters=[{
            'color': 'GREEN',
            **mapper_roi,
        }]
    )

    display_node = Node(
        package='ai_map_package',
        executable='map_display',
        name='ai_map_displayer',
        output='screen'
    )

    saver_node = Node(
        package='ai_map_package',
        executable='map_save',
        name='ai_map_saver',
        output='screen'
    )
    
    return LaunchDescription([
        costmap,
        mapper_mgn,
        mapper_red,
        mapper_grn,
        display_node,
        saver_node
    ])
