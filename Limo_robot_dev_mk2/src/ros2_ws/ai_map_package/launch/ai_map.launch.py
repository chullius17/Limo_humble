from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    costmap = Node(
        package='map_package',
        executable='costmap',
        name='costmap_boardwalk',
        output='screen',
        parameters=[{
            'AI_mode': True
        }]
    )

    mapper_mgn = Node(
        package='map_package',
        executable='mapper',
        name='mapper_boardwalk',
        output='screen',
        parameters=[{
            'color': 'MAGENTA'
        }]
    )

    mapper_red = Node(
        package='map_package',
        executable='mapper',
        name='mapper_solid',
        output='screen',
        parameters=[{
            'color': 'RED'
        }]
    )

    mapper_grn = Node(
        package='map_package',
        executable='mapper',
        name='mapper_dashed',
        output='screen',
        parameters=[{
            'color': 'GREEN'
        }]
    )

    display_node = Node(
        package='map_package',
        executable='map_display',
        name='map_displayer',
        output='screen',
        parameters=[{
            'AI_mode': True
        }]
    )

    saver_node = Node(
        package='map_package',
        executable='map_save',
        name='map_saver',
        output='screen',
        parameters=[{
            'AI_mode': True
        }]  
    )
    
    return LaunchDescription([
        costmap,
        mapper_mgn,
        mapper_red,
        mapper_grn,
        display_node,
        saver_node
    ])
