from launch import LaunchDescription
from launch.actions import  RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node

def generate_launch_description():
    lane_node = Node(
        package='ai_cv_package',
        executable='lane_detector',
        name='lane_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'enable_telemetry': False,
            'rgb_topic': '/rgb/image_raw',
        }]
    )
    
    boundary_node = Node( 
        package='ai_cv_package',
        executable='boundaries',
        name='boundary_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'enable_telemetry': False,
            'roi_y_min': 0.54,
            'roi_y_max': 0.98,
        }]
    )

    bev_node = Node(
        package='ai_cv_package',
        executable='bev_node',
        name='bev_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'enable_telemetry': False,
            'camera_info_topic': '/rgb/camera_info',
            'depth_topic': '/depth_camera/depth/image_raw'
        }]
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
