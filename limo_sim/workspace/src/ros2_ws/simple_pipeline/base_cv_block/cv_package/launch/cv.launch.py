from launch import LaunchDescription
from launch.actions import RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node


def generate_launch_description():
    lane_node = Node(
            package='cv_package',
            executable='lane_detector',
            name='lane_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'enable_telemetry': False,
                'rgb_topic': '/rgb/image_raw',
                'roi_y_min': 0.1,
                'roi_y_max': 1.0,
            }]
        )

    boundary_node = Node(
        package='cv_package',
        executable='boundaries',
        name='boundary_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'enable_telemetry': True,
            'roi_y_min': 0.0,
            'roi_y_max': 1.0,
            'point_voxel_size': 5,
            'camera_info_topic': '/rgb/camera_info',
            'depth_topic': (
                'limo/cv_package/depth_correction/depth_corrected/raw'
            ),
            'fallback_depth_topic': '/depth_camera/depth/image_raw',
            'corrected_depth_timeout_sec': 1.0,
            'pointcloud_topic': 'limo/cv_package/boundaries/points',
            'input_crop_y_min': 0.5,
            'pointcloud_min_depth_m': 0.1,
            'pointcloud_max_depth_m': 5.0,
            'max_depth_time_delta_sec': 0.1,
            'blue_radius_min_m': 0.10,
            'blue_radius_max_m': 0.16,
        }]
    )

    depth_correction_node = Node(
        package='cv_package',
        executable='depth_correction',
        name='depth_correction',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'input_topic': '/depth_camera/depth/image_raw',
            'camera_info_topic': '/depth_camera/depth/camera_info',
        }],
    )

    bev_node = Node(
        package='cv_package',
        executable='bev_node',
        name='simple_bev',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'enable_telemetry': True,
            'telemetry_window_size': 60,
            'telemetry_log_interval_frames': 30,
            'pointcloud_topic': 'limo/cv_package/boundaries/points',
            'bev_pointcloud_topic': 'limo/cv_package/bev/points',
            'plane_frame': 'base_link',
            'plane_z': 0.0,
        }],
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
            on_start=[TimerAction(period=10.0, actions=[bev_node])]
        )
    )

    return LaunchDescription([
        lane_node,
        depth_correction_node,
        boundary_trigger,
        bev_trigger
    ])
