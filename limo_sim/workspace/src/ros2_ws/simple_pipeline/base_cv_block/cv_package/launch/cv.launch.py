from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    classification_blue_distance_threshold_px = LaunchConfiguration(
        'classification_blue_distance_threshold_px'
    )
    classification_magenta_distance_threshold_px = LaunchConfiguration(
        'classification_magenta_distance_threshold_px'
    )
    classification_blue_max_distance_threshold_px = LaunchConfiguration(
        'classification_blue_max_distance_threshold_px'
    )

    classification_blue_distance_threshold = DeclareLaunchArgument(
        'classification_blue_distance_threshold_px',
        default_value='10.0',
        description='Distance from blue beyond which white becomes magenta',
    )
    classification_magenta_distance_threshold = DeclareLaunchArgument(
        'classification_magenta_distance_threshold_px',
        default_value='10.0',
        description='Distance used to propagate the magenta classification',
    )
    classification_blue_max_distance_threshold = DeclareLaunchArgument(
        'classification_blue_max_distance_threshold_px',
        default_value='16.0',
        description='Maximum blue distance; farther white is discarded',
    )

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
            'enable_telemetry': False,
            'roi_y_min': 0.0,
            'roi_y_max': 1.0,
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
        executable='bev_and_clas',
        name='bev_and_clas',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'enable_telemetry': True,
            'telemetry_window_size': 60,
            'telemetry_log_interval_frames': 30,
            'camera_info_topic': '/rgb/camera_info',
            'depth_topic': (
                'limo/cv_package/depth_correction/depth_corrected/raw'
            ),
            'input_crop_y_min': 0.5,
            'bev_width': 600,
            'bev_height': 300,
            'bev_resolution': 0.01,
            'projection_stride': 3,
            'use_gpu': True,
            'max_processing_fps': 12.0,
            'blue_distance_threshold_px': ParameterValue(
                classification_blue_distance_threshold_px,
                value_type=float,
            ),
            'blue_max_distance_threshold_px': ParameterValue(
                classification_blue_max_distance_threshold_px,
                value_type=float,
            ),
            'magenta_distance_threshold_px': ParameterValue(
                classification_magenta_distance_threshold_px,
                value_type=float,
            ),
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
        classification_blue_distance_threshold,
        classification_blue_max_distance_threshold,
        classification_magenta_distance_threshold,
        lane_node,
        depth_correction_node,
        boundary_trigger,
        bev_trigger
    ])
