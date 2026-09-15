from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
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
    classification_blue_distance_threshold = DeclareLaunchArgument(
        'classification_blue_distance_threshold_px',
        default_value='5.0',
        description='Maximum distance in pixels from a blue pixel before a white pixel is classified as magenta',
    )
    classification_magenta_distance_threshold = DeclareLaunchArgument(
        'classification_magenta_distance_threshold_px',
        default_value='5.0',
        description='Maximum distance in pixels from magenta for propagating the classification',
    )

    lane_node = Node(
            package='nav_cv_package',
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
        package='nav_cv_package',
        executable='boundaries',
        name='boundary_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'enable_telemetry': False,
            'roi_y_min': 0.5,
            'roi_y_max': 1.0,
        }]
    )

    bev_node = Node(
        package='nav_cv_package',
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

    classification_node = Node(
        package='nav_cv_package',
        executable='classification',
        name='classification',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'blue_distance_threshold_px': ParameterValue(
                classification_blue_distance_threshold_px,
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
            on_start=[bev_node]
        )
    )

    return LaunchDescription([
        classification_blue_distance_threshold,
        classification_magenta_distance_threshold,
        lane_node,
        classification_node,
        boundary_trigger,
        bev_trigger
    ])
