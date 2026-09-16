from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)
    lane_node = Node(
            package='cv_package',
            executable='lane_detector',
            name='lane_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'use_sim_time': use_sim_time,
                'enable_telemetry': False,
                'rgb_topic': '/rgb/image_raw',
                'roi_y_min': 0.1,
                'roi_y_max': 1.0,
                'opencv_num_threads': 1,
            }]
        )

    visual_ptcld_node = Node(
        package='cv_package',
        executable='visual_ptcld',
        name='visual_ptcld',
        output='screen',
        emulate_tty=True,
        additional_env={
            'OPENBLAS_NUM_THREADS': '1',
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'BLIS_NUM_THREADS': '1',
        },
        parameters=[{
            'use_sim_time': use_sim_time,
            'opencv_num_threads': 1,
            'enable_telemetry': True,
            'roi_y_min': 0.0,
            'roi_y_max': 1.0,
            # Optional images are disabled; PointCloud2 is always published.
            'enable_debug_publications': False,
            'point_voxel_size': 5,
            # Downsample all published BEV classes in class-aware 2 cm cells.
            'pointcloud_voxel_size_m': 0.02,
            # 7x7 keeps an approximately three-pixel-wide inner blue boundary.
            'blue_boundary_kernel_size': 7,
            'camera_info_topic': '/rgb/camera_info',
            'depth_topic': (
                'limo/cv_package/depth_correction/depth_corrected/raw'
            ),
            'fallback_depth_topic': '/depth_camera/depth/image_raw',
            'corrected_depth_timeout_sec': 1.0,
            'fallback_depth_width': 320,
            'fallback_depth_height': 120,
            'pointcloud_topic': 'limo/cv_package/visual_ptcld/points',
            'input_crop_y_min': 0.5,
            'pointcloud_min_depth_m': 0.1,
            'pointcloud_max_depth_m': 2.5,
            'blue_radius_min_m': 0.15,
            'blue_radius_max_m': 0.25,
            # White points in the blue distance band seed class 4 (boardwalk).
            # Restore exact metric point distances: blue neighbors, then seed neighbors.
            'enable_boardwalk': True,
            'boardwalk_propagation_radius_m': 0.15,
            # OpenCV applies the 7x7 blue filter; cKDTree runs both point passes.
            'telemetry_window_size': 60,
            'telemetry_log_interval_frames': 30,
        }]
    )

    depth_correction_node = Node(
        package='cv_package',
        executable='depth_correction',
        name='depth_correction',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'use_sim_time': use_sim_time,
            'input_topic': '/depth_camera/depth/image_raw',
            'camera_info_topic': '/depth_camera/depth/camera_info',
            'enable_telemetry': False,
        }],
    )

    boundary_trigger = RegisterEventHandler(
        OnProcessStart(
            target_action=lane_node,
            on_start=[visual_ptcld_node]
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        lane_node,
        depth_correction_node,
        boundary_trigger,
    ])
