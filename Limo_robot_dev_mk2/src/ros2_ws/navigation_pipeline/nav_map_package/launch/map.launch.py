import signal

from launch import LaunchDescription
from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown, matches_action
from launch.events.process import SignalProcess
from launch_ros.actions import Node

def generate_launch_description():
    protected_process = {
        'prefix': 'python3 -m nav_map_package.signal_guard',
        # The normal launch escalation must not kill the map saver while the
        # final request is still running. OnProcessExit below stops it sooner.
        'sigterm_timeout': '60.0',
        'sigkill_timeout': '5.0',
    }

    filtering_roi = {
        'roi_x_min_m': 0.0,
        'roi_x_max_m': 1.85,
        'roi_width_near_m': 0.60,
        'roi_width_far_m': 2.65,
    }

    metric_bev = Node(
        package='nav_map_package',
        executable='metric_bev',
        name='metric_bev',
        output='screen',
        parameters=[{
            'enable_telemetry': False,
        }],
        **protected_process,
    )

    filtering_turquoise = Node(
        package='nav_map_package',
        executable='filtering',
        name='filtering_turquoise',
        output='screen',
        parameters=[{
            'color': 'TURQUOISE',
            **filtering_roi,
        }],
        **protected_process,
    )

    filtering_white = Node(
        package='nav_map_package',
        executable='filtering',
        name='filtering_white',
        output='screen',
        parameters=[{
            'color': 'WHITE',
            **filtering_roi,
        }],
        **protected_process,
    )

    filtering_magenta = Node(
        package='nav_map_package',
        executable='filtering',
        name='filtering_magenta',
        output='screen',
        parameters=[{
            'color': 'MAGENTA',
            **filtering_roi,
        }],
        **protected_process,
    )

    cv_map_display = Node(
        package='nav_map_package',
        executable='cv_map_display',
        name='cv_map_display',
        output='screen',
        parameters=[{
            'enable_telemetry': False,
        }],
        **protected_process,
    )

    nav_map = Node(
        package='nav_map_package',
        executable='nav_map',
        name='nav_map',
        output='screen',
        parameters=[{
            'global_frame': 'map',
            'static_map_topic': '/map',
            'cv_grid_topic': '/limo/nav_map_package/cv_map_display/cv_map_occupancy_grid',
            'scan_topic': '/scan',
            'output_topic': '/limo/nav_map_package/nav_map/combined_grid',
            'publish_rate_hz': 10.0,
            'lidar_cost': 100,
        }]
    )

    nav2_map_saver = Node(
        package='nav2_map_server',
        executable='map_saver_server',
        name='map_saver',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'save_map_timeout': 30.0,
            'free_thresh_default': 0.25,
            'occupied_thresh_default': 0.65,
            'map_subscribe_transient_local': True,
        }],
        **protected_process,
    )

    map_saver_lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_nav_map_saver',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': ['map_saver'],
        }],
        **protected_process,
    )

    saver_node = Node(
        package='nav_map_package',
        executable='map_saver',
        name='nav_map_saver',
        output='screen',
        parameters=[{
            'map_topic': '/limo/nav_map_package/nav_map/combined_grid',
            'request_service': '/limo/nav_map_package/map_saver/save_map',
            'nav2_service': '/map_saver/save_map',
            'map_name': 'limo_map',
        }],
        **protected_process,
    )

    protected_nodes = (
        metric_bev,
        filtering_turquoise,
        filtering_white,
        filtering_magenta,
        cv_map_display,
        nav2_map_saver,
        map_saver_lifecycle,
        saver_node,
    )

    stop_protected_nodes = [
        EmitEvent(
            event=SignalProcess(
                signal_number=signal.SIGTERM,
                process_matcher=matches_action(node),
            )
        )
        for node in protected_nodes
    ]

    shutdown_after_nav_map = RegisterEventHandler(
        OnProcessExit(
            target_action=nav_map,
            on_exit=stop_protected_nodes + [
                EmitEvent(
                    event=Shutdown(
                        reason='nav_map finished after final map save'
                    )
                )
            ],
        )
    )
    
    return LaunchDescription([
        metric_bev,
        filtering_turquoise,
        filtering_white,
        filtering_magenta,
        cv_map_display,
        nav_map,
        nav2_map_saver,
        map_saver_lifecycle,
        saver_node,
        shutdown_after_nav_map,
    ])
