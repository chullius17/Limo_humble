from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from pathlib import Path


def find_project_root(start: Path):
    """Find the project root from source, build, or install paths."""
    for candidate in [start] + list(start.parents):
        if (candidate / "src" / "ros2_ws").is_dir():
            return candidate
    return None


def generate_launch_description():

    project_root = find_project_root(Path(__file__).resolve())
    if project_root is None:
        raise RuntimeError("Root del progetto LIMO non trovata")

    default_map_path = str(project_root / "ai_ros2_maps" / "limo_map.yaml")

    map_yaml_file = LaunchConfiguration('map_yaml_file')

    declare_map_yaml_cmd = DeclareLaunchArgument(
        'map_yaml_file',
        default_value=default_map_path,
        description='Path to map yaml'
    )

    use_sim_time = {'use_sim_time': True}
    rviz_config = PathJoinSubstitution([
        FindPackageShare('ai_limo_rviz'),
        'config',
        'ai_limo_rviz.rviz',
    ])

    # =========================
    # TF STATICI
    # =========================

    tf_map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='ai_static_tf_map_to_odom',
        arguments=['0', '0', '0', 
                   '0', '0', '0', 
                   'map', 
                   'odom'],
        output='screen',
        parameters=[use_sim_time]
    )

    tf_launch = TimerAction(
        period=0.5,
        actions=[tf_map_to_odom]
    )

    # =========================
    # MAP SERVER (Nav2)
    # =========================

    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='ai_map_server',
        output='screen',
        parameters=[
            {'yaml_filename': map_yaml_file},
            {'frame_id': 'map'},
            use_sim_time
        ]
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='ai_lifecycle_manager_map_server',
        output='screen',
        parameters=[
            use_sim_time,
            {'autostart': True},
            {'node_names': ['ai_map_server']}
        ]
    )

    map_launch = TimerAction(
        period=2.0,
        actions=[map_server]
    )

    lifecycle_launch = TimerAction(
        period=3.5,
        actions=[lifecycle_manager]
    )

    # =========================
    # RVIZ
    # =========================

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='ai_rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[use_sim_time]
    )

    rviz_launch = TimerAction(
        # RViz must subscribe before the static map is published. Its default
        # Map display uses volatile durability and cannot retrieve old samples.
        period=0.5,
        actions=[rviz]
    )

    # =========================
    # LAUNCH FINAL
    # =========================

    return LaunchDescription([
        declare_map_yaml_cmd,

        tf_launch,
        map_launch,
        lifecycle_launch,
        rviz_launch
    ])
