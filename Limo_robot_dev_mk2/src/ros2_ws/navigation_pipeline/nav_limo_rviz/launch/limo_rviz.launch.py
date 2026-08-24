from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
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

    map_yaml_file = str(
        project_root
        / 'ros2_maps'
        / 'nav_pipeline'
        / 'limo_map.yaml'
    )

    use_sim_time = {'use_sim_time': True}
    start_map_server = LaunchConfiguration('start_map_server')
    publish_static_map_to_odom = LaunchConfiguration(
        'publish_static_map_to_odom'
    )
    rviz_config = PathJoinSubstitution([
        FindPackageShare('nav_limo_rviz'),
        'config',
        'online_nav_map.rviz',
    ])

    # =========================
    # TF STATICI
    # =========================

    tf_map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_to_odom',
        arguments=['0', '0', '0', 
                   '0', '0', '0', 
                   'map', 
                   'odom'],
        output='screen',
        parameters=[use_sim_time],
        condition=IfCondition(publish_static_map_to_odom),
    )

    # =========================
    # MAP SERVER (Nav2)
    # =========================

    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
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
        name='lifecycle_manager_map_server',
        output='screen',
        parameters=[
            use_sim_time,
            {'autostart': True},
            {'node_names': ['map_server']}
        ]
    )

    # =========================
    # RVIZ
    # =========================

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[use_sim_time]
    )

    # =========================
    # LAUNCH DELAYED
    # =========================

    tf_launch = TimerAction(
        period=0.5,
        actions=[tf_map_to_odom]
        )

    map_launch = TimerAction(
        period=1.0,
        actions=[map_server],
        condition=IfCondition(start_map_server),
    )

    rviz_launch = TimerAction(
        period=1.5,
        actions=[rviz]                 
    )

    lifecycle_launch = TimerAction(
        period=6.0,
        actions=[lifecycle_manager],
        condition=IfCondition(start_map_server),
    )

    # =========================
    # LAUNCH FINAL
    # =========================

    return LaunchDescription([
        DeclareLaunchArgument(
            'start_map_server',
            default_value='true',
            description='Start the default single Nav2 map server.',
        ),
        DeclareLaunchArgument(
            'publish_static_map_to_odom',
            default_value='true',
            description=(
                'Publish a static map-to-odom transform. Disable this when '
                'AMCL broadcasts the transform.'
            ),
        ),
        tf_launch,
        map_launch,
        lifecycle_launch,
        rviz_launch
    ])
