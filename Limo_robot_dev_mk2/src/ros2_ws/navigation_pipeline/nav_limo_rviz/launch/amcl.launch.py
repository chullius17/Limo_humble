"""Launch and automatically activate AMCL for the LIMO robot."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    base_frame_id = LaunchConfiguration('base_frame_id')
    odom_frame_id = LaunchConfiguration('odom_frame_id')
    global_frame_id = LaunchConfiguration('global_frame_id')
    scan_topic = LaunchConfiguration('scan_topic')
    map_topic = LaunchConfiguration('map_topic')
    tf_broadcast = LaunchConfiguration('tf_broadcast')

    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[{
            'use_sim_time': ParameterValue(
                use_sim_time,
                value_type=bool,
            ),
            'base_frame_id': base_frame_id,
            'odom_frame_id': odom_frame_id,
            'global_frame_id': global_frame_id,
            'scan_topic': scan_topic,
            'map_topic': map_topic,
            'tf_broadcast': ParameterValue(
                tf_broadcast,
                value_type=bool,
            ),
        }],
    )

    # AMCL is a lifecycle node. This manager configures and activates it
    # automatically as soon as both processes are running.
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_amcl',
        output='screen',
        parameters=[{
            'use_sim_time': ParameterValue(
                use_sim_time,
                value_type=bool,
            ),
            'autostart': True,
            'node_names': ['amcl'],
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use the simulation or rosbag clock.',
        ),
        DeclareLaunchArgument(
            'base_frame_id',
            default_value='base_link',
            description='Robot base frame used by AMCL.',
        ),
        DeclareLaunchArgument(
            'odom_frame_id',
            default_value='odom',
            description='Odometry frame used by AMCL.',
        ),
        DeclareLaunchArgument(
            'global_frame_id',
            default_value='map',
            description='Global localization frame used by AMCL.',
        ),
        DeclareLaunchArgument(
            'scan_topic',
            default_value='/scan',
            description='LaserScan input topic.',
        ),
        DeclareLaunchArgument(
            'map_topic',
            default_value='/map',
            description='OccupancyGrid input topic.',
        ),
        DeclareLaunchArgument(
            'tf_broadcast',
            default_value='false',
            description=(
                'Publish map to odom from AMCL. Keep false while the LIMO '
                'launch publishes the existing static map-to-odom TF.'
            ),
        ),
        amcl,
        lifecycle_manager,
    ])
