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
    cv_enabled = LaunchConfiguration('cv_enabled')
    cv_map_topic = LaunchConfiguration('cv_map_topic')
    cv_cloud_topic = LaunchConfiguration('cv_cloud_topic')
    cv_sync_tolerance = LaunchConfiguration('cv_sync_tolerance')
    laser_weight_factor = LaunchConfiguration('laser_weight_factor')
    cv_weight_factor = LaunchConfiguration('cv_weight_factor')

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
            'cv_enabled': ParameterValue(
                cv_enabled,
                value_type=bool,
            ),
            'cv_map_topic': cv_map_topic,
            'cv_cloud_topic': cv_cloud_topic,
            'cv_sync_tolerance': ParameterValue(
                cv_sync_tolerance,
                value_type=float,
            ),
            'laser_weight_factor': ParameterValue(
                laser_weight_factor,
                value_type=float,
            ),
            'cv_weight_factor': ParameterValue(
                cv_weight_factor,
                value_type=float,
            ),
            'cv_z_hit': 0.5,
            'cv_z_rand': 0.5,
            'cv_sigma_hit': 0.2,
            'cv_max_occ_dist': 2.0,
            'cv_sensor_max_range': 10.0,
            'cv_voxel_leaf_size': 0.05,
            'cv_max_points': 600,
            'cv_occupied_threshold': 50,
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
            default_value='true',
            description='Publish the dynamic map-to-odom transform from AMCL.',
        ),
        DeclareLaunchArgument(
            'cv_enabled',
            default_value='true',
            description='Fuse synchronized CV obstacle likelihoods.',
        ),
        DeclareLaunchArgument(
            'cv_map_topic',
            default_value=(
                '/limo/nav_map_package/online/maps/cv_map'
            ),
            description='Static CV occupancy map topic.',
        ),
        DeclareLaunchArgument(
            'cv_cloud_topic',
            default_value=(
                '/limo/nav_map_package/online/cv_2_ptcld/points'
            ),
            description='Robot-relative CV obstacle cloud topic.',
        ),
        DeclareLaunchArgument(
            'cv_sync_tolerance',
            default_value='0.10',
            description='Maximum laser-to-CV timestamp error in seconds.',
        ),
        DeclareLaunchArgument(
            'laser_weight_factor',
            default_value='1.0',
            description='Exponent applied to the normalized laser weight.',
        ),
        DeclareLaunchArgument(
            'cv_weight_factor',
            default_value='0.25',
            description='Exponent applied to the raw CV likelihood.',
        ),
        amcl,
        lifecycle_manager,
    ])
