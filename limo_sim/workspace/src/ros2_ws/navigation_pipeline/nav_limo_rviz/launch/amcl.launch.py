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
    cv_obstacle_grid_topic = LaunchConfiguration('cv_obstacle_grid_topic')
    cv_street_map_topic = LaunchConfiguration('cv_street_map_topic')
    cv_street_grid_topic = LaunchConfiguration('cv_street_grid_topic')
    cv_sync_tolerance = LaunchConfiguration('cv_sync_tolerance')
    laser_weight_factor = LaunchConfiguration('laser_weight_factor')
    cv_weight_factor = LaunchConfiguration('cv_weight_factor')
    cv_obstacle_weight_factor = LaunchConfiguration(
        'cv_obstacle_weight_factor'
    )
    cv_street_weight_factor = LaunchConfiguration('cv_street_weight_factor')
    cv_sad_gain = LaunchConfiguration('cv_sad_gain')
    cv_sad_cell_size = LaunchConfiguration('cv_sad_cell_size')
    cv_sad_min_cell_occupancy = LaunchConfiguration(
        'cv_sad_min_cell_occupancy'
    )
    cv_sad_min_positive_mass = LaunchConfiguration(
        'cv_sad_min_positive_mass'
    )
    max_particles = LaunchConfiguration('max_particles')
    min_particles = LaunchConfiguration('min_particles')
    workload_logging_enabled = LaunchConfiguration(
        'workload_logging_enabled'
    )
    alpha1 = LaunchConfiguration('alpha1')
    alpha2 = LaunchConfiguration('alpha2')
    alpha3 = LaunchConfiguration('alpha3')
    alpha4 = LaunchConfiguration('alpha4')

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
            'cv_obstacle_grid_topic': cv_obstacle_grid_topic,
            'cv_street_map_topic': cv_street_map_topic,
            'cv_street_grid_topic': cv_street_grid_topic,
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
            'cv_obstacle_weight_factor': ParameterValue(
                cv_obstacle_weight_factor,
                value_type=float,
            ),
            'cv_street_weight_factor': ParameterValue(
                cv_street_weight_factor,
                value_type=float,
            ),
            'cv_sad_gain': ParameterValue(cv_sad_gain, value_type=float),
            'cv_sad_cell_size': ParameterValue(
                cv_sad_cell_size,
                value_type=float,
            ),
            'cv_sad_min_cell_occupancy': ParameterValue(
                cv_sad_min_cell_occupancy,
                value_type=float,
            ),
            'cv_sad_min_positive_mass': ParameterValue(
                cv_sad_min_positive_mass,
                value_type=float,
            ),
            'max_particles': ParameterValue(
                max_particles,
                value_type=int,
            ),
            'min_particles': ParameterValue(
                min_particles,
                value_type=int,
            ),
            'workload_logging_enabled': ParameterValue(
                workload_logging_enabled,
                value_type=bool,
            ),
            # Ackermann motion is approximated by AMCL's non-holonomic
            # DifferentialMotionModel. alpha5 is intentionally omitted because
            # it is only consumed by the omnidirectional model.
            'alpha1': ParameterValue(alpha1, value_type=float),
            'alpha2': ParameterValue(alpha2, value_type=float),
            'alpha3': ParameterValue(alpha3, value_type=float),
            'alpha4': ParameterValue(alpha4, value_type=float),
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
            description='Fuse the synchronized CV grid SAD likelihood.',
        ),
        DeclareLaunchArgument(
            'cv_map_topic',
            default_value=(
                '/limo/nav_map_package/online/maps/cv_map'
            ),
            description='Static CV occupancy map topic.',
        ),
        DeclareLaunchArgument(
            'cv_obstacle_grid_topic',
            default_value=(
                '/limo/nav_map_package/online/metric_bev/'
                'cost_grid_binary_obstacles'
            ),
            description='Robot-local obstacle OccupancyGrid SAD template.',
        ),
        DeclareLaunchArgument(
            'cv_street_map_topic',
            default_value=(
                '/limo/nav_map_package/online/maps/street_map'
            ),
            description='Static street occupancy map topic.',
        ),
        DeclareLaunchArgument(
            'cv_street_grid_topic',
            default_value=(
                '/limo/nav_map_package/online/metric_bev/'
                'cost_grid_binary_street'
            ),
            description='Robot-local street OccupancyGrid SAD template.',
        ),
        DeclareLaunchArgument(
            'cv_sync_tolerance',
            default_value='0.10',
            description='Maximum laser-to-CV timestamp error in seconds.',
        ),
        DeclareLaunchArgument(
            'laser_weight_factor',
            default_value='2.0',
            description='Exponent applied to the normalized laser weight.',
        ),
        DeclareLaunchArgument(
            'cv_weight_factor',
            default_value='0.25',
            description='Exponent applied to the CV SAD likelihood.',
        ),
        DeclareLaunchArgument(
            'cv_obstacle_weight_factor',
            default_value='1.0',
            description=(
                'Relative obstacle evidence factor; zero disables obstacle SAD.'
            ),
        ),
        DeclareLaunchArgument(
            'cv_street_weight_factor',
            default_value='1.0',
            description=(
                'Relative street evidence factor; zero disables street SAD.'
            ),
        ),
        DeclareLaunchArgument(
            'cv_sad_gain',
            default_value='20.0',
            description='Gain converting normalized SAD into likelihood.',
        ),
        DeclareLaunchArgument(
            'cv_sad_cell_size',
            default_value='0.075',
            description='Regular SAD template sampling size in metres.',
        ),
        DeclareLaunchArgument(
            'cv_sad_min_cell_occupancy',
            default_value='0.1',
            description=(
                'Minimum occupied fraction retained in a local SAD cell.'
            ),
        ),
        DeclareLaunchArgument(
            'cv_sad_min_positive_mass',
            default_value='5.0',
            description=(
                'Minimum foreground mass required for a CV class to vote.'
            ),
        ),
        DeclareLaunchArgument(
            'max_particles',
            default_value='2000',
            description='Maximum number of particles maintained by AMCL.',
        ),
        DeclareLaunchArgument(
            'min_particles',
            default_value='300',
            description='Minimum number of particles maintained by AMCL.',
        ),
        DeclareLaunchArgument(
            'workload_logging_enabled',
            default_value='true',
            description='Publish throttled AMCL and CV workload counters.',
        ),
        DeclareLaunchArgument(
            'alpha1',
            default_value='0.2',
            description='Rotation noise caused by Ackermann rotation.',
        ),
        DeclareLaunchArgument(
            'alpha2',
            default_value='0.2',
            description='Rotation/steering noise caused by translation.',
        ),
        DeclareLaunchArgument(
            'alpha3',
            default_value='0.2',
            description='Translation noise caused by translation.',
        ),
        DeclareLaunchArgument(
            'alpha4',
            default_value='0.2',
            description='Translation noise caused by Ackermann rotation.',
        ),
        amcl,
        lifecycle_manager,
    ])
