"""Launch AMCL with laser and voxelized semantic PointCloud2 fusion."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Values are read when AMCL is configured; restart to change CV tuning.
    settings = {
        'use_sim_time': ('true', bool, 'Use the simulation or rosbag clock.'),
        'base_frame_id': ('base_link', str, 'Robot base frame.'),
        'odom_frame_id': ('odom', str, 'Odometry frame for motion compensation.'),
        'global_frame_id': ('map', str, 'Global localization frame.'),
        'scan_topic': ('/scan', str, 'LaserScan input.'),
        'map_topic': ('/limo/map_package/online/maps/laser_map', str,
                      'Static laser occupancy map.'),
        'tf_broadcast': ('true', bool, 'Publish map to odom.'),
        'cv_enabled': ('true', bool, 'Fuse semantic cloud evidence before resampling.'),
        'cv_map_topic': ('/limo/map_package/online/maps/cv_obstacle', str,
                         'Binary static CV obstacle map.'),
        'cv_cloud_topic': ('/limo/cv_package/visual_ptcld/points', str,
                           'Metric cloud with x/y/z and class_id fields.'),
        'cv_buffer_size': ('10', int, 'Number of clouds retained for synchronization.'),
        'cv_sync_tolerance': ('0.20', float, 'Maximum cloud/laser timestamp difference, seconds.'),
        'cv_voxel_size': ('0.075', float, 'XY voxel size in metres after class filtering.'),
        'cv_min_points': ('5.0', float, 'Minimum obstacle/road voxels required for CV fusion.'),
        'cv_occupied_threshold': ('50', int, 'Occupied threshold in the binary CV map.'),
        'laser_weight_factor': ('1.0', float, 'Laser exponent; zero skips laser likelihood updates.'),
        'cv_weight_factor': (
            '0.0', float, 'Exponent of the CV likelihood (0 disables CV weight fusion).'),
        'cv_sad_gain': ('20.0', float, 'Gain converting mean obstacle/road mismatch to likelihood.'),
        'cv_quality_gate_enabled': ('true', bool, 'Reject ambiguous CV updates before fusion.'),
        'cv_min_information': ('0.02', float, 'Minimum CV divergence from uniform, in nats.'),
        'cv_max_position_stddev': ('0.5', float, 'Maximum CV support XY standard deviation, metres.'),
        'cv_max_yaw_stddev': ('0.5', float, 'Maximum CV support circular yaw deviation, radians.'),
        'max_particles': ('2000', int, 'Maximum AMCL particle count.'),
        'min_particles': ('300', int, 'Minimum AMCL particle count.'),
        'workload_logging_enabled': ('true', bool, 'Log throttled cloud and particle counts.'),
        'alpha1': ('0.2', float, 'Rotation noise caused by rotation.'),
        'alpha2': ('0.2', float, 'Rotation noise caused by translation.'),
        'alpha3': ('0.2', float, 'Translation noise caused by translation.'),
        'alpha4': ('0.2', float, 'Translation noise caused by rotation.'),
    }
    parameters = {
        name: ParameterValue(LaunchConfiguration(name), value_type=kind)
        for name, (_, kind, _) in settings.items()
    }
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=default, description=description)
          for name, (default, _, description) in settings.items()],
        Node(
            package='nav2_amcl', executable='amcl', name='amcl',
            output='screen', parameters=[parameters],
        ),
        Node(
            package='nav2_lifecycle_manager', executable='lifecycle_manager',
            name='lifecycle_manager_amcl', output='screen',
            parameters=[{
                'use_sim_time': parameters['use_sim_time'],
                'autostart': True,
                'node_names': ['amcl'],
            }],
        ),
    ])
