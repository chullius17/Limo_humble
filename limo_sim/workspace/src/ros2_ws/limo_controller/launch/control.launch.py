"""Launch the Foxy-compatible Nav2 DWB controller for the physical LIMO."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from nav2_common.launch import RewrittenYaml


def model_overrides(profile):
    """Match the different physical and Gazebo Ackermann geometries."""
    if profile == 'real':
        return {}  # Preserve the physical/custom values in the supplied YAML.
    if profile == 'sim':
        return {
            'FollowPath.MPC.wheelbase': 0.24,
            'FollowPath.MPC.rear_axle_to_base': 0.12,
            'FollowPath.AckermannKinematics.min_turning_radius': 0.55,
        }
    raise ValueError('robot_model must be sim or real')


def controller_node(context, configured_params, log_level):
    """Apply simulation geometry only when explicitly selected by the launch."""
    profile = LaunchConfiguration('robot_model').perform(context)
    return [Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[configured_params, model_overrides(profile)],
        arguments=['--ros-args', '--log-level', log_level],
        remappings=[('cmd_vel', '/cmd_vel_autonomy')],
    )]


def generate_launch_description():
    """Create the controller server and its lifecycle manager."""
    package_share = get_package_share_directory('limo_controller')
    default_params_file = os.path.join(
        package_share,
        'config',
        'dwb_params.yaml',
    )
    twist_mux_params_file = os.path.join(
        package_share,
        'config',
        'twist_mux.yaml',
    )

    controller_params_file = LaunchConfiguration('controller_params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    log_level = LaunchConfiguration('log_level')
    start_gui = LaunchConfiguration('start_gui')

    configured_params = RewrittenYaml(
        source_file=controller_params_file,
        root_key='',
        param_rewrites={'use_sim_time': use_sim_time},
        convert_types=True,
    )

    controller_server = OpaqueFunction(
        function=controller_node,
        args=[configured_params, log_level],
    )

    cmd_vel_mux = Node(
        package='limo_controller',
        executable='cmd_vel_mux',
        name='twist_mux',
        output='screen',
        parameters=[
            twist_mux_params_file,
            {'use_sim_time': ParameterValue(use_sim_time, value_type=bool)},
        ],
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_limo_controller',
        output='screen',
        parameters=[{
            'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
            'autostart': ParameterValue(autostart, value_type=bool),
            'node_names': ['controller_server'],
        }],
        arguments=['--ros-args', '--log-level', log_level],
    )

    path_executor = Node(
        package='limo_controller',
        executable='path_executor',
        name='path_executor',
        output='screen',
        parameters=[{
            'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
        }],
    )

    control_gui = Node(
        package='limo_controller',
        executable='control_gui',
        name='control_gui',
        output='screen',
        condition=IfCondition(start_gui),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'controller_params_file',
            default_value=default_params_file,
            description='Absolute path to the LIMO controller parameters.',
        ),
        DeclareLaunchArgument(
            'robot_model',
            default_value='real',
            description='sim for limo_car Gazebo geometry; real preserves the YAML geometry.',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use Gazebo/rosbag time instead of the robot clock.',
        ),
        DeclareLaunchArgument(
            'autostart',
            default_value='true',
            description='Configure and activate controller nodes automatically.',
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='ROS log level for controller processes.',
        ),
        DeclareLaunchArgument(
            'start_gui',
            default_value='true',
            description='Open the local control window.',
        ),
        controller_server,
        cmd_vel_mux,
        # Foxy lifecycle_manager performs startup only once.  When the whole
        # application starts at the same time as mapping, planning and RViz,
        # Fast DDS discovery can expose controller_server after that attempt.
        # Give the server time to advertise its lifecycle services first.
        TimerAction(period=3.0, actions=[lifecycle_manager]),
        path_executor,
        control_gui,
    ])
