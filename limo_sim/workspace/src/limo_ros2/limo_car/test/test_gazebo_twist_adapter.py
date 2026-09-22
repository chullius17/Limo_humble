"""Check the yaw-rate contract at the Gazebo Foxy actuator boundary."""

import math
from pathlib import Path
import time
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import pytest

from geometry_msgs.msg import Twist
import rclpy

from gazebo_twist_adapter import GazeboTwistAdapter, gazebo_command


@pytest.mark.parametrize('velocity', [0.10, 0.25, 0.50, -0.10, -0.25])
@pytest.mark.parametrize('curvature', [-1.5, 0.0, 1.5])
def test_plugin_executes_requested_curvature_in_both_directions(velocity, curvature):
    desired_yaw_rate = velocity * curvature
    speed, angle = gazebo_command(velocity, desired_yaw_rate, 0.24, math.atan(0.24 / 0.55))
    # gazebo_ros_ackermann_drive.cpp: target_rot *= copysign(1, target_linear).
    actual_steering = angle * math.copysign(1.0, speed)
    actual_yaw_rate = speed * math.tan(actual_steering) / 0.24
    assert actual_yaw_rate == pytest.approx(desired_yaw_rate)
    assert actual_yaw_rate / speed == pytest.approx(curvature)


def test_old_direct_twist_interface_oversteered_at_cruising_speed():
    speed, desired_yaw_rate = 0.5, 0.3
    wrong_yaw_rate = speed * math.tan(desired_yaw_rate) / 0.24
    assert wrong_yaw_rate > 2 * desired_yaw_rate
    speed, angle = gazebo_command(speed, desired_yaw_rate, 0.24, math.atan(0.24 / 0.55))
    assert speed * math.tan(angle) / 0.24 == pytest.approx(desired_yaw_rate)


@pytest.mark.parametrize('velocity,yaw_rate', [
    (0.0, 1.0), (0.0001, -1.0), (float('nan'), 0.0), (0.1, float('inf')),
])
def test_stopped_and_nonfinite_commands_stop_the_robot(velocity, yaw_rate):
    assert gazebo_command(velocity, yaw_rate, 0.24, math.atan(0.24 / 0.55)) == (0.0, 0.0)


def test_steering_saturation_respects_simulated_inner_wheel_limit():
    speed, angle = gazebo_command(0.1, 2.0, 0.24, math.atan(0.24 / 0.55))
    radius = 0.24 / math.tan(angle)
    assert radius >= 0.55 - 1e-12
    assert math.atan(0.24 / (radius - (0.168 + 0.045) / 2)) < math.pi / 6
    assert speed == 0.1


def test_only_the_adapter_topic_reaches_the_gazebo_plugin():
    model = Path(__file__).resolve().parents[1] / 'gazebo' / 'ackermann.xacro'
    root = ET.parse(model).getroot()
    plugin = root.find(".//plugin[@name='ackermann_controller']")
    assert plugin.findtext('ros/remapping') == 'cmd_vel:=/cmd_vel_gazebo'


def test_node_converts_input_and_stops_once_on_timeout(monkeypatch):
    rclpy.init()
    node = GazeboTwistAdapter()
    publisher = Mock()
    node.publisher = publisher
    try:
        now = time.monotonic()
        monkeypatch.setattr(time, 'monotonic', lambda: now)
        command = Twist()
        command.linear.x = 0.5
        command.angular.z = 0.3
        node._command(command)
        converted = publisher.publish.call_args[0][0]
        assert converted.linear.x == 0.5
        assert converted.angular.z == pytest.approx(math.atan(0.24 * 0.3 / 0.5))
        node._check_timeout()
        assert publisher.publish.call_count == 1
        now += 0.6
        node._check_timeout()
        node._check_timeout()
        assert publisher.publish.call_count == 2
        stopped = publisher.publish.call_args[0][0]
        assert stopped.linear.x == 0.0
        assert stopped.angular.z == 0.0
    finally:
        node.destroy_node()
        rclpy.shutdown()
