#!/usr/bin/env python3
# Copyright 2026 Giulio Cataldo
# Licensed under the Apache License, Version 2.0.

"""Convert standard Twist yaw rate to Gazebo Foxy's steering-angle interface."""

import math
import time

from geometry_msgs.msg import Twist
import rclpy
from rclpy.clock import Clock
from rclpy.clock import ClockType
from rclpy.node import Node


def gazebo_command(velocity, yaw_rate, wheelbase, max_steering_angle):
    """Return (speed, plugin angle), including Gazebo's reverse sign handling."""
    if not math.isfinite(velocity) or not math.isfinite(yaw_rate):
        return 0.0, 0.0
    if abs(velocity) <= 0.001:
        return 0.0, 0.0
    # Gazebo internally multiplies angular.z by sign(linear.x). Its input is
    # therefore atan(L*w/abs(v)), not standard yaw rate and not atan(L*w/v).
    angle = math.atan(wheelbase * yaw_rate / abs(velocity))
    return velocity, max(-max_steering_angle, min(max_steering_angle, angle))


class GazeboTwistAdapter(Node):
    """Keep /cmd_vel in SI units; publish the plugin-specific command separately."""

    def __init__(self):
        super().__init__('gazebo_twist_adapter')
        self.declare_parameter('input_topic', '/cmd_vel')
        self.declare_parameter('output_topic', '/cmd_vel_gazebo')
        self.declare_parameter('wheelbase', 0.24)
        # R >= 0.55 m leaves margin for the collision-center offsets in Gazebo.
        self.declare_parameter('max_steering_angle', math.atan(0.24 / 0.55))
        self.declare_parameter('command_timeout', 0.5)
        self.wheelbase = self.get_parameter('wheelbase').value
        self.max_angle = self.get_parameter('max_steering_angle').value
        self.timeout = self.get_parameter('command_timeout').value
        for value in (self.wheelbase, self.max_angle, self.timeout):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError('Adapter geometry and timeout must be finite and positive')
        if self.max_angle >= math.pi / 2:
            raise ValueError('max_steering_angle must be below pi/2')
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        if not input_topic or not output_topic or input_topic == output_topic:
            raise ValueError('Input and output topics must be nonempty and different')
        self.publisher = self.create_publisher(Twist, output_topic, 10)
        self.subscription = self.create_subscription(Twist, input_topic, self._command, 10)
        self.last_command_time = None
        self.stop_sent = True
        self.watchdog_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.watchdog = self.create_timer(0.05, self._check_timeout, clock=self.watchdog_clock)

    def _command(self, message):
        command = Twist()
        command.linear.x, command.angular.z = gazebo_command(
            message.linear.x, message.angular.z, self.wheelbase, self.max_angle)
        self.publisher.publish(command)
        self.last_command_time = time.monotonic()
        self.stop_sent = command.linear.x == 0.0 and command.angular.z == 0.0

    def _check_timeout(self):
        if (self.last_command_time is not None and not self.stop_sent
                and time.monotonic() - self.last_command_time > self.timeout):
            self.publisher.publish(Twist())
            self.stop_sent = True


def main(args=None):
    """Run the simulation-only command converter."""
    rclpy.init(args=args)
    node = GazeboTwistAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publisher.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
