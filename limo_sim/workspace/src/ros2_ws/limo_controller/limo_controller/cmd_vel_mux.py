# Copyright 2026 Giulio Cataldo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Arbitrate autonomy and teleop velocity commands by priority and timeout."""

import copy
from dataclasses import dataclass
import math
import time

from geometry_msgs.msg import Twist
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


@dataclass
class CommandInput:
    """Configuration and latest value for one velocity source."""

    priority: int
    timeout: float
    message: Twist = None
    received_at: float = None


def select_source(inputs, now):
    """Return the name of the freshest highest-priority command source."""
    available = [
        (state.priority, name)
        for name, state in inputs.items()
        if state.message is not None
        and state.received_at is not None
        and 0.0 <= now - state.received_at <= state.timeout
    ]
    return max(available)[1] if available else None


class CmdVelMux(Node):
    """Give teleoperation priority while keeping autonomous control alive."""

    def __init__(self):
        super().__init__('twist_mux')
        defaults = {
            'output_topic': '/cmd_vel',
            'publish_rate': 50.0,
            'topics.autonomy.topic': '/cmd_vel_autonomy',
            'topics.autonomy.timeout': 0.5,
            'topics.autonomy.priority': 10,
            'topics.teleop.topic': '/cmd_vel_teleop',
            'topics.teleop.timeout': 0.5,
            'topics.teleop.priority': 100,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        values = {
            name: self.get_parameter(name).value for name in defaults
        }
        if (not values['output_topic']
                or not values['topics.autonomy.topic']
                or not values['topics.teleop.topic']):
            raise ValueError('Velocity mux topics must not be empty')
        if (not math.isfinite(values['publish_rate'])
                or values['publish_rate'] <= 0.0):
            raise ValueError('publish_rate must be finite and greater than zero')
        for source in ('autonomy', 'teleop'):
            timeout = values[f'topics.{source}.timeout']
            priority = values[f'topics.{source}.priority']
            if not math.isfinite(timeout) or timeout <= 0.0:
                raise ValueError(f'{source} timeout must be finite and positive')
            if (not isinstance(priority, int) or isinstance(priority, bool)
                    or priority < 0):
                raise ValueError(f'{source} priority must be a nonnegative integer')

        self.inputs = {
            source: CommandInput(
                priority=values[f'topics.{source}.priority'],
                timeout=values[f'topics.{source}.timeout'],
            )
            for source in ('autonomy', 'teleop')
        }
        self.output_publisher = self.create_publisher(
            Twist, values['output_topic'], 10)
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.source_publisher = self.create_publisher(
            String, '/limo/control/cmd_vel_source', status_qos)
        self._input_subscriptions = [
            self.create_subscription(
                Twist,
                values[f'topics.{source}.topic'],
                lambda message, selected=source: self._command_callback(
                    selected, message),
                10,
            )
            for source in ('autonomy', 'teleop')
        ]
        self.active_source = '__uninitialized__'
        self.stop_published = False
        self.timer = self.create_timer(
            1.0 / values['publish_rate'], self._publish_selected)
        self._set_active_source(None)
        self.get_logger().info(
            'Velocity mux: autonomy={} (priority {}, timeout {:g}s), '
            'teleop={} (priority {}, timeout {:g}s) -> {}'.format(
                values['topics.autonomy.topic'],
                self.inputs['autonomy'].priority,
                self.inputs['autonomy'].timeout,
                values['topics.teleop.topic'],
                self.inputs['teleop'].priority,
                self.inputs['teleop'].timeout,
                values['output_topic'],
            )
        )

    def _command_callback(self, source, message):
        state = self.inputs[source]
        state.message = copy.deepcopy(message)
        state.received_at = time.monotonic()
        self._publish_selected(state.received_at)

    def _set_active_source(self, source):
        if source == self.active_source:
            return
        self.active_source = source
        label = source if source is not None else 'idle'
        self.source_publisher.publish(String(data=label))
        self.get_logger().info(f'CMD_VEL source: {label}')

    def _publish_selected(self, now=None):
        if now is None:
            now = time.monotonic()
        source = select_source(self.inputs, now)
        self._set_active_source(source)
        if source is None:
            if not self.stop_published:
                self.output_publisher.publish(Twist())
                self.stop_published = True
            return
        self.output_publisher.publish(self.inputs[source].message)
        self.stop_published = False

    def destroy_node(self):
        """Publish a final stop before releasing the output publisher."""
        self.output_publisher.publish(Twist())
        return super().destroy_node()


def main(args=None):
    """Run the priority velocity multiplexer."""
    rclpy.init(args=args)
    node = CmdVelMux()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
