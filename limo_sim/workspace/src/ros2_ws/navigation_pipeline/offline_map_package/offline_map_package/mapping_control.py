#!/usr/bin/env python3

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Bool
from std_srvs.srv import SetBool


class MappingControl(Node):
    """Expose one service controlling updates in all offline map filters."""

    def __init__(self):
        super().__init__('mapping_control')
        self.declare_parameter('mapping_enabled', True)
        self.mapping_enabled = bool(
            self.get_parameter('mapping_enabled').value
        )

        control_qos = QoSProfile(depth=1)
        control_qos.reliability = QoSReliabilityPolicy.RELIABLE
        control_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.state_publisher = self.create_publisher(
            Bool,
            '/limo/nav_map_package/offline/mapping_enabled',
            control_qos,
        )
        self.service = self.create_service(
            SetBool,
            '/limo/nav_map_package/offline/set_mapping_enabled',
            self.set_mapping_enabled,
        )
        self.publish_state()

    def publish_state(self):
        msg = Bool()
        msg.data = self.mapping_enabled
        self.state_publisher.publish(msg)

    def set_mapping_enabled(self, request, response):
        self.mapping_enabled = request.data
        self.publish_state()
        state = 'enabled' if self.mapping_enabled else 'paused'
        response.success = True
        response.message = f'Offline mapping updates {state}'
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = MappingControl()
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
