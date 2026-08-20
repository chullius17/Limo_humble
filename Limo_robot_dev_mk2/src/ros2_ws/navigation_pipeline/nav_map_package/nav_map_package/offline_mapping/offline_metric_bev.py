import rclpy
from rclpy.executors import ExternalShutdownException

from nav_map_package.metric_bev import MetricBEV


def main(args=None):
    rclpy.init(args=args)
    node = MetricBEV(
        node_name='offline_metric_bev',
        publish_individual=True,
        publish_combined=False,
        topic_namespace='offline/metric_bev',
        frame_prefix='offline_metric_bev_origin',
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
