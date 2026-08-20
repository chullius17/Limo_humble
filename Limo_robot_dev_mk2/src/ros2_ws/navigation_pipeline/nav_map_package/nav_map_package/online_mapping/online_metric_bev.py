import rclpy
from rclpy.executors import ExternalShutdownException

from nav_map_package.metric_bev import MetricBEV


def main(args=None):
    rclpy.init(args=args)
    node = MetricBEV(
        node_name='online_metric_bev',
        publish_individual=False,
        publish_combined=True,
        topic_namespace='online/metric_bev',
        frame_prefix='online_metric_bev_origin',
        default_publish_debug=False,
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
