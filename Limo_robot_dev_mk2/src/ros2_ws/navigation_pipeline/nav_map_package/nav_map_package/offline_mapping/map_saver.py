"""Service bridge used to save the combined map through Nav2."""

from pathlib import Path
from threading import Event

import rclpy
from nav2_msgs.srv import SaveMap
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger


def find_project_root(start: Path):
    """Find the LIMO project root from source or install paths."""
    for candidate in [start] + list(start.parents):
        if (candidate / 'src' / 'ros2_ws').is_dir():
            return candidate
    return None


class MapSaver(Node):
    """Expose a simple service and delegate the actual map write to Nav2."""

    def __init__(self):
        super().__init__('nav_map_saver')

        self.declare_parameter(
            'request_service',
            '/limo/nav_map_package/offline/map_saver/save_map',
        )
        self.declare_parameter('nav2_service', '/map_saver/save_map')
        self.declare_parameter(
            'map_topic',
            '/limo/nav_map_package/offline/nav_map/combined_grid',
        )
        self.declare_parameter('save_directory', '')
        self.declare_parameter('map_name', 'limo_map')
        self.declare_parameter('image_format', 'pgm')
        self.declare_parameter('map_mode', 'scale')
        self.declare_parameter('free_thresh', 0.25)
        self.declare_parameter('occupied_thresh', 0.65)
        self.declare_parameter('nav2_service_timeout_sec', 10.0)
        self.declare_parameter('save_timeout_sec', 30.0)
        self.declare_parameter(
            'status_topic',
            '/limo/nav_map_package/offline/map_saver/status',
        )

        save_directory = self.get_parameter('save_directory').value
        if save_directory:
            self.save_directory = Path(save_directory).expanduser().resolve()
        else:
            project_root = find_project_root(Path(__file__).resolve())
            if project_root is None:
                raise RuntimeError(
                    'Cannot locate project root; set save_directory explicitly'
                )
            self.save_directory = project_root / 'ros2_maps' / 'nav_pipeline'
        self.save_directory.mkdir(parents=True, exist_ok=True)

        self.callback_group = ReentrantCallbackGroup()
        nav2_service = self.get_parameter('nav2_service').value
        request_service = self.get_parameter('request_service').value
        status_topic = self.get_parameter('status_topic').value
        status_qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.status_publisher = self.create_publisher(
            String,
            status_topic,
            status_qos,
        )
        self.save_client = self.create_client(
            SaveMap,
            nav2_service,
            callback_group=self.callback_group,
        )
        self.save_service = self.create_service(
            Trigger,
            request_service,
            self.save_callback,
            callback_group=self.callback_group,
        )
        self.saving = False

        self.publish_status(
            'INFO',
            f'Ready on {request_service}; Nav2 target is {nav2_service}'
        )

    def publish_status(self, level, message):
        """Publish a saver event for the GUI terminal and ROS logs."""
        status = String()
        status.data = f'[{level}] {message}'
        self.status_publisher.publish(status)
        logger = (
            self.get_logger().error
            if level == 'ERROR'
            else self.get_logger().info
        )
        logger(message)

    def save_callback(self, _request, response):
        if self.saving:
            response.success = False
            response.message = 'A map save request is already running'
            self.publish_status('ERROR', response.message)
            return response

        self.saving = True
        self.publish_status('INFO', 'Map save request received')
        try:
            service_timeout = float(
                self.get_parameter('nav2_service_timeout_sec').value
            )
            self.publish_status(
                'INFO',
                f"Waiting for Nav2 service '{self.save_client.srv_name}'",
            )
            if not self.save_client.wait_for_service(
                timeout_sec=service_timeout
            ):
                response.success = False
                response.message = (
                    f"Nav2 service '{self.save_client.srv_name}' unavailable"
                )
                self.publish_status('ERROR', response.message)
                return response

            map_name = self.get_parameter('map_name').value
            save_path = self.save_directory / map_name
            request = SaveMap.Request()
            request.map_topic = self.get_parameter('map_topic').value
            request.map_url = str(save_path)
            request.image_format = self.get_parameter('image_format').value
            request.map_mode = self.get_parameter('map_mode').value
            request.free_thresh = float(
                self.get_parameter('free_thresh').value
            )
            request.occupied_thresh = float(
                self.get_parameter('occupied_thresh').value
            )

            self.publish_status(
                'INFO',
                f"Saving {request.map_topic} as '{save_path}'",
            )
            future = self.save_client.call_async(request)
            completed = Event()
            future.add_done_callback(lambda _future: completed.set())
            save_timeout = float(
                self.get_parameter('save_timeout_sec').value
            )
            if not completed.wait(timeout=save_timeout):
                response.success = False
                response.message = 'Timed out while Nav2 was saving the map'
                self.publish_status('ERROR', response.message)
                return response

            try:
                nav2_response = future.result()
            except Exception as exc:
                response.success = False
                response.message = f'Nav2 SaveMap call failed: {exc}'
                self.publish_status('ERROR', response.message)
                return response

            response.success = bool(nav2_response.result)
            if response.success:
                response.message = f'Map saved as {save_path}.yaml/.pgm'
                self.publish_status('SUCCESS', response.message)
            else:
                response.message = 'Nav2 received the request but failed to save'
                self.publish_status('ERROR', response.message)
            return response
        finally:
            self.saving = False


def main(args=None):
    rclpy.init(args=args)
    node = MapSaver()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
