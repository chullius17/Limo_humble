"""Service bridge used to save all maps produced by ``nav_map``."""

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
    """Expose one service that saves all four offline map products."""

    def __init__(self):
        super().__init__('nav_map_saver')

        self.declare_parameter(
            'request_service',
            '/limo/nav_map_package/offline/map_saver/save_map',
        )
        self.declare_parameter('nav2_service', '/map_saver/save_map')
        self.declare_parameter(
            'combined_map_topic',
            '/limo/nav_map_package/offline/nav_map/combined_grid',
        )
        self.declare_parameter(
            'laser_map_topic',
            '/limo/nav_map_package/offline/nav_map/laser_map',
        )
        self.declare_parameter(
            'cv_map_topic',
            '/limo/nav_map_package/offline/nav_map/cv_map',
        )
        self.declare_parameter(
            'street_map_topic',
            '/limo/nav_map_package/offline/nav_map/street_map',
        )
        self.declare_parameter('save_directory', '')
        self.declare_parameter('combined_map_name', 'limo_map_combined')
        self.declare_parameter('laser_map_name', 'limo_map_laser')
        self.declare_parameter('cv_map_name', 'limo_map_cv')
        self.declare_parameter('street_map_name', 'limo_map_street')
        self.declare_parameter('image_format', 'pgm')
        self.declare_parameter('combined_map_mode', 'scale')
        self.declare_parameter('laser_map_mode', 'trinary')
        self.declare_parameter('cv_map_mode', 'trinary')
        self.declare_parameter('street_map_mode', 'trinary')
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

            maps = (
                (
                    'combined',
                    'combined_map_topic',
                    'combined_map_name',
                    'combined_map_mode',
                ),
                (
                    'laser',
                    'laser_map_topic',
                    'laser_map_name',
                    'laser_map_mode',
                ),
                ('CV', 'cv_map_topic', 'cv_map_name', 'cv_map_mode'),
                (
                    'street',
                    'street_map_topic',
                    'street_map_name',
                    'street_map_mode',
                ),
            )
            saved = []
            failed = []
            for label, topic_parameter, name_parameter, mode_parameter in maps:
                topic = self.get_parameter(topic_parameter).value
                map_name = self.get_parameter(name_parameter).value
                map_mode = self.get_parameter(mode_parameter).value
                if self._save_map(label, topic, map_name, map_mode):
                    saved.append(map_name)
                else:
                    failed.append(map_name)

            response.success = not failed
            if response.success:
                response.message = 'Saved all four maps: ' + ', '.join(saved)
                self.publish_status('SUCCESS', response.message)
            else:
                response.message = (
                    f"Saved {len(saved)}/4 maps; failed: {', '.join(failed)}"
                )
                self.publish_status('ERROR', response.message)
            return response
        finally:
            self.saving = False

    def _save_map(self, label, topic, map_name, map_mode):
        """Save one OccupancyGrid through Nav2 and report its result."""
        save_path = self.save_directory / map_name
        request = SaveMap.Request()
        request.map_topic = topic
        request.map_url = str(save_path)
        request.image_format = self.get_parameter('image_format').value
        request.map_mode = map_mode
        request.free_thresh = float(self.get_parameter('free_thresh').value)
        request.occupied_thresh = float(
            self.get_parameter('occupied_thresh').value
        )

        self.publish_status(
            'INFO',
            f"Saving {label} map in {map_mode} mode from {topic} "
            f"as '{save_path}'",
        )
        future = self.save_client.call_async(request)
        completed = Event()
        future.add_done_callback(lambda _future: completed.set())
        save_timeout = float(self.get_parameter('save_timeout_sec').value)
        if not completed.wait(timeout=save_timeout):
            self.publish_status(
                'ERROR',
                f'Timed out while Nav2 was saving the {label} map',
            )
            return False

        try:
            nav2_response = future.result()
        except Exception as exc:
            self.publish_status(
                'ERROR',
                f'Nav2 failed while saving the {label} map: {exc}',
            )
            return False

        if not nav2_response.result:
            self.publish_status(
                'ERROR',
                f'Nav2 could not save the {label} map from {topic}',
            )
            return False

        image_format = self.get_parameter('image_format').value
        self.publish_status(
            'SUCCESS',
            f"Saved {label} map as {save_path}.yaml/.{image_format}",
        )
        return True


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
