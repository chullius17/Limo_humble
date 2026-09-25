"""Check profile selection, topic wiring and the real waterfall ROS connection."""

import importlib.util
from pathlib import Path
import time

import numpy as np
import pytest
import yaml

pytest.importorskip('launch_ros')
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument

PACKAGE = Path(__file__).resolve().parents[1]
PACKAGES = PACKAGE.parent


def load_launch(name):
    spec = importlib.util.spec_from_file_location('cv_launch_test', PACKAGE / 'launch' / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.get_package_share_directory = lambda package: str(PACKAGES / package)
    return module


def pipeline(monkeypatch, profile='real', **overrides):
    module = load_launch('cv.launch.py')
    context = LaunchContext()
    context.launch_configurations.update({
        'config_file': str(PACKAGE / 'config' / f'cv_{profile}.yaml'), **overrides})
    for action in module.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    monkeypatch.setattr(module, 'Node', lambda **kwargs: kwargs)
    monkeypatch.setattr(module, 'OnProcessStart', lambda **kwargs: kwargs)
    monkeypatch.setattr(module, 'RegisterEventHandler', lambda handler: {'handler': handler})
    actions = module._launch_cv(context)
    nodes = {action['name']: action for action in actions if 'name' in action}
    for action in actions:
        if 'handler' in action:
            assert action['handler']['target_action'] is nodes['lane_node']
            for node in action['handler']['on_start']:
                nodes[node['name']] = node
    return nodes


def test_real_profile_selects_waterfall_and_connects_to_cloud(monkeypatch):
    nodes = pipeline(monkeypatch)
    assert set(nodes) == {'lane_node', 'depth_correction', 'visual_ptcld'}
    lane = nodes['lane_node']
    assert lane['executable'] == 'lane_detector_waterfall'
    assert lane['remappings'] == [(
        'limo/cv_package/detection/lane_waterfall_labels/raw',
        'limo/cv_package/detection/lane_labels/raw')]
    params = lane['parameters'][0]
    assert params['seed_erosion_iterations'] == 0
    assert params['barrier_closing_iterations'] == 1
    assert params['gradient_threshold'] == 15.0
    assert 'road_max_value' not in params and 'yellow_min_saturation' not in params
    assert all(node['parameters'][0]['use_sim_time'] is False for node in nodes.values())
    cloud = nodes['visual_ptcld']['parameters'][0]
    assert cloud['road_boardwalk_only'] is True
    assert cloud['pointcloud_topic'] == 'limo/cv_package/visual_ptcld/points'
    assert cloud['input_crop_y_min'] == 0.5


@pytest.mark.parametrize('clock', ['true', 'false'])
def test_sim_detector_does_not_depend_on_clock(monkeypatch, clock):
    nodes = pipeline(monkeypatch, 'sim', use_sim_time=clock)
    assert nodes['lane_node']['executable'] == 'lane_detector'
    assert nodes['lane_node']['remappings'] == []
    assert 'seed_max_gray' not in nodes['lane_node']['parameters'][0]
    assert nodes['lane_node']['parameters'][0]['use_sim_time'] is (clock == 'true')


def test_desktop_starts_only_viewer_and_backend_ignores_rviz(monkeypatch):
    assert set(pipeline(monkeypatch, mode='desktop')) == {'cv_rviz'}
    nodes = pipeline(monkeypatch, mode='backend', start_rviz='true',
                     visual_ptcld_enable_telemetry='false')
    assert 'cv_rviz' not in nodes
    assert nodes['visual_ptcld']['parameters'][0]['enable_telemetry'] is False


@pytest.mark.parametrize('filename,profile,mode', [
    ('cv_real.launch.py', 'real', 'backend'),
    ('cv_sim.launch.py', 'sim', None),
    ('desktop_cv.launch.py', 'real', 'desktop'),
])
def test_wrappers_select_correct_profile(filename, profile, mode):
    include = load_launch(filename).generate_launch_description().entities[0]
    values = dict(include.launch_arguments)
    assert values['config_file'] == str(PACKAGE / 'config' / f'cv_{profile}.yaml')
    assert values.get('mode') == mode


def test_unknown_detector_is_rejected(monkeypatch, tmp_path):
    config = yaml.safe_load((PACKAGE / 'config/cv_real.yaml').read_text())
    config['launch']['lane_detector'] = 'typo'
    path = tmp_path / 'invalid.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='lane_detector'):
        pipeline(monkeypatch, config_file=str(path))


def test_real_profile_publishes_labels_on_the_consumer_topic():
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from cv_package.lane_detector_waterfall import WaterfallLaneDetector

    profile = yaml.safe_load((PACKAGE / 'config/cv_real.yaml').read_text())
    arguments = ['--ros-args', '-r', '__node:=lane_node', '-r',
                 'limo/cv_package/detection/lane_waterfall_labels/raw:='
                 '/limo/cv_package/detection/lane_labels/raw']
    for name, value in profile['lane_detector'].items():
        value = str(value).lower() if isinstance(value, bool) else str(value)
        arguments.extend(['-p', f'{name}:={value}'])
    rclpy.init(args=arguments)
    detector = None
    peer = None
    executor = SingleThreadedExecutor()
    try:
        detector = WaterfallLaneDetector()
        assert detector.get_name() == 'lane_node'
        peer = Node('cv_launch_consumer_test', use_global_arguments=False)
        received = []
        peer.create_subscription(Image, '/limo/cv_package/detection/lane_labels/raw',
                                 received.append, 1)
        camera = peer.create_publisher(Image, '/rgb/image_raw', qos_profile_sensor_data)
        executor.add_node(detector)
        executor.add_node(peer)
        msg = detector.bridge.cv2_to_imgmsg(np.full((480, 640, 3), 40, np.uint8), 'bgr8')
        msg.header.frame_id = 'camera_optical_frame'
        msg.header.stamp = peer.get_clock().now().to_msg()
        deadline = time.monotonic() + 10
        while not received and time.monotonic() < deadline:
            camera.publish(msg)
            executor.spin_once(timeout_sec=0.02)
        assert received
        labels = received[-1]
        assert (labels.width, labels.height, labels.encoding) == (320, 120, 'mono8')
        assert labels.header == msg.header
        assert set(np.frombuffer(labels.data, np.uint8)) == {0, 1}
    finally:
        executor.shutdown()
        if detector is not None:
            detector.destroy_node()
        if peer is not None:
            peer.destroy_node()
        rclpy.shutdown()
