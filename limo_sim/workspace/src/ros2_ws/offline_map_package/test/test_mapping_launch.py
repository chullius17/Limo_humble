"""Check profile selection, parameter forwarding and desktop/backend separation."""

import importlib.util
from pathlib import Path

import pytest
import yaml

pytest.importorskip('launch_ros')
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch_ros.utilities import evaluate_parameters, normalize_parameters


PACKAGE = Path(__file__).resolve().parents[1]


def load_launch(filename):
    spec = importlib.util.spec_from_file_location('mapping_launch', PACKAGE / 'launch' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Read the source checkout under test, not another installed workspace.
    module.get_package_share_directory = lambda package: str(
        PACKAGE if package == 'offline_map_package' else PACKAGE.parent / package)
    return module


def mapping(monkeypatch, profile='sim', **overrides):
    module = load_launch('map.launch.py')
    context = LaunchContext()
    context.launch_configurations.update({
        'config_file': str(PACKAGE / 'config' / ('mapping_' + profile + '.yaml')),
        **overrides,
    })
    for action in module.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    # Inspect the actual launch inputs without starting processes or moving hardware.
    monkeypatch.setattr(module, 'Node', lambda **kwargs: kwargs)
    # CV is an included launch; inspect the direct node actions here.
    nodes = [node for node in module._launch_mapping(context) if isinstance(node, dict)]
    for node in nodes:
        parameters = evaluate_parameters(context, normalize_parameters(node['parameters']))
        node['values'] = {key: value for params in parameters for key, value in params.items()}
    return {node['name']: node for node in nodes}


def test_simulation_preserves_mapping_parameters(monkeypatch):
    nodes = mapping(monkeypatch)
    assert set(nodes) == {'slam_toolbox', 'semantic_mapper', 'rviz2', 'map_save_gui'}
    assert all(node['values']['use_sim_time'] is True for node in nodes.values())
    assert nodes['slam_toolbox']['values']['base_frame'] == 'base_link'
    assert nodes['slam_toolbox']['values']['minimum_time_interval'] == 0.1
    mapper = nodes['semantic_mapper']['values']
    assert (mapper['turquoise_cost'], mapper['white_cost'], mapper['boardwalk_cost']) == (60, 30, 90)
    assert mapper['save_directory'] == ''
    assert mapper['publish_rate_hz'] == 4.0


def test_real_profile_is_headless_with_wall_clock(monkeypatch):
    nodes = mapping(monkeypatch, 'real')
    assert set(nodes) == {'slam_toolbox', 'semantic_mapper'}
    assert all(node['values']['use_sim_time'] is False for node in nodes.values())
    assert nodes['slam_toolbox']['values']['base_frame'] == 'base_link'


@pytest.mark.parametrize('profile,sim_time', [('sim', True), ('real', False)])
def test_desktop_never_starts_mapping_nodes(monkeypatch, profile, sim_time):
    nodes = mapping(monkeypatch, profile, mode='desktop', start_slam='true', start_mapper='true')
    assert set(nodes) == {'rviz2', 'map_save_gui'}
    assert all(node['values']['use_sim_time'] is sim_time for node in nodes.values())
    assert nodes['map_save_gui']['values']['save_service'].endswith('/map_saver/save_map')


def test_cli_overrides_and_backend_guards(monkeypatch):
    nodes = mapping(monkeypatch, mode='backend', use_sim_time='false', resolution='0.1',
                    pose_source='cartographer', trajectory_id='2', save_directory='/tmp/maps',
                    start_rviz='true', start_gui='true')
    assert set(nodes) == {'slam_toolbox', 'semantic_mapper'}
    assert all(node['values']['use_sim_time'] is False for node in nodes.values())
    assert all(node['values']['resolution'] == 0.1 for node in nodes.values())
    assert nodes['semantic_mapper']['values']['pose_source'] == 'cartographer'
    assert nodes['semantic_mapper']['values']['trajectory_id'] == 2
    assert nodes['semantic_mapper']['values']['save_directory'] == '/tmp/maps'


def test_disable_single_desktop_window(monkeypatch):
    nodes = mapping(monkeypatch, 'real', mode='desktop', start_gui='false', fixed_frame='odom')
    assert set(nodes) == {'rviz2'}
    assert nodes['rviz2']['arguments'][-2:] == ['-f', 'odom']


def test_custom_profile_parameters_reach_nodes(monkeypatch, tmp_path):
    profile = yaml.safe_load((PACKAGE / 'config' / 'mapping_sim.yaml').read_text())
    profile['semantic_mapper']['white_cost'] = 42
    profile['slam_toolbox']['scan_topic'] = '/test_scan'
    profile['map_save_gui']['save_service'] = '/test_save'
    config = tmp_path / 'profile.yaml'
    config.write_text(yaml.safe_dump(profile))
    nodes = mapping(monkeypatch, config_file=str(config))
    assert nodes['semantic_mapper']['values']['white_cost'] == 42
    assert nodes['slam_toolbox']['values']['scan_topic'] == '/test_scan'
    assert nodes['map_save_gui']['values']['save_service'] == '/test_save'


@pytest.mark.parametrize('filename,profile,mode', [
    ('map_sim.launch.py', 'sim', None),
    ('map_real.launch.py', 'real', 'backend'),
    ('desktop_offline.launch.py', 'real', 'desktop'),
])
def test_wrappers_select_profile_and_role(filename, profile, mode):
    module = load_launch(filename)
    actions = module.generate_launch_description().entities
    assert len(actions) == 1
    expected = {
        'config_file': str(PACKAGE / 'config' / ('mapping_' + profile + '.yaml'))}
    if mode is not None:
        expected['mode'] = mode
    assert dict(actions[0].launch_arguments) == expected


@pytest.mark.parametrize('overrides', [{'use_sim_time': 'typo'}, {'mode': 'typo'}])
def test_invalid_launch_settings_fail(monkeypatch, overrides):
    with pytest.raises(ValueError):
        mapping(monkeypatch, **overrides)
