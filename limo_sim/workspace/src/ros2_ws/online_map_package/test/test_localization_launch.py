"""Check online profiles, wrappers and desktop/backend separation."""

import importlib.util
from pathlib import Path

import pytest
import yaml

pytest.importorskip('launch_ros')
from launch import LaunchContext  # noqa: E402
from launch.actions import (  # noqa: E402
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch_ros.utilities import (  # noqa: E402
    evaluate_parameters,
    normalize_parameters,
)


PACKAGE = Path(__file__).resolve().parents[1]
PACKAGES = PACKAGE.parent


def load_launch(filename, package=PACKAGE):
    spec = importlib.util.spec_from_file_location(
        'localization_launch', package / 'launch' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, 'get_package_share_directory'):
        module.get_package_share_directory = lambda name: str(PACKAGES / name)
    return module


def online(monkeypatch, profile='sim', **overrides):
    module = load_launch('online_map.launch.py')
    context = LaunchContext()
    context.launch_configurations.update({
        'config_file': str(
            PACKAGE / 'config' / ('mapping_' + profile + '.yaml')),
        **overrides,
    })
    for action in module.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    monkeypatch.setattr(module, 'Node', lambda **kwargs: kwargs)
    actions = module._launch_online(context)
    nodes = {}
    includes = []
    for action in actions:
        if isinstance(action, dict):
            parameters = evaluate_parameters(
                context, normalize_parameters(action['parameters']))
            action['values'] = {
                key: value for group in parameters
                for key, value in group.items()
            }
            nodes[action['name']] = action
        elif isinstance(action, IncludeLaunchDescription):
            includes.append(action)
    return nodes, includes


def test_simulation_starts_cv_maps_amcl_and_rviz(monkeypatch):
    nodes, includes = online(monkeypatch)
    assert set(nodes) == {
        'complete_map_server', 'laser_map_server', 'cv_map_server',
        'lifecycle_manager_online_maps', 'rviz2',
    }
    assert all(node['values']['use_sim_time'] is True
               for node in nodes.values())
    assert len(includes) == 2
    include_arguments = [dict(action.launch_arguments) for action in includes]
    assert {'use_sim_time': 'true'} in include_arguments
    amcl = next(values for values in include_arguments
                if 'cv_voxel_size' in values)
    assert amcl['cv_voxel_size'] == '0.075'
    assert amcl['map_topic'].endswith('/laser_map')
    assert amcl['cv_map_topic'].endswith('/cv_obstacle')
    assert nodes['rviz2']['arguments'][-2:] == ['-f', 'map']


def test_real_profile_is_headless_and_does_not_restart_cv(monkeypatch):
    nodes, includes = online(monkeypatch, 'real', mode='backend')
    assert set(nodes) == {
        'complete_map_server', 'laser_map_server', 'cv_map_server',
        'lifecycle_manager_online_maps',
    }
    assert all(node['values']['use_sim_time'] is False
               for node in nodes.values())
    assert len(includes) == 1
    amcl = dict(includes[0].launch_arguments)
    assert amcl['use_sim_time'] == 'false'
    assert amcl['cv_enabled'] == 'true'


def test_desktop_online_starts_only_rviz(monkeypatch):
    nodes, includes = online(monkeypatch, 'real', mode='desktop')
    assert set(nodes) == {'rviz2'}
    assert nodes['rviz2']['values']['use_sim_time'] is False
    assert nodes['rviz2']['arguments'][-2:] == ['-f', 'map']
    assert includes == []


def test_custom_profile_and_cli_overrides_reach_consumers(
        monkeypatch, tmp_path):
    profile = yaml.safe_load(
        (PACKAGE / 'config' / 'mapping_sim.yaml').read_text())
    profile['map_servers']['name'] = 'custom'
    profile['map_servers']['laser_topic'] = '/test/laser_map'
    profile['amcl']['cv_voxel_size'] = 0.12
    config = tmp_path / 'profile.yaml'
    config.write_text(yaml.safe_dump(profile))
    nodes, includes = online(
        monkeypatch, config_file=str(config),
        map_directory=str(tmp_path), max_particles='600')
    assert nodes['complete_map_server']['values']['yaml_filename'] == str(
        tmp_path / 'custom_complete.yaml')
    assert ('map', '/test/laser_map') in (
        nodes['laser_map_server']['remappings'])
    amcl = next(dict(action.launch_arguments) for action in includes
                if 'cv_voxel_size' in dict(action.launch_arguments))
    assert amcl['cv_voxel_size'] == '0.12'
    assert amcl['max_particles'] == '600'
    assert amcl['map_topic'] == '/test/laser_map'


@pytest.mark.parametrize('filename,profile,mode', [
    ('online_map_sim.launch.py', 'sim', None),
    ('online_map_real.launch.py', 'real', 'backend'),
    ('desktop_online.launch.py', 'real', 'desktop'),
])
def test_wrappers_select_profile_and_role(filename, profile, mode):
    module = load_launch(filename)
    actions = module.generate_launch_description().entities
    assert len(actions) == 1
    expected = {
        'config_file': str(
            PACKAGE / 'config' / ('mapping_' + profile + '.yaml')),
    }
    if mode is not None:
        expected['mode'] = mode
    assert dict(actions[0].launch_arguments) == expected


@pytest.mark.parametrize('overrides', [
    {'use_sim_time': 'typo'},
    {'mode': 'typo'},
    {'max_particles': 'not-an-integer'},
])
def test_invalid_launch_settings_fail(monkeypatch, overrides):
    with pytest.raises(ValueError):
        online(monkeypatch, **overrides)


def test_amcl_launch_passes_typed_cloud_parameters(monkeypatch):
    module = load_launch(
        'amcl.launch.py', PACKAGES / 'limo_rviz')
    monkeypatch.setattr(module, 'Node', lambda **kwargs: kwargs)
    description = module.generate_launch_description()
    context = LaunchContext()
    context.launch_configurations.update({
        'cv_voxel_size': '0.12',
        'cv_enabled': 'false',
        'cv_cloud_topic': '/test/cloud',
        'use_sim_time': 'false',
        'max_particles': '600',
    })
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    nodes = {}
    for node in description.entities:
        if isinstance(node, dict):
            parameters = evaluate_parameters(
                context, normalize_parameters(node['parameters']))
            nodes[node['name']] = {
                key: value for group in parameters
                for key, value in group.items()
            }
    params = nodes['amcl']
    assert params['cv_enabled'] is False
    assert params['use_sim_time'] is False
    assert params['cv_voxel_size'] == 0.12
    assert params['cv_cloud_topic'] == '/test/cloud'
    assert params['max_particles'] == 600
    assert params['cv_map_topic'].endswith('/cv_obstacle')
    assert params['map_topic'].endswith('/laser_map')
