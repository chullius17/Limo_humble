"""Verify both application launches compose all LIMO subsystems."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip('launch')
from launch import LaunchContext  # noqa: E402
from launch.actions import DeclareLaunchArgument, OpaqueFunction  # noqa: E402

from user_package import app_launch  # noqa: E402


PACKAGE = Path(__file__).resolve().parents[1]


def load_launch(profile):
    filename = 'limo_app.launch.py' if profile == 'legacy' else 'limo_app_{}.launch.py'.format(profile)
    path = PACKAGE / 'launch' / filename
    spec = importlib.util.spec_from_file_location(
        'limo_app_{}_launch'.format(profile), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compose(monkeypatch, profile, **overrides):
    module = load_launch(profile)
    description = module.generate_launch_description()
    context = LaunchContext()
    context.launch_configurations.update(overrides)
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    monkeypatch.setattr(
        app_launch, '_include',
        lambda package, launch, arguments=None: {
            'package': package,
            'launch': launch,
            'arguments': arguments or {},
        })
    opaque_action = next(
        action for action in description.entities
        if isinstance(action, OpaqueFunction))
    return opaque_action.execute(context)


def test_sim_launches_map_trajectory_and_controller(monkeypatch):
    actions = compose(monkeypatch, 'sim')
    assert [(item['package'], item['launch']) for item in actions] == [
        ('online_map_package', 'online_map_sim.launch.py'),
        ('traj_package', 'trajectory.launch.py'),
        ('limo_controller', 'control.launch.py'),
    ]
    assert actions[1]['arguments'] == {
        'map_topic': '/map',
        'use_sim_time': 'true',
        'autostart': 'true',
    }
    assert actions[2]['arguments']['start_gui'] == 'true'
    assert actions[2]['arguments']['robot_model'] == 'sim'


def test_real_uses_wall_clock_and_robot_feedback(monkeypatch):
    actions = compose(monkeypatch, 'real')
    assert actions[0]['launch'] == 'online_map_real.launch.py'
    assert actions[1]['arguments']['use_sim_time'] == 'false'
    assert actions[2]['arguments']['use_sim_time'] == 'false'
    assert actions[2]['arguments']['start_gui'] == 'false'
    assert actions[2]['arguments']['robot_model'] == 'real'


def test_application_overrides_are_forwarded(monkeypatch):
    actions = compose(
        monkeypatch, 'real', map_topic='/custom_map',
        start_control_gui='true')
    assert actions[1]['arguments']['map_topic'] == '/custom_map'
    assert actions[2]['arguments']['start_gui'] == 'true'


def test_invalid_control_gui_setting_fails(monkeypatch):
    with pytest.raises(ValueError):
        compose(monkeypatch, 'sim', start_control_gui='invalid')


def test_invalid_internal_profile_fails():
    with pytest.raises(ValueError):
        app_launch.generate_app_launch_description('invalid')


@pytest.mark.parametrize('profile', ['real', 'sim', 'legacy'])
def test_cv_is_explicit_and_forwarded_once(monkeypatch, profile):
    actions = compose(monkeypatch, profile)
    assert actions[0]['arguments']['start_cv'] == 'false'
    actions = compose(monkeypatch, profile, start_cv='true', cv_config='/tmp/custom_cv.yaml')
    assert len(actions) == 3
    assert actions[0]['arguments']['start_cv'] == 'true'
    assert actions[0]['arguments']['cv_config'] == '/tmp/custom_cv.yaml'
    assert all(item['package'] != 'cv_package' for item in actions)
    if profile == 'legacy':
        assert actions[0]['launch'] == 'online_map_sim.launch.py'
        assert actions[0]['arguments']['use_sim_time'] == 'false'


def test_invalid_cv_switch_fails(monkeypatch):
    with pytest.raises(ValueError):
        compose(monkeypatch, 'real', start_cv='typo')
