"""Verify the application launch composes independent LIMO subsystems."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip('launch')
from launch import LaunchContext  # noqa: E402
from launch.actions import DeclareLaunchArgument  # noqa: E402


PACKAGE = Path(__file__).resolve().parents[1]


def load_launch():
    spec = importlib.util.spec_from_file_location(
        'limo_app_launch', PACKAGE / 'launch' / 'limo_app.launch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compose(monkeypatch, **overrides):
    module = load_launch()
    context = LaunchContext()
    context.launch_configurations.update(overrides)
    for action in module.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    monkeypatch.setattr(
        module, '_include',
        lambda package, launch, arguments=None: {
            'package': package,
            'launch': launch,
            'arguments': arguments or {},
        })
    return module._launch_app(context)


def test_sim_profile_launches_map_trajectory_and_controller(monkeypatch):
    actions = compose(monkeypatch)
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


def test_real_profile_uses_wall_clock_and_robot_feedback(monkeypatch):
    actions = compose(monkeypatch, profile='real')
    assert actions[0]['launch'] == 'online_map_real.launch.py'
    assert actions[1]['arguments']['use_sim_time'] == 'false'
    assert actions[2]['arguments']['use_sim_time'] == 'false'
    assert actions[2]['arguments']['start_gui'] == 'false'
    assert actions[2]['arguments']['robot_model'] == 'real'


@pytest.mark.parametrize('overrides', [
    {'profile': 'invalid'},
    {'start_control_gui': 'invalid'},
])
def test_invalid_application_settings_fail(monkeypatch, overrides):
    with pytest.raises(ValueError):
        compose(monkeypatch, **overrides)
