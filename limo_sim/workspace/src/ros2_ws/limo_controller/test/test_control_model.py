"""Keep simulator overrides separate from physical/custom controller profiles."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip('launch')


def load_launch():
    path = Path(__file__).resolve().parents[1] / 'launch' / 'control.launch.py'
    spec = importlib.util.spec_from_file_location('control_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_simulation_uses_gazebo_wheelbase_and_turning_limits():
    params = load_launch().model_overrides('sim')
    assert params['FollowPath.MPC.wheelbase'] == 0.24
    assert params['FollowPath.MPC.rear_axle_to_base'] == 0.12
    assert params['FollowPath.AckermannKinematics.min_turning_radius'] == 0.55


def test_real_profile_preserves_custom_yaml_geometry():
    assert load_launch().model_overrides('real') == {}


def test_unknown_model_fails():
    with pytest.raises(ValueError):
        load_launch().model_overrides('differential')
