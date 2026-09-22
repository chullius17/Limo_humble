"""Launch the complete LIMO application with the simulation map profile."""

from user_package.app_launch import generate_app_launch_description


def generate_launch_description():
    """Launch mapping, planning, and control for simulation."""
    return generate_app_launch_description('sim')
