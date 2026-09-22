"""Launch the complete LIMO application with the physical-robot map profile."""

from user_package.app_launch import generate_app_launch_description


def generate_launch_description():
    """Launch mapping, planning, and control for the physical LIMO."""
    return generate_app_launch_description('real')
