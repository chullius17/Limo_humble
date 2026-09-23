"""Launch the simulation application profile using the wall clock."""

from user_package.app_launch import generate_app_launch_description


def generate_launch_description():
    """Use the sim profile and RViz with use_sim_time disabled."""
    return generate_app_launch_description('sim', use_sim_time=False)
