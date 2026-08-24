import os
from glob import glob

from setuptools import find_packages, setup


package_name = 'online_map_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        (
            'share/' + package_name + '/launch',
            glob(os.path.join('launch', '*.launch.py')),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Giulio Cataldo',
    maintainer_email='giulio.cataldo@studio.unibo.it',
    description='Online map processing nodes for the LIMO navigation pipeline.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'online_metric_bev = online_map_package.online_metric_bev:main',
            'cv_amcl_debug = online_map_package.cv_amcl_debug:main',
            'online_nav_map = online_map_package.online_nav_map:main',
        ],
    },
)
