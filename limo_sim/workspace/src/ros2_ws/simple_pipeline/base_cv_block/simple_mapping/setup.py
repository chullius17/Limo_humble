import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'simple_mapping'

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
            os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py')),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Giulio Cataldo',
    maintainer_email='giulio.cataldo@studio.unibo.it',
    description=(
        'Simple mapping utilities for the LIMO base computer-vision pipeline.'
    ),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'metric_bev = simple_mapping.metric_bev:main',
            'mapper = simple_mapping.mapper:main',
        ],
    },
)
