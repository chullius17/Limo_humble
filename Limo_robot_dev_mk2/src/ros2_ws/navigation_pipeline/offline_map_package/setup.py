from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'offline_map_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob(os.path.join('launch', '*.launch.py'))),
        ('share/' + package_name + '/config', glob(os.path.join('config', '*.yaml')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'offline_metric_bev = offline_map_package.offline_metric_bev:main',
            'filtering = offline_map_package.filtering:main',
            'cv_map_display = offline_map_package.cv_map_display:main',
            'nav_map = offline_map_package.nav_map:main',
            'map_saver = offline_map_package.map_saver:main',
            'map_save_gui = offline_map_package.map_save_gui:main',
        ],
    },
)
