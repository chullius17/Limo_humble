from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'nav_map_package'

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
            'offline_metric_bev = nav_map_package.offline_mapping.offline_metric_bev:main',
            'online_metric_bev = nav_map_package.online_mapping.online_metric_bev:main',
            'cv_2_ptcld = nav_map_package.online_mapping.cv_2_ptcld:main',
            'laser_cv_fusion = nav_map_package.online_mapping.laser_cv_fusion:main',
            'filtering = nav_map_package.offline_mapping.filtering:main',
            'cv_map_display = nav_map_package.offline_mapping.cv_map_display:main',
            'nav_map = nav_map_package.nav_map:main',
            'online_nav_map = nav_map_package.online_nav_map:main',
            'map_saver = nav_map_package.offline_mapping.map_saver:main',
            'map_save_gui = nav_map_package.offline_mapping.map_save_gui:main',
        ],
    },
)
