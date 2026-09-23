from setuptools import setup
import os
from glob import glob

package_name = 'cv_package'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob(os.path.join('launch', '*.launch.py'))),
        ('share/' + package_name + '/config',
            glob(os.path.join('config', '*.yaml')) +
            glob(os.path.join('config', '*.rviz'))),
    ],
    install_requires=['setuptools', 'PyTurboJPEG'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_detector = cv_package.lane_detector:main',
            'visual_ptcld = cv_package.visual_ptcld:main',
            'depth_correction = cv_package.depth_correction:main',
        ],
    },
)
