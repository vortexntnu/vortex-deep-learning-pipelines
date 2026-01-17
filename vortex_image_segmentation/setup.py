
from setuptools import find_packages, setup
import glob

package_name = 'vortex_image_segmentation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob('launch/*.launch.py')),
        ('share/' + package_name + '/params', glob.glob('params/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mads',
    maintainer_email='mjengesv@ntnu.no',
    description='ROS 2 package that provides a YOLO-based instance segmentation node (yolo_seg_node) for real-time segmentation.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_seg_node = vortex_image_segmentation.yolo_seg_node:main',
        ],
    },
)
