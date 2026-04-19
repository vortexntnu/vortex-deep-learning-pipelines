from glob import glob

from setuptools import find_packages, setup

package_name = 'yolo_ultralytics_pose_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/yolo_ultralytics_pose_node'],
        ),
        ('share/yolo_ultralytics_pose_node', ['package.xml']),
        ('share/yolo_ultralytics_pose_node/launch', glob('launch/*.launch.py')),
        ('share/yolo_ultralytics_pose_node/config', glob('config/*.yaml')),
        ('share/yolo_ultralytics_pose_node/model', glob('model/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ashishagain',
    maintainer_email='ashishbhardwaj2005@gmail.com',
    description='YOLO pose estimation for valve orientation using keypoints.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_ultralytics_pose_node=yolo_ultralytics_pose_node.yolo_ultralytics_pose_node:main',
        ],
    },
)
