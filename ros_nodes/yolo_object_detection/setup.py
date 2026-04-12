from setuptools import setup, find_packages
from glob import glob

package_name = 'yolo_object_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/yolo_object_detection']),
        ('share/yolo_object_detection', ['package.xml']),
        ('share/yolo_object_detection/launch', glob('launch/*.launch.py')),
        ('share/yolo_object_detection/config', glob('config/*.yaml')),
        ('share/yolo_object_detection/model', glob('model/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kluge7',
    maintainer_email='89779148+kluge7@users.noreply.github.com',
    description='YOLO object detection on images, publishing detections and annotated outputs.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_object_detection_node=yolo_object_detection.yolo_object_detection_node:main',
        ],
    },
)
