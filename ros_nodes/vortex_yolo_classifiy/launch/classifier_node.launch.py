#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ALLOWED_DEVICES = ['cpu', '0']


def validate_device(device: str):
    if device not in ALLOWED_DEVICES:
        raise RuntimeError(
            f"Invalid device '{device}'. Choose one of: {', '.join(ALLOWED_DEVICES)}"
        )


def launch_setup(context, *args, **kwargs):
    device = LaunchConfiguration('device').perform(context)
    validate_device(device)

    classifier_params = os.path.join(
        get_package_share_directory('vortex_yolo_classifiy'),
        'config/vortex_yolo_classify_params.yaml',
    )

    classifier_node = Node(
        package='vortex_yolo_classifiy',
        executable='classifier_node',
        name='classifier_node',
        namespace='yolo',
        output='screen',
        parameters=[
            classifier_params,
            {'device': device},
        ],
    )

    return [classifier_node]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'device',
                default_value='cpu',
                description="Device to run YOLO classifier on ('0' for GPU or 'cpu')",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
