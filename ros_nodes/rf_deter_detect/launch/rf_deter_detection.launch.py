#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory('rf_deter_detect'),
        'config',
        'rf_deter_detection_params.yaml',
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'device',
                default_value='cpu',
                description=(
                    "RF-DETR device: 'cpu', 'cuda', 'cuda:N', 'mps', or GPU index"
                ),
            ),
            Node(
                package='rf_deter_detect',
                executable='rf_deter_detection_node.py',
                name='rf_deter_detection',
                namespace='rf_detr',
                output='screen',
                parameters=[
                    params_file,
                    {
                        'device': ParameterValue(
                            LaunchConfiguration('device'), value_type=str
                        )
                    },
                ],
            ),
        ]
    )
