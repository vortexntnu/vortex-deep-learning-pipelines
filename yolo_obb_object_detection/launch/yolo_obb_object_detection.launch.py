#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node




def validate_device(device: str):
    if device in ("cpu", "cuda", "mps"):
        return

    if device.isdigit():
        return

    if device.startswith("cuda:") and device.split(":")[1].isdigit():
        return

    raise RuntimeError(
        f"Invalid device '{device}'. Use 'cpu', GPU index (0,1,...), "
        "'cuda', 'cuda:N', or 'mps'."
    )

def launch_setup(context, *args, **kwargs):
    device = LaunchConfiguration('device').perform(context)
    validate_device(device)

    yolo_params = os.path.join(
        get_package_share_directory('yolo_obb_object_detection'),
        'config/yolo_obb_object_detection_params.yaml',
    )

    yolo_node = Node(
        package='yolo_obb_object_detection',
        executable='yolo_obb_object_detection_node.py',
        name='yolo_obb_object_detection',
        namespace='yolo',
        output='screen',
        parameters=[
            yolo_params,
            {'device': device},
        ],
    )

    return [yolo_node]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'device',
                default_value='cpu',
                description="Device to run YOLO on: 'cpu', GPU index (0,1,...), 'cuda', 'cuda:N', or 'mps' (Mac GPU)",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
