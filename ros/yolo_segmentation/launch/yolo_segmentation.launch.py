import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    yolo_params = os.path.join(
        get_package_share_directory("yolo_segmentation"),
        "config",
        "yolo_segmentation_params.yaml",
    )

    return LaunchDescription(
        [
            Node(
                package="yolo_segmentation",
                executable="yolo_seg_node",
                name="yolo_segmentation_node",
                output="screen",
                parameters=[yolo_params],
            )
        ]
    )
