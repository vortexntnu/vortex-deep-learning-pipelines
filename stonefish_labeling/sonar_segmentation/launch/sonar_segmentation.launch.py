import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("sonar_segmentation"),
        "config",
        "sonar_segmentation.yaml",
    )

    return LaunchDescription(
        [
            Node(
                package="sonar_segmentation",
                executable="sonar_segmentation_node",
                name="sonar_segmentation_node",
                parameters=[config],
                output="screen",
            )
        ]
    )
