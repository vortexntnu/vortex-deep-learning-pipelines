from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('vortex_image_segmentation')
    yolo_params = os.path.join(pkg_share, 'params', 'yolo_params.yaml')

    return LaunchDescription([
        Node(
            package='vortex_image_segmentation',
            executable='yolo_seg_node',
            name='yolo_segmentation_node',
            output='screen',
            parameters=[yolo_params]
        )
    ])
