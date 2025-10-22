from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vortex_image_segmentation',
            executable='unet_node',
            name='unet_segmentation_node',
            parameters=[
                'config/common_params.yaml',
                'config/unet_params.yaml'
            ]
        ),
        Node(
            package='vortex_image_segmentation',
            executable='yolo_node',
            name='yolo_segmentation_node',
            parameters=[
                'config/common_params.yaml',
                'config/yolo_params.yaml'
            ]
        )
    ])
