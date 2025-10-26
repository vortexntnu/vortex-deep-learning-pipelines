from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vortex_image_segmentation',
            executable='unet_node',
            name='unet_segmentation_node',
            output='screen',
            parameters=[
                '/ros2_ws/vortex_image_segmentation/config/common_params.yaml'
                '/ros2_ws/vortex_image_segmentation/config/unet_params.yaml'
            ]
        )
    ])
