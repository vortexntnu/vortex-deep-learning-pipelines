from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vortex_image_segmentation',
            executable='yolo_seg_node',
            name='yolo_segmentation_node',
            output='screen',
            parameters=[
                '/ros2_ws/vortex_image_segmentation/params/yolo_params.yaml'
            ]
        )
    ])
