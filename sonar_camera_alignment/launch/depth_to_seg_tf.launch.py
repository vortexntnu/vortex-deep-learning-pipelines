from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="depth_to_seg_tf",
            arguments=[
                "0", "0", "0",
                "0", "0", "0",
                "camera_rig/Dcam",
                "camera_rig/segmentation_camera_front",
            ],
            output="screen"
        ),
    ])
