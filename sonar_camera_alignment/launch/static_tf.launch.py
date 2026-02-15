from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_to_sonar_tf',
            arguments=[
                '0', '0', '0',          # x y z
                '0', '0', '0',          # roll pitch yaw
                'camera_rig/segmentation_camera_front',
                'camera_rig/front_fls'
            ],
            output='screen'
        )
    ])
