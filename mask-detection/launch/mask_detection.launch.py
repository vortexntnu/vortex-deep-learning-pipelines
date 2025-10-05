import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription

home = os.environ.get('HOME', '/tmp')
out_dir = os.path.join(home, 'seg_frames')

def generate_launch_description():
    mask_detection_node = Node(
        package='mask_detection',
        executable='mask_detection_node',
        name='mask_detection_node',
        parameters=[
            {'output_dir': out_dir},
            os.path.join(
                get_package_share_directory('mask_detection'),
                'config',
                'mask_detection_params.yaml',
                
            )
        ],
        output='screen',
    )
    return LaunchDescription([mask_detection_node])
