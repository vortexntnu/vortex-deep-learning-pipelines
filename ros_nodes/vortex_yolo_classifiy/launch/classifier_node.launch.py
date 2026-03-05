from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vortex_yolo_classifiy',
            executable='classifier_node',
            name='classifier_node',
            output='screen',
            parameters=[
                {
                    'model_path': '/home/gard/ros2_ws/src/vortex-deep-learning-pipelines/runs/classify/results/classify-20260304-154252/weights/best.pt'
                }
            ]
        )
    ])