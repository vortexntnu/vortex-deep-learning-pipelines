import os

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Get the home directory
    home = os.environ.get("HOME", "/tmp")
    legend_csv_path = os.path.join(home, "seg_frames", "legend.csv")
    label_map = os.path.join(home, "seg_frames", "id_label_map.json")
    return LaunchDescription([
        Node(
            package='mask_classifier',
            executable='node.py',
            name='mask_classifier_node',
            output='screen',
            parameters=[{
                # Publisher from C++ that gives the RGB image with masks
                'segmentation_image_color_sub_topic': 'segmentation_image_color',
                'segmentation_image_id_sub_topic': 'segmentation_image_id',
                # Path where your C++ saves the legend (optional but useful)
                'legend_csv_path': legend_csv_path,
                'id_label_map_path': label_map,
                # Minimum pixel threshold to consider an object
                'min_pixels_per_object': 200,
            }],
        )
    ])
