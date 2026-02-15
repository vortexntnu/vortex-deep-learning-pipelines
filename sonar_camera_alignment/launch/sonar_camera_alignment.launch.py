from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('sonar_camera_alignment')
    default_params = os.path.join(pkg_share, 'config', 'sonar_camera_alignment.yaml')
    tf_launch = os.path.join(pkg_share, 'launch', 'static_tf.launch.py')
    tf_launch_depth = os.path.join(pkg_share, 'launch', 'depth_to_seg_tf.launch.py')

    params_file = LaunchConfiguration('params_file')
    sonar_topic = LaunchConfiguration('sonar_topic')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Path to the YAML parameter file.'
        ),

        # IMPORTANT: Choose the fan-shaped sonar topic here.
        # If your sim swaps topics (544 vs 768), set this to whichever is the fan in that run.
        DeclareLaunchArgument(
            'sonar_topic',
            default_value='/front_sonar/display_mono',
            description='Fan-shaped sonar image topic to use as reference (must match projected_seg size).'
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(tf_launch)
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(tf_launch_depth)
        ),


        Node(
            package='sonar_camera_alignment',
            executable='sonar_camera_alignment_node',
            name='sonar_camera_alignment_projector',
            output='screen',
            parameters=[
                params_file,
                {
                    'sonar_topic': sonar_topic,
                    'camera_frame_override': 'camera_rig/segmentation_camera_front',
                    'sonar_frame_override': 'camera_rig/front_fls',

                    # Force output size to match the fan-shaped reference image
                    'out_width': -1,
                    'out_height': -1,
                }
            ],
        ),

        Node(
            package='sonar_camera_alignment',
            executable='sonar_projected_overlay_node',
            name='sonar_projected_overlay',
            output='screen',
            parameters=[
                params_file,
                {
                    # Overlay base must be the SAME sonar topic used as reference (fan-shaped)
                    'sonar_topic': sonar_topic,
                    'projected_topic': '/front_sonar/projected_seg',
                    'overlay_topic': '/front_sonar/overlay',

                    # Keep false to avoid distorting the mask; fix topics/size instead
                    'resize_projected_to_sonar': False,

                    'alpha': 0.65,
                    'dilate_iter': 1,
                    'colorize_labels': True,
                    'mask_to_sonar_fan': True,
                    'fan_mask_threshold': 5,
                }
            ],
        ),
        Node(
            package='sonar_camera_alignment',
            executable='seg_depth_sync_node',
            name='seg_depth_sync',
            output='screen',
            parameters=[{
                'seg_topic': '/front_camera_seg/image_raw',
                'depth_topic': '/depth_camera/image_depth',
                'camera_info_topic': '/front_camera_seg/camera_info',
                'out_topic': '/synced/seg_depth_packet',
                # More tolerant sync for sim timestamp jitter
                'sync_queue_size': 400,
                'max_sync_interval_s': 5.0,
                'max_depth_m': 10.0,
            }]
        ),
        Node(
            package='sonar_camera_alignment',
            executable='seg_label_legend_node',
            name='seg_label_legend',
            output='screen',
            parameters=[
                params_file,
                {
                    'packet_topic': '/synced/seg_depth_packet',
                }
            ],
        ),
    ])
