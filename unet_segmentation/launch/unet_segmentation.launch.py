# launch/unet_segmentation.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    pkg = DeclareLaunchArgument("package", default_value="unet_segmentation")
    params_file = DeclareLaunchArgument(
        "params_file",
        default_value=PathJoinSubstitution([
            FindPackageShare(LaunchConfiguration("package")),
            "config",
            "unet_segmentation.yaml",
        ]),
    )
    input_topic = DeclareLaunchArgument("input_topic", default_value="/image_color")

    node = Node(
        package=LaunchConfiguration("package"),
        executable="interface.py",
        name="unet_segmentation_node",
        parameters=[LaunchConfiguration("params_file")],
        remappings=[
            ("/image_color", LaunchConfiguration("input_topic")),
            # Overlay and mask topics come from YAML; remap here only if needed:
            # ("/segmentation/overlay", "/your/overlay"),
            # ("/segmentation/mask", "/your/mask"),
        ],
        output="screen",
    )

    return LaunchDescription([pkg, params_file, input_topic, node])
