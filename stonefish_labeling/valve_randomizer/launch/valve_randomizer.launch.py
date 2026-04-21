from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("period", default_value="5.0"),
            DeclareLaunchArgument("min_angle", default_value="-1.57"),
            DeclareLaunchArgument("max_angle", default_value="1.57"),
            Node(
                package="valve_randomizer",
                executable="valve_randomizer_node",
                name="valve_randomizer",
                output="screen",
                parameters=[
                    {
                        "valves": ["valve1", "valve2"],
                        "period": LaunchConfiguration("period"),
                        "min_angle": LaunchConfiguration("min_angle"),
                        "max_angle": LaunchConfiguration("max_angle"),
                    }
                ],
            ),
        ]
    )
