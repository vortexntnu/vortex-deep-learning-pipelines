import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

ALLOWED_DEVICES = ['cpu', '0']


def validate_device(device: str):
    if device not in ALLOWED_DEVICES:
        raise RuntimeError(
            f"Invalid device '{device}'. Choose one of: {', '.join(ALLOWED_DEVICES)}"
        )


def launch_setup(context, *args, **kwargs):
    device = LaunchConfiguration('device').perform(context)
    validate_device(device)

    unet_params = os.path.join(
        get_package_share_directory('unet_segmentation'),
        'config/unet_segmentation.yaml',
    )
    model_path = PathJoinSubstitution(
        [
            FindPackageShare('unet_segmentation'),
            'model',
            'unet-simple-320-240-l-5-e10-b16(1).pth',
        ]
    )

    unet_node = Node(
        package='unet_segmentation',
        executable='unet_segmentation_node.py',
        name='unet_segmentation',
        namespace='unet',
        output='screen',
        parameters=[
            unet_params,
            {'device': device},
            {'model_path': model_path},
        ],
    )

    return [unet_node]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'device',
                default_value='0',
                description='run unet segmentation',
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
