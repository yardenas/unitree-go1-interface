import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("unitree_go1_interface"),
        "config",
        "params.yaml",
    )
    return LaunchDescription(
        [
            Node(
                package="unitree_go1_interface",
                executable="robot",
                parameters=[config],
                arguments=["--simulator"],
            ),
            Node(
                package="crl_unitree_monitor",
                executable="monitor",
                remappings=[("monitor_joystick", "remote")],
            ),
        ]
    )
