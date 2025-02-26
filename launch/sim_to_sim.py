import os
from copy import deepcopy
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("unitree_go1_interface"),
        "config",
        "params.yaml",
    )
    print("Config path:", config)
    config = apply_yaml_alias(config)
    return LaunchDescription(
        [
            Node(
                package="unitree_go1_interface",
                executable="robot",
                namespace="go1_sim",
                parameters=[config],
                arguments=["--simulator"],
            ),
            Node(
                package="crl_unitree_monitor",
                namespace="go1_sim",
                executable="monitor",
                remappings=[("monitor_joystick", "remote")],
            ),
        ]
    )


def apply_yaml_alias(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    export_config = deepcopy(config)

    for key, value in config["/**"].items():
        is_ros_param = False

        for key_ in value.keys():
            if "ros__parameters" in key_:
                is_ros_param = True

        if not is_ros_param:
            # remove anchor delcaration
            export_config["/**"].pop(key)
        else:
            export_config["/**"][key] = value
            for key__ in export_config["/**"][key]["ros__parameters"].keys():
                export_config["/**"][key]["ros__parameters"][key__] = deepcopy(
                    value["ros__parameters"][key__]
                )

    # export export_config
    tmp_dir = Path(config_path).parent / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(tmp_dir, "tmp.yaml"), "w") as file:
        yaml.dump(export_config, file, default_flow_style=False)

    export_config_path = tmp_dir / "tmp.yaml"

    return export_config_path.__str__()
