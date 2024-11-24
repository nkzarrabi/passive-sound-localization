from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    movement_config = os.path.join(
        get_package_share_directory("movement_library"),
        "config",
        "movement_params.yaml",
    )

    slam_config = os.path.join(
        get_package_share_directory("slam_toolbox"),
        "config",
        "mapper_params_online_sync.yaml",
    )

    return LaunchDescription(
        [
            Node(
                package="slam_toolbox",
                executable="sync_slam_toolbox_node",
                name="slam_toolbox",
                output="screen",
                parameters=[slam_config],
            ),
            Node(
                package="nav2_bringup",
                executable="bringup_launch.py",
                name="nav2",
                output="screen",
            ),
            Node(
                package="movement_library",
                executable="movement_node",
                name="movement_node",
                output="screen",
                parameters=[movement_config],
            ),
        ]
    )
