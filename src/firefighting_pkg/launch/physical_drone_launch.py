import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('firefighting_pkg')
    model_path = os.path.join(pkg_dir, 'models', 'best.blob')

    return LaunchDescription([
        # 1. The Web Bridge Node
        Node(
            package='firefighting_pkg',
            executable='firebase_bridge',
            name='firebase_bridge_node',
            output='screen'
        ),

        # 2. The OAK-D Vision Node
        Node(
            package='firefighting_pkg',
            executable='oakd_yolo',
            name='oakd_yolo_node',
            output='screen',
            parameters=[{
                'model_path': model_path,
                'confidence_threshold': 0.65
            }]
        ),

        # 3. The Flight Controller Node
        Node(
            package='firefighting_pkg',
            executable='controller',
            name='mavlink_controller_node',
            output='screen',
            parameters=[{
                'connection': '/dev/ttyAMA0', # Physical UART pin on Pi 5
                'baudrate': 57600,
                'search_alt': 15.0,           # Safe altitude above trees
                'drop_alt': 5.0               # Safe drop altitude
            }]
        )
    ])