from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config_file', description='e-MDB experiment yaml'),
        DeclareLaunchArgument('random_seed', default_value='0'),
        DeclareLaunchArgument('visualize', default_value='True'),

        Node(
            package='ur5_mujoco',
            executable='sim_ur5',
            name='sim_ur5_mujoco',
            output='screen',
            parameters=[{
                'config_file': LaunchConfiguration('config_file'),
                'random_seed': LaunchConfiguration('random_seed'),
                'visualize': LaunchConfiguration('visualize'),
            }],
        ),
    ])
