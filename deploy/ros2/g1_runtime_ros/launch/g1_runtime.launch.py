"""ros2 launch g1_runtime_ros g1_runtime.launch.py io:=unitree iface:=eth0 models:=/home/unitree/g1_models  (UI on 127.0.0.1: SSH tunnel)

Starts the operator runtime DISARMED. Nothing moves until an operator arms in the web UI with the typed code.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGS = {
    "io": ("fake", "fake | unitree"),
    "iface": ("", "network interface to the robot (unitree), e.g. eth0"),
    "models": ("", "directory of policy dirs / packs (policy.onnx + layout.json)"),
    "shadow": ("", "policy dir run in shadow mode (computes and logs, never sends)"),
    "camera": ("none", "none | fake | realsense"),
    "lan": ("false", "true: serve the UI on 0.0.0.0 (default 127.0.0.1)"),
    "port": ("8080", "web UI port"),
    "token": ("", "fixed UI token (default: random, printed in the log)"),
    "allow_lowcmd": ("false", "create the gated rt/lowcmd path (Hold step; policy controller on the robot)"),
    "config_dir": ("", "directory with g1_limits.yaml and robot_runtime.yaml (default: the package share)"),
    "log_root": ("", "run log root (default from robot_runtime.yaml)"),
    "foxglove": ("-1", "Foxglove WebSocket port on 127.0.0.1 (-1 = off)"),
    "lan_plain_http_ack": ("false", "true: acknowledge that lan:=true is plain HTTP (required with lan:=true)"),
    "limits": ("", "S1 + operator limits file (default: config_dir/g1_limits.yaml; first walk: g1_limits_firstwalk.yaml)"),
}


def generate_launch_description():
    decl = [DeclareLaunchArgument(k, default_value=v, description=d) for k, (v, d) in ARGS.items()]
    node = Node(package="g1_runtime_ros", executable="g1_runtime_node", name="g1_runtime", output="screen",
                parameters=[{k: LaunchConfiguration(k) for k in ARGS}])
    return LaunchDescription(decl + [node])
