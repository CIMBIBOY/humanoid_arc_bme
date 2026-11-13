"""g1_runtime_ros: thin ROS 2 wrapper. Importing this package never imports rclpy; `node` does (and needs ROS 2)."""

import importlib.util

HAVE_RCLPY = importlib.util.find_spec("rclpy") is not None
