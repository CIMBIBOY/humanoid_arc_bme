"""ROS 2 node: runs the same operator runtime + web UI as `python -m humanoid_arc_bme.onboard.runtime serve`, plus ROS topics.

    ros2 launch g1_runtime_ros g1_runtime.launch.py io:=unitree iface:=eth0   (UI on 127.0.0.1: SSH tunnel)
    ros2 run g1_runtime_ros g1_runtime_node --ros-args -p io:=fake

Topics:
  sub /cmd_vel (geometry_msgs/Twist)        used only when the UI's input mode is "ROS cmd_vel", and only while the
                                            UI heartbeat (the deadman) keeps arriving. ROS cannot arm the robot.
  sub /g1_runtime/stop (std_msgs/Empty)     STOP from ROS (always allowed: stopping is always safe)
  pub /g1_runtime/status (std_msgs/String)  JSON: supervisor state, command, preflight green, 2 Hz
  pub /g1_runtime/cmd (geometry_msgs/Twist) the command sent to the controller (zero while disarmed), 10 Hz
"""

from __future__ import annotations

import json

import rclpy
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from std_msgs.msg import Empty, String

from humanoid_arc_bme.onboard.runtime.cli import build_operator

from .params import PARAMS, to_args, twist_to_cmd


class G1RuntimeNode(Node):
    def __init__(self):
        super().__init__("g1_runtime")
        for k, (default, _kind) in PARAMS.items():
            # dynamic typing: launch arguments are YAML-parsed strings (a numeric token or "" must not crash the node)
            self.declare_parameter(k, default, ParameterDescriptor(dynamic_typing=True))
        args = to_args({k: self.get_parameter(k).value for k in PARAMS})
        self.rt, self.app, token = build_operator(args)
        self.app.make_server()
        self.rt.start()
        self.app.start()
        self.get_logger().info(f"operator UI on {self.app.host}:{self.app.port}, token {token} (DISARMED); run log {self.rt.runlog.dir}")
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        self.create_subscription(Empty, "/g1_runtime/stop", lambda _m: self.rt.stop("ros"), 10)
        self.pub_status = self.create_publisher(String, "/g1_runtime/status", 10)
        self.pub_cmd = self.create_publisher(Twist, "/g1_runtime/cmd", 10)
        self.create_timer(0.5, self._status)
        self.create_timer(0.1, self._cmd)

    def _on_cmd_vel(self, m: Twist):
        try:
            self.rt.set_external_cmd(*twist_to_cmd(m.linear.x, m.linear.y, m.angular.z))
        except ValueError as e:
            self.get_logger().warn(f"/cmd_vel rejected: {e}")

    def _status(self):
        s = self.rt.status()
        self.pub_status.publish(String(data=json.dumps({"state": s["supervisor"]["state"], "reason": s["supervisor"]["reason"],
                                                        "controller": s["controller"], "cmd": s["cmd"], "preflight_green": s["preflight_green"],
                                                        "mode": s["mode"]}, default=str)))

    def _cmd(self):
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = (float(x) for x in self.rt.last_cmd)
        self.pub_cmd.publish(t)

    def close(self):
        self.rt.close()
        self.app.close()


def main(argv=None):
    rclpy.init(args=argv)
    node = G1RuntimeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()  # disarms ("shutdown"): controller safe state, logs closed
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
