"""ROS 2 Humble wrapper (stub): nav_msgs/Path + nav_msgs/Odometry in, geometry_msgs/Twist out.

    ros2 run ... or: python -m humanoid_arc_bme.onboard.nav.ros2_node --controller rpp --rate 20

Topics (names are parameters): `/nav/path` (nav_msgs/Path, frame = odom), `/odom` (nav_msgs/Odometry from FAST-LIO /
Point-LIO / the vendor bridge), `/nav/obstacle_distance` (std_msgs/Float32, nearest obstacle ahead; optional),
out `/cmd_vel` (geometry_msgs/Twist) and `/nav/status` (std_msgs/String, JSON). `/nav/stop` (std_msgs/Empty).
Without fresh odometry the follower fails with `no_pose` and publishes zero.

`rclpy` is imported lazily: the conversion helpers below work on any object with the same attribute names, so they
are tested without ROS (tests/core/test_nav_ros2.py). This node has NOT been run against ROS 2 (no ROS 2 Humble on
this workstation; /opt/ros has jazzy only and rclpy is not importable from the project python).
"""

from __future__ import annotations

import json
import math

from .odometry import ExternalPoseOdometry, yaw_from_quat
from .path import Path, PathPlan, Pose2D


def path_from_msg(msg, direction: int = 1) -> PathPlan:
    """nav_msgs/Path -> PathPlan. Direction is not in nav_msgs/Path: by default forward; a path whose first pose
    orientation is opposite to its travel tangent is taken as backward (the planner set the body yaw)."""
    xs = [p.pose.position.x for p in msg.poses]
    ys = [p.pose.position.y for p in msg.poses]
    if len(xs) < 2:
        raise ValueError("nav_msgs/Path needs at least 2 poses")
    q = msg.poses[0].pose.orientation
    yaw0 = yaw_from_quat((q.w, q.x, q.y, q.z))
    tang = math.atan2(ys[-1] - ys[0], xs[-1] - xs[0]) if len(xs) < 3 else math.atan2(ys[1] - ys[0], xs[1] - xs[0])
    if direction == 0:  # auto
        direction = -1 if math.cos(yaw0 - tang) < -0.5 else 1
    qe = msg.poses[-1].pose.orientation
    return PathPlan([Path(xs, ys, direction, yaw_from_quat((qe.w, qe.x, qe.y, qe.z)))])


def pose_from_odom(msg) -> Pose2D:
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    return Pose2D(p.x, p.y, yaw_from_quat((q.w, q.x, q.y, q.z)))


def twist_fields(cmd) -> dict:
    return {"linear": (cmd.vx, cmd.vy, 0.0), "angular": (0.0, 0.0, cmd.wz)}


def main(argv=None):  # pragma: no cover - needs ROS 2
    import argparse

    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry as OdomMsg
    from nav_msgs.msg import Path as PathMsg
    from rclpy.node import Node
    from std_msgs.msg import Empty, Float32, String

    from .follower import PathFollower
    from .limits import NavLimits
    from .obstacle import SpeedSeparationGuard

    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="regulated_pure_pursuit")
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--direction", type=int, default=0, help="1 forward, -1 backward, 0 auto from the first pose")
    ap.add_argument("--obstacle", action="store_true", help="require /nav/obstacle_distance (fail closed)")
    args = ap.parse_args(argv)

    class NavNode(Node):
        def __init__(self):
            super().__init__("humanoid_arc_nav")
            self.odom = ExternalPoseOdometry(stale_s=0.5)
            self.dist = None
            guard = SpeedSeparationGuard(lambda d: self.dist, require_sensor=True) if args.obstacle else None
            self.fol = PathFollower(self.odom, args.controller, NavLimits(), guard)
            self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
            self.status = self.create_publisher(String, "/nav/status", 10)
            self.create_subscription(PathMsg, "/nav/path", self.on_path, 1)
            self.create_subscription(OdomMsg, "/odom", self.on_odom, 10)
            self.create_subscription(Float32, "/nav/obstacle_distance", self.on_dist, 10)
            self.create_subscription(Empty, "/nav/stop", lambda _m: self.fol.stop(), 1)
            self.create_timer(1.0 / args.rate, self.on_timer)

        def now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def on_path(self, msg):
            self.fol.follow(path_from_msg(msg, args.direction), self.now())

        def on_odom(self, msg):
            self.odom.set_pose(self.now(), pose_from_odom(msg))

        def on_dist(self, msg):
            self.dist = float(msg.data)

        def on_timer(self):
            cmd = self.fol.step(self.now())
            t = Twist()
            t.linear.x, t.linear.y, t.angular.z = cmd.vx, cmd.vy, cmd.wz
            self.pub.publish(t)
            self.status.publish(String(data=json.dumps(self.fol.last, default=float)))

    rclpy.init()
    node = NavNode()
    try:
        rclpy.spin(node)
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
