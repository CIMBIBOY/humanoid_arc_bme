"""ROS-free navigation core for a walking robot commanded by body velocity (vx, vy, wz).

Design, options, ranking and the robot plan: docs/plan/navstack.md. Stdlib only, so it can ship to the Jetson in
the runtime bundle; the ROS 2 wrapper (`ros2_node.py`) imports rclpy lazily and is optional.

    path.py         Path (sampled, s/heading/curvature/direction), builders: waypoints, line, arc, clothoid, spline
    limits.py       NavLimits, curvature-limited speed profile, CommandShaper (ramps, curvature-keeping saturation)
    controllers.py  PurePursuit, RegulatedPurePursuit, Stanley, ProfileTracker, MPPI behind one interface
    follower.py     PathFollower state machine: align, track, stop-at-goal + settle, obstacle cap, failure reasons
    odometry.py     Odometry interface: truth, dead reckoning (command/IMU yaw), external pose, fused
    obstacle.py     speed-and-separation obstacle-stop hook (A2.6 rule)
    runtime_hook.py NavSession for the web UI / humanoid_arc_bme.onboard.runtime (JSON in/out), NavPolicy (runtime Policy protocol)
    sim.py          CPU harness: unicycle + lag + delay + deadband + sway + odometry drift, metrics
    ros2_node.py    nav_msgs/Path in, geometry_msgs/Twist out (needs rclpy)
"""

from .controllers import CONTROLLERS, make_controller  # noqa: F401
from .follower import FollowerConfig, PathFollower  # noqa: F401
from .limits import CommandShaper, NavLimits, speed_profile  # noqa: F401
from .path import Path, PathPlan, Pose2D, arc, catmull_rom, clothoid, from_waypoints, line, standard_paths  # noqa: F401
