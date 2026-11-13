"""Robot-facing interface: the contract shared by the simulator and the real robot, and the safety layer.

Nothing in this package may import isaaclab, torch or rclpy: it must import on the robot's dev computer
and in simulation alike.
"""
