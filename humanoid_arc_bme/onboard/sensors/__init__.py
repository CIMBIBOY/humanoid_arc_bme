"""Real-sensor adapters for the onboard pipeline: RealSense head depth camera, Livox MID-360 lidar, DLIO odometry.

Each adapter turns one device's native data into the pipeline's interface types (`DepthFrame`, `LidarFrame`,
`LocalizerEstimate`). The hardware and ROS imports (`pyrealsense2`, `rclpy`) are lazy: importing this package needs
only the stdlib and numpy, and every class accepts an injected fake so the adapters are testable off the robot.
"""
