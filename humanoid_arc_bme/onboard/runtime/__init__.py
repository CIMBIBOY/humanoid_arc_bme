"""humanoid_arc_bme.onboard.runtime: what runs ON the G1's dev computer (Jetson).

Ports and adapters: the control loop talks to `RobotIO` / `CameraSource` / `TelemetrySink` interfaces; adapters for the
real robot (unitree_sdk2py, RealSense), for tests (fakes) and for the operator's viewer (Foxglove, JSONL) plug in behind them.
The robot-facing messages and the safety layer are the shared stdlib-only contract `humanoid_arc_bme.onboard.api`.
Design and rules: docs/plan/runtime-design.md.
"""

__version__ = "0.2.0"
