"""Code that runs on the robot (Jetson Orin NX, JetPack 6.2, Python 3.10): the safety layer, the G1 operator runtime,
navigation, the perception/grasp pipeline, skills and the delivery tools. Standard library + numpy at import time; heavy
backends (onnxruntime, TensorRT, ROS, FastAPI) are imported lazily inside the functions that need them.

This is the standalone university baseline. The private `arc` package may import it and replace its parts through the
Protocols in `humanoid_arc_bme.onboard.pipeline.interfaces`; it must never be the other way round (tests/test_layering.py).
"""
