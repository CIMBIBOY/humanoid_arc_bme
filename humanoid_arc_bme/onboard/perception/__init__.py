"""Perception models: colour naming, YOLO instance segmentation (ONNX), and scene assembly into body-frame objects.

Lightweight at import time (numpy only). onnxruntime and cv2 are imported lazily inside the functions that need them.
"""
