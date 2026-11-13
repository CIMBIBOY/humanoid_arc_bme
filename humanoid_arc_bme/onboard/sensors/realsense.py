"""Head depth camera (Intel RealSense D435i class, looking down): aligned colour and metric depth as a pipeline source.

Stdlib + numpy at import time; `pyrealsense2` is imported only by the live pipeline's `start()`. Frames: depth is in
the camera optical frame (x right, y down, z forward), metres; `cam_to_body` maps it to the body frame (calibrated
extrinsics, see `camera_pose_from_mount` in pipeline/perception/sensing.py). Stamps are host monotonic seconds taken
when the frameset is received; the driver's hardware timestamps are not used.

Intrinsics convention: the colour stream's principal point (ppx, ppy) is used AS GIVEN, i.e. read as the OpenCV centre
convention (pixel (u, v) centred at integer coordinates). UNVERIFIED on the device: if the driver reports the pixel-edge
convention (ppx close to W/2, not (W-1)/2) every back-projected point is off by half a pixel. Check on the bench.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Protocol

import numpy as np

from humanoid_arc_bme.onboard.pipeline.interfaces import DepthFrame, Intrinsics, Pose
from humanoid_arc_bme.onboard.validation import is_num

MAX_RANGE_M = 10.0      # depth beyond this is treated as no return (D435i useful range; placeholder, see config)
CAMERA = "down"


def depth_to_metres(raw_z16: np.ndarray, depth_scale: float) -> np.ndarray:
    """Raw z16 depth units -> float32 metres (`depth_scale` metres per unit, from the device). Raw 0 (no depth) and
    values beyond MAX_RANGE_M become NaN. Raises ValueError on a non-integer or non-2-D array or a bad scale."""
    if not (is_num(depth_scale) and depth_scale > 0):
        raise ValueError("depth_scale: must be a finite number > 0")
    raw = np.asarray(raw_z16)
    if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("raw_z16: expected a 2-D integer array")
    metres = raw.astype(np.float32) * np.float32(depth_scale)
    invalid = (raw <= 0) | (metres > MAX_RANGE_M)
    return np.where(invalid, np.float32(np.nan), metres).astype(np.float32)


class FramePipeline(Protocol):
    """What RealSenseSource needs from a camera. After start() the pipeline exposes the colour intrinsics (aligned
    depth shares them) and the depth scale. wait_for_frames() blocks for the next frameset and returns
    (rgb uint8 (H, W, 3), depth z16 uint16 (H, W)); it raises RuntimeError on a timeout."""

    depth_scale: float
    intrinsics: Intrinsics

    def start(self) -> None: ...

    def wait_for_frames(self) -> tuple[np.ndarray, np.ndarray]: ...

    def stop(self) -> None: ...


class RealSenseSource:
    """Newest aligned (rgb, depth) pair from the head camera. `latest()` waits for the next frameset (the pipeline's
    timeout bounds the wait). A timeout or malformed frameset returns (None, None) and increments `dropped`; it never
    raises. Construction raises if the pipeline cannot start or reports a bad depth scale."""

    def __init__(self, cam_to_body: Pose, width: int = 640, height: int = 480, fps: int = 15,
                 serial: str | None = None, pipeline: FramePipeline | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self._cam_to_body = cam_to_body
        self._clock = clock
        self._pipe = pipeline if pipeline is not None else _RealSensePipeline(width, height, fps, serial)
        self._pipe.start()
        self._intr: Intrinsics = self._pipe.intrinsics
        self._scale = self._pipe.depth_scale
        if not (is_num(self._scale) and self._scale > 0):
            raise ValueError("depth_scale: device reported no valid depth scale")
        self._h, self._w = self._intr.height, self._intr.width
        self._closed = False
        self.dropped = 0

    def latest(self) -> tuple[np.ndarray | None, DepthFrame | None]:
        """(rgb uint8 (H, W, 3), DepthFrame in metres with NaN for invalid pixels), or (None, None) on a missing frame."""
        if self._closed:
            return None, None
        try:
            rgb, raw = self._pipe.wait_for_frames()
            stamp = self._clock()
            rgb = np.asarray(rgb)
            if rgb.dtype != np.uint8 or rgb.shape != (self._h, self._w, 3):
                raise ValueError("rgb: expected uint8 (H, W, 3) matching the intrinsics")
            depth = DepthFrame(stamp, depth_to_metres(raw, self._scale), self._intr, self._cam_to_body, CAMERA)
        except (RuntimeError, ValueError, TypeError):   # RuntimeError: driver timeout; the rest: malformed frameset
            self.dropped += 1
            return None, None
        return rgb, depth

    def close(self) -> None:
        """Stop the pipeline. Idempotent; `latest()` returns (None, None) afterwards."""
        if not self._closed:
            self._closed = True
            self._pipe.stop()


class _RealSensePipeline:
    """Live librealsense pipeline: depth z16 and colour rgb8 at the same size, depth aligned to colour."""

    def __init__(self, width: int, height: int, fps: int, serial: str | None):
        self._cfg = (width, height, fps, serial)
        self._pipe = None
        self._align = None
        self.depth_scale = math.nan
        self.intrinsics: Intrinsics | None = None

    def start(self) -> None:
        import pyrealsense2 as rs   # lazy: only the live pipeline needs the SDK

        width, height, fps, serial = self._cfg
        cfg = rs.config()
        if serial is not None:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        pipe = rs.pipeline()
        profile = pipe.start(cfg)
        self._pipe = pipe
        self._align = rs.align(rs.stream.color)
        self.depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        ci = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = Intrinsics(ci.fx, ci.fy, ci.ppx, ci.ppy, ci.width, ci.height)

    def wait_for_frames(self) -> tuple[np.ndarray, np.ndarray]:
        frames = self._align.process(self._pipe.wait_for_frames())
        depth, colour = frames.get_depth_frame(), frames.get_color_frame()
        if not depth or not colour:
            raise RuntimeError("incomplete frameset")
        # copy: librealsense recycles the frame buffers once the frameset is released
        return np.array(colour.get_data(), copy=True), np.array(depth.get_data(), copy=True)

    def stop(self) -> None:
        if self._pipe is not None:
            self._pipe.stop()
