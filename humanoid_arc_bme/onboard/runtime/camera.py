"""Camera sources. The G1's head camera is an Intel RealSense D435i per Unitree's xr_teleoperate docs (UNKNOWN for this robot until B0.1/B5 check it)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .types import CameraFrame


class CameraSource(ABC):
    name: str

    @abstractmethod
    def read(self) -> CameraFrame | None:
        """Newest frame not yet returned, or None. Never blocks longer than a few ms."""

    def close(self) -> None:  # noqa: B027
        pass


class FakeCamera(CameraSource):
    def __init__(self, clock, name="head", width=64, height=48):
        self.clock, self.name, self.w, self.h = clock, name, width, height

    def read(self):
        v = int(self.clock() * 50) % 256
        return CameraFrame(self.name, self.clock(), self.w, self.h, "rgb8", bytes([v, 255 - v, 128]) * (self.w * self.h))


class RealSenseCamera(CameraSource):
    """Colour stream only (pyrealsense2, imported lazily). Not run on a robot yet."""

    def __init__(self, clock, name="head", width=640, height=480, fps=30, serial: str | None = None):
        import pyrealsense2 as rs

        self.clock, self.name, self.w, self.h = clock, name, width, height
        self._pipe, cfg = rs.pipeline(), rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self._pipe.start(cfg)

    def read(self):
        ok, frames = self._pipe.try_wait_for_frames(timeout_ms=5)
        if not ok:
            return None
        color = frames.get_color_frame()
        return CameraFrame(self.name, self.clock(), self.w, self.h, "rgb8", bytes(color.get_data()))

    def close(self):
        self._pipe.stop()
