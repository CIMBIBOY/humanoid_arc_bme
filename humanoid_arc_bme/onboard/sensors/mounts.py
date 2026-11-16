"""Sensor mounts: one file (config/sensor_mounts.yaml) for the simulator and for the stack's extrinsics.

Pure Python + numpy. Frames are given in the pelvis frame of the standing robot; the stack works in a ground-referenced body frame
(origin on the floor under the pelvis, x forward, z up), so every pose here takes the standing pelvis height.
The camera convention is converted in ONE place: `OPTICAL_FROM_USD` (a USD / Isaac camera looks along -z with +y up; the stack's
optical frame is x right, y down, z forward, as RealSense and OpenCV use).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from humanoid_arc_bme.onboard.pipeline.interfaces import Intrinsics, Pose
from humanoid_arc_bme.onboard.pipeline.perception.sensing import DepthNoise, camera_pose_from_mount, matrix_to_quat

from humanoid_arc_bme.onboard.paths import CONFIG_DIR

DEFAULT = CONFIG_DIR / "sensor_mounts.yaml"
OPTICAL_FROM_USD = np.diag([1.0, -1.0, -1.0])        # p_optical = OPTICAL_FROM_USD @ p_usd_camera (its own inverse)
# A camera prim placed in a link frame (x forward, y left, z up) so that it looks along the link's +x: columns are the USD camera's
# axes (x right, y up, z backward) expressed in the link frame.
USD_CAMERA_IN_LINK = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def usd_depth_to_optical(points_usd: np.ndarray) -> np.ndarray:
    """(N, 3) points in a USD camera frame -> the optical frame (x right, y down, z forward)."""
    return np.asarray(points_usd, float) @ OPTICAL_FROM_USD.T


def intrinsics_from_fov(width: int, height: int, hfov_deg: float) -> Intrinsics:
    """Square pixels, principal point at the image centre in the pixel-centre convention (cx = (W - 1) / 2)."""
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return Intrinsics(f, f, (width - 1) / 2.0, (height - 1) / 2.0, width, height)


@dataclass(frozen=True)
class Mounts:
    raw: dict

    def frame(self, name: str) -> dict:
        return self.raw["frames"][name]

    def pelvis_height(self, where: str = "robot") -> float:
        return float(self.raw["pelvis_height_m"][where])

    def camera_pose(self, name: str, where: str = "robot") -> Pose:
        """Optical-to-body pose of a camera for a robot standing at its nominal height (what the stack uses as extrinsics)."""
        f = self.frame(name)
        x, y, z = f["xyz_pelvis"]
        return camera_pose_from_mount((x, y, z + self.pelvis_height(where)), math.radians(f["pitch_down_deg"]))

    def lidar_pose(self, where: str = "robot") -> Pose:
        """Lidar-to-body pose: the mount position above the ground and a rotation about y by the down-pitch."""
        f = self.frame("mid360")
        a = math.radians(f["pitch_down_deg"])
        x, y, z = f["xyz_pelvis"]
        return Pose((x, y, z + self.pelvis_height(where)), (math.cos(a / 2.0), 0.0, math.sin(a / 2.0), 0.0))

    def intrinsics(self, name: str) -> Intrinsics:
        c = self.raw["cameras"][name]
        return intrinsics_from_fov(int(c["width"]), int(c["height"]), float(c["hfov_deg"]))

    def depth_noise(self) -> DepthNoise:
        return DepthNoise(**{k: float(v) for k, v in self.raw["depth_noise"].items()})

    def autonomy_overrides(self, where: str = "robot") -> dict:
        """The keys of config/autonomy.yaml that describe geometry, taken from this file (so there is one source)."""
        d, m = self.frame("d435"), self.frame("mid360")
        h = self.pelvis_height(where)
        return {"pelvis_height_m": h, "camera": {"mount_torso_xyz": list(d["xyz_pelvis"]), "pitch_down_deg": d["pitch_down_deg"],
                                                  "width": self.raw["cameras"]["d435"]["width"], "height": self.raw["cameras"]["d435"]["height"],
                                                  "fps": self.raw["cameras"]["d435"]["rate_hz"]},
                "lidar": {"enabled": True, "mount_xyz": [m["xyz_pelvis"][0], m["xyz_pelvis"][1], m["xyz_pelvis"][2] + h], "pitch_down_deg": m["pitch_down_deg"]}}


def load_mounts(path=None) -> Mounts:
    import yaml

    raw = yaml.safe_load(Path(path or DEFAULT).read_text())
    for name, f in raw["frames"].items():
        if len(f["xyz_pelvis"]) != 3 or not all(math.isfinite(float(v)) for v in f["xyz_pelvis"]) or not -90.0 <= float(f["pitch_down_deg"]) <= 90.0:
            raise ValueError(f"sensor_mounts: bad frame {name}")
        if f["tag"] not in ("ASSET", "PLACEHOLDER", "MEASURED"):
            raise ValueError(f"sensor_mounts: frame {name} needs a tag (ASSET | PLACEHOLDER | MEASURED)")
    return Mounts(raw)


def link_rotation_for_pitch(pitch_down_deg: float) -> np.ndarray:
    """Rotation of a sensor frame (x forward) pitched down by the angle, in its parent frame."""
    a = math.radians(pitch_down_deg)
    return np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])


def quat_of(R: np.ndarray) -> tuple:
    return matrix_to_quat(np.asarray(R, float))
