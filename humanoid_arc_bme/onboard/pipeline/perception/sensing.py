"""Depth sensing helpers: back-projection, a first-principles depth noise model, camera extrinsics config.

Stdlib + numpy only. Conventions match nav/depth_guard.py: camera optical frame x right, y down, z forward;
body frame x forward, y left, z up; metres, radians; quaternions (w, x, y, z).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from humanoid_arc_bme.onboard.pipeline.interfaces import DepthFrame, Pose, QUAT_NORM_TOL, finite_vec, quat_to_matrix  # noqa: F401 (re-export)


def matrix_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 rotation matrix -> unit quaternion (w, x, y, z), w >= 0."""
    m = np.asarray(R, dtype=np.float64)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    q = (w / n, x / n, y / n, z / n)
    return q if q[0] >= 0 else tuple(-c for c in q)


def depth_to_points(frame: DepthFrame, max_range: float = 5.0, stride: int = 1, min_range: float = 0.1) -> np.ndarray:
    """Back-project valid pixels to 3-D points in the BODY frame. Returns (N, 3) float32 (computed in float64).

    Pinhole model: x = (u - cx) z / fx, y = (v - cy) z / fy, z = depth; then p_body = R p + t with
    `frame.cam_to_body`. Pixels that are not finite or outside [min_range, max_range] are dropped. `stride`
    subsamples rows and columns. An all-invalid frame gives shape (0, 3).
    """
    if not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError("stride: must be an int >= 1")
    if not (min_range <= max_range):
        raise ValueError("min_range: must be <= max_range")
    k = frame.intrinsics
    z = np.asarray(frame.depth)[::stride, ::stride].astype(np.float64, copy=False)
    v, u = np.mgrid[0:frame.depth.shape[0]:stride, 0:frame.depth.shape[1]:stride]
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(z) & (z >= min_range) & (z <= max_range)
    zs = z[ok]
    if zs.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    cam = np.empty((zs.size, 3))
    cam[:, 0] = (u[ok] - k.cx) * zs / k.fx
    cam[:, 1] = (v[ok] - k.cy) * zs / k.fy
    cam[:, 2] = zs
    R = frame.cam_to_body.rotation_matrix()
    body = cam @ R.T + np.asarray(frame.cam_to_body.position)
    return body.astype(np.float32)


@dataclass(frozen=True)
class DepthNoise:
    """Stereo/ToF depth error model. ALL VALUES ARE UNMEASURED PLACEHOLDERS until calibrated on the real camera.

    sigma(z) = sigma0 + sigma_k * z^2 (stereo triangulation error grows with the square of range).
    p_drop: per-pixel probability of dropout to NaN on smooth surfaces; p_edge: the same where a neighbouring pixel
    differs by more than `edge_thresh` metres (depth discontinuities). `quant_step` > 0 rounds depth to that step.
    """

    sigma0: float = 0.002  # UNMEASURED
    sigma_k: float = 0.0015  # UNMEASURED, 1/m
    p_drop: float = 0.01  # UNMEASURED
    p_edge: float = 0.30  # UNMEASURED
    edge_thresh: float = 0.10  # UNMEASURED, metres
    quant_step: float = 0.0  # UNMEASURED, metres; 0 disables

    def __post_init__(self):
        for n in ("sigma0", "sigma_k", "edge_thresh", "quant_step"):
            v = getattr(self, n)
            if not (math.isfinite(v) and v >= 0):
                raise ValueError(f"{n}: must be finite and >= 0")
        for n in ("p_drop", "p_edge"):
            v = getattr(self, n)
            if not (math.isfinite(v) and 0.0 <= v <= 1.0):
                raise ValueError(f"{n}: must be in [0, 1]")


def _edge_mask(d: np.ndarray, thresh: float) -> np.ndarray:
    """True where any 4-neighbour is valid and differs by more than thresh."""
    edge = np.zeros(d.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        dx = np.abs(np.diff(d, axis=1)) > thresh
        dy = np.abs(np.diff(d, axis=0)) > thresh
    dx &= np.isfinite(d[:, 1:]) & np.isfinite(d[:, :-1])
    dy &= np.isfinite(d[1:, :]) & np.isfinite(d[:-1, :])
    edge[:, 1:] |= dx
    edge[:, :-1] |= dx
    edge[1:, :] |= dy
    edge[:-1, :] |= dy
    return edge


def apply_depth_noise(depth: np.ndarray, cfg: DepthNoise, rng: np.random.Generator) -> np.ndarray:
    """Return a noisy copy of `depth` (same dtype/shape). Input is not modified; NaN/inf/<=0 pixels stay invalid.
    Deterministic for a given rng state."""
    d = np.asarray(depth)
    out = d.astype(np.float64, copy=True)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(out) & (out > 0)
    edge = _edge_mask(np.where(valid, out, np.nan), cfg.edge_thresh)
    # draw all randoms up front so the stream does not depend on the data
    g = rng.standard_normal(out.shape)
    u = rng.random(out.shape)
    sigma = cfg.sigma0 + cfg.sigma_k * np.where(valid, out, 0.0) ** 2
    out = out + g * sigma
    if cfg.quant_step > 0:
        out = np.round(out / cfg.quant_step) * cfg.quant_step
    drop = u < np.where(edge, cfg.p_edge, cfg.p_drop)
    out[drop & valid] = np.nan
    out[~valid] = d[~valid]
    return out.astype(d.dtype)


# ---------------------------------------------------------------- extrinsics

_CAM_KEYS = ("xyz", "quat_wxyz")


def _ext_vec(where: str, v, n: int) -> tuple[float, ...]:
    if isinstance(v, (str, bytes, dict)):
        raise ValueError(f"{where}: expected a list of {n} numbers")
    return finite_vec(where, v, n)


def load_extrinsics(d: dict) -> dict[str, Pose]:
    """{"cameras": {name: {"xyz": [3], "quat_wxyz": [4]}}} -> {name: Pose} (camera optical frame -> body)."""
    if not isinstance(d, dict):
        raise ValueError("extrinsics: expected a dict")
    extra = set(d) - {"cameras"}
    if extra:
        raise ValueError(f"extrinsics: unknown key(s) {sorted(extra)}")
    if "cameras" not in d:
        raise ValueError("extrinsics.cameras: missing")
    cams = d["cameras"]
    if not isinstance(cams, dict):
        raise ValueError("extrinsics.cameras: expected a dict")
    out: dict[str, Pose] = {}
    for name, c in cams.items():
        where = f"extrinsics.cameras.{name}"
        if not isinstance(c, dict):
            raise ValueError(f"{where}: expected a dict")
        extra = set(c) - set(_CAM_KEYS)
        if extra:
            raise ValueError(f"{where}: unknown key(s) {sorted(extra)}")
        for key in _CAM_KEYS:
            if key not in c:
                raise ValueError(f"{where}.{key}: missing")
        xyz = _ext_vec(f"{where}.xyz", c["xyz"], 3)
        q = _ext_vec(f"{where}.quat_wxyz", c["quat_wxyz"], 4)
        if abs(math.sqrt(sum(x * x for x in q)) - 1.0) > QUAT_NORM_TOL:
            raise ValueError(f"{where}.quat_wxyz: must be unit length")
        out[name] = Pose(xyz, q)
    return out


def dump_extrinsics(poses: dict[str, Pose]) -> dict:
    return {"cameras": {n: {"xyz": list(p.position), "quat_wxyz": list(p.quat)} for n, p in poses.items()}}


def camera_pose_from_mount(xyz, pitch_down_rad: float, yaw_rad: float = 0.0) -> Pose:
    """Optical-to-body pose for a camera whose optical axis points along body +x, pitched down.

    Derivation: with zero pitch/yaw the optical axes map to body as x_opt (right) = -y_body, y_opt (down) = -z_body,
    z_opt (forward) = +x_body, i.e. R0 has columns (0,-1,0), (0,0,-1), (1,0,0). Pitching down by a is a rotation about
    the body y axis by +a (Ry(a) takes +x toward -z), yaw is a rotation about body z: R = Rz(yaw) Ry(pitch) R0.
    """
    R0 = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    c, s = math.cos(pitch_down_rad), math.sin(pitch_down_rad)
    Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Pose(tuple(float(v) for v in xyz), matrix_to_quat(Rz @ Ry @ R0))
