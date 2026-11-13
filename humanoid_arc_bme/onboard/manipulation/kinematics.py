"""Arm kinematics of the Unitree G1 (29 DoF) for the wrist and palm targets of the task layer.

Forward kinematics, the geometric Jacobian and damped least-squares inverse kinematics, all in the PELVIS frame.

Frames and units: lengths in metres, angles in radians. A pose is a position and a unit quaternion (w, x, y, z), the
contract's order. The tool point is `tool_offset` expressed in the wrist_yaw link frame (an assumed value until the
palm point is measured on the robot). The chain is read once from the kinematics-only URDF
(robots/g1/g1_29dof_with_hand_only_kinematics.urdf). Joint ranges are the URDF limits, shrunk by `margin_rad`, intersected
with the safety layer's ranges (`Limits.joint_range`, config/g1_limits.yaml): every solution lies inside the safety envelope.

Only numpy and xml.etree; no scipy, no torch, no pinocchio.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import numpy as np

from ..api.safety import Limits, load_limits
from ..g1_layout import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, WAIST_JOINTS
from ..geometry import quat_to_matrix
from ..paths import BME_ROOT, DEFAULT_LIMITS
from ..pipeline.interfaces import Pose

SIDES = ("left", "right")
DEFAULT_URDF = BME_ROOT / "humanoid_arc_bme" / "robots" / "g1" / "g1_29dof_with_hand_only_kinematics.urdf"
DEFAULT_TOOL_OFFSET = (0.10, 0.0, 0.0)  # m in the wrist_yaw link frame; ASSUMED, to be measured on the palm
_LINK_PREFIX = "g1_29dof_with_hand_rev_1_0_"
_ANGLE_EPS = 1e-12  # below this |sin(angle)| the rotation vector uses its first-order form
_LAMBDA_MIN = 1e-9  # lower bound of the Levenberg-Marquardt factor
_MAX_TRIALS = 8  # rejected steps per iteration before the solver gives up


@lru_cache(maxsize=1)
def default_limits() -> Limits:
    """The shipped safety limits (config/g1_limits.yaml), parsed once."""
    return load_limits(DEFAULT_LIMITS)


@dataclass(frozen=True)
class JointSpec:
    """One revolute joint of the chain. `rot` and `pos` place the joint frame in the frame of the previous revolute joint
    (the fixed URDF joints in between are composed in), with the joint at zero. `axis` is a unit vector in the joint frame.
    `lower` and `upper` are the URDF limits in rad."""

    name: str
    rot: np.ndarray
    pos: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


@dataclass(frozen=True)
class ArmChain:
    """The serial chain pelvis -> waist (yaw, roll, pitch) -> torso -> one arm, in walk order. `tail_rot` and `tail_pos`
    place the wrist_yaw link frame in the frame of the last revolute joint."""

    side: str
    joints: tuple[JointSpec, ...]
    tail_rot: np.ndarray
    tail_pos: np.ndarray


@dataclass(frozen=True)
class _UrdfJoint:
    name: str
    kind: str
    parent: str
    child: str
    rot: np.ndarray
    pos: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


@dataclass(frozen=True)
class IKResult:
    """Inverse kinematics outcome. `q` is the best configuration found (inside the joint ranges); `pos_err_m` and
    `ori_err_rad` are the residuals at `q`; `iters` counts the solver iterations."""

    q: np.ndarray
    converged: bool
    pos_err_m: float
    ori_err_rad: float
    iters: int


class _Eval(NamedTuple):
    q: np.ndarray
    cost: float
    pos_err: float
    ori_err: float
    wj: np.ndarray  # weighted Jacobian, (6, n)
    we: np.ndarray  # weighted error, (6,)


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF rpy (extrinsic X-Y-Z roll, pitch, yaw) -> rotation matrix R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _skew(axes: np.ndarray) -> np.ndarray:
    """Vectors (..., 3) -> cross-product matrices (..., 3, 3), so that skew(a) @ v == a x v."""
    x, y, z = axes[..., 0], axes[..., 1], axes[..., 2]
    zero = np.zeros_like(x)
    return np.stack([np.stack([zero, -z, y], -1), np.stack([z, zero, -x], -1), np.stack([-y, x, zero], -1)], -2)


def _rodrigues(skew: np.ndarray, skew2: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rotations by `angles` (..., ) about unit axes given as cross-product matrices `skew` and `skew2 = skew @ skew`."""
    s = np.sin(angles)[..., None, None]
    c = np.cos(angles)[..., None, None]
    return np.eye(3) + s * skew + (1.0 - c) * skew2


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix by `angle` about a unit `axis` (Rodrigues)."""
    k = _skew(axis)
    return _rodrigues(k, k @ k, np.atleast_1d(float(angle)))[0]


def _matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z) with w >= 0 (Shepperd's method, branch on the largest term)."""
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0.0:
        s = 2.0 * math.sqrt(1.0 + tr)
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = ((r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s)
    out = np.array(q, dtype=np.float64)
    if out[0] < 0.0:
        out = -out
    return out / np.linalg.norm(out)


def _rotvec(r: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis times angle, angle in [0, pi]), via the quaternion."""
    q = _matrix_to_quat(r)
    w, v = q[0], q[1:]
    s = float(np.linalg.norm(v))
    if s < _ANGLE_EPS:
        return 2.0 * v
    return v * (2.0 * math.atan2(s, w) / s)


def _as_vec3(x, label: str) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ValueError(f"{label}: expected 3 finite numbers, got {x!r}")
    return v


@lru_cache(maxsize=4)
def _read_urdf(path: str) -> dict[str, _UrdfJoint]:
    """The URDF joints keyed by child link (each link has one parent joint). Parsed once per path."""
    joints: dict[str, _UrdfJoint] = {}
    for el in ET.parse(path).getroot().iter("joint"):
        origin = el.find("origin")
        xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        rpy = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
        axis_el = el.find("axis")
        axis = np.array([float(v) for v in (axis_el.get("xyz", "1 0 0") if axis_el is not None else "1 0 0").split()])
        axis = axis / np.linalg.norm(axis)
        limit = el.find("limit")
        lower = float(limit.get("lower", 0.0)) if limit is not None else 0.0
        upper = float(limit.get("upper", 0.0)) if limit is not None else 0.0
        child = el.find("child").get("link")
        joints[child] = _UrdfJoint(
            name=el.get("name"),
            kind=el.get("type", ""),
            parent=el.find("parent").get("link"),
            child=child,
            rot=_rpy_matrix(*(float(v) for v in rpy.split())),
            pos=np.array([float(v) for v in xyz.split()]),
            axis=axis,
            lower=lower,
            upper=upper,
        )
    return joints


@lru_cache(maxsize=8)
def load_arm_chain(urdf_path: str, side: str) -> ArmChain:
    """The chain from the URDF root (through the pelvis) to the wrist_yaw link of one arm, read from the URDF and checked
    against the joint names of g1_layout."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    by_child = _read_urdf(urdf_path)
    link = f"{_LINK_PREFIX}{side}_wrist_yaw_link"
    upward: list[_UrdfJoint] = []
    while link in by_child:
        upward.append(by_child[link])
        link = by_child[link].parent
    rot, pos = np.eye(3), np.zeros(3)
    specs: list[JointSpec] = []
    for joint in reversed(upward):
        rot, pos = rot @ joint.rot, pos + rot @ joint.pos
        if joint.kind == "revolute":
            specs.append(JointSpec(joint.name, rot, pos, joint.axis, joint.lower, joint.upper))
            rot, pos = np.eye(3), np.zeros(3)
        elif joint.kind != "fixed":
            raise ValueError(f"{urdf_path}: joint {joint.name!r} of type {joint.kind!r} is not supported in the arm chain")
    arm = LEFT_ARM_JOINTS if side == "left" else RIGHT_ARM_JOINTS
    if tuple(s.name for s in specs) != WAIST_JOINTS + arm:
        raise ValueError(f"{urdf_path}: the {side} chain joints {[s.name for s in specs]} do not match the expected layout")
    return ArmChain(side, tuple(specs), rot, pos)


class G1ArmKinematics:
    """Forward kinematics, Jacobian and inverse kinematics of one G1 arm, in the pelvis frame.

    `use_waist=False` fixes the waist at `waist` (yaw, roll, pitch in rad) and the solution vector has the 7 arm joints.
    `use_waist=True` solves for the waist too (10 joints, waist first). `joint_names`, `lower` and `upper` give the order
    and the ranges of the solution vector.
    """

    def __init__(self, side: str, urdf_path=DEFAULT_URDF, tool_offset=DEFAULT_TOOL_OFFSET, use_waist: bool = False,
                 margin_rad: float = 0.05, waist=(0.0, 0.0, 0.0)):
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        if not (math.isfinite(margin_rad) and margin_rad >= 0.0):
            raise ValueError(f"margin_rad must be a non-negative finite number, got {margin_rad!r}")
        chain = load_arm_chain(str(urdf_path), side)
        limits = default_limits()
        self.side = side
        self.tool_offset = _as_vec3(tool_offset, "tool_offset")
        waist_values = _as_vec3(waist, "waist")
        bounds: dict[str, tuple[float, float]] = {}
        for spec in chain.joints:
            safe_lo, safe_hi = limits.joint_range(spec.name)
            lo = max(spec.lower + margin_rad, safe_lo)
            hi = min(spec.upper - margin_rad, safe_hi)
            if lo >= hi:
                raise ValueError(f"joint {spec.name}: no range left after margin_rad={margin_rad}")
            bounds[spec.name] = (lo, hi)
        first = 0 if use_waist else len(WAIST_JOINTS)
        origin_rot, origin_pos, axes = [], [], []
        frame_rot, frame_pos = np.eye(3), np.zeros(3)
        for k, spec in enumerate(chain.joints):
            rot = frame_rot @ spec.rot
            pos = frame_pos + frame_rot @ spec.pos
            if k < first:  # a fixed waist joint: fold its rotation into the frames that follow
                lo, hi = bounds[spec.name]
                if not lo <= waist_values[k] <= hi:
                    raise ValueError(f"waist {spec.name}={waist_values[k]} is outside [{lo}, {hi}]")
                frame_rot = rot @ _axis_rotation(spec.axis, waist_values[k])
                frame_pos = pos
            else:
                origin_rot.append(rot)
                origin_pos.append(pos)
                axes.append(spec.axis)
                frame_rot, frame_pos = np.eye(3), np.zeros(3)
        self.joint_names = tuple(spec.name for spec in chain.joints[first:])
        self.lower = np.array([bounds[n][0] for n in self.joint_names])
        self.upper = np.array([bounds[n][1] for n in self.joint_names])
        self._origin_rot = np.stack(origin_rot)
        self._origin_pos = np.stack(origin_pos)
        self._axis = np.stack(axes)
        self._skew = _skew(self._axis)
        self._skew2 = self._skew @ self._skew
        self._tail_rot = frame_rot @ chain.tail_rot
        self._tail_pos = frame_pos + frame_rot @ chain.tail_pos

    def _as_q(self, q) -> np.ndarray:
        arr = np.asarray(q, dtype=np.float64)
        if arr.shape != (len(self.joint_names),):
            raise ValueError(f"q: expected {len(self.joint_names)} joint values, got shape {arr.shape}")
        return arr

    def _chain(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (R, p_tool, Z, P): the wrist_yaw link rotation and tool point in the pelvis frame, and per joint the
        world axis Z[i] and pivot P[i]."""
        rotations = _rodrigues(self._skew, self._skew2, q)
        n = len(q)
        z_axes = np.empty((n, 3))
        pivots = np.empty((n, 3))
        rot, pos = np.eye(3), np.zeros(3)
        for i in range(n):
            rot_origin = rot @ self._origin_rot[i]
            pos = pos + rot @ self._origin_pos[i]
            z_axes[i] = rot_origin @ self._axis[i]
            pivots[i] = pos
            rot = rot_origin @ rotations[i]
        rot_link = rot @ self._tail_rot
        pos_link = pos + rot @ self._tail_pos
        return rot_link, pos_link + rot_link @ self.tool_offset, z_axes, pivots

    @staticmethod
    def _jacobian_from(z_axes: np.ndarray, pivots: np.ndarray, p_tool: np.ndarray) -> np.ndarray:
        jac = np.empty((6, z_axes.shape[0]))
        jac[:3] = np.cross(z_axes, p_tool - pivots).T
        jac[3:] = z_axes.T
        return jac

    def fk(self, q) -> Pose:
        """Tool pose in the pelvis frame. Raises ValueError for a non-finite result."""
        rot, p_tool, _, _ = self._chain(self._as_q(q))
        return Pose(position=tuple(float(x) for x in p_tool), quat=tuple(float(x) for x in _matrix_to_quat(rot)))

    def jacobian(self, q) -> np.ndarray:
        """Geometric Jacobian (6, n) at the tool point: linear velocity rows, then angular velocity rows (world frame)."""
        _, p_tool, z_axes, pivots = self._chain(self._as_q(q))
        return self._jacobian_from(z_axes, pivots, p_tool)

    def ik(self, target: Pose, q_init, *, pos_weight: float = 1.0, ori_weight: float = 0.3, damping: float = 0.05,
           max_iters: int = 60, pos_tol: float = 1e-3, ori_tol: float = 2e-2, step_max: float = 0.3,
           posture=None, posture_weight: float = 0.02) -> IKResult:
        """Damped least-squares (Levenberg-Marquardt) solve of the tool pose in the pelvis frame, warm-started at `q_init`.

        The task cost is ||W e||^2 with e = [position error (m); rotation error (rad, world frame)] and
        W = diag(pos_weight, ori_weight). A step is accepted only if the task cost decreases; otherwise the damping grows.
        The posture (default `q_init`) is a null-space pull: each step adds `posture_weight` times the damped null-space
        projector applied to `posture - q`, so it never biases the converged pose. Each step is capped at `step_max` rad on
        its largest joint and clipped to the joint ranges.

        Convergence: position error below `pos_tol` and, when `ori_weight > 0`, orientation error below `ori_tol`.
        An unreachable target, or a non-finite target (which a `Pose` cannot hold, but a bypassed one could), never raises:
        the result is `converged=False` with the best configuration found (for a non-finite target, `q_init` clipped).
        Raises ValueError for a malformed `q_init` or a parameter outside its range.
        """
        q_start = self._as_q(q_init)
        if not np.all(np.isfinite(q_start)):
            raise ValueError("q_init: must be finite")
        q_start = np.clip(q_start, self.lower, self.upper)
        target_pos = np.asarray(target.position, dtype=np.float64)
        target_quat = np.asarray(target.quat, dtype=np.float64)
        if not (np.all(np.isfinite(target_pos)) and np.all(np.isfinite(target_quat))):
            return IKResult(q_start, False, math.nan, math.nan, 0)
        for label, value in (("pos_weight", pos_weight), ("ori_weight", ori_weight), ("posture_weight", posture_weight)):
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{label} must be a non-negative finite number, got {value!r}")
        for label, value in (("damping", damping), ("step_max", step_max), ("pos_tol", pos_tol), ("ori_tol", ori_tol)):
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{label} must be a positive finite number, got {value!r}")
        if max_iters < 0:
            raise ValueError(f"max_iters must be non-negative, got {max_iters}")
        q_post = q_start.copy() if posture is None else self._as_q(posture)
        target_rot = quat_to_matrix(target_quat)
        weights = np.array([pos_weight] * 3 + [ori_weight] * 3, dtype=np.float64)

        def evaluate(q: np.ndarray) -> _Eval:
            rot, p_tool, z_axes, pivots = self._chain(q)
            e_pos = target_pos - p_tool
            e_ori = _rotvec(target_rot @ rot.T)
            err = np.concatenate([e_pos, e_ori])
            jac = self._jacobian_from(z_axes, pivots, p_tool)
            cost = float(np.dot(weights * err, weights * err))
            return _Eval(q, cost, float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_ori)),
                         jac * weights[:, None], err * weights)

        def converged(e: _Eval) -> bool:
            return e.pos_err < pos_tol and (ori_weight == 0.0 or e.ori_err < ori_tol)

        cur = evaluate(np.clip(q_start, self.lower, self.upper))
        lam = damping * damping
        iters = 0
        while iters < max_iters and not converged(cur):
            iters += 1
            moved = False
            for _ in range(_MAX_TRIALS):
                gram = cur.wj.T @ cur.wj + lam * np.eye(len(self.joint_names))
                dq = np.linalg.solve(gram, cur.wj.T @ cur.we)
                # Damped null-space projector N = lam (J^T J + lam I)^-1; N (posture - q) pulls only inside the null space.
                dq += posture_weight * lam * np.linalg.solve(gram, q_post - cur.q)
                largest = float(np.max(np.abs(dq)))
                if largest > step_max:
                    dq *= step_max / largest
                trial = evaluate(np.clip(cur.q + dq, self.lower, self.upper))
                if trial.cost < cur.cost:
                    cur = trial
                    lam = max(lam / 3.0, _LAMBDA_MIN)
                    moved = True
                    break
                lam *= 4.0
            if not moved:
                break
        return IKResult(cur.q.copy(), converged(cur), cur.pos_err, cur.ori_err, iters)

    def reachable(self, target: Pose, q_init) -> bool:
        """True when `ik` converges to `target` from the warm start `q_init` (default solver settings)."""
        return self.ik(target, q_init).converged
