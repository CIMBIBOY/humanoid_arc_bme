"""The robot-facing contract: commands out, state in, skill status.

Plain frozen dataclasses with validation, identical for the simulator and the real robot
(company/physical_api/README.md). Imports only the standard library.

Conventions, stated once:
- Lengths in metres, angles in radians, velocities in m/s and rad/s, time in seconds.
- Quaternions are (w, x, y, z), the order IsaacLab 2.3.2 uses, and must be unit length.
- A wrist pose is (x, y, z, qw, qx, qy, qz) in the named frame.
- Messages round-trip through plain dicts (`to_dict` / `from_dict`) and so through JSON. `from_dict` rejects
  unknown keys, wrong lengths and missing keys with ValueError; the only keys that may be missing are fields added in
  schema v2 that have a default (so a v1 dict still loads, and means what it meant in v1).

Schema v2 (audit P0-4, 2026-10-23) adds: `CameraFrame`, `TaskSpec`, `JointTargets`, `BaseVelocityHeight`,
`WholeBodyCommand`, `ActionChunk`; `HandCommand.joint_targets` (7 Dex3 joints, or the v1 scalar `closure`, never both);
`RobotState.base_vel_source` (what `base_vel` is: `sim_truth`, `estimator` or `none` = placeholder) and
`RobotState.foot_contacts`. v1 classes and their positional signatures are unchanged.
"""

from __future__ import annotations

import base64
import math
from dataclasses import MISSING, dataclass, fields

SCHEMA_VERSION = 2

SIDES = ("left", "right")
SKILL_STATES = ("running", "succeeded", "failed")
BASE_VEL_SOURCES = ("sim_truth", "estimator", "none")
DEX3_JOINTS = 7
ENCODINGS = {"rgb8": 3, "bgr8": 3, "mono8": 1, "depth16": 2}
CHUNK_REPRESENTATIONS = ("abs_pelvis", "rel_eef", "joint_pos")
QUAT_NORM_TOL = 1e-3


def _vec(name: str, values, n: int | None = None) -> tuple[float, ...]:
    try:
        out = tuple(float(v) for v in values)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name}: expected a sequence of numbers ({e})") from None
    if n is not None and len(out) != n:
        raise ValueError(f"{name}: expected {n} values, got {len(out)}")
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name}: values must be finite")
    return out


def _scalar(name: str, value) -> float:
    return _vec(name, [value], 1)[0]


def _side(value) -> str:
    if value not in SIDES:
        raise ValueError(f"side: expected one of {SIDES}, got {value!r}")
    return value


def _quat(name: str, values) -> tuple[float, float, float, float]:
    q = _vec(name, values, 4)
    if abs(math.sqrt(sum(c * c for c in q)) - 1.0) > QUAT_NORM_TOL:
        raise ValueError(f"{name}: quaternion (w, x, y, z) must have unit length")
    return q


def _set(obj, **kwargs) -> None:
    for k, v in kwargs.items():
        object.__setattr__(obj, k, v)


class _Message:
    """Dict round trip with strict key checking."""

    def to_dict(self) -> dict:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, _Message):
                v = v.to_dict()
            elif isinstance(v, dict):
                v = {k: list(x) if isinstance(x, tuple) else x for k, x in v.items()}
            elif isinstance(v, tuple):
                v = list(v)
            out[f.name] = v
        return out

    @classmethod
    def from_dict(cls, data: dict):
        if not isinstance(data, dict):
            raise ValueError(f"{cls.__name__}: expected a dict, got {type(data).__name__}")
        names = {f.name for f in fields(cls)}
        optional = {f.name for f in fields(cls) if f.default is not MISSING or f.default_factory is not MISSING}
        missing, extra = names - data.keys() - optional, data.keys() - names
        if missing or extra:
            raise ValueError(f"{cls.__name__}: missing keys {sorted(missing)}, unknown keys {sorted(extra)}")
        return cls(**data)


@dataclass(frozen=True)
class ArmCommand(_Message):
    """Target wrist pose for one arm. `wrist_pose` = (x, y, z, qw, qx, qy, qz) in `frame`."""

    side: str
    wrist_pose: tuple
    frame: str

    def __post_init__(self):
        pose = _vec("wrist_pose", self.wrist_pose, 7)
        _quat("wrist_pose quaternion", pose[3:])
        if not isinstance(self.frame, str) or not self.frame:
            raise ValueError("frame: expected a non-empty string")
        _set(self, side=_side(self.side), wrist_pose=pose)


@dataclass(frozen=True)
class HandCommand(_Message):
    """Hand command: v1 `closure` (0 = fully open, 1 = fully closed, dimensionless) OR v2 `joint_targets` (the 7 Dex3
    joint positions in rad, in the robot's joint order for that hand). Exactly one of the two."""

    side: str
    closure: float | None = None
    joint_targets: tuple | None = None

    def __post_init__(self):
        if (self.closure is None) == (self.joint_targets is None):
            raise ValueError("HandCommand: give exactly one of closure or joint_targets")
        c, j = None, None
        if self.closure is not None:
            c = _scalar("closure", self.closure)
            if not 0.0 <= c <= 1.0:
                raise ValueError(f"closure: must be in [0, 1], got {c}")
        else:
            j = _vec("joint_targets", self.joint_targets, DEX3_JOINTS)
        _set(self, side=_side(self.side), closure=c, joint_targets=j)


@dataclass(frozen=True)
class BaseVelocity(_Message):
    """Planar base velocity in the base frame: forward vx, left vy (m/s), yaw rate wz (rad/s)."""

    vx: float
    vy: float
    wz: float

    def __post_init__(self):
        _set(self, vx=_scalar("vx", self.vx), vy=_scalar("vy", self.vy), wz=_scalar("wz", self.wz))


@dataclass(frozen=True)
class SkillStatus(_Message):
    """State of a skill. A failed skill must say why (`reason` is a short machine-readable string)."""

    state: str
    reason: str = ""

    def __post_init__(self):
        if self.state not in SKILL_STATES:
            raise ValueError(f"state: expected one of {SKILL_STATES}, got {self.state!r}")
        if not isinstance(self.reason, str):
            raise ValueError("reason: expected a string")
        if self.state == "failed" and not self.reason:
            raise ValueError("a failed skill needs a reason")


@dataclass(frozen=True)
class Imu(_Message):
    """Orientation quaternion (w, x, y, z), angular rate (rad/s), linear acceleration (m/s^2), body frame."""

    quat: tuple
    gyro: tuple
    accel: tuple

    def __post_init__(self):
        _set(self, quat=_quat("quat", self.quat), gyro=_vec("gyro", self.gyro, 3), accel=_vec("accel", self.accel, 3))


@dataclass(frozen=True)
class RobotState(_Message):
    """One time-stamped snapshot. `stamp` in seconds on the common clock; joint arrays follow `joint_names`.

    `wrist_poses` maps side -> (x, y, z, qw, qx, qy, qz) in the pelvis frame.
    v2: `base_vel_source` says what `base_vel` is (`none` = a placeholder; the real G1 measures no base velocity);
    `foot_contacts` = (left, right) contact flags when an adapter knows them, else None.
    """

    stamp: float
    joint_names: tuple
    joint_pos: tuple
    joint_vel: tuple
    base_vel: tuple
    imu: Imu
    wrist_poses: dict
    base_vel_source: str = "none"
    foot_contacts: tuple | None = None

    def __post_init__(self):
        if self.base_vel_source not in BASE_VEL_SOURCES:
            raise ValueError(f"base_vel_source: expected one of {BASE_VEL_SOURCES}, got {self.base_vel_source!r}")
        if self.foot_contacts is not None:
            fc = tuple(self.foot_contacts)
            if len(fc) != 2 or not all(isinstance(x, bool) for x in fc):
                raise ValueError("foot_contacts: expected (left, right) booleans or None")
            _set(self, foot_contacts=fc)
        names = tuple(self.joint_names)
        if not all(isinstance(n, str) and n for n in names) or len(set(names)) != len(names):
            raise ValueError("joint_names: expected unique non-empty strings")
        n = len(names)
        imu = Imu.from_dict(self.imu) if isinstance(self.imu, dict) else self.imu
        if not isinstance(imu, Imu):
            raise ValueError("imu: expected an Imu or its dict")
        if not isinstance(self.wrist_poses, dict):
            raise ValueError("wrist_poses: expected a dict side -> pose")
        poses = {}
        for side, pose in self.wrist_poses.items():
            p = _vec(f"wrist_poses[{side}]", pose, 7)
            _quat(f"wrist_poses[{side}] quaternion", p[3:])
            poses[_side(side)] = p
        _set(
            self,
            stamp=_scalar("stamp", self.stamp),
            joint_names=names,
            joint_pos=_vec("joint_pos", self.joint_pos, n),
            joint_vel=_vec("joint_vel", self.joint_vel, n),
            base_vel=_vec("base_vel", self.base_vel, 3),
            imu=imu,
            wrist_poses=poses,
        )


# ---------------------------------------------------------------- schema v2 messages


def _str(name: str, value, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{name}: expected a {'' if allow_empty else 'non-empty '}string")
    return value


@dataclass(frozen=True)
class CameraFrame(_Message):
    """One image. `data` is row-major, no padding, width * height * channels(encoding) bytes (depth16: little-endian
    uint16 millimetres). `K` = 3x3 intrinsics row-major (9 values) and `T_pelvis_camera` = (x, y, z, qw, qx, qy, qz) of
    the camera in the pelvis frame, both None when not calibrated (a provisional sim pose is still a pose: flag it in
    the producer). In dicts `data` is base64 text."""

    name: str
    stamp: float
    width: int
    height: int
    encoding: str
    data: bytes
    K: tuple | None = None
    T_pelvis_camera: tuple | None = None

    def __post_init__(self):
        _str("name", self.name)
        if self.encoding not in ENCODINGS:
            raise ValueError(f"encoding: expected one of {tuple(ENCODINGS)}, got {self.encoding!r}")
        if not (isinstance(self.width, int) and isinstance(self.height, int)) or self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive integers")
        data = self.data
        if isinstance(data, str):
            data = base64.b64decode(data.encode("ascii"))
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError("data: expected bytes")
        want = self.width * self.height * ENCODINGS[self.encoding]
        if len(data) != want:
            raise ValueError(f"data: expected {want} bytes for {self.width}x{self.height} {self.encoding}, got {len(data)}")
        k = None if self.K is None else _vec("K", self.K, 9)
        t = None
        if self.T_pelvis_camera is not None:
            t = _vec("T_pelvis_camera", self.T_pelvis_camera, 7)
            _quat("T_pelvis_camera quaternion", t[3:])
        _set(self, stamp=_scalar("stamp", self.stamp), data=bytes(data), K=k, T_pelvis_camera=t)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["data"] = base64.b64encode(self.data).decode("ascii")
        return d


@dataclass(frozen=True)
class TaskSpec(_Message):
    """What to do: a language `instruction`, a stable `task_id` (for metrics and conditioning), and the goal either as a
    named anchor (`goal_name`, e.g. "plate_left") and/or a pose (x, y, z, qw, qx, qy, qz) in `frame`."""

    instruction: str
    task_id: str
    goal_name: str = ""
    goal_pose: tuple | None = None
    frame: str = "pelvis"

    def __post_init__(self):
        _str("instruction", self.instruction)
        _str("task_id", self.task_id)
        _str("goal_name", self.goal_name, allow_empty=True)
        _str("frame", self.frame)
        p = None
        if self.goal_pose is not None:
            p = _vec("goal_pose", self.goal_pose, 7)
            _quat("goal_pose quaternion", p[3:])
        if not self.goal_name and p is None:
            raise ValueError("TaskSpec: give a goal_name, a goal_pose, or both")
        _set(self, goal_pose=p)


@dataclass(frozen=True)
class JointTargets(_Message):
    """Joint position targets (rad) by name: the output of a whole-body / leg controller (L1 -> L0)."""

    names: tuple
    positions: tuple

    def __post_init__(self):
        names = tuple(self.names)
        if not names or not all(isinstance(n, str) and n for n in names) or len(set(names)) != len(names):
            raise ValueError("names: expected unique non-empty strings")
        _set(self, names=names, positions=_vec("positions", self.positions, len(names)))


@dataclass(frozen=True)
class BaseVelocityHeight(_Message):
    """The AGILE / Decoupled-WBC / W4 lower-body command: vx, vy (m/s), wz (rad/s) in the base frame, pelvis height h (m)."""

    vx: float
    vy: float
    wz: float
    height: float

    def __post_init__(self):
        v = _vec("velocity_height", (self.vx, self.vy, self.wz, self.height), 4)
        if not 0.2 <= v[3] <= 1.2:
            raise ValueError(f"height: {v[3]} m is not a pelvis height")
        _set(self, vx=v[0], vy=v[1], wz=v[2], height=v[3])


@dataclass(frozen=True)
class WholeBodyCommand(_Message):
    """L2 -> L1. Exactly one of: `velocity_height` (legs walk / crouch, arms elsewhere), `wrists` (ArmCommands for one or
    both arms; legs keep balance), `joints` (direct joint targets). Motion tokens (SONIC) are not in v2: added when a
    SONIC adapter exists."""

    stamp: float
    velocity_height: BaseVelocityHeight | None = None
    wrists: tuple | None = None
    joints: JointTargets | None = None

    def __post_init__(self):
        vh = self.velocity_height
        if isinstance(vh, dict):
            vh = BaseVelocityHeight.from_dict(vh)
        j = self.joints
        if isinstance(j, dict):
            j = JointTargets.from_dict(j)
        w = None
        if self.wrists is not None:
            w = tuple(ArmCommand.from_dict(a) if isinstance(a, dict) else a for a in self.wrists)
            if not w or not all(isinstance(a, ArmCommand) for a in w) or len({a.side for a in w}) != len(w):
                raise ValueError("wrists: expected one ArmCommand per side")
        if sum(x is not None for x in (vh, w, j)) != 1:
            raise ValueError("WholeBodyCommand: exactly one of velocity_height, wrists, joints")
        _set(self, stamp=_scalar("stamp", self.stamp), velocity_height=vh, wrists=w, joints=j)

    def to_dict(self) -> dict:
        d = super().to_dict()
        if self.wrists is not None:
            d["wrists"] = [a.to_dict() for a in self.wrists]
        return d


@dataclass(frozen=True)
class ActionChunk(_Message):
    """K future actions from a chunking policy or VLA (L3/L2). Row k applies at `stamp + k * dt`; the consumer executes
    the first `execute_horizon` rows and then expects a new chunk. `names` labels the A action dimensions (e.g.
    "left_wrist_x", "right_hand_index_0_joint"); `representation` says what the numbers are."""

    stamp: float
    dt: float
    representation: str
    names: tuple
    actions: tuple
    execute_horizon: int

    def __post_init__(self):
        if self.representation not in CHUNK_REPRESENTATIONS:
            raise ValueError(f"representation: expected one of {CHUNK_REPRESENTATIONS}")
        names = tuple(self.names)
        if not names or len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
            raise ValueError("names: expected unique non-empty strings")
        rows = tuple(_vec(f"actions[{k}]", r, len(names)) for k, r in enumerate(self.actions))
        if not rows:
            raise ValueError("actions: a chunk needs at least one row")
        dt = _scalar("dt", self.dt)
        if dt <= 0:
            raise ValueError("dt must be positive")
        if not isinstance(self.execute_horizon, int) or not 1 <= self.execute_horizon <= len(rows):
            raise ValueError("execute_horizon: expected an integer in 1..K")
        _set(self, stamp=_scalar("stamp", self.stamp), dt=dt, names=names, actions=rows)

    @property
    def K(self) -> int:
        return len(self.actions)
