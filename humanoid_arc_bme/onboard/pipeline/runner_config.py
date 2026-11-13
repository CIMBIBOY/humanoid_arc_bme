"""Value types and configuration of the PipelineRunner: the tick input/output, the safety status and the validated config.

Split out of runner.py so the runner holds only the behaviour. Everything here is re-exported from `runner`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from humanoid_arc_bme.onboard.pipeline.interfaces import DepthFrame, GraspCandidate, LocalizerEstimate
from humanoid_arc_bme.onboard.pipeline.perception.lidar import LidarFrame
from humanoid_arc_bme.onboard.validation import is_num as _is_num

OK, SLOW, STOP = "ok", "slow", "stop"
STAGES = ("pose", "depth", "costmap", "guard", "grasp", "total")
ZERO = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SafetyStatus:
    level: str                      # "ok" | "slow" | "stop"
    reasons: tuple[str, ...] = ()

    @property
    def stop(self) -> bool:
        return self.level == STOP


@dataclass(frozen=True)
class TickInput:
    stamp: float
    forward_depth: DepthFrame | None
    down_depth: DepthFrame | None
    pose: LocalizerEstimate | None
    roll: float
    pitch: float
    rgb: np.ndarray | None = None
    command: tuple[float, float, float] = ZERO   # (vx, vy, wz)
    lidar: LidarFrame | None = None              # body-frame Livox scan; adds near-field detection only


@dataclass(frozen=True)
class TickOutput:
    command_out: tuple[float, float, float]
    safety: SafetyStatus
    clearance: object | None        # ground.ClearanceResult or None
    grasp_candidates: list[GraspCandidate] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    lidar: object | None = None     # lidar.LidarClearance of this tick (None = no usable lidar); .near_observed says if the near field was seen


@dataclass(frozen=True)
class RunnerConfig:
    hold_ticks: int = 3             # ticks a stop persists after its cause clears
    a_stop: float = 0.3             # placeholder deceleration (m/s^2), see nav/obstacle.py
    d_margin: float = 0.3
    corridor_half_width: float = 0.35
    max_range: float = 3.0          # how far ahead the guard camera is trusted; a floor the camera cannot see beyond this reads as a drop edge
    approach_clearance_m: float = 0.12   # approach mode: stop when the RAW nearest obstacle (no robot-radius inflation) is this close (body front ~0.10 m ahead of the pelvis origin; UNMEASURED)
    approach_max_speed: float = 0.06     # approach mode: speed cap (m/s)
    approach_ground_frac: float = 0.12   # approach mode: floor share of the lower band needed to fit the ground (default 0.3). A table
                                         # fills the view near it; the fit's height prior (floor at body-origin height) still refuses a tabletop
    approach_plane_age_s: float = 300.0  # approach mode: reuse the last good floor plane this long when the fit fails (a table fills the view)...
    approach_plane_travel_m: float = 1.0 # ...and only while the robot is within this distance of where it was fitted
    guard_camera: str = "forward"   # "forward" (neck camera) or "down" (down-tilted camera only: set max_range to its real floor look-ahead
                                    # and require_lidar=True, so the Livox covers what the camera cannot see)
    points_max_range: float = 6.0
    points_stride: int = 2
    min_valid_fraction: float = 0.5
    max_events: int = 500
    timing_window: int = 2000
    grasp_max_range: float = 1.5
    memory_ttl_s: float = 30.0      # how long an obstacle that left the camera's view stays remembered
    memory_lateral_margin: float = 0.2
    max_stamp_jump_s: float = 1.0   # larger forward jump of the tick stamp = clock fault -> latched bad_stamp
    max_depth_age_s: float = 0.3    # |tick stamp - depth frame stamp| limit
    # Absolute output envelope; defaults equal config/g1_limits.yaml base_vel_max (a test pins the equality). Not
    # imported from api/safety.py because that module has no default Limits and the yaml needs pyyaml.
    max_vx: float = 0.2
    max_vy: float = 0.1
    max_wz: float = 0.3
    # Turn-in-place exception (see module docstring). PLACEHOLDERS until the footprint is measured.
    turn_clearance_m: float = 0.45  # no inflated-occupied cell within this radius of the body centre
    turn_wz_max: float | None = None  # None -> max_wz / 2
    # Lidar fusion (see module docstring). Lidar only ADDS detection; the camera requirements are unchanged.
    require_lidar: bool = False     # True: a missing/stale/invalid lidar, or an unobserved near field, is a hard stop
    max_lidar_age_s: float = 0.3    # |tick stamp - lidar frame stamp| limit
    lidar_min_near_ground_points: int = 10   # floor returns in corridor x in (0.3, 1.2) needed to call the near field observed
    lidar_min_near_coverage: float = 0.8     # and the fraction of 0.1 m x-bins of that range holding a floor return
    lidar_self_radius_m: float = 0.25  # PLACEHOLDER body exclusion cylinder radius for lidar returns (0 disables), heights 0.05..1.5 m
    lidar_h_min: float = 0.10       # obstacle threshold above the ground for lidar returns (m); 0.10 = lidar noise margin

    @property
    def turn_wz(self) -> float:
        return self.max_wz / 2.0 if self.turn_wz_max is None else float(self.turn_wz_max)


def _validate_cfg(c) -> None:
    """Every knob is checked here (the guard is built from them): NaN a_stop would disable the speed cap, NaN
    hold_ticks the sticky stop. Raises ValueError naming the knob."""
    if not isinstance(c, RunnerConfig):
        raise ValueError("cfg: expected a RunnerConfig")
    pos = ("a_stop", "corridor_half_width", "max_range", "points_max_range", "grasp_max_range", "memory_ttl_s",
           "max_stamp_jump_s", "max_depth_age_s", "max_lidar_age_s", "lidar_h_min", "max_vx", "max_vy", "max_wz", "turn_clearance_m")
    for n in pos:
        v = getattr(c, n)
        if not _is_num(v) or v <= 0.0:
            raise ValueError(f"{n}: must be a finite number > 0, got {v!r}")
    if not _is_num(c.approach_ground_frac) or not 0.0 < c.approach_ground_frac <= 1.0:
        raise ValueError(f"approach_ground_frac: must be in (0, 1], got {c.approach_ground_frac!r}")
    for n in ("approach_clearance_m", "approach_max_speed", "approach_plane_age_s", "approach_plane_travel_m"):
        v = getattr(c, n)
        if not _is_num(v) or v <= 0.0:
            raise ValueError(f"{n}: must be a finite number > 0, got {v!r}")
    for n in ("d_margin", "memory_lateral_margin", "lidar_self_radius_m"):
        v = getattr(c, n)
        if not _is_num(v) or v < 0.0:
            raise ValueError(f"{n}: must be a finite number >= 0, got {v!r}")
    if c.turn_wz_max is not None and (not _is_num(c.turn_wz_max) or not 0.0 < c.turn_wz_max <= c.max_wz):
        raise ValueError(f"turn_wz_max: must be None or a finite number in (0, max_wz], got {c.turn_wz_max!r}")
    if not _is_num(c.min_valid_fraction) or not 0.0 <= c.min_valid_fraction <= 1.0:
        raise ValueError(f"min_valid_fraction: must be in [0, 1], got {c.min_valid_fraction!r}")
    if c.guard_camera not in ("forward", "down"):
        raise ValueError(f"guard_camera: must be 'forward' or 'down', got {c.guard_camera!r}")
    if type(c.require_lidar) is not bool:
        raise ValueError(f"require_lidar: must be a bool, got {c.require_lidar!r}")
    if not _is_num(c.lidar_min_near_coverage) or not 0.0 < c.lidar_min_near_coverage <= 1.0:
        raise ValueError(f"lidar_min_near_coverage: must be in (0, 1], got {c.lidar_min_near_coverage!r}")
    for n, lo in (("lidar_min_near_ground_points", 1), ("hold_ticks", 0), ("max_events", 1), ("timing_window", 1), ("points_stride", 1)):
        v = getattr(c, n)
        if type(v) is not int or v < lo:   # rejects bool, float, NaN, numpy ints
            raise ValueError(f"{n}: must be an int >= {lo}, got {v!r}")
