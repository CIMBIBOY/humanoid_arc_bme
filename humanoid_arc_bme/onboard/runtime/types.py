"""Runtime-only types. CameraFrame now lives in the contract (v2) and is re-exported here for old imports."""

from __future__ import annotations

from dataclasses import dataclass, field

from humanoid_arc_bme.onboard.api.contract import ENCODINGS, CameraFrame  # noqa: F401 - moved into contract v2 (2026-10-23)


@dataclass
class StateMeta:
    """What the adapter could NOT measure. A value of 'none' means the RobotState field is a placeholder, not a measurement."""

    base_vel_source: str = "none"  # none | estimator | sim_truth
    wrist_pose_source: str = "none"  # none | fk
    mode_machine: int | None = None
    extra: dict = field(default_factory=dict)
