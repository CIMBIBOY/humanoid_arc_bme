"""Domain-randomization ranges for the pick-place task (no Isaac Sim needed).

Pure Python. Every range is an unmeasured guess until the matching value is
checked on the real robot or scene; `measured` marks the ones that are. `sample`
draws parameter sets in three modes: "in" (inside the ranges), "edge" (the
corners) and "heldout" (a band just outside the ranges, to test extrapolation).
"""

import json
import random
from dataclasses import asdict, dataclass

HELDOUT_FRACTION = 0.25
MODES = ("in", "edge", "heldout")


@dataclass(frozen=True)
class Param:
    name: str
    low: float
    high: float
    measured: bool
    reason: str
    floor: float | None = None  # physical lower limit; held-out draws never go below it

    def __post_init__(self):
        if not self.low < self.high:
            raise ValueError(f"{self.name}: low {self.low} must be below high {self.high}")
        if not self.reason.strip():
            raise ValueError(f"{self.name}: reason must not be empty")


PARAMS: tuple[Param, ...] = (
    Param("object_dx", -0.10, 0.10, False,
          "Metres offset of the object from its nominal start along x, as the grasp spot varies between resets."),
    Param("object_dy", -0.10, 0.10, False,
          "Metres offset of the object from its nominal start along y, as the placement on the table varies."),
    Param("object_dz", 0.0, 0.02, False,
          "Metres of height above the table surface, standing in for the object resting on a slight lift or a thick pad."),
    Param("object_yaw", -3.1416, 3.1416, False,
          "Radians of yaw about the vertical axis, covering any orientation the object can land in."),
    Param("object_roll_pitch", -0.15, 0.15, False,
          "Radians of roll and pitch tilt, standing in for the object not lying flat on its base."),
    Param("object_scale", 0.8, 1.2, False,
          "Dimensionless uniform size factor, covering the spread of real object dimensions between instances."),
    Param("object_mass", 0.05, 0.60, False,
          "Kilograms of object mass, covering light packaging through a full-size object.", floor=0.0),
    Param("object_friction", 0.4, 1.2, False,
          "Dimensionless static friction coefficient of the object surface, varying with material and wear."),
    Param("table_dz", -0.05, 0.05, False,
          "Metres of table height error, covering levelling and mounting tolerance of the table."),
    Param("table_friction", 0.4, 1.2, False,
          "Dimensionless friction coefficient of the table surface, varying with cloth, paint and dust."),
    Param("goal_dx", -0.05, 0.05, False,
          "Metres offset of the place target along x, as the marked spot is placed by hand."),
    Param("goal_dy", -0.05, 0.05, False,
          "Metres offset of the place target along y, as the marked spot is placed by hand."),
    Param("camera_dx", -0.03, 0.03, False,
          "Metres of camera mounting offset along x, covering small shifts of the head or camera bracket."),
    Param("camera_dy", -0.03, 0.03, False,
          "Metres of camera mounting offset along y, covering small shifts of the head or camera bracket."),
    Param("camera_dz", -0.03, 0.03, False,
          "Metres of camera mounting offset along z, covering small shifts of the head or camera bracket."),
    Param("camera_yaw", -0.1, 0.1, False,
          "Radians of camera yaw error from mounting, a small rotation of the view."),
    Param("light_intensity_scale", 0.5, 1.5, False,
          "Dimensionless multiplier on scene light intensity, covering room lighting changes over a day."),
    Param("depth_noise_std", 0.0, 0.01, False,
          "Metres of standard deviation of per-pixel depth noise, covering sensor noise at working distance.", floor=0.0),
    Param("depth_dropout", 0.0, 0.1, False,
          "Fraction of depth pixels lost to missing returns, as on shiny or dark surfaces.", floor=0.0),
    Param("camera_delay_s", 0.0, 0.15, False,
          "Seconds of latency between the rendered image and the state used by the policy, from camera pipelines.", floor=0.0),
    Param("joint_friction_scale", 0.5, 2.0, False,
          "Dimensionless multiplier on joint friction, covering gearbox and wear differences between joints."),
    Param("wrist_payload_kg", 0.0, 0.3, False,
          "Kilograms of extra mass at the wrist, standing in for a tool, cable or gripper load not in the model.", floor=0.0),
    Param("action_delay_steps", 0.0, 4.0, False,
          "Control steps (float, rounded by the consumer) of delay between the commanded and the applied action, from bus latency.", floor=0.0),
    Param("hand_gain_scale", 0.7, 1.3, False,
          "Dimensionless multiplier on hand position-controller gains, covering tuning and motor variation."),
)

_BY_NAME = {p.name: p for p in PARAMS}


def sample(seed: int, n: int, mode: str) -> list[dict[str, float]]:
    """`n` parameter sets drawn with random.Random(seed) in the given mode."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if n < 1:
        raise ValueError("n must be at least 1")
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        row = {}
        for p in PARAMS:
            width = p.high - p.low
            if mode == "in":
                x = p.low + width * rng.random()
                row[p.name] = min(p.high, max(p.low, x))
            elif mode == "edge":
                row[p.name] = rng.choice((p.low, p.high))
            else:
                # 1 - random() is in (0, 1], so the draw never lands on the range edge.
                u = 1.0 - rng.random()
                band = HELDOUT_FRACTION * width * u
                below = p.low - band
                use_high = rng.random() < 0.5 or (p.floor is not None and below < p.floor)
                row[p.name] = p.high + band if use_high else below
        out.append(row)
    return out


def inside(sample: dict) -> bool:
    """True iff every value is within its [low, high] range (inclusive).

    Raises KeyError if a parameter is missing or unknown.
    """
    unknown = set(sample) - set(_BY_NAME)
    if unknown:
        raise KeyError(f"unknown parameters: {sorted(unknown)}")
    return all(_BY_NAME[p.name].low <= sample[p.name] <= p.high for p in PARAMS)


def to_json(path: str) -> None:
    with open(path, "w") as f:
        json.dump([asdict(p) for p in PARAMS], f, indent=2)


def from_json(path: str) -> tuple[Param, ...]:
    with open(path) as f:
        return tuple(Param(**d) for d in json.load(f))


def describe() -> str:
    lines = []
    for p in PARAMS:
        status = "measured" if p.measured else "GUESS (unmeasured)"
        lines.append(f"{p.name} [{p.low}, {p.high}] {status}: {p.reason}")
    return "\n".join(lines)
