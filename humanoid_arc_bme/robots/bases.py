"""Base plug-ins: what carries the upper body.

Manipulation tasks control the arms, hands and torso through Pink IK. What
the upper body stands on is a plug-in, chosen by name on the env config
(`robot_base`), so the same task, skills and datasets run on different bases:

- `agile_legs`: the G1's own legs, driven by NVIDIA's AGILE lower-body
  policy. Appends a 4-dim command `[vx, vy, wz, hip_height]` to the action;
  `idle_command` stands still. This is the layout of every recorded dataset
  and checkpoint so far (32 = 28 upper body + 4).
- `fixed`: pelvis fixed in space, legs held at their default pose, no base
  action (28 dims). Upper-body-only experiments and a base for other carriers.

The metadata (`command_dim`, `idle_command`) is pure Python; `apply()` edits
an IsaacLab env config and imports IsaacLab lazily.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Base:
    name: str
    command_dim: int
    # The base's slice of the action vector that holds it still (appended after the upper body).
    idle_command: tuple[float, ...]
    apply: Callable[[object], None]


def _apply_agile_legs(env_cfg) -> None:
    from .assets import AGILE_LOCOMOTION_POLICY

    env_cfg.actions.lower_body_joint_pos.policy_path = str(AGILE_LOCOMOTION_POLICY)


def _apply_fixed(env_cfg) -> None:
    # The upper-body IK solves in the pelvis frame, so it works unchanged on a fixed pelvis.
    env_cfg.actions.lower_body_joint_pos = None
    env_cfg.observations.lower_body_policy = None  # only the AGILE action term reads it
    robot = env_cfg.scene.robot
    props = robot.spawn.articulation_props.replace(fix_root_link=True)
    env_cfg.scene.robot = robot.replace(spawn=robot.spawn.replace(articulation_props=props))


AGILE_LEGS = Base("agile_legs", command_dim=4, idle_command=(0.0, 0.0, 0.0, 0.72), apply=_apply_agile_legs)
FIXED = Base("fixed", command_dim=0, idle_command=(), apply=_apply_fixed)

BASES: dict[str, Base] = {b.name: b for b in (AGILE_LEGS, FIXED)}


def get_base(name: str) -> Base:
    try:
        return BASES[name]
    except KeyError:
        raise ValueError(f"unknown robot_base {name!r}; choose from {sorted(BASES)}") from None
