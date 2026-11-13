"""Skill API (plan A4.5): the request side of pick and place.

`SkillApi.pick` and `SkillApi.place` validate a request and, if it is valid, store it in `self.pending`
for a later executor. They never move anything and never change a state machine: validation happens
before anything else, and a rejected request leaves `pending` as it was.

`status_from_machine` maps a running `PickPlaceMachine` (skills/state_pick_place.py) to a `SkillStatus`.

Stdlib only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..api.contract import SkillStatus
from ..api.safety import Limits
from .state_pick_place import PickPlaceMachine, State


@dataclass(frozen=True)
class Request:
    action: str  # "pick" or "place"
    target: object  # object name for pick; (x, y, z) pelvis-frame metres for place


class SkillApi:
    def __init__(self, limits: Limits, known_objects: set[str]):
        self.limits = limits
        self.known_objects = set(known_objects)
        self.pending: Request | None = None

    def pick(self, name: str) -> SkillStatus:
        if not isinstance(name, str) or name not in self.known_objects:
            return SkillStatus("failed", "unknown_object")
        self.pending = Request("pick", name)
        return SkillStatus("running")

    def place(self, location) -> SkillStatus:
        point = self._in_workspace(location)
        if point is None:
            return SkillStatus("failed", "out_of_workspace")
        self.pending = Request("place", point)
        return SkillStatus("running")

    def _in_workspace(self, location) -> tuple[float, float, float] | None:
        try:
            values = tuple(location)
        except TypeError:
            return None
        if len(values) != 3:
            return None
        try:
            xyz = tuple(float(v) for v in values)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in xyz):
            return None
        for v, lo, hi in zip(xyz, self.limits.ws_min, self.limits.ws_max):
            if not lo <= v <= hi:
                return None
        return xyz


def status_from_machine(machine: PickPlaceMachine) -> SkillStatus:
    if machine.state is State.DONE:
        return SkillStatus("succeeded")
    if machine.state is State.FAILED:
        return SkillStatus("failed", machine.reason)
    return SkillStatus("running")
