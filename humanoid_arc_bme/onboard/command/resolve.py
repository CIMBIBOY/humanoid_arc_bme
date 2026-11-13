"""Resolve a Pick against the objects currently visible. Pure: the scene is passed in, perception is not imported.

A scene item needs the attributes of `VisibleObject`. `color` is a colour word from the grammar, None when the
object's colour was not given, or "unknown". An unknown colour never matches a colour the operator asked for, but it
does match a colour-less request when it is the only candidate. Only a single match resolves; anything else is a
Reject that tells the operator what to change.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .types import Pick, Reject

UNKNOWN_COLOR = "unknown"


class VisibleObject(Protocol):
    """What the resolver reads from one visible object. `range_m` and `confidence` come from perception."""

    label: str
    color: str | None
    range_m: float  # metres from the camera
    confidence: float  # perception score in [0, 1]


def _describe(label: str, color: str | None) -> str:
    return label if color in (None, UNKNOWN_COLOR) else f"{color} {label}"


def _plural(noun: str) -> str:
    return noun + ("es" if noun.endswith(("s", "x", "ch", "sh")) else "s")


def summarize(scene: Sequence[VisibleObject]) -> str:
    """Short list of what is visible, e.g. "blue cup, cup" (an unknown colour is shown without one), or "nothing"."""
    names = [_describe(obj.label, obj.color) for obj in scene]
    return ", ".join(names) if names else "nothing"


def resolve_pick(cmd: Pick, scene: Sequence[VisibleObject]) -> VisibleObject | Reject:
    """The one visible object that `cmd` names, or a Reject explaining why there is not exactly one."""
    items = list(scene)
    candidates = [obj for obj in items if obj.label == cmd.label]
    matches = candidates if cmd.color is None else [obj for obj in candidates if obj.color == cmd.color]
    if not matches:
        return Reject(f"I do not see any {_describe(cmd.label, cmd.color)}. I see: {summarize(items)}")
    if len(matches) == 1:
        return matches[0]
    colors = {obj.color for obj in matches}
    if len(colors) == 1:
        (color,) = colors
        shown = _describe(cmd.label, None if color == UNKNOWN_COLOR else color)
        return Reject(f"{len(matches)} {_plural(shown)} are visible; remove one so only one is in view")
    return Reject(f"{len(matches)} {_plural(cmd.label)} are visible, say which colour")
