"""Command types: validation on construction, frozen values, intent names, limits."""

import dataclasses
import math

import pytest

from humanoid_arc_bme.onboard.command import (
    INTENTS, PLACE_SIDES, CommandLimits, Move, Pick, Place, Reject, Status, Stop, Turn, intent_of,
)


@pytest.mark.parametrize("make", [
    lambda: Move(0.0, 0.0),
    lambda: Move(math.nan, 0.0),
    lambda: Move(0.0, math.inf),
    lambda: Move("1", 0.0),
    lambda: Move(True, 0.0),
    lambda: Move(10**400, 0.0),
    lambda: Turn(0.0),
    lambda: Turn(math.nan),
    lambda: Turn(None),
    lambda: Pick(""),
    lambda: Pick("   "),
    lambda: Pick("cup", ""),
    lambda: Place("up"),
    lambda: Place(None),
    lambda: Reject(""),
    lambda: Reject("  "),
    lambda: CommandLimits(max_move_m=0.0),
    lambda: CommandLimits(max_turn_rad=-1.0),
    lambda: CommandLimits(max_move_m=math.inf),
])
def test_invalid_values_are_rejected_at_construction(make):
    with pytest.raises(ValueError):
        make()


def test_numbers_are_stored_as_floats():
    assert Move(2, 0).forward_m == 2.0 and isinstance(Move(2, 0).forward_m, float)
    assert Turn(1).angle_rad == 1.0 and isinstance(Turn(1).angle_rad, float)


def test_move_allows_backwards_and_leftwards_values():
    assert Move(-1.5, 0.0).forward_m == -1.5
    assert Move(0.0, -0.25).left_m == -0.25


def test_defaults():
    assert Pick("cup").color is None
    assert Place().side == "front"
    assert PLACE_SIDES == ("front", "left", "right")


def test_commands_are_frozen():
    cmd = Move(1.0, 0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cmd.forward_m = 2.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        Pick("cup").label = "bowl"


def test_equal_values_compare_equal():
    assert Turn(math.pi / 2) == Turn(math.pi / 2)
    assert Stop() == Stop()
    assert Status() == Status()


@pytest.mark.parametrize("cmd, intent", [
    (Move(1.0, 0.0), "move"), (Turn(1.0), "turn"), (Pick("cup"), "pick"), (Place(), "place"),
    (Stop(), "stop"), (Status(), "status"), (Reject("no"), "reject"),
])
def test_intent_of(cmd, intent):
    assert intent_of(cmd) == intent


def test_intent_of_rejects_non_commands():
    with pytest.raises(TypeError):
        intent_of("move")


def test_intents_are_the_six_actions():
    assert INTENTS == ("move", "turn", "pick", "place", "stop", "status")


def test_limits_default_to_three_metres_and_half_a_turn():
    limits = CommandLimits()
    assert limits.max_move_m == 3.0
    assert limits.max_turn_rad == pytest.approx(math.pi)
