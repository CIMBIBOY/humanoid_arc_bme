"""Resolving a Pick against the visible scene: unique match, no match, duplicates, unknown colours."""

from dataclasses import dataclass

import pytest

from humanoid_arc_bme.onboard.command import Pick, Reject, resolve_pick, summarize


@dataclass(frozen=True)
class Seen:
    label: str
    color: str | None
    range_m: float = 1.0
    confidence: float = 0.9


RED_CUP = Seen("cup", "red")
BLUE_CUP = Seen("cup", "blue")
RED_BOTTLE = Seen("bottle", "red")
UNKNOWN_CUP = Seen("cup", "unknown")


def test_unique_colour_match_resolves_to_that_object():
    scene = [BLUE_CUP, RED_BOTTLE, RED_CUP]
    assert resolve_pick(Pick("cup", "red"), scene) is RED_CUP


def test_colourless_request_resolves_when_one_candidate():
    assert resolve_pick(Pick("bottle"), [BLUE_CUP, RED_BOTTLE]) is RED_BOTTLE


def test_unknown_colour_matches_colourless_request_when_only_candidate():
    assert resolve_pick(Pick("cup"), [UNKNOWN_CUP, RED_BOTTLE]) is UNKNOWN_CUP


def test_unknown_colour_never_matches_a_colour_request():
    reject = resolve_pick(Pick("cup", "red"), [UNKNOWN_CUP])
    assert isinstance(reject, Reject)
    assert reject.reason == "I do not see any red cup. I see: cup"


def test_colourless_request_with_unknown_and_known_candidates_is_ambiguous():
    reject = resolve_pick(Pick("cup"), [UNKNOWN_CUP, RED_CUP])
    assert isinstance(reject, Reject)
    assert reject.reason == "2 cups are visible, say which colour"


def test_no_match_says_what_is_visible():
    reject = resolve_pick(Pick("cup", "red"), [BLUE_CUP, RED_BOTTLE])
    assert isinstance(reject, Reject)
    assert reject.reason == "I do not see any red cup. I see: blue cup, red bottle"


def test_no_match_on_empty_scene():
    reject = resolve_pick(Pick("bowl"), [])
    assert reject == Reject("I do not see any bowl. I see: nothing")


def test_two_same_label_same_colour_is_refused():
    reject = resolve_pick(Pick("cup", "red"), [RED_CUP, Seen("cup", "red", range_m=0.6)])
    assert reject == Reject("2 red cups are visible; remove one so only one is in view")


def test_colourless_request_with_two_same_colour_candidates_is_refused():
    reject = resolve_pick(Pick("cup"), [RED_CUP, Seen("cup", "red")])
    assert reject == Reject("2 red cups are visible; remove one so only one is in view")


def test_two_different_colours_ask_which_colour():
    reject = resolve_pick(Pick("cup"), [RED_CUP, BLUE_CUP])
    assert reject == Reject("2 cups are visible, say which colour")


def test_plural_of_words_ending_in_x_gets_es():
    boxes = [Seen("box", "red"), Seen("box", "red")]
    assert resolve_pick(Pick("box", "red"), boxes) == Reject(
        "2 red boxes are visible; remove one so only one is in view")


def test_resolution_returns_the_very_object_passed_in():
    scene = [RED_CUP]
    assert resolve_pick(Pick("cup", "red"), scene) is RED_CUP


def test_scene_may_be_any_sequence_not_only_a_list():
    assert resolve_pick(Pick("cup", "red"), (RED_CUP, RED_BOTTLE)) is RED_CUP


@pytest.mark.parametrize("scene, text", [
    ([], "nothing"),
    ([RED_CUP], "red cup"),
    ([RED_CUP, RED_BOTTLE], "red cup, red bottle"),
    ([UNKNOWN_CUP], "cup"),
    ([Seen("can", None)], "can"),
])
def test_summarize(scene, text):
    assert summarize(scene) == text
