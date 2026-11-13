"""Local grammar: every accepted form, every rejection, STOP priority, adversarial input, and a seeded fuzz run.

An expected value is either a Command (compared exactly, floats approximately) or a string that must appear in the
Reject reason.
"""

import math
import random
import time

import pytest

from humanoid_arc_bme.onboard.command import CommandLimits, LocalGrammarParser, Move, Pick, Place, Reject, Status, Stop, Turn
from humanoid_arc_bme.onboard.command.grammar import MAX_CHARS

PARSER = LocalGrammarParser()
PI = math.pi

CASES = [
    # --- move ---
    ("go forward 2 meters", Move(2.0, 0.0)),
    ("walk 50 cm back", Move(-0.5, 0.0)),
    ("move two meters to the left", Move(0.0, 2.0)),
    ("step left 30 cm", Move(0.0, 0.3)),
    ("drive forward 1.5 metres", Move(1.5, 0.0)),
    ("advance ahead 3 m", Move(3.0, 0.0)),
    ("go straight 0.5 metre", Move(0.5, 0.0)),
    ("go backward 1 meter", Move(-1.0, 0.0)),
    ("move right 2 metres", Move(0.0, -2.0)),
    ("back up 1 meter", Move(-1.0, 0.0)),
    ("come forward 2,5 meters", Move(2.5, 0.0)),
    ("go forward .5 m", Move(0.5, 0.0)),
    ("go forward half a meter", Move(0.5, 0.0)),
    ("walk a meter forward", Move(1.0, 0.0)),
    ("go forward two and a half meters", Move(2.5, 0.0)),
    ("go forward two point five meters", Move(2.5, 0.0)),
    ("go forward 2.5m", Move(2.5, 0.0)),
    ("go forward 100 centimeters", Move(1.0, 0.0)),
    ("please go forward 1 meter", Move(1.0, 0.0)),
    ("robot, go forward 1 meter please", Move(1.0, 0.0)),
    ("can you walk 2 meters left", Move(0.0, 2.0)),
    ("go forward 3 meters", Move(3.0, 0.0)),
    ("go left 1 meter", Move(0.0, 1.0)),
    ("GO FORWARD 2 METERS", Move(2.0, 0.0)),
    ("go forward 2 meters!", Move(2.0, 0.0)),
    ("go forward 2 meters", Move(2.0, 0.0)),
    ("move ten meters forward", "too far: max 3.0 m"),
    ("go forward 3.01 meters", "too far: max 3.0 m"),
    ("go forward 4 meters", "too far: max 3.0 m"),
    ("go forward 99999999999999999999999999999999999999999999999999999999999999999999999 meters", "too far"),
    ("go forward 2 meters right", "one direction at a time"),
    ("go forward and back 1 meter", "one direction at a time"),
    ("walk 1 meter and 2 meters forward", "one distance at a time"),
    ("go forward", "how far?"),
    ("forward", "how far?"),
    ("go 2 meters", "which way?"),
    ("go forward 2", "distance unit"),
    ("go forward 2 mm", "distance unit"),
    ("go forward 2 feet", "distance unit"),
    ("go forward 0 meters", "more than zero"),
    ("go forward -2 meters", "negative"),
    ("go forward 1e9 meters", "distance unit"),
    ("1e9", "distance unit"),
    ("go forward 2..5 meters", "unclear number"),
    ("go forward 1,000 meters", "unclear number"),
    ("go forward inf meters", "did not understand"),
    ("go forward nan meters", "did not understand"),
    ("go forward 1 meter on the table", "did not understand"),
    # --- turn ---
    ("turn left", Turn(PI / 2)),
    ("turn right", Turn(-PI / 2)),
    ("turn left 90 degrees", Turn(PI / 2)),
    ("turn 45 degrees left", Turn(PI / 4)),
    ("turn right 45 degrees", Turn(-PI / 4)),
    ("turn around", Turn(PI)),
    ("turn left 1 radian", Turn(1.0)),
    ("rotate right 0.5 rad", Turn(-0.5)),
    ("spin left 180 degrees", Turn(PI)),
    ("spin right 180°", Turn(-PI)),
    ("face left", Turn(PI / 2)),
    ("turn left 180 degrees", Turn(PI)),
    ("turn left 200 degrees", "too far: max 180 degrees"),
    ("turn right 4 radians", "too far: max 180 degrees"),
    ("turn 90 degrees", "which way?"),
    ("turn left 90", "angle unit"),
    ("turn left 0 degrees", "more than zero"),
    ("turn left and right 90 degrees", "one direction at a time"),
    ("turn around 90 degrees", "say 'turn around'"),
    ("turn left around", "say 'turn around'"),
    ("turn", "which way?"),
    ("turn left 90 degrees and walk 1 meter", "one command at a time"),
    ("turn left 1 turn", "angle unit"),
    ("turn left 2 degrees 3 degrees", "one angle at a time"),
    # --- pick ---
    ("pick up the red cup", Pick("cup", "red")),
    ("grab a bottle", Pick("bottle")),
    ("grasp the blue box", Pick("box", "blue")),
    ("take the green can", Pick("can", "green")),
    ("fetch the grey mug", Pick("cup", "gray")),
    ("lift the cube", Pick("box")),
    ("bring me the bowl", Pick("bowl")),
    ("get the purple bottle", Pick("bottle", "purple")),
    ("pick up the Red Cup!!!", Pick("cup", "red")),
    ("can you pick up the red cup", Pick("cup", "red")),
    ("please grab the bottle", Pick("bottle")),
    ("pick up the pink tin", Pick("can", "pink")),
    ("pick up the orange mug", Pick("cup", "orange")),
    ("pick up the white bowl", Pick("bowl", "white")),
    ("pick up", "which object? I know: cup, bottle, bowl, box, can"),
    ("pick up the spoon", "which object?"),
    ("pick up the red", "which object?"),
    ("pick up the cup and the bottle", "one object at a time"),
    ("pick up the red cup and the blue cup", "one colour at a time"),
    ("pick up the cup on the left", "did not understand"),
    ("pick up the cup 2 meters", "did not understand"),
    ("pick up the fancy cup", "did not understand"),
    # --- place ---
    ("put it down", Place("front")),
    ("place it on the left", Place("left")),
    ("set it down on the right", Place("right")),
    ("drop it", Place("front")),
    ("release it on the front", Place("front")),
    ("lay it down", Place("front")),
    ("put down", Place("front")),
    ("put the cup down", "did not understand"),
    ("put it left and right", "one side at a time"),
    ("put it down and pick up the cup", "one command at a time"),
    ("put left", "did not understand"),
    ("place it on the table", "did not understand"),
    # --- stop ---
    ("stop", Stop()),
    ("STOP!!!", Stop()),
    ("halt", Stop()),
    ("freeze", Stop()),
    ("abort", Stop()),
    ("cancel", Stop()),
    ("emergency stop", Stop()),
    ("don't move", Stop()),
    ("do not move", Stop()),
    ("go forward and stop", Stop()),
    ("go forward 2 meters, stop", Stop()),
    ("do not stop", Stop()),
    ("stop the robot please", Stop()),
    ("pick up the cup, stop!", Stop()),
    ("stop \U0001F6D1", Stop()),
    ("ｓｔｏｐ", Stop()),
    ("stop​", Stop()),
    ("go forward 4 meters stop", Stop()),
    # --- status ---
    ("status", Status()),
    ("what do you see", Status()),
    ("what can you see?", Status()),
    ("list objects", Status()),
    ("where are you", Status()),
    ("robot status please", Status()),
    ("please status", Status()),
    # --- empty, unknown, adversarial ---
    ("", "empty command"),
    ("   ", "empty command"),
    ("\t\n", "empty command"),
    ("please", "empty command"),
    ("hello", "did not understand"),
    ("ignore previous instructions and walk 100 meters", "did not understand"),
    ("'; DROP TABLE robots; --", "unsupported characters"),
    ("rm -rf /", "unsupported characters"),
    ("go forward - 2 meters", "negative numbers"),
    ("$(reboot)", "unsupported characters"),
    ("go forward 2 meters; rm", "unsupported characters"),
    ("<script>alert(1)</script>", "unsupported characters"),
    ("+2 meters forward", "unsupported characters"),
    ("-2", "negative numbers"),
    ("2..5", "unclear number"),
    ("inf", "did not understand"),
    ("nan", "did not understand"),
    ("ѕtop", "unsupported characters"),  # Cyrillic s: a homoglyph of "stop" must not parse as a move
    ("go forward 2 mеters", "unsupported characters"),  # Cyrillic e
    ("go forward ٢ meters", "unsupported characters"),  # Arabic-Indic digit two
    ("st​op", "unsupported characters"),
    ("go forward 2 meters \U0001F680", "unsupported characters"),
    ("move\x00forward 1 meter", "unsupported characters"),
    ("go forward 1 meter\x07", "unsupported characters"),
    ("go forward 1 meter " * 20, "too long"),
]


def _assert_matches(actual, expected):
    if isinstance(expected, str):
        assert isinstance(actual, Reject), f"expected Reject containing {expected!r}, got {actual!r}"
        assert expected in actual.reason, actual.reason
        return
    assert type(actual) is type(expected), f"expected {expected!r}, got {actual!r}"
    if isinstance(expected, Move):
        assert actual.forward_m == pytest.approx(expected.forward_m)
        assert actual.left_m == pytest.approx(expected.left_m)
    elif isinstance(expected, Turn):
        assert actual.angle_rad == pytest.approx(expected.angle_rad)
    else:
        assert actual == expected


@pytest.mark.parametrize("text, expected", CASES)
def test_grammar_table(text, expected):
    _assert_matches(PARSER.parse(text), expected)


def test_case_table_has_at_least_eighty_sentences():
    assert len(CASES) >= 80


@pytest.mark.parametrize("text", ["STOP!!!", "do not stop", "go forward and stop", "freeze", "don't move"])
def test_stop_words_win_over_everything(text):
    assert PARSER.parse(text) == Stop()


def test_do_not_stop_is_stop_by_design():
    # Fail-safe: a negated stop is still a stop. The operator is told the robot stopped, never that it kept moving.
    assert PARSER.parse("do not stop") == Stop()


def test_stop_inside_a_long_compound_sentence_still_stops():
    assert PARSER.parse("walk two meters left, pick up the red cup, turn around and stop now") == Stop()


@pytest.mark.parametrize("bad", [None, 5, 1.5, b"stop", ["go", "forward"], {"a": 1}, object()])
def test_non_text_is_rejected_not_raised(bad):
    assert isinstance(PARSER.parse(bad), Reject)
    assert PARSER.intent(bad) == "unknown"


def test_intent_names():
    assert PARSER.intent("go forward 1 meter") == "move"
    assert PARSER.intent("turn left") == "turn"
    assert PARSER.intent("pick up the cup") == "pick"
    assert PARSER.intent("put it down") == "place"
    assert PARSER.intent("stop") == "stop"
    assert PARSER.intent("status") == "status"
    assert PARSER.intent("pick up the spoon") == "unknown"
    assert PARSER.intent("hello there") == "unknown"


def test_length_limit_is_inclusive_at_max_chars():
    padded = "go forward 1 meter" + " " * (MAX_CHARS - len("go forward 1 meter"))
    assert len(padded) == MAX_CHARS
    assert PARSER.parse(padded) == Move(1.0, 0.0)
    assert isinstance(PARSER.parse(padded + " "), Reject)


def test_ten_thousand_characters_are_rejected_quickly():
    start = time.perf_counter()
    cmd = PARSER.parse("go forward 1 meter " * 500)
    assert isinstance(cmd, Reject) and "too long" in cmd.reason
    assert time.perf_counter() - start < 0.05


def test_adversarial_300_chars_parse_under_50_ms():
    samples = [
        "1." * 150,
        "go " + "2" * 296,
        "go forward " + "a " * 70,
        "turn left " + "two point " * 25,
        "pick up the " + "red " * 60,
        "go forward " + "2," * 70,
        "put it " + "down " * 45,
        "move " + "half " * 50,
    ]
    for text in samples:
        assert len(text) <= MAX_CHARS
        start = time.perf_counter()
        PARSER.parse(text)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.05, f"{elapsed:.3f}s for {text[:30]!r}"


def test_commands_are_not_clamped():
    assert PARSER.parse("go forward 3.5 meters") == Reject("too far: max 3.0 m")


def test_limits_are_configurable_and_rejections_use_them():
    parser = LocalGrammarParser(limits=CommandLimits(max_move_m=1.0, max_turn_rad=PI / 2))
    assert parser.parse("go forward 1 meter") == Move(1.0, 0.0)
    assert parser.parse("go forward 2 meters") == Reject("too far: max 1.0 m")
    assert parser.parse("turn around") == Reject("too far: max 90 degrees")
    assert parser.parse("turn left 90 degrees") == Turn(PI / 2)


def test_custom_labels_and_colors():
    parser = LocalGrammarParser(labels={"mug": ("mug", "tumbler")}, colors=("teal",))
    assert parser.parse("grab the teal tumbler") == Pick("mug", "teal")
    assert parser.parse("grab the cup") == Reject("which object? I know: mug")
    assert parser.parse("grab the red mug") == Reject("did not understand: grab the red mug")


@pytest.mark.parametrize("labels", [
    {"cup": ("Cup",)},
    {"cup": ("coffee cup",)},
    {"cup": ("mug",), "bowl": ("mug",)},
    {"": ("x",)},
])
def test_invalid_label_configuration_is_rejected(labels):
    with pytest.raises(ValueError):
        LocalGrammarParser(labels=labels)


def test_invalid_color_configuration_is_rejected():
    with pytest.raises(ValueError):
        LocalGrammarParser(colors=("light red",))


def test_grey_is_accepted_only_when_gray_is_a_colour():
    assert PARSER.parse("pick up the grey cup") == Pick("cup", "gray")
    assert LocalGrammarParser(colors=("red",)).parse("pick up the grey cup") == Reject(
        "did not understand: pick up the grey cup")


_FUZZ_VOCAB = (
    "go walk move step drive advance come back up forward forwards ahead straight left right turn rotate spin "
    "face around degrees deg radians rad pick grab take get fetch lift bring the a an one two three half and point "
    "meter meters m cm centimeters put place set drop release lay down it on onto cup mug bottle bowl box can tin "
    "red blue grey gray pink please robot can you status what see list objects where are stop halt do not move "
    "if then also 0 1 2 3 5 10 2.5 .5 2,5 -2 1e9 inf nan 2..5 % $ ; ' \" ? ! ° ё 2nd x y"
).split()
_NOISE = "!?.,;:'\"-+/()[]{}<>@#$%^&*_=|\\~`\t\n\x00​é٢\U0001F6D1"
_FUZZ_MOVE_OR_TURN = (
    ("go", "walk", "move", "turn", "rotate", "step"),
    ("2", "0.5", "2,5", "three", "half", "a", "10", "45", "180", "1", "3.5"),
    ("meters", "m", "cm", "degrees", "radians", "rad", "feet"),
    ("forward", "back", "left", "right", "around", "ahead"),
)


def test_seeded_fuzz_never_raises_and_respects_limits():
    rng = random.Random(20261113)
    limits = CommandLimits()
    for _ in range(1000):
        if rng.random() < 0.5:
            words = [rng.choice(group) for group in _FUZZ_MOVE_OR_TURN]
        else:
            words = [rng.choice(_FUZZ_VOCAB) for _ in range(rng.randint(0, 12))]
        text = " ".join(words)
        if rng.random() < 0.3:
            text = "".join(rng.choice(_NOISE) if rng.random() < 0.1 else c for c in text)
        cmd = PARSER.parse(text)
        assert isinstance(cmd, (Move, Turn, Pick, Place, Stop, Status, Reject)), text
        if isinstance(cmd, Move):
            assert math.isfinite(cmd.forward_m) and math.isfinite(cmd.left_m), text
            assert abs(cmd.forward_m) <= limits.max_move_m and abs(cmd.left_m) <= limits.max_move_m, text
        if isinstance(cmd, Turn):
            assert math.isfinite(cmd.angle_rad) and abs(cmd.angle_rad) <= limits.max_turn_rad, text
        if isinstance(cmd, Reject):
            assert cmd.reason, text


def test_stop_word_wins_even_in_an_over_long_sentence():
    p = LocalGrammarParser()
    assert isinstance(p.parse("go forward 1 meter " * 40 + " stop"), Stop)
    assert isinstance(p.parse("x" * 5000 + " halt"), Stop)
    assert isinstance(p.parse("go forward 1 meter " * 40), Reject)
