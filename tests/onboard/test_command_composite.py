import pytest

from humanoid_arc_bme.onboard.command import CompositeParser, LocalGrammarParser, Move, Pick, Reject, Stop, Turn


class Fake:
    def __init__(self, intent="pick", prob=0.95, exc=None):
        self.intent, self.prob, self.exc, self.calls = intent, prob, exc, []

    def classify(self, text, options, context):
        self.calls.append((text, tuple(options), context))
        if self.exc:
            raise self.exc
        return self.intent, self.prob


def test_the_grammar_wins_when_it_understands_and_the_classifier_is_never_asked():
    c = Fake()
    p = CompositeParser(classifier=c)
    assert p.parse("go forward 2 meters") == Move(2.0, 0.0) and not c.calls and p.last_source == "local"


def test_stop_comes_from_the_grammar_first_and_from_the_classifier_at_a_lower_bar():
    c = Fake(intent="stop", prob=0.6)
    assert isinstance(CompositeParser(classifier=c).parse("halt right now"), Stop) and not c.calls
    assert isinstance(CompositeParser(classifier=c).parse("please make it cease immediately"), Stop)


def test_classifier_supplies_the_intent_but_the_grammar_supplies_the_arguments():
    c = Fake(intent="pick", prob=0.95)
    p = CompositeParser(classifier=c)
    cmd = p.parse("could you fetch me that red mug over there", visible=["cup"])
    assert cmd == Pick("cup", "red") and p.last_source == "classifier"
    assert c.calls[0][2] == {"visible": ["cup"]} and "none" in c.calls[0][1]


def test_missing_arguments_are_a_reject_never_a_guess():
    out = CompositeParser(classifier=Fake(intent="move", prob=0.99)).parse("get me going somewhere nice")
    assert isinstance(out, Reject) and "understood as 'move'" in out.reason


@pytest.mark.parametrize("intent,prob", [("pick", 0.5), ("none", 0.99), ("dance", 0.99), ("pick", float("nan")), ("pick", 1.5)])
def test_low_confidence_unknown_or_malformed_answers_fall_back_to_the_grammar_rejection(intent, prob):
    out = CompositeParser(classifier=Fake(intent=intent, prob=prob)).parse("fetch the red mug when you can")
    assert isinstance(out, Reject)


@pytest.mark.parametrize("exc", [TimeoutError("slow"), ConnectionError("offline"), ValueError("bad json")])
def test_a_failing_classifier_never_breaks_parsing(exc):
    p = CompositeParser(classifier=Fake(exc=exc))
    assert isinstance(p.parse("fetch the red mug when you can"), Reject) and "classifier" in p.last_error


def test_classifier_can_never_inject_numbers():
    """"go a long way": the grammar has no distance, so even a 0.99 'move' stays a reject."""
    assert isinstance(CompositeParser(classifier=Fake(intent="move", prob=0.99)).parse("go a long long way"), Reject)


def test_parse_as_extracts_only_what_the_sentence_says():
    g = LocalGrammarParser()
    assert g.parse_as("turn", "swing round to the left 45 degrees") in (Turn(0.7853981633974483), Reject("did not understand: swing round to the left 45 degrees"))
    assert isinstance(g.parse_as("dance", "x"), Reject) and isinstance(g.parse_as("stop", "anything"), Stop)


def test_negation_is_never_dropped_as_filler():
    g = LocalGrammarParser()
    for t in ("don't pick the red cup", "do not take the red cup", "never grab the red cup", "pick up anything but not the red cup"):
        assert isinstance(g.parse_as("pick", t), Reject), t
