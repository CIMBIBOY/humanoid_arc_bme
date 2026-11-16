import json

import pytest

from humanoid_arc_bme.onboard.command import INTENTS, CompositeParser, LocalGrammarParser, Pick, Reject, Stop
from humanoid_arc_bme.onboard.command.jev import (FakeJev, JevClassifier, JevConfig, JevUnavailable, ReplayJev, build_request,
                                                  parse_response, record)

OPTS = (*INTENTS, "none")
KEY = "sk-secret-123"
HARD = "could you fetch me that red mug when you can"  # the grammar rejects it on its own


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", KEY)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make(reply=None, exc=None, clock=None, **cfg):
    seen = []

    def transport(url, body, headers, timeout_s):
        seen.append((url, body, headers, timeout_s))
        if exc:
            raise exc
        return reply

    jev = JevClassifier(JevConfig("http://x.invalid", **cfg), transport, clock or Clock())
    return jev, seen


def test_request_response_round_trip():
    body = json.loads(build_request("fetch it", OPTS, {"visible": ["cup"]}))
    assert body == {"input": "fetch it", "choices": list(OPTS), "context": {"visible": ["cup"]}}
    reply = json.dumps({"choice": "pick", "probabilities": {"pick": 0.9, "none": 0.1}}).encode()
    jev, seen = make(reply)
    assert jev.classify("fetch it", OPTS, {}) == ("pick", 0.9) and seen[0][3] == 0.8
    assert parse_response(b'{"choice": "none", "probability": 1}', OPTS) == ("none", 1.0)
    assert jev.calls == 1 and jev.failures == 0 and jev.last_latency_s == 0.0 and jev.state == "closed"


@pytest.mark.parametrize("raw", [b"nope", b"[]", b"\xff", b'{"choice": 3, "probability": 0.5}', b'{"choice": "dance", "probability": 0.5}',
                                 b'{"choice": "pick"}', b'{"choice": "pick", "probability": "0.9"}', b'{"choice": "pick", "probability": true}',
                                 b'{"choice": "pick", "probability": NaN}', b'{"choice": "pick", "probability": 1.5}',
                                 b'{"choice": "pick", "probability": -0.1}', b'{"choice": "pick", "probabilities": {"none": 1}}',
                                 b'{"choice": "pick", "probabilities": [1]}'])
def test_malformed_responses_raise_value_error(raw):
    with pytest.raises(ValueError):
        parse_response(raw, OPTS)


def test_the_key_goes_only_into_the_header_never_into_the_body_or_an_error():
    jev, seen = make(exc=ConnectionError(f"boom {KEY}"))
    with pytest.raises(JevUnavailable) as info:
        jev.classify("x", OPTS, {})
    assert KEY not in str(info.value) and KEY not in repr(seen[0][1]) and KEY in seen[0][2]["Authorization"]
    with pytest.raises(JevUnavailable) as info:
        make(b"{}")[0].classify("x", OPTS, {})
    assert KEY not in str(info.value)


def test_missing_key_makes_no_network_call(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY")
    jev, seen = make(b"{}")
    with pytest.raises(JevUnavailable):
        jev.classify("x", OPTS, {})
    assert not seen


@pytest.mark.parametrize("kw", [dict(exc=TimeoutError("slow")), dict(exc=OSError("down")), dict(reply=b"garbage"),
                                dict(reply=b'{"choice": "pick", "probability": 0.2}')])
def test_every_failure_ends_in_the_grammar_reject(kw):
    jev, _ = make(**kw)
    p = CompositeParser(classifier=jev)
    assert p.parse(HARD) == LocalGrammarParser().parse(HARD) and isinstance(p.parse(HARD), Reject)


def test_a_sentence_the_grammar_understands_never_calls_the_classifier():
    jev, seen = make(exc=TimeoutError())
    assert CompositeParser(classifier=jev).parse("go forward 2 meters") and not seen and jev.calls == 0


@pytest.mark.parametrize("text", ["stop", "please stop right now", "stop and fetch the cup"])
def test_stop_is_the_grammars_and_never_waits_for_the_classifier(text):
    jev, seen = make(exc=TimeoutError())
    assert isinstance(CompositeParser(classifier=jev).parse(text), Stop) and not seen


def test_circuit_breaker_opens_stays_silent_and_closes_after_the_cooldown():
    clock = Clock()
    jev, seen = make(exc=TimeoutError(), clock=clock)
    for _ in range(3):
        with pytest.raises(JevUnavailable):
            jev.classify("x", OPTS, {})
    assert jev.state == "open" and len(seen) == 3 and jev.failures == 3
    with pytest.raises(JevUnavailable):
        jev.classify("x", OPTS, {})
    assert len(seen) == 3
    clock.t += 31.0
    assert jev.state == "closed"
    with pytest.raises(JevUnavailable):
        jev.classify("x", OPTS, {})  # one trial call, which fails and reopens
    assert len(seen) == 4 and jev.state == "open"
    clock.t += 31.0
    jev._transport = lambda *a: b'{"choice": "none", "probability": 0.9}'
    assert jev.classify("x", OPTS, {}) == ("none", 0.9) and jev.state == "closed"


def test_fake_jev_modes_and_clock():
    clock = Clock()
    fake = FakeJev(clock_advance=lambda s: setattr(clock, "t", clock.t + s))
    assert fake.classify("Fetch it", OPTS, {}) == ("pick", 0.9) and fake.classify("halt", OPTS, {})[0] == "stop"
    assert fake.classify("hello", OPTS, {})[0] == "none" and abs(clock.t - 100.6) < 1e-9 and len(fake.calls) == 3
    with pytest.raises(TimeoutError):
        FakeJev(mode="timeout").classify("x", OPTS, {})
    with pytest.raises(ConnectionError):
        FakeJev(mode="error").classify("x", OPTS, {})
    assert FakeJev(mode="malformed").classify("x", OPTS, {}) == "pick"
    assert FakeJev(mode="low_prob").classify("fetch", OPTS, {}) == ("pick", 0.3)
    out = FakeJev(mode="outage_after:1")
    out.classify("fetch", OPTS, {})
    with pytest.raises(ConnectionError):
        out.classify("fetch", OPTS, {})


@pytest.mark.parametrize("mode", ["malformed", "low_prob", "timeout", "error"])
def test_fake_failure_modes_fall_back_to_the_grammar(mode):
    assert isinstance(CompositeParser(classifier=FakeJev(mode=mode)).parse(HARD), Reject)


def test_go_to_the_red_cup_with_the_grammar_alone_and_with_fake_jev():
    text = "go to the red cup"
    alone = CompositeParser().parse(text)
    fake = FakeJev()
    with_jev = CompositeParser(classifier=fake).parse(text)
    print("grammar alone:", alone, "| with FakeJev:", with_jev, "| classifier asked:", bool(fake.calls))
    if isinstance(alone, Reject):
        assert with_jev == Pick("cup", "red")
    else:
        assert with_jev == alone and not fake.calls


def test_replay_round_trip(tmp_path):
    path = str(tmp_path / "rec.json")
    record(path, FakeJev(), ["fetch the red mug", "hello"], OPTS)
    replay = ReplayJev(path)
    assert replay.classify("fetch the red mug", OPTS, {}) == ("pick", 0.9)
    with pytest.raises(KeyError):
        replay.classify("unrecorded", OPTS, {})
    assert isinstance(CompositeParser(classifier=replay).parse("unrecorded sentence"), Reject)
    assert CompositeParser(classifier=replay).parse("fetch the red mug") == Pick("cup", "red")
