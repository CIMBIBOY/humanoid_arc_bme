"""Jev adapter: an optional cloud intent classifier for `CompositeParser` (semantic tier only).

PLACEHOLDER SCHEMA: the real Jev API specification and key have not been supplied. Every constant tagged PLACEHOLDER below
(URL, auth header, request and response field names, the probability format) is a guess to be replaced. To confirm against
the real API: URL, auth header and scheme, field names, whether probabilities are returned at all, rate limits.

Safety rule: Jev is never in the stop path and the robot never depends on it. The grammar sees every sentence first
(STOP included); Jev only gets sentences the grammar rejected, and every failure here is an exception that
`CompositeParser` turns into the grammar's own rejection. The API key is read from the environment at call time and is
never logged or put into an exception message. Stdlib only.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

PLACEHOLDER_AUTH_HEADER = "Authorization"  # PLACEHOLDER
PLACEHOLDER_AUTH_SCHEME = "Bearer "  # PLACEHOLDER
PLACEHOLDER_REQUEST_KEYS = ("input", "choices", "context")  # PLACEHOLDER
PLACEHOLDER_RESPONSE_KEYS = ("choice", "probabilities", "probability")  # PLACEHOLDER


class JevUnavailable(RuntimeError):
    """No usable answer: missing key, open circuit breaker or a failed call. Never carries the key."""


@dataclass(frozen=True)
class JevConfig:
    url: str  # PLACEHOLDER: real endpoint unknown
    api_key_env: str = "JEV_API_KEY"
    timeout_s: float = 0.8
    max_failures: int = 3
    cooldown_s: float = 30.0
    model: str | None = None


def build_request(text: str, options: Sequence[str], context: Mapping) -> bytes:
    return json.dumps({"input": text, "choices": list(options), "context": dict(context)}).encode("utf-8")  # PLACEHOLDER


def parse_response(raw: bytes, options: Sequence[str]) -> tuple[str, float]:
    """(choice, probability) from a placeholder-schema reply; ValueError on anything malformed."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError, UnicodeDecodeError):
        raise ValueError("reply is not JSON") from None
    if not isinstance(data, dict) or not isinstance(data.get("choice"), str):
        raise ValueError("reply has no string 'choice'")
    choice = data["choice"]
    if choice not in options:
        raise ValueError("unknown choice")
    probs = data.get("probabilities")
    prob = probs.get(choice) if isinstance(probs, dict) else data.get("probability")
    if isinstance(prob, bool) or not isinstance(prob, (int, float)) or not math.isfinite(prob) or not 0.0 <= prob <= 1.0:
        raise ValueError("missing or invalid probability")
    return choice, float(prob)


def _urllib_transport(url: str, body: bytes, headers: dict, timeout_s: float) -> bytes:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 - the URL comes from the operator's config
        return resp.read(1_000_000)


class JevClassifier:
    """`IntentClassifier` over the injectable `transport(url, body, headers, timeout_s) -> bytes`. One attempt per call."""

    def __init__(self, cfg: JevConfig, transport: Callable[[str, bytes, dict, float], bytes] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg, self._transport, self._clock = cfg, transport or _urllib_transport, clock
        self.calls = self.failures = self._streak = 0
        self.last_latency_s: float | None = None
        self._opened_at = 0.0

    @property
    def state(self) -> str:
        open_ = self._streak >= self.cfg.max_failures and self._clock() - self._opened_at < self.cfg.cooldown_s
        return "open" if open_ else "closed"

    def classify(self, text: str, options: Sequence[str], context: Mapping) -> tuple[str, float]:
        if self.state == "open":
            raise JevUnavailable("circuit open: too many consecutive failures")
        key = os.environ.get(self.cfg.api_key_env, "")
        if not key:
            raise JevUnavailable(f"no API key in ${self.cfg.api_key_env}")
        body = build_request(text, options, dict(context))
        headers = {"Content-Type": "application/json", PLACEHOLDER_AUTH_HEADER: PLACEHOLDER_AUTH_SCHEME + key}
        if self.cfg.model:
            headers["X-Model"] = self.cfg.model  # PLACEHOLDER
        self.calls += 1
        start = self._clock()
        try:
            out = parse_response(self._transport(self.cfg.url, body, headers, self.cfg.timeout_s), options)
        except Exception as exc:  # noqa: BLE001 - every failure counts toward the breaker; the message never includes the key
            self.failures += 1
            self._streak += 1
            self._opened_at = self._clock()
            raise JevUnavailable(f"jev call failed: {type(exc).__name__}") from None
        finally:
            self.last_latency_s = self._clock() - start
        self._streak = 0
        return out


DEFAULT_RULES = (("go to", "pick"), ("walk to", "pick"), ("bring", "pick"), ("fetch", "pick"), ("grab", "pick"),
                 ("halt", "stop"), ("freeze", "stop"))


class FakeJev:
    """Deterministic offline classifier. mode: ok | timeout | error | malformed | low_prob | outage_after:N."""

    def __init__(self, rules: Sequence[tuple[str, str]] | None = None, mode: str = "ok", latency_s: float = 0.2,
                 clock_advance: Callable[[float], None] | None = None) -> None:
        self.rules, self.mode, self.latency_s, self.clock_advance = tuple(DEFAULT_RULES if rules is None else rules), mode, latency_s, clock_advance
        self.calls: list[tuple] = []

    def classify(self, text: str, options: Sequence[str], context: Mapping):
        self.calls.append((text, tuple(options), dict(context)))
        if self.clock_advance is not None:
            self.clock_advance(self.latency_s)
        mode = self.mode
        if mode.startswith("outage_after:") and len(self.calls) > int(mode.split(":", 1)[1]):
            mode = "error"
        if mode == "timeout":
            raise TimeoutError("fake jev timeout")
        if mode == "error":
            raise ConnectionError("fake jev error")
        if mode == "malformed":
            return "pick"
        low = text.lower()
        intent = next((i for phrase, i in self.rules if phrase in low), "none")
        return intent, 0.3 if mode == "low_prob" else 0.9


class ReplayJev:
    """Answers from a recorded file `{text: [choice, probability]}`; an unrecorded sentence raises KeyError."""

    def __init__(self, path: str) -> None:
        with open(path, encoding="utf-8") as fh:
            self._answers = json.load(fh)

    def classify(self, text: str, options: Sequence[str], context: Mapping) -> tuple[str, float]:
        choice, prob = self._answers[text]
        return choice, float(prob)


def record(path: str, classifier, sentences: Sequence[str], options: Sequence[str]) -> None:
    answers = {s: list(classifier.classify(s, options, {})) for s in sentences}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(answers, fh, indent=1)
