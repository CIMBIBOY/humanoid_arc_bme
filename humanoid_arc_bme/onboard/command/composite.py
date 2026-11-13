"""CompositeParser: the local grammar first, an optional intent classifier only for what the grammar does not understand.

The classifier (e.g. the Jev decision model, a cloud service; or any local model) answers ONE question: which intent, out
of a closed list, does this sentence express? It never produces a number, a label or a colour: those come from the local
grammar (`LocalGrammarParser.parse_as`), so a model can never invent a distance or an object. Rules:

* the grammar's own result is final whenever it understood the sentence;
* STOP is never delayed or filtered: the grammar sees it first, and a classifier that says "stop" stops (an unneeded stop
  is harmless);
* any other classifier answer needs probability >= `min_prob`, a known intent and arguments the grammar can extract;
  a timeout, an exception or a malformed answer falls back to the grammar's rejection. The robot never depends on it.
"""

from __future__ import annotations

import math
from typing import Mapping, Protocol, Sequence

from .grammar import LocalGrammarParser
from .types import INTENTS, Command, Reject, Stop


class IntentClassifier(Protocol):
    def classify(self, text: str, options: Sequence[str], context: Mapping) -> tuple[str, float]:
        """(option, probability in [0, 1]); `options` includes "none" for "not a robot command"."""


class CompositeParser:
    def __init__(self, local: LocalGrammarParser | None = None, classifier: IntentClassifier | None = None,
                 min_prob: float = 0.8, stop_prob: float = 0.5):
        if not (0.0 < min_prob <= 1.0 and 0.0 < stop_prob <= 1.0):
            raise ValueError("probabilities must be in (0, 1]")
        self.local = local if local is not None else LocalGrammarParser()
        self.classifier, self.min_prob, self.stop_prob = classifier, float(min_prob), float(stop_prob)
        self.last_source = "local"
        self.last_error = ""

    def parse(self, text: object, visible: Sequence[str] = ()) -> Command:
        """`visible`: labels of the objects currently in view, passed to the classifier as context only."""
        cmd = self.local.parse(text)
        self.last_source, self.last_error = "local", ""
        if not isinstance(cmd, Reject) or self.classifier is None or not isinstance(text, str):
            return cmd
        try:
            intent, prob = self.classifier.classify(text, (*INTENTS, "none"), {"visible": list(visible)})
            prob = float(prob)
        except Exception as exc:  # noqa: BLE001 - timeout, network, malformed answer: fall back, never fail
            self.last_error = f"classifier: {type(exc).__name__}: {exc}"
            return cmd
        if not (isinstance(intent, str) and math.isfinite(prob) and 0.0 <= prob <= 1.0):
            self.last_error = "classifier: malformed answer"
            return cmd
        if intent == "stop" and prob >= self.stop_prob:
            self.last_source = "classifier"
            return Stop()
        if intent not in INTENTS or prob < self.min_prob:
            return cmd
        out = self.local.parse_as(intent, text)
        if not isinstance(out, Reject):
            self.last_source = "classifier"
            return out
        return Reject(f"understood as '{intent}' but {out.reason}")
