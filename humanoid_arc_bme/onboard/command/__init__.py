"""Natural-language command layer: operator sentences become typed, validated commands through a deterministic grammar.

No model is needed. `LocalGrammarParser.parse` returns a Command for every input; anything it does not understand is
a `Reject` with a reason. `resolve_pick` checks a pick against the objects currently visible. `CompositeParser` can add an
optional intent classifier (Jev or any other) for sentences the grammar does not understand; numbers and objects always
come from the grammar.
"""

from .composite import CompositeParser, IntentClassifier
from .grammar import DEFAULT_COLORS, DEFAULT_LABELS, MAX_CHARS, LocalGrammarParser
from .resolve import VisibleObject, resolve_pick, summarize
from .types import INTENTS, PLACE_SIDES, Command, CommandLimits, Move, Pick, Place, Reject, Status, Stop, Turn, intent_of

__all__ = [
    "Command", "CommandLimits", "CompositeParser", "IntentClassifier", "DEFAULT_COLORS", "DEFAULT_LABELS", "INTENTS", "LocalGrammarParser", "MAX_CHARS",
    "Move", "PLACE_SIDES", "Pick", "Place", "Reject", "Status", "Stop", "Turn", "VisibleObject", "intent_of",
    "resolve_pick", "summarize",
]
