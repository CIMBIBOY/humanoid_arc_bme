"""Deterministic grammar: one operator sentence in, one typed Command out. No model, no network, stdlib only.

A sentence is folded (NFKC, lower case, single spaces), checked for characters outside plain ASCII and basic
punctuation, stripped of polite prefixes, tokenised, and matched to one action: move, turn, pick, place or a status
question. A STOP word wins over everything else in the sentence. `parse` always returns a Command. Whatever the
grammar does not accept is a `Reject` with a reason. Numbers are read from the sentence and never clamped: a distance
or angle above `CommandLimits` is rejected.

Accepted examples: "go forward 2 meters", "walk 50 cm back", "move two meters to the left", "turn left 90 degrees",
"turn around", "pick up the red cup", "grab a bottle", "put it down", "place it on the left", "stop", "what do you see".
"""

from __future__ import annotations

import math
import re
import string
import unicodedata
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from .types import PLACE_SIDES, Command, CommandLimits, Move, Pick, Place, Reject, Status, Stop, Turn, intent_of

MAX_CHARS = 300
STOP_SCAN_CHARS = 100_000   # bound on the text scanned for a stop word (the scan is linear)

DEFAULT_LABELS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "cup": ("cup", "mug"),
    "bottle": ("bottle",),
    "bowl": ("bowl",),
    "box": ("box", "cube", "block"),
    "can": ("can", "tin"),
})
DEFAULT_COLORS = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "white", "gray", "black")
_COLOR_ALIASES = MappingProxyType({"grey": "gray"})

_ALLOWED = frozenset(string.ascii_lowercase + string.digits + " .,!?'")
_NUMERIC_START = "0123456789."
# A STOP word anywhere wins. Inflections and "do not move" are included: a missed stop is the dangerous error.
_STOP_RE = re.compile(
    r"\b(?:stop|stops|stopped|stopping|halt|halts|halted|freeze|freezes|abort|cancel|emergency"
    r"|do(?:n\W?t|\s+not)\s+move)\b"
)
_NEGATIVE = re.compile(r"-\s*[\d.]")
_DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d{1,2}(?!\d))")  # "2,5" -> "2.5"; "1,000" is left alone and rejected
_TOKEN = re.compile(r"\.?\d[\d.,]*|[a-z]+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?|\.\d+")

_NUMBER_WORDS = MappingProxyType({word: n for n, word in enumerate(
    ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"), start=1)})
_DIGIT_WORDS = MappingProxyType({word: str(n) for n, word in enumerate(
    ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"))})
_AMOUNT_WORDS = frozenset({"a", "an", "half"})
_METRES = MappingProxyType({
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01, "centimetre": 0.01, "centimetres": 0.01,
})
_DEGREES = frozenset({"degree", "degrees", "deg"})
_RADIANS = frozenset({"radian", "radians", "rad"})

# Move directions -> (axis, sign): "x" is forward (+) / back (-), "y" is left (+) / right (-).
_MOVE_DIRECTION = MappingProxyType({
    "forward": ("x", 1), "forwards": ("x", 1), "ahead": ("x", 1), "straight": ("x", 1),
    "back": ("x", -1), "backward": ("x", -1), "backwards": ("x", -1),
    "left": ("y", 1), "right": ("y", -1),
})
_TURN_SIGN = MappingProxyType({"left": 1, "right": -1})

# "back" is listed as a move verb for "back up"; "walk 50 cm back" reads it as a direction.
_VERB_ACTION = MappingProxyType({
    **dict.fromkeys(("go", "move", "walk", "step", "drive", "advance", "come", "back"), "move"),
    **dict.fromkeys(("turn", "rotate", "spin", "face"), "turn"),
    **dict.fromkeys(("pick", "grab", "grasp", "take", "get", "fetch", "lift", "bring"), "pick"),
    **dict.fromkeys(("put", "place", "set", "drop", "release", "lay"), "place"),
})
_FILLER = frozenset({"a", "an", "the", "to", "by", "on", "onto", "it", "up", "down", "now", "just", "some",
                     "me", "and", "please", "this", "that"})
_POLITE_WORDS = frozenset({"please", "robot", "hey"})
_NEGATION = frozenset({"not", "no", "never", "dont", "don", "cant", "wont", "without", "nothing", "nobody", "instead"})
_POLITE_PAIRS = frozenset({("can", "you"), ("could", "you"), ("would", "you"), ("will", "you")})
_STATUS_PHRASES = frozenset({"status", "what do you see", "what can you see", "list objects", "where are you"})


def _fold(text: str) -> str:
    """NFKC, lower case, the degree sign as the word "degrees", and single spaces between words."""
    folded = unicodedata.normalize("NFKC", text).lower().replace("°", " degrees ")
    return " ".join(folded.split())


def _printable_ascii(folded: str) -> str:
    """The sentence with every character outside printable ASCII replaced by a space. Used for the STOP check only."""
    return "".join(c if " " <= c <= "~" else " " for c in folded)


def _tokens(folded: str) -> list[str]:
    """Numbers (with their dots and commas) and words, in order. Other punctuation separates tokens and is dropped."""
    return [tok.rstrip(".,") for tok in _TOKEN.findall(_DECIMAL_COMMA.sub(".", folded))]


def _strip_polite(tokens: list[str]) -> list[str]:
    """Drop leading "please", "robot", "hey", "can you" and similar, and a trailing "please"."""
    start = 0
    while start < len(tokens):
        if tokens[start] in _POLITE_WORDS:
            start += 1
        elif tuple(tokens[start:start + 2]) in _POLITE_PAIRS:
            start += 2
        else:
            break
    tokens = tokens[start:]
    return tokens[:-1] if tokens[-1:] == ["please"] else tokens


def _starts_amount(tok: str) -> bool:
    return tok in _AMOUNT_WORDS or tok in _NUMBER_WORDS or _NUMBER.fullmatch(tok) is not None


def _digit(tok: str) -> str | None:
    if tok in _DIGIT_WORDS:
        return _DIGIT_WORDS[tok]
    return tok if len(tok) == 1 and tok.isdigit() else None


def _read_amount(tokens: list[str], i: int) -> tuple[float, int]:
    """Read the number phrase at tokens[i], which `_starts_amount` accepted. Returns (value, index after the phrase).

    Forms: digits ("2", "2.5", ".5"), number words ("one" to "ten"), "a" or "an" (= 1), "half" or "a half" (= 0.5),
    "X and a half", and "X point Y" where Y is single digits (words or numerals) and X is a whole number.
    """
    tok = tokens[i]
    if tok == "half":
        j = i + 1
        return 0.5, (j + 1 if tokens[j:j + 1] in (["a"], ["an"]) else j)
    if tok in ("a", "an") and tokens[i + 1:i + 2] == ["half"]:
        return 0.5, i + 2
    if tok in ("a", "an"):
        value = 1.0
    elif tok in _NUMBER_WORDS:
        value = float(_NUMBER_WORDS[tok])
    else:
        value = float(tok)
    j = i + 1
    if tokens[j:j + 3] == ["and", "a", "half"]:
        value += 0.5
        j += 3
    if value.is_integer() and tokens[j:j + 1] == ["point"]:
        digits: list[str] = []
        k = j + 1
        while k < len(tokens) and _digit(tokens[k]) is not None:
            digits.append(_digit(tokens[k]))
            k += 1
        if digits:
            value += float("0." + "".join(digits))
            j = k
    return value, j


def _read_measure(tokens: list[str], i: int) -> tuple[float, str, int]:
    """Read an amount and the word right after it (the unit, or "" at the end). Returns (value, unit, next index)."""
    value, j = _read_amount(tokens, i)
    return value, (tokens[j] if j < len(tokens) else ""), j + 1


def _checked_turn(angle: float, limit: float) -> Command:
    if angle == 0.0:
        return Reject("angle must be more than zero")
    if abs(angle) > limit:
        return Reject(f"too far: max {math.degrees(limit):.0f} degrees")
    return Turn(angle)


def _unknown(raw: str) -> Reject:
    snippet = "".join(c if c.isascii() and c.isprintable() else "?" for c in raw[:40]).strip()
    return Reject(f"did not understand: {snippet}")


def _check_word(word: object) -> None:
    if not (isinstance(word, str) and word.isascii() and word.isalpha() and word.islower()):
        raise ValueError(f"grammar word must be one lower-case ASCII word, got {word!r}")


class LocalGrammarParser:
    """Turns operator sentences into Commands. Build once and reuse: `parse` and `intent` are pure.

    `labels` maps each canonical object label to its synonyms (each a single lower-case word); the canonical label
    is also accepted as a word. `colors` lists the colour words; "grey" is accepted when "gray" is in the list.
    """

    def __init__(self, limits: CommandLimits = CommandLimits(), labels: Mapping[str, Sequence[str]] | None = None,
                 colors: Sequence[str] = DEFAULT_COLORS) -> None:
        self._limits = limits
        source = DEFAULT_LABELS if labels is None else labels
        self._labels = tuple(source)
        self._label_of: dict[str, str] = {}
        for canonical, synonyms in source.items():
            for word in (canonical, *synonyms):
                _check_word(word)
                if self._label_of.setdefault(word, canonical) != canonical:
                    raise ValueError(f"label word {word!r} belongs to two labels")
        for color in colors:
            _check_word(color)
        self._color_of = {color: color for color in colors}
        for alias, color in _COLOR_ALIASES.items():
            if color in self._color_of:
                self._color_of[alias] = color

    def intent(self, text: object) -> str:
        """One of INTENTS for an accepted sentence, or "unknown" for a Reject."""
        cmd = self.parse(text)
        return "unknown" if isinstance(cmd, Reject) else intent_of(cmd)

    def _tokens(self, text: object) -> "list[str] | Command":
        """Shared front end: the tokens of one sentence, or the Command (Stop / Reject) that ends parsing early."""
        if not isinstance(text, str):
            return Reject("command must be text")
        # STOP wins over everything, including the length limit: a stop word anywhere in the first STOP_SCAN_CHARS
        # characters stops the robot, however long the sentence is.
        if _STOP_RE.search(_printable_ascii(_fold(text[:STOP_SCAN_CHARS]))):
            return Stop()
        if len(text) > MAX_CHARS:
            return Reject(f"too long: max {MAX_CHARS} characters")
        folded = _fold(text)
        if not folded:
            return Reject("empty command")
        if _NEGATIVE.search(folded):
            return Reject("negative numbers are not accepted: say back, left or right")
        if not set(folded) <= _ALLOWED:
            return Reject("unsupported characters: use plain letters, digits and . , ! ?")
        tokens = _tokens(folded)
        for tok in tokens:
            if tok[0] in _NUMERIC_START and _NUMBER.fullmatch(tok) is None:
                return Reject(f"unclear number: {tok[:20]}")
        tokens = _strip_polite(tokens)
        if not tokens:
            return Reject("empty command")
        return tokens

    def _dispatch(self, action: str, tokens: list[str], text: str) -> Command:
        if action == "turn":
            return self._turn(tokens, text)
        if action == "pick":
            return self._pick(tokens, text)
        if action == "place":
            return self._place(tokens, text)
        return self._move(tokens, text)

    def parse(self, text: object) -> Command:
        """The Command for one sentence. Never raises: anything not understood is a Reject."""
        tokens = self._tokens(text)
        if not isinstance(tokens, list):
            return tokens
        if " ".join(tokens) in _STATUS_PHRASES:
            return Status()
        actions = {_VERB_ACTION[tok] for tok in tokens if tok in _VERB_ACTION}
        if len(actions) > 1:
            return Reject("one command at a time")
        return self._dispatch(next(iter(actions), "move"), tokens, text)

    def parse_as(self, intent: str, text: object) -> Command:
        """Extract the arguments of `text` for an intent decided elsewhere (the optional classifier). The numbers, labels
        and colours still come from this grammar; an intent without usable arguments is a Reject, never a guess."""
        tokens = self._tokens(text)
        if not isinstance(tokens, list):
            return tokens
        if intent == "stop":
            return Stop()
        if intent == "status":
            return Status()
        if intent not in ("move", "turn", "pick", "place"):
            return Reject(f"unknown intent {intent!r}")
        if _NEGATION & set(tokens):
            return Reject("negated sentences are not accepted: say what you want done")
        # The intent is given, so filler words ("me", "over there") may be skipped. Only words the grammar knows are kept:
        # every number, unit, direction, label and colour still comes from the sentence through the same rules.
        known = self._known_words()
        return self._dispatch(intent, [t for t in tokens if t in known or t[0] in _NUMERIC_START], text)

    def _known_words(self) -> frozenset:
        return frozenset(self._label_of) | frozenset(self._color_of) | frozenset(_VERB_ACTION) | _FILLER | frozenset(PLACE_SIDES) \
            | frozenset(_MOVE_DIRECTION) | frozenset(_TURN_SIGN) | frozenset(_NUMBER_WORDS) | frozenset(_DIGIT_WORDS) \
            | _AMOUNT_WORDS | frozenset(_METRES) | _DEGREES | _RADIANS | frozenset({"around", "point", "and"})

    def _move(self, tokens: list[str], raw: str) -> Command:
        direction: tuple[str, int] | None = None
        distance: float | None = None
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in _MOVE_DIRECTION:
                if direction not in (None, _MOVE_DIRECTION[tok]):
                    return Reject("one direction at a time")
                direction = _MOVE_DIRECTION[tok]
                i += 1
            elif _starts_amount(tok):
                if distance is not None:
                    return Reject("one distance at a time")
                value, unit, i = _read_measure(tokens, i)
                if unit not in _METRES:
                    return Reject("distance unit missing or unknown: say metres or centimetres, e.g. 2 meters")
                distance = value * _METRES[unit]
            elif tok in _VERB_ACTION or tok in _FILLER:
                i += 1
            else:
                return _unknown(raw)
        if direction is None:
            return Reject("which way? forward, back, left or right")
        if distance is None:
            return Reject("how far?")
        if distance == 0.0:
            return Reject("distance must be more than zero")
        if distance > self._limits.max_move_m:
            return Reject(f"too far: max {self._limits.max_move_m} m")
        axis, sign = direction
        metres = sign * distance
        return Move(forward_m=metres if axis == "x" else 0.0, left_m=metres if axis == "y" else 0.0)

    def _turn(self, tokens: list[str], raw: str) -> Command:
        sign: int | None = None
        around = False
        angle: float | None = None
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in _TURN_SIGN:
                if sign not in (None, _TURN_SIGN[tok]):
                    return Reject("one direction at a time")
                sign = _TURN_SIGN[tok]
                i += 1
            elif tok == "around":
                around = True
                i += 1
            elif _starts_amount(tok):
                if angle is not None:
                    return Reject("one angle at a time")
                value, unit, i = _read_measure(tokens, i)
                if unit in _DEGREES:
                    angle = math.radians(value)
                elif unit in _RADIANS:
                    angle = value
                else:
                    return Reject("angle unit missing or unknown: say degrees or radians, e.g. 90 degrees")
            elif tok in _VERB_ACTION or tok in _FILLER:
                i += 1
            else:
                return _unknown(raw)
        limit = self._limits.max_turn_rad
        if around:
            if sign is not None or angle is not None:
                return Reject("say 'turn around' alone, or an angle with left or right")
            return _checked_turn(math.pi, limit)
        if sign is None:
            return Reject("which way? say turn left or turn right")
        return _checked_turn(sign * (math.pi / 2 if angle is None else angle), limit)

    def _pick(self, tokens: list[str], raw: str) -> Command:
        labels: list[str] = []
        colors: list[str] = []
        unknown = False
        for tok in tokens:
            if tok in self._label_of:
                labels.append(self._label_of[tok])
            elif tok in self._color_of:
                colors.append(self._color_of[tok])
            elif tok not in _VERB_ACTION and tok not in _FILLER:
                unknown = True
        if len(set(labels)) > 1:
            return Reject("one object at a time")
        if len(set(colors)) > 1:
            return Reject("one colour at a time")
        if not labels:
            return Reject(f"which object? I know: {', '.join(self._labels)}")
        if unknown:
            return _unknown(raw)
        return Pick(label=labels[0], color=colors[0] if colors else None)

    def _place(self, tokens: list[str], raw: str) -> Command:
        sides: list[str] = []
        for tok in tokens:
            if tok in PLACE_SIDES:
                sides.append(tok)
            elif tok not in _VERB_ACTION and tok not in _FILLER:
                return _unknown(raw)
        if "it" not in tokens and "down" not in tokens:
            return _unknown(raw)
        if len(set(sides)) > 1:
            return Reject("one side at a time")
        return Place(sides[0] if sides else "front")
