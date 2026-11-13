"""D1: strict task-layer skill contract (stdlib only).

A skill runs only if its typed preconditions hold, is judged only by sensor-based postconditions (never by the
policy's own claim), has a timeout, and ends in a defined outcome with a failure code. Everything fails closed:
a missing, non-finite or erroring fact/check counts as NOT satisfied.

Failure codes: ``precondition:<name>``, ``timeout``, ``postcondition_lost`` (a safety invariant stopped holding),
``exception`` (step raised), ``invalid_dt``, ``not_started``, ``stale_facts`` (facts carry no stamp or are older than
``max_fact_age_s`` by the injected clock: a frozen sensor must not satisfy conditions forever). ``abort()`` is
called exactly once per started run that reaches a terminal state; it is NOT called when a precondition or stale
facts block the start, nor when ``SkillRunner.abort()`` is invoked on a never-started runner (nothing was actuated).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Optional, Protocol

from humanoid_arc_bme.onboard.api.contract import SkillStatus


class WorldFacts:
    """Immutable mapping of named bool/float facts from sensors. Non-finite or non-numeric values are dropped
    (= unknown). `stamp` is the time (same clock as the runner) at which the facts were measured; None = unstamped,
    which the runner treats as stale."""

    def __init__(self, facts=None, stamp: Optional[float] = None):
        if stamp is not None and (isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp)):
            stamp = None
        self.stamp = None if stamp is None else float(stamp)
        clean = {}
        for k, v in dict(facts or {}).items():
            if isinstance(v, bool):
                clean[str(k)] = v
            elif isinstance(v, (int, float)) and math.isfinite(v):
                clean[str(k)] = float(v)
        self._d = MappingProxyType(clean)

    def get(self, name: str):
        """Value, or None when unknown."""
        return self._d.get(name)

    def __contains__(self, name):
        return name in self._d

    def __len__(self):
        return len(self._d)

    def as_dict(self) -> dict:
        return dict(self._d)


@dataclass(frozen=True)
class Condition:
    name: str
    check: Callable[[WorldFacts], bool]

    @staticmethod
    def is_true(name: str, fact: Optional[str] = None) -> "Condition":
        fact = fact or name
        return Condition(name, lambda f: f.get(fact) is True)

    @staticmethod
    def below(name: str, fact: str, limit: float) -> "Condition":
        def chk(f):
            v = f.get(fact)
            return v is not None and not isinstance(v, bool) and v < limit

        return Condition(name, chk)


class Skill(Protocol):
    name: str
    preconditions: list
    postconditions: list
    timeout_s: float

    def step(self, facts: WorldFacts, dt: float) -> None: ...

    def abort(self) -> None: ...


@dataclass(frozen=True)
class SkillResult:
    state: str  # running | succeeded | failed
    code: str = ""  # failure code, "" unless failed
    elapsed: float = 0.0

    @property
    def terminal(self) -> bool:
        return self.state != "running"


def to_status(result: SkillResult) -> SkillStatus:
    return SkillStatus(result.state, result.code if result.state == "failed" else "")


class SkillLog:
    """Bounded append-only run log of (stamp, event, detail); JSON-serialisable. Oldest entries drop first."""

    def __init__(self, maxlen: int = 256):
        if maxlen < 1:
            raise ValueError("maxlen must be >= 1")
        self.maxlen, self.dropped, self._e = maxlen, 0, []

    def append(self, stamp: float, event: str, detail: str = "") -> None:
        self._e.append((float(stamp), str(event), str(detail)))
        if len(self._e) > self.maxlen:
            del self._e[0]
            self.dropped += 1

    @property
    def entries(self) -> tuple:
        return tuple(self._e)

    def to_json_obj(self) -> dict:
        return {"dropped": self.dropped, "entries": [list(e) for e in self._e]}


class SkillRunner:
    def __init__(self, skill, clock: Callable[[], float], invariants=None, log: Optional[SkillLog] = None,
                 max_fact_age_s: float = 1.0):
        if isinstance(max_fact_age_s, bool) or not isinstance(max_fact_age_s, (int, float)) \
                or not math.isfinite(max_fact_age_s) or max_fact_age_s <= 0:
            raise ValueError(f"max_fact_age_s must be finite and > 0, got {max_fact_age_s!r}")
        self.max_fact_age_s = float(max_fact_age_s)
        t = skill.timeout_s
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0:
            raise ValueError(f"timeout_s must be finite and > 0, got {t!r}")
        self.skill, self.clock = skill, clock
        self.invariants = list(invariants or [])
        self.log = log if log is not None else SkillLog()
        self._t0: Optional[float] = None
        self._result: Optional[SkillResult] = None
        self._aborted = False

    # -- helpers
    def _holds(self, cond: Condition, facts: WorldFacts) -> bool:
        try:
            return cond.check(facts) is True
        except Exception as exc:  # fail closed, record
            self.log.append(self.clock(), "check_error", f"{cond.name}: {type(exc).__name__}: {exc}")
            return False

    def _fresh(self, facts: WorldFacts) -> bool:
        st = getattr(facts, "stamp", None)
        return st is not None and abs(self.clock() - st) <= self.max_fact_age_s

    def _abort_once(self) -> None:
        if self._aborted:
            return
        self._aborted = True
        try:
            self.skill.abort()
        except Exception as exc:
            self.log.append(self.clock(), "abort_error", f"{type(exc).__name__}: {exc}")

    def _finish(self, state: str, code: str = "", abort: bool = True) -> SkillResult:
        now = self.clock()
        self._result = SkillResult(state, code, now - (self._t0 if self._t0 is not None else now))
        if abort:
            self._abort_once()
        self.log.append(now, state, code)
        return self._result

    # -- API
    def start(self, facts: WorldFacts) -> SkillResult:
        if self._t0 is not None or self._result is not None:
            return self._result or SkillResult("running")
        if not self._fresh(facts):
            self._t0 = self.clock()
            return self._finish("failed", "stale_facts", abort=False)
        for c in list(self.skill.preconditions) + self.invariants:
            if not self._holds(c, facts):
                self._t0 = self.clock()
                return self._finish("failed", f"precondition:{c.name}", abort=False)
        self._t0 = self.clock()
        self.log.append(self._t0, "start", self.skill.name)
        return SkillResult("running")

    def tick(self, facts: WorldFacts, dt: float) -> SkillResult:
        if self._result is not None:
            return self._result
        if self._t0 is None:
            return SkillResult("failed", "not_started")
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
            return self._finish("failed", "invalid_dt")
        if not self._fresh(facts):
            return self._finish("failed", "stale_facts")
        for c in self.invariants:
            if not self._holds(c, facts):
                return self._finish("failed", "postcondition_lost")
        try:
            self.skill.step(facts, float(dt))
        except Exception as exc:
            self.log.append(self.clock(), "step_error", f"{type(exc).__name__}: {exc}")
            return self._finish("failed", "exception")
        if all(self._holds(c, facts) for c in self.skill.postconditions) and self.skill.postconditions:
            return self._finish("succeeded")
        if self.clock() - self._t0 >= self.skill.timeout_s:
            return self._finish("failed", "timeout")
        return SkillResult("running", "", self.clock() - self._t0)

    def abort(self) -> SkillResult:
        """External safe stop. Idempotent; the run ends as failed('aborted') unless already terminal. On a runner
        that was never started nothing was actuated, so skill.abort() is NOT called."""
        if self._result is None:
            started = self._t0 is not None
            if not started:
                self._t0 = self.clock()
            return self._finish("failed", "aborted", abort=started)
        return self._result


class ReachSkill:
    """Fake example skill: step moves a position toward a target; success is judged from the supplied facts."""

    name = "reach"

    def __init__(self, timeout_s: float = 5.0):
        self.preconditions = [Condition.is_true("gripper_empty"), Condition.is_true("object_visible")]
        self.postconditions = [Condition.below("hand_near_object", "hand_object_dist", 0.02)]
        self.timeout_s = timeout_s
        self.steps = 0
        self.aborts = 0

    def step(self, facts, dt):
        self.steps += 1

    def abort(self):
        self.aborts += 1
