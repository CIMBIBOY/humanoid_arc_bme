"""Phase-based model-group scheduler for a memory-limited device (stdlib only).

All sizes are in bytes. A resident set is loaded on ``start()`` and stays loaded until ``stop()``. Optional
model groups are loaded per phase by ``switch()``.

Invariants:
  * ``resident_bytes + sum(est_bytes of the phase's distinct models) <= budget_bytes`` for every accepted switch
    (declared sizes, checked before anything is touched).
  * Within a switch, unloads happen before loads, so declared memory never exceeds the budget mid-switch.
  * A model shared by the old and new phase is neither unloaded nor reloaded.
  * After a failed switch only the resident set is loaded (every phase model is unloaded, best effort) and
    ``phase is None`` (safe state); a model whose rollback unload also fails stays loaded and is cleaned by the
    next ``switch`` or ``stop``.
Models are identified by object identity.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class ModelHandle(Protocol):
    """A loadable model. ``est_bytes`` is its declared memory budget in bytes."""

    est_bytes: int
    loaded: bool

    def load(self) -> None: ...

    def unload(self) -> None: ...


class BudgetError(ValueError):
    """The requested phase does not fit in the memory budget."""


def _name(h: ModelHandle) -> str:
    return str(getattr(h, "name", type(h).__name__))


def _distinct(handles: list[ModelHandle]) -> list[ModelHandle]:
    seen: set[int] = set()
    out = []
    for h in handles:
        if id(h) not in seen:
            seen.add(id(h))
            out.append(h)
    return out


class PhaseScheduler:
    """Keeps ``resident`` loaded and swaps optional model groups by phase.

    ``memory_probe`` (optional) returns current process/device memory in bytes; it is called after every
    load/unload. ``max_log`` bounds the in-memory event log (oldest entries dropped).
    """

    def __init__(
        self,
        resident: list[ModelHandle],
        phases: dict[str, list[ModelHandle]],
        budget_bytes: int,
        memory_probe: Callable[[], int] | None = None,
        max_log: int = 1000,
    ) -> None:
        if budget_bytes <= 0 or max_log <= 0:
            raise ValueError("budget_bytes and max_log must be positive")
        self._resident = list(resident)
        if len(_distinct(self._resident)) != len(self._resident):
            raise ValueError("duplicate model in resident set")
        resident_ids = {id(h) for h in self._resident}
        self._phases: dict[str, list[ModelHandle]] = {}
        for name, models in phases.items():
            if len(_distinct(models)) != len(models):
                raise ValueError(f"duplicate model in phase {name!r}")
            if any(id(h) in resident_ids for h in models):
                raise ValueError(f"resident model listed in phase {name!r}")
            self._phases[name] = list(models)
        self._resident_bytes = sum(h.est_bytes for h in self._resident)
        if self._resident_bytes > budget_bytes:
            raise BudgetError(f"resident set {self._resident_bytes} B exceeds budget {budget_bytes} B")
        self._budget = budget_bytes
        self._probe = memory_probe
        self._log: deque[tuple[str | None, str, int | None]] = deque(maxlen=max_log)
        self._peak = 0
        self._phase: str | None = None
        self._started = False
        self._lock = threading.RLock()

    @property
    def phase(self) -> str | None:
        """Current phase, or None (resident only / safe state)."""
        with self._lock:
            return self._phase

    @property
    def peak_bytes(self) -> int:
        """Largest probe reading seen, in bytes (0 without a probe)."""
        with self._lock:
            return self._peak

    def _record(self, event: str, h: ModelHandle) -> None:
        used = self._probe() if self._probe is not None else None
        if used is not None:
            self._peak = max(self._peak, used)
        self._log.append((self._phase, f"{event}:{_name(h)}", used))

    def _load(self, h: ModelHandle) -> None:
        h.load()
        self._record("load", h)

    def _unload(self, h: ModelHandle) -> None:
        h.unload()
        self._record("unload", h)

    def start(self) -> None:
        """Load the resident set. Idempotent. On failure, unload what was loaded and re-raise."""
        with self._lock:
            if self._started:
                return
            try:
                for h in self._resident:
                    self._load(h)
            except BaseException:
                # Roll back by actual state, not by bookkeeping: a model whose load() succeeded but whose
                # probe/record step raised is loaded and must be unloaded too.
                for h in reversed(self._resident):
                    if h.loaded:
                        self._rollback_unload(h)
                raise
            self._started = True

    def _rollback_unload(self, h: ModelHandle) -> None:
        try:
            self._unload(h)
        except Exception:  # best effort; the original error is the one that matters
            self._log.append((self._phase, f"rollback_unload_failed:{_name(h)}", None))

    def switch(self, phase: str) -> None:
        """Move to ``phase``; an unknown phase raises ValueError. Validates first (state untouched on rejection), then unloads, then loads."""
        with self._lock:
            if not self._started:
                raise RuntimeError("scheduler not started")
            if phase not in self._phases:
                raise ValueError(f"unknown phase {phase!r}")
            need = self._phases[phase]
            total = self._resident_bytes + sum(h.est_bytes for h in need)
            if total > self._budget:
                raise BudgetError(f"phase {phase!r} needs {total} B, budget {self._budget} B")
            if phase == self._phase:
                return
            need_ids = {id(h) for h in need}
            self._phase = None  # safe state until the switch completes
            try:
                # Every loaded phase model not needed is stale, whichever phase it came from.
                for h in self._phase_models():
                    if id(h) not in need_ids and h.loaded:
                        self._unload(h)
                for h in need:
                    if not h.loaded:
                        self._load(h)
            except BaseException:
                # Safe state = resident only: unload every phase model, best effort, then re-raise.
                for h in reversed(self._phase_models()):
                    if h.loaded:
                        self._rollback_unload(h)
                raise
            self._phase = phase

    def _phase_models(self) -> list[ModelHandle]:
        return _distinct([h for ms in self._phases.values() for h in ms])

    def stop(self) -> None:
        """Unload everything loaded (phase models, then resident). Idempotent; raises a combined error at the end."""
        with self._lock:
            errors: list[str] = []
            everything = _distinct([h for ms in self._phases.values() for h in ms]) + self._resident
            for h in everything:
                if h.loaded:
                    try:
                        self._unload(h)
                    except Exception as exc:
                        errors.append(f"{_name(h)}: {exc!r}")
            self._phase = None
            self._started = False
            if errors:
                raise RuntimeError("unload failed for: " + "; ".join(errors))

    def report(self) -> dict:
        """JSON-serializable snapshot: phase, budget/peak bytes, loaded model names, bounded event log."""
        with self._lock:
            everything = _distinct([h for ms in self._phases.values() for h in ms]) + self._resident
            return {
                "phase": self._phase,
                "started": self._started,
                "budget_bytes": self._budget,
                "peak_bytes": self._peak,
                "resident_bytes": self._resident_bytes,
                "loaded": [_name(h) for h in everything if h.loaded],
                "log": [{"phase": p, "event": e, "bytes": b} for p, e, b in self._log],
            }
