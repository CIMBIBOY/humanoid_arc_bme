"""Closed-loop pick-and-place as a state machine (plan A4.1).

Pure Python, no simulator: it reads an `Observation` each control step and
returns the `Command` for that step (a named wrist waypoint and a hand
closure). The same logic drives the sim env (A4.2) and the robot, and later
acts as the DAgger expert (A5).

Rules:
  * a transition fires only on a named predicate of the observation;
  * every state has a timeout; on timeout the state is re-entered (a retry)
    until `max_retries`, then the machine goes to FAILED with a reason;
  * losing the object while it should be held sends the machine back to
    LOCATE, and counts as a retry of the state where it was lost;
  * DONE and FAILED are terminal and never change.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class State(str, Enum):
    LOCATE = "LOCATE"
    PREGRASP = "PREGRASP"
    APPROACH = "APPROACH"
    CLOSE = "CLOSE"
    LIFT = "LIFT"
    CARRY = "CARRY"
    PLACE = "PLACE"
    RELEASE = "RELEASE"
    WITHDRAW = "WITHDRAW"
    DONE = "DONE"
    FAILED = "FAILED"


TERMINAL = (State.DONE, State.FAILED)
# States in which the object must be in the hand.
HOLDING = (State.LIFT, State.CARRY, State.PLACE)

# state -> (waypoint name of scripted_pick_place.WAYPOINTS, hand closure 0..1)
COMMANDS = {
    State.LOCATE: ("rest", 0.0),
    State.PREGRASP: ("above_pick", 0.0),
    State.APPROACH: ("pick", 0.0),
    State.CLOSE: ("pick", 1.0),
    State.LIFT: ("above_pick", 1.0),
    State.CARRY: ("above_place", 1.0),
    State.PLACE: ("place", 1.0),
    State.RELEASE: ("place", 0.0),
    State.WITHDRAW: ("ready_place", 0.0),
    State.DONE: ("rest", 0.0),
    State.FAILED: ("rest", 0.0),
}

# state -> (predicate that leaves it, next state)
TRANSITIONS = {
    State.LOCATE: ("object_seen", State.PREGRASP),
    State.PREGRASP: ("pregrasp_reached", State.APPROACH),
    State.APPROACH: ("grasp_pose_reached", State.CLOSE),
    State.CLOSE: ("closure_confirmed", State.LIFT),
    State.LIFT: ("lifted", State.CARRY),
    State.CARRY: ("place_pose_reached", State.PLACE),
    State.PLACE: ("placed", State.RELEASE),
    State.RELEASE: ("hand_open", State.WITHDRAW),
    State.WITHDRAW: ("hand_clear", State.DONE),
}

# Seconds. `(proposed)`: about twice the duration of the matching phase in the
# timer script; A4.2 measures the real durations and the log replaces these.
DEFAULT_TIMEOUTS = {
    State.LOCATE: 3.0,
    State.PREGRASP: 6.0,
    State.APPROACH: 4.0,
    State.CLOSE: 3.0,
    State.LIFT: 4.0,
    State.CARRY: 6.0,
    State.PLACE: 4.0,
    State.RELEASE: 3.0,
    State.WITHDRAW: 4.0,
}


@dataclass(frozen=True)
class Observation:
    """What the machine may look at. Thresholds live in the producer of the
    observation (pregrasp within 1.5 cm, lifted 3 cm above rest, placed within
    2 cm of the goal and at rest); the machine only reads the booleans."""

    object_seen: bool = False
    pregrasp_reached: bool = False
    grasp_pose_reached: bool = False
    closure_confirmed: bool = False
    object_held: bool = False
    lifted: bool = False
    place_pose_reached: bool = False
    placed: bool = False
    hand_open: bool = False
    hand_clear: bool = False


@dataclass(frozen=True)
class Command:
    waypoint: str
    hand_closure: float


class PickPlaceMachine:
    def __init__(self, timeouts: Mapping[State, float] = DEFAULT_TIMEOUTS, max_retries: int = 2):
        missing = [s for s in TRANSITIONS if s not in timeouts]
        if missing:
            raise ValueError(f"no timeout for states {[s.value for s in missing]}")
        if any(t <= 0 for t in timeouts.values()):
            raise ValueError("timeouts must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.timeouts = dict(timeouts)
        self.max_retries = max_retries
        self.state = State.LOCATE
        self.reason = ""
        self.retries = {s: 0 for s in TRANSITIONS}
        self._entered = None  # time of entry; set on the first step

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL

    def command(self) -> Command:
        wp, closure = COMMANDS[self.state]
        return Command(wp, closure)

    def _enter(self, state: State, now: float) -> None:
        self.state = state
        self._entered = now

    def _retry_or_fail(self, failed_state: State, cause: str, reenter: State, now: float) -> None:
        if self.retries[failed_state] >= self.max_retries:
            self.reason = f"{cause}:{failed_state.value}"
            self._enter(State.FAILED, now)
            return
        self.retries[failed_state] += 1
        self._enter(reenter, now)

    def step(self, now: float, obs: Observation) -> Command:
        """Advance by one control step at time `now` (seconds, non-decreasing)."""
        if self.terminal:
            return self.command()
        if self._entered is None:
            self._entered = now
        elif now < self._entered:
            raise ValueError("time went backwards")
        st = self.state
        if st in HOLDING and not obs.object_held:
            self._retry_or_fail(st, "object_dropped", State.LOCATE, now)
        else:
            pred, nxt = TRANSITIONS[st]
            if getattr(obs, pred):
                self._enter(nxt, now)
            elif now - self._entered > self.timeouts[st]:
                self._retry_or_fail(st, "timeout", st, now)
        return self.command()
