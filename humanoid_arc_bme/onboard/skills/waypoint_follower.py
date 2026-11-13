"""Route and timing of the right wrist between the named waypoints (plan A4.2).

Pure Python. The waypoints are the ones of scripted_pick_place.WAYPOINTS. The
wrist only moves along the edges below, which are the motions the timer script
already proved safe (it rises to `ready_*` before the hand turns toward the
object, so the fingers do not sweep it). A new target is routed from the node
the wrist is heading to: the segment in flight always finishes first, so the
commanded motion never jumps.
"""

from collections import deque

# (a, b): seconds to travel. Durations are the timer script's phase durations.
EDGE_SECONDS = {
    ("rest", "ready_pick"): 1.5,
    ("ready_pick", "above_pick"): 1.5,
    ("above_pick", "pick"): 1.5,
    ("above_pick", "above_place"): 2.0,
    ("above_place", "place"): 1.5,
    ("above_place", "ready_place"): 1.0,
    ("ready_place", "rest"): 1.5,
}
EDGES = {}
for (_a, _b), _t in EDGE_SECONDS.items():
    EDGES.setdefault(_a, {})[_b] = _t
    EDGES.setdefault(_b, {})[_a] = _t
NODES = tuple(EDGES)


def route(start: str, goal: str) -> list:
    """Shortest list of nodes after `start` that ends at `goal` (empty if equal)."""
    if start not in EDGES or goal not in EDGES:
        raise ValueError(f"unknown waypoint: {start!r} -> {goal!r}")
    prev, queue = {start: None}, deque([start])
    while queue:
        n = queue.popleft()
        for m in EDGES[n]:
            if m not in prev:
                prev[m] = n
                queue.append(m)
    path, n = [], goal
    while n != start:
        path.append(n)
        n = prev[n]
    return path[::-1]


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


class WaypointFollower:
    def __init__(self, start: str = "rest", speed: float = 1.0):
        if start not in EDGES:
            raise ValueError(f"unknown waypoint {start!r}")
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.speed = speed
        self.node = start      # last node reached
        self._next = None      # node being travelled to
        self._t = 0.0
        self._dur = 0.0

    @property
    def settled(self) -> str | None:
        """The node the wrist sits at, or None while it is travelling."""
        return self.node if self._next is None else None

    def step(self, target: str, dt: float):
        """Advance dt seconds toward `target`. Returns (from_node, to_node, s)
        with s in [0, 1] already smoothed; at rest it is (node, node, 0.0)."""
        if target not in EDGES:
            raise ValueError(f"unknown waypoint {target!r}")
        if dt < 0:
            raise ValueError("dt must be >= 0")
        if self._next is None and target != self.node:
            self._next = route(self.node, target)[0]
            self._t = 0.0
            self._dur = EDGES[self.node][self._next] / self.speed
        if self._next is None:
            return self.node, self.node, 0.0
        self._t += dt
        a, b = self.node, self._next
        s = smoothstep(self._t / self._dur)
        if self._t >= self._dur:
            self.node, self._next = b, None
        return a, b, s


class HandRamp:
    """Hand closure command, rate limited; `reached(hold)` is true once the
    command has sat at the target for `hold` seconds."""

    def __init__(self, rate: float = 1.0, closure: float = 0.0):
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate, self.closure, self._target, self._held = rate, closure, closure, 0.0

    def step(self, target: float, dt: float) -> float:
        if not 0.0 <= target <= 1.0:
            raise ValueError("closure target must be in [0, 1]")
        if target != self._target:
            self._target, self._held = target, 0.0
        delta = target - self.closure
        self.closure += max(-self.rate * dt, min(self.rate * dt, delta))
        self._held = self._held + dt if self.closure == target else 0.0
        return self.closure

    def reached(self, hold: float) -> bool:
        return self._held >= hold
