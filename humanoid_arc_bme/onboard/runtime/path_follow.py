"""Path-following inputs for the operator runtime.

- Curvature: (v, kappa) -> (vx, 0, v * kappa), clipped by `OperatorLimits.clip_curvature` (keeps the path, lowers speed).
- Waypoints: the navigation module named in config/robot_runtime.yaml (`nav_module`, default
  humanoid_arc_bme.onboard.nav.runtime_hook, written by the navstack agent, docs/plan/navstack.md). Its `NavSession`
  (load / start / tick / stop / status) is wrapped by `NavSessionAdapter` into the runtime's follower protocol:
  `set_waypoints(list of (x, y))`, `command(state) -> (vx, vy, wz)`, `stop()`, `status() -> dict`.
  Waypoints are relative to the robot's pose when the path starts (x forward, y left, metres). The path starts on the
  first ARMED tick in waypoint mode; a disarm stops it (re-send the waypoints to go again). Odometry is dead reckoning
  on the commanded velocity + IMU yaw (stage 1 of the navstack plan): the G1 reports no position, so the real path
  drifts; UNMEASURED until the robot (the navstack plan's stage 2 adds external pose).
  A module may instead provide `make_follower(config) -> follower` with the protocol above.
"""

from __future__ import annotations

import importlib
import time


class NavSessionAdapter:
    def __init__(self, session, clock=time.monotonic):
        self.s, self.clock = session, clock
        self._pending = False
        self._active = False  # False after stop(): command() is zero until new waypoints (no decaying ramp on re-arm)
        self.loaded: dict | None = None

    def set_waypoints(self, waypoints) -> dict:
        pts = [[float(w[0]), float(w[1])] for w in waypoints]
        if len(pts) == 1:
            pts = [[0.0, 0.0]] + pts  # a single goal: straight from the start pose
        self.loaded = self.s.load({"waypoints": pts, "frame": "start", "smoothing": "corner"})
        self._pending, self._active = True, True
        return self.loaded

    def command(self, state):
        if not self._active:
            return (0.0, 0.0, 0.0)
        now = self.clock()
        if self._pending:
            self.s.start(now)
            self._pending = False
        c = self.s.tick(now, state)
        return (c.vx, c.vy, c.wz)

    def stop(self) -> None:
        self._pending, self._active = False, False
        self.s.stop()

    def status(self) -> dict:
        d = self.s.status()
        d.update({"loaded": self.loaded, "pending_start": self._pending, "active": self._active})
        return d


def load_nav(module_name: str | None, config: dict | None = None, clock=time.monotonic):
    """(follower, "ok") or (None, reason)."""
    if not module_name:
        return None, "no nav_module configured"
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:  # noqa: BLE001
        return None, f"cannot import {module_name}: {type(e).__name__}: {e}"
    if hasattr(mod, "NavSession"):
        f = NavSessionAdapter(mod.NavSession(), clock)
    elif hasattr(mod, "make_follower"):
        f = mod.make_follower(config or {})
    else:
        return None, f"{module_name} has neither NavSession nor make_follower()"
    for m in ("set_waypoints", "command", "status"):
        if not callable(getattr(f, m, None)):
            return None, f"{module_name} follower lacks {m}()"
    return f, "ok"
