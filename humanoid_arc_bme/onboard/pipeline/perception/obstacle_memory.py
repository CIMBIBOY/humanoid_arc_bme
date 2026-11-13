"""Obstacle memory for a forward-looking depth camera with a near-field blind zone.

An obstacle that was seen and then leaves the camera view (robot gets close, camera looks over/past it) would
otherwise read as "clear". The memory keeps the nearest depth-seen obstacle as a map-frame point and re-projects it
with the gated pose; `blocked_distance` returns its forward distance (inf when nothing is remembered).

Drop rules (a remembered obstacle is forgotten ONLY on evidence):
  * visible-empty: the camera observes that place (beyond the blind zone) and sees free space well past it;
  * behind the robot (forward distance <= 0) or off the corridor (|lateral| > half width + margin);
  * TTL since it was last seen.
Never dropped because of: a missing/non-finite pose (memory frozen, `blocked_distance` returns 0.0 if anything is
remembered so the caller stops), a NaN/backwards/non-finite stamp (observation ignored), NaN depth input.

Limits (documented, deliberate):
  * Default is a single nearest obstacle, not a grid: it covers exactly the failure it exists for (walking into the
    last thing seen) with O(1) state and no map-drift accumulation. `max_items` > 1 keeps several points (oldest
    evicted), but there is no free-space carving of a grid; the costmap remains the source for dense structure.
  * Static world only: a moving obstacle is remembered where it was last seen until it is seen-empty or TTL expires.
  * Dead-reckoning quality is that of the gated pose; the lateral margin absorbs small pose drift.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from humanoid_arc_bme.onboard.validation import is_num as _num


@dataclass
class ObstacleMemoryConfig:
    ttl_s: float = 30.0
    corridor_half_width: float = 0.35
    lateral_margin: float = 0.2
    empty_margin: float = 0.3       # free range must exceed the obstacle's distance by this to call it gone
    blind_margin: float = 0.1       # beyond blind_until by this before "visible"
    max_items: int = 1              # cap on remembered obstacles; oldest dropped
    merge_radius: float = 0.15      # new sighting this close to a remembered one replaces it

    def __post_init__(self) -> None:
        for n in ("ttl_s", "corridor_half_width", "lateral_margin", "empty_margin", "blind_margin", "merge_radius"):
            v = getattr(self, n)
            if not _num(v) or v < 0.0:
                raise ValueError(f"ObstacleMemoryConfig.{n} must be a finite number >= 0, got {v!r}")
        if not _num(self.ttl_s) or self.ttl_s <= 0.0:
            raise ValueError("ttl_s must be > 0")
        if not isinstance(self.max_items, int) or isinstance(self.max_items, bool) or self.max_items < 1:
            raise ValueError("max_items must be an int >= 1")


def _pose_ok(p) -> bool:
    try:
        return len(p) == 3 and all(_num(v) for v in p)
    except TypeError:
        return False


class ObstacleMemory:
    def __init__(self, cfg: ObstacleMemoryConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else ObstacleMemoryConfig()
        self._items: list[tuple[float, float, float]] = []   # map x, y, stamp last seen
        self._last_stamp = -math.inf

    def reset(self) -> None:
        self._items = []
        self._last_stamp = -math.inf

    def __len__(self) -> int:
        return len(self._items)

    def _body(self, item, pose):
        px, py, yaw = pose
        cy, sy = math.cos(yaw), math.sin(yaw)
        dx, dy = item[0] - px, item[1] - py
        return dx * cy + dy * sy, -dx * sy + dy * cy

    def _expired_or_off(self, item, pose, stamp) -> bool:
        c = self.cfg
        xb, yb = self._body(item, pose)
        return (stamp - item[2] > c.ttl_s) or xb <= 0.0 or abs(yb) > c.corridor_half_width + c.lateral_margin

    def observe(self, stamp, pose_xy_yaw, nearest_in_body=None, visible_free_range=None, *, blind_until: float) -> None:
        """Update with one frame. `pose_xy_yaw` None or non-finite -> memory frozen (caller must stop).
        `nearest_in_body`: (x, y) of the nearest depth-seen obstacle in the body frame, or None.
        `visible_free_range`: forward range up to which the camera saw free space (None/NaN = no evidence).
        `blind_until` is a REQUIRED keyword (no default): a caller that forgets the blind zone must get a TypeError,
        never a silent "visible and empty". A non-finite value is treated as "everything blind" (nothing dropped)."""
        if not _num(stamp) or stamp < self._last_stamp or not _pose_ok(pose_xy_yaw):
            return
        self._last_stamp = stamp
        c = self.cfg
        pose = tuple(float(v) for v in pose_xy_yaw)
        blind = float(blind_until) if _num(blind_until) else math.inf   # unknown blind zone: never claim visibility
        free = float(visible_free_range) if _num(visible_free_range) or visible_free_range == math.inf else None
        kept = []
        for it in self._items:
            if self._expired_or_off(it, pose, stamp):
                continue
            xb, _ = self._body(it, pose)
            if free is not None and xb > blind + c.blind_margin and free > xb + c.empty_margin:
                continue                                        # place in view and empty: it went away
            kept.append(it)
        self._items = kept
        if nearest_in_body is not None:
            try:
                bx, by = float(nearest_in_body[0]), float(nearest_in_body[1])
            except (TypeError, ValueError, IndexError):
                bx = by = math.nan
            if math.isfinite(bx) and math.isfinite(by):
                cy, sy = math.cos(pose[2]), math.sin(pose[2])
                new = (pose[0] + bx * cy - by * sy, pose[1] + bx * sy + by * cy, float(stamp))
                near = min((self._body(i, pose)[0] for i in self._items), default=math.inf)
                if c.max_items == 1 and bx >= near:
                    return                                      # single-obstacle mode keeps the nearer one
                self._items = [i for i in self._items if math.hypot(i[0] - new[0], i[1] - new[1]) > c.merge_radius]
                self._items.append(new)
                if len(self._items) > c.max_items:
                    self._items.sort(key=lambda i: i[2])
                    self._items = self._items[-c.max_items:]    # oldest dropped

    def blocked_distance(self, stamp, pose_xy_yaw) -> float:
        """Forward distance to the nearest remembered obstacle; inf if none. Read-only. Unknown pose with something
        remembered -> 0.0 (fail closed). An invalid/backwards stamp skips the TTL test (never forgets on it)."""
        if not self._items:
            return math.inf
        if not _pose_ok(pose_xy_yaw):
            return 0.0
        pose = tuple(float(v) for v in pose_xy_yaw)
        stamp_ok = _num(stamp) and stamp >= self._last_stamp
        c = self.cfg
        rem = math.inf
        for it in self._items:
            xb, yb = self._body(it, pose)
            if xb <= 0.0 or abs(yb) > c.corridor_half_width + c.lateral_margin:
                continue
            if stamp_ok and stamp - it[2] > c.ttl_s:
                continue
            rem = min(rem, xb)
        return rem
