"""Path representation: a densely sampled planar curve with arc length, heading, curvature and a travel direction.

Every builder (waypoints, line, arc, clothoid, Catmull-Rom spline) produces the same `Path`: points spaced about
`ds` apart, so the controllers and the speed profile only ever see one representation. Pure Python, stdlib only (it
ships to the Jetson inside the runtime bundle).

Direction: `direction = +1` means the robot walks forward along the curve (body heading = tangent);
`direction = -1` means it walks backward (body heading = tangent + pi, commanded vx < 0). A path is one direction;
`PathPlan` holds a list of paths for out-and-back routes (a stop-and-settle at each cusp).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def wrap(a: float) -> float:
    """Angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float

    def to_local(self, px: float, py: float) -> tuple[float, float]:
        """World point -> this pose's body frame (x forward, y left)."""
        dx, dy = px - self.x, py - self.y
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return c * dx + s * dy, -s * dx + c * dy

    def flipped(self) -> "Pose2D":
        return Pose2D(self.x, self.y, wrap(self.yaw + math.pi))


@dataclass
class Path:
    """Sampled path. `s` cumulative arc length, `heading` tangent angle, `kappa` signed curvature (1/m, + = left)."""

    x: list
    y: list
    direction: int = 1
    goal_yaw: float | None = None  # body yaw wanted at the end (None: whatever the path ends with)
    s: list = field(default_factory=list)
    heading: list = field(default_factory=list)
    kappa: list = field(default_factory=list)

    def __post_init__(self):
        if len(self.x) != len(self.y) or len(self.x) < 2:
            raise ValueError("a path needs at least 2 points and equal x/y lengths")
        if self.direction not in (1, -1):
            raise ValueError("direction must be +1 (forward) or -1 (backward)")
        if not all(math.isfinite(v) for v in (*self.x, *self.y)):
            raise ValueError("path points must be finite")
        self._derive()

    def _derive(self):
        x, y = list(self.x), list(self.y)
        # drop duplicate consecutive points (zero-length segments break the tangent)
        keep_x, keep_y = [x[0]], [y[0]]
        for a, b in zip(x[1:], y[1:]):
            if math.hypot(a - keep_x[-1], b - keep_y[-1]) > 1e-9:
                keep_x.append(a)
                keep_y.append(b)
        if len(keep_x) < 2:
            raise ValueError("path has zero length")
        self.x, self.y = keep_x, keep_y
        n = len(keep_x)
        s = [0.0]
        for i in range(1, n):
            s.append(s[-1] + math.hypot(keep_x[i] - keep_x[i - 1], keep_y[i] - keep_y[i - 1]))
        seg = [math.atan2(keep_y[i + 1] - keep_y[i], keep_x[i + 1] - keep_x[i]) for i in range(n - 1)]
        heading = [seg[0]] + [seg[i - 1] + 0.5 * wrap(seg[i] - seg[i - 1]) for i in range(1, n - 1)] + [seg[-1]]
        kappa = [0.0] * n
        for i in range(1, n - 1):
            ds = 0.5 * (s[i + 1] - s[i - 1])
            kappa[i] = wrap(seg[i] - seg[i - 1]) / ds if ds > 1e-9 else 0.0
        if n > 2:
            kappa[0], kappa[-1] = kappa[1], kappa[-2]
        self.s, self.heading, self.kappa = s, [wrap(h) for h in heading], kappa

    @property
    def length(self) -> float:
        return self.s[-1]

    @property
    def end(self) -> tuple[float, float]:
        return self.x[-1], self.y[-1]

    def __len__(self):
        return len(self.x)

    def body_yaw(self, i: int) -> float:
        """Body heading the robot should have at sample i (tangent, flipped when walking backward)."""
        return wrap(self.heading[i] + (0.0 if self.direction > 0 else math.pi))

    def final_yaw(self) -> float:
        return self.goal_yaw if self.goal_yaw is not None else self.body_yaw(len(self) - 1)

    def nearest(self, px: float, py: float, start: int = 0, window: int | None = None) -> tuple[int, float, float]:
        """Closest point on the polyline at or after segment `start` (within `window` segments if given).
        Returns (segment index i, fraction t in [0,1] on segment i..i+1, arc length s of the projection)."""
        n = len(self.x)
        hi = n - 1 if window is None else min(n - 1, start + window)
        best = (float("inf"), start, 0.0)
        for i in range(max(0, start), hi):
            ax, ay, bx, by = self.x[i], self.y[i], self.x[i + 1], self.y[i + 1]
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            t = 0.0 if L2 <= 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
            qx, qy = ax + t * dx, ay + t * dy
            d2 = (px - qx) ** 2 + (py - qy) ** 2
            if d2 < best[0]:
                best = (d2, i, t)
        _, i, t = best
        i = min(i, n - 2)
        return i, t, self.s[i] + t * (self.s[i + 1] - self.s[i])

    def index_at(self, s: float) -> int:
        """Largest sample index with s[i] <= s (binary search)."""
        lo, hi = 0, len(self.s) - 1
        if s <= 0:
            return 0
        if s >= self.s[-1]:
            return hi
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.s[mid] <= s:
                lo = mid
            else:
                hi = mid
        return lo

    def point_at(self, s: float) -> tuple[float, float, float, float]:
        """(x, y, heading, kappa) at arc length s, linearly interpolated; clamped to the ends."""
        s = max(0.0, min(self.s[-1], s))
        i = self.index_at(s)
        if i >= len(self.s) - 1:
            return self.x[-1], self.y[-1], self.heading[-1], self.kappa[-1]
        L = self.s[i + 1] - self.s[i]
        t = 0.0 if L <= 0 else (s - self.s[i]) / L
        h = self.heading[i] + t * wrap(self.heading[i + 1] - self.heading[i])
        return (self.x[i] + t * (self.x[i + 1] - self.x[i]), self.y[i] + t * (self.y[i + 1] - self.y[i]), wrap(h),
                self.kappa[i] + t * (self.kappa[i + 1] - self.kappa[i]))

    def errors(self, pose: Pose2D, start: int = 0, window: int | None = None) -> tuple[float, float, float, int]:
        """(signed cross-track error e_y (+ = robot left of path), body heading error, s of projection, segment)."""
        i, t, s = self.nearest(pose.x, pose.y, start, window)
        px, py, h, _ = self.point_at(s)
        ey = -math.sin(h) * (pose.x - px) + math.cos(h) * (pose.y - py)
        want = wrap(h + (0.0 if self.direction > 0 else math.pi))
        return ey, wrap(pose.yaw - want), s, i


# ---------------------------------------------------------------- builders


def _resample(xs, ys, ds):
    """Points about ds apart along the polyline (keeps the vertices so corners stay sharp)."""
    ox, oy = [xs[0]], [ys[0]]
    for i in range(1, len(xs)):
        ax, ay, bx, by = xs[i - 1], ys[i - 1], xs[i], ys[i]
        L = math.hypot(bx - ax, by - ay)
        k = max(1, int(math.ceil(L / ds)))
        for j in range(1, k + 1):
            ox.append(ax + (bx - ax) * j / k)
            oy.append(ay + (by - ay) * j / k)
    return ox, oy


def from_waypoints(points, ds: float = 0.05, direction: int = 1, corner_radius: float = 0.0,
                   goal_yaw: float | None = None) -> Path:
    """Polyline through (x, y) waypoints. `corner_radius > 0` replaces each corner with a tangent circular arc (the
    radius shrinks where segments are too short), which removes the infinite-curvature corner."""
    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) < 2:
        raise ValueError("need at least 2 waypoints")
    if corner_radius <= 0 or len(pts) == 2:
        xs, ys = _resample([p[0] for p in pts], [p[1] for p in pts], ds)
        return Path(xs, ys, direction, goal_yaw)
    xs, ys = [pts[0][0]], [pts[0][1]]
    for k in range(1, len(pts) - 1):
        (ax, ay), (bx, by), (cx, cy) = (xs[-1], ys[-1]), pts[k], pts[k + 1]
        h1, h2 = math.atan2(by - ay, bx - ax), math.atan2(cy - by, cx - bx)
        turn = wrap(h2 - h1)
        if abs(turn) < 1e-6:
            xs.append(bx)
            ys.append(by)
            continue
        tan_len = corner_radius * math.tan(abs(turn) / 2)
        lim = 0.5 * min(math.hypot(bx - ax, by - ay), math.hypot(cx - bx, cy - by))
        r = corner_radius
        if tan_len > lim:
            r *= lim / tan_len
            tan_len = lim
        sx, sy = bx - tan_len * math.cos(h1), by - tan_len * math.sin(h1)
        xs.append(sx)
        ys.append(sy)
        sign = 1.0 if turn > 0 else -1.0
        cxr, cyr = sx - sign * r * math.sin(h1), sy + sign * r * math.cos(h1)
        steps = max(2, int(math.ceil(r * abs(turn) / ds)))
        for j in range(1, steps + 1):
            a = h1 + turn * j / steps
            xs.append(cxr + sign * r * math.sin(a))
            ys.append(cyr - sign * r * math.cos(a))
    xs.append(pts[-1][0])
    ys.append(pts[-1][1])
    xs, ys = _resample(xs, ys, ds)
    return Path(xs, ys, direction, goal_yaw)


def line(length: float, start: Pose2D = Pose2D(0, 0, 0), ds: float = 0.05, direction: int = 1) -> Path:
    """Straight segment of `length` m along the start pose's tangent (`start.yaw` is the travel direction)."""
    ex, ey = start.x + length * math.cos(start.yaw), start.y + length * math.sin(start.yaw)
    return from_waypoints([(start.x, start.y), (ex, ey)], ds, direction)


def arc(radius: float, angle: float, start: Pose2D = Pose2D(0, 0, 0), ds: float = 0.05, direction: int = 1) -> Path:
    """Circular arc; angle > 0 turns left."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    n = max(2, int(math.ceil(radius * abs(angle) / ds)))
    sign = 1.0 if angle > 0 else -1.0
    cx, cy = start.x - sign * radius * math.sin(start.yaw), start.y + sign * radius * math.cos(start.yaw)
    xs, ys = [], []
    for j in range(n + 1):
        a = start.yaw + angle * j / n
        xs.append(cx + sign * radius * math.sin(a))
        ys.append(cy - sign * radius * math.cos(a))
    return Path(xs, ys, direction)


def clothoid(length: float, kappa0: float, kappa1: float, start: Pose2D = Pose2D(0, 0, 0), ds: float = 0.02,
             direction: int = 1) -> Path:
    """Euler spiral: curvature changes linearly from kappa0 to kappa1 over `length` (midpoint integration)."""
    n = max(2, int(math.ceil(length / ds)))
    h = length / n
    x, y, th = start.x, start.y, start.yaw
    xs, ys = [x], [y]
    for j in range(n):
        km = kappa0 + (kappa1 - kappa0) * (j + 0.5) / n
        thm = th + 0.5 * h * km
        x += h * math.cos(thm)
        y += h * math.sin(thm)
        th += h * km
        xs.append(x)
        ys.append(y)
    return Path(xs, ys, direction)


def catmull_rom(points, ds: float = 0.05, direction: int = 1, alpha: float = 0.5, goal_yaw: float | None = None) -> Path:
    """Centripetal Catmull-Rom spline through the waypoints (passes through every point, no overshoot loops)."""
    P = [(float(p[0]), float(p[1])) for p in points]
    if len(P) < 2:
        raise ValueError("need at least 2 waypoints")
    if len(P) == 2:
        return from_waypoints(P, ds, direction, goal_yaw=goal_yaw)
    P = [(2 * P[0][0] - P[1][0], 2 * P[0][1] - P[1][1])] + P + [(2 * P[-1][0] - P[-2][0], 2 * P[-1][1] - P[-2][1])]
    xs, ys = [P[1][0]], [P[1][1]]

    def tj(ti, a, b):
        return ti + max(1e-9, math.hypot(b[0] - a[0], b[1] - a[1])) ** alpha

    for k in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[k - 1], P[k], P[k + 1], P[k + 2]
        t0 = 0.0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)
        steps = max(2, int(math.ceil(math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / ds)))
        for j in range(1, steps + 1):
            t = t1 + (t2 - t1) * j / steps
            pt = []
            for d in (0, 1):
                a1 = (t1 - t) / (t1 - t0) * p0[d] + (t - t0) / (t1 - t0) * p1[d]
                a2 = (t2 - t) / (t2 - t1) * p1[d] + (t - t1) / (t2 - t1) * p2[d]
                a3 = (t3 - t) / (t3 - t2) * p2[d] + (t - t2) / (t3 - t2) * p3[d]
                b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
                b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
                pt.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
            xs.append(pt[0])
            ys.append(pt[1])
    return Path(xs, ys, direction, goal_yaw)


def concat(*paths: Path) -> Path:
    """Join same-direction paths end to start (the next path is translated so it starts where the last ended)."""
    if not paths or len({p.direction for p in paths}) != 1:
        raise ValueError("concat needs paths of one direction")
    xs, ys = list(paths[0].x), list(paths[0].y)
    for p in paths[1:]:
        ox, oy = xs[-1] - p.x[0], ys[-1] - p.y[0]
        xs += [v + ox for v in p.x[1:]]
        ys += [v + oy for v in p.y[1:]]
    return Path(xs, ys, paths[0].direction, paths[-1].goal_yaw)


def end_pose(p: Path) -> Pose2D:
    return Pose2D(p.x[-1], p.y[-1], p.heading[-1])


@dataclass
class PathPlan:
    """A route of one or more single-direction paths; the follower stops and settles at each junction (cusp)."""

    segments: list

    def __post_init__(self):
        if not self.segments or not all(isinstance(p, Path) for p in self.segments):
            raise ValueError("PathPlan needs at least one Path")


# ---------------------------------------------------------------- the benchmark paths


def standard_paths(ds: float = 0.05) -> dict:
    """The six test paths of the brief (+ an out-and-back cusp). All start at the origin facing +x."""
    o = Pose2D(0, 0, 0)
    s1 = arc(1.0, math.pi / 2, o, ds)
    s2 = arc(1.0, -math.pi / 2, end_pose(s1), ds)
    s_curve = concat(line(1.0, o, ds), s1, s2, line(1.0, end_pose(s2), ds))
    circle = arc(1.0, 2 * math.pi - 1e-3, o, ds)
    a1 = arc(1.0, 2 * math.pi - 1e-3, o, ds)
    a2 = arc(1.0, -(2 * math.pi - 1e-3), end_pose(a1), ds)
    eight = concat(a1, a2)
    # walk backward 2 m: the path runs toward -x while the body keeps facing +x
    back = line(2.0, Pose2D(0, 0, math.pi), ds, direction=-1)
    corner = from_waypoints([(0, 0), (2.0, 0), (2.0, 2.0)], ds)
    return {
        "straight": PathPlan([line(4.0, o, ds)]),
        "backward": PathPlan([back]),
        "s_curve": PathPlan([s_curve]),
        "circle": PathPlan([circle]),
        "figure_eight": PathPlan([eight]),
        "sharp_corner": PathPlan([corner]),
        "out_and_back": PathPlan([line(2.0, o, ds), line(2.0, Pose2D(2.0, 0, math.pi), ds, direction=-1)]),
    }


def split_at_corners(path: Path, angle: float = 0.8, within: float = 0.15) -> list:
    """Split a path where the heading turns by more than `angle` rad within `within` m (a polyline corner). A walking
    robot stops, turns in place and walks on (stop-turn-go) instead of crawling round a near-zero-radius curve that a
    gait deadband would stall on. Returns a list of paths (the input alone when there is no corner)."""
    n = len(path)
    cuts, i = [], 1
    while i < n - 1:
        j = i
        while j < n - 1 and path.s[j] - path.s[i] < within:
            j += 1
        turn = wrap(path.heading[j] - path.heading[max(0, i - 1)])
        if abs(turn) > angle:
            # the vertex: the sample with the largest |kappa| in the window
            k = max(range(i, j + 1), key=lambda m: abs(path.kappa[m]))
            if 0 < k < n - 1 and (not cuts or k - cuts[-1] > 1):
                cuts.append(k)
            i = j + 1
        else:
            i += 1
    if not cuts:
        return [path]
    out, a = [], 0
    for k in cuts + [n - 1]:
        if k - a >= 1:
            gy = path.goal_yaw if k == n - 1 else None
            out.append(Path(path.x[a:k + 1], path.y[a:k + 1], path.direction, gy))
        a = k
    return out
