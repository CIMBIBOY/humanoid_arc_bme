"""Path-tracking laws behind one interface: `compute(ctx) -> (vx, wz)` (raw, before the shared shaper and stop rules).

All laws work for a unicycle (vx, wz); vy is left at 0 (the G1 can side-step, but sideways walking is slower and
none of these laws need it). Backward walking is handled once, here: for a `direction = -1` path the law sees a
"virtual" robot whose heading is flipped by pi and that drives forward; its forward speed is then negated. For a
unicycle this is exact: the virtual robot has the same position and the same yaw rate.

Options (ranked in docs/plan/navstack.md by tools/sim/nav_bench.py):
- `PurePursuit`: constant speed, fixed lookahead, kappa = 2 y_L / L^2 (Coulter 1992).
- `RegulatedPurePursuit`: Nav2's RPP rules re-implemented: velocity-scaled lookahead, curvature-regulated speed,
  approach slow-down, rotate-to-heading when the carrot is far off the nose.
- `Stanley`: heading error + atan(k e / (k_s + v)) mapped to a yaw rate, plus the path's feed-forward curvature.
- `ProfileTracker`: time-optimal curvature-limited speed profile (limits.speed_profile) + Kanayama/Samson-style
  feedback on cross-track and heading error.
- `MPPI`: sampling MPC over a unicycle with first-order velocity lag and the accel limits, tracking the profile's
  time-parameterised reference. Pure Python (K samples x H steps); deterministic given its seed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .limits import NavLimits, speed_profile
from .path import Path, Pose2D


@dataclass
class TrackContext:
    """What a controller gets each tick. `pose` is the odometry estimate; `s` its projection on the path;
    `ey`/`eyaw` the signed cross-track and body-heading errors; `v_last`/`w_last` the last shaped command."""

    pose: Pose2D
    path: Path
    s: float
    ey: float
    eyaw: float
    v_last: float
    w_last: float
    dt: float
    limits: NavLimits
    profile: list = field(default_factory=list)

    @property
    def remaining(self) -> float:
        return max(0.0, self.path.length - self.s)

    def vpose(self) -> Pose2D:
        """The virtual forward-driving pose (flipped when walking backward)."""
        return self.pose if self.path.direction > 0 else self.pose.flipped()

    def v_ref(self, s: float | None = None) -> float:
        s = self.s if s is None else s
        if not self.profile:
            return self.limits.vmax(self.path.direction)
        i = self.path.index_at(s)
        if i >= len(self.profile) - 1:
            return self.profile[-1]
        s0, s1 = self.path.s[i], self.path.s[i + 1]
        t = 0.0 if s1 <= s0 else (s - s0) / (s1 - s0)
        return self.profile[i] + t * (self.profile[i + 1] - self.profile[i])

    def carrot(self, ahead: float) -> tuple[float, float]:
        """Point `ahead` metres past the projection; beyond the end it continues along the final tangent (so the
        lookahead never collapses onto the goal and the heading stays defined)."""
        s = self.s + ahead
        p = self.path
        if s <= p.length:
            x, y, _, _ = p.point_at(s)
            return x, y
        h = p.heading[-1]
        extra = s - p.length
        return p.x[-1] + extra * math.cos(h), p.y[-1] + extra * math.sin(h)


class Controller:
    name = "base"

    def reset(self) -> None:
        pass

    def compute(self, ctx: TrackContext) -> tuple[float, float]:
        """Return the raw body command (vx, wz); vx < 0 on backward paths."""
        v, w = self._forward(ctx)
        return (v if ctx.path.direction > 0 else -v), w

    def _forward(self, ctx: TrackContext) -> tuple[float, float]:
        raise NotImplementedError


def _pp_curvature(vpose: Pose2D, cx: float, cy: float) -> tuple[float, float, float]:
    lx, ly = vpose.to_local(cx, cy)
    L2 = lx * lx + ly * ly
    return (2.0 * ly / L2 if L2 > 1e-9 else 0.0), lx, ly


@dataclass
class PurePursuit(Controller):
    lookahead: float = 0.4  # m
    name: str = "pure_pursuit"

    def _forward(self, ctx):
        k, lx, ly = _pp_curvature(ctx.vpose(), *ctx.carrot(self.lookahead))
        v = ctx.limits.vmax(ctx.path.direction)
        if lx < 0:  # carrot behind: turn toward it, do not walk away from it
            return 0.0, math.copysign(ctx.limits.wz_max, ly if abs(ly) > 1e-9 else 1.0)
        return v, v * k


@dataclass
class RegulatedPurePursuit(Controller):
    lookahead_time: float = 1.5  # s (velocity-scaled lookahead L = t * |v|)
    min_lookahead: float = 0.3
    max_lookahead: float = 0.9
    regulated_min_radius: float = 0.9  # m: below this turning radius the speed is scaled by R / r_min
    approach_dist: float = 0.6  # m: slow down linearly over the last approach_dist metres
    min_approach_v: float = 0.05
    rotate_min_angle: float = 0.785  # rad: carrot this far off the nose -> rotate in place
    rotate_wz: float | None = None  # None -> limits.wz_max
    name: str = "regulated_pure_pursuit"

    def _forward(self, ctx):
        L = ctx.limits
        vmax = L.vmax(ctx.path.direction)
        Ld = max(self.min_lookahead, min(self.max_lookahead, self.lookahead_time * abs(ctx.v_last)))
        cx, cy = ctx.carrot(Ld)
        k, lx, ly = _pp_curvature(ctx.vpose(), cx, cy)
        ang = math.atan2(ly, lx)
        if abs(ang) > self.rotate_min_angle:
            return 0.0, math.copysign(self.rotate_wz or L.wz_max, ang)
        v = vmax
        if abs(k) > 1e-9:
            R = 1.0 / abs(k)
            if R < self.regulated_min_radius:
                v *= R / self.regulated_min_radius
        if ctx.remaining < self.approach_dist:
            v = max(min(v, self.min_approach_v), v * ctx.remaining / self.approach_dist)
        return v, v * k


@dataclass
class Stanley(Controller):
    k: float = 1.5  # 1/s gain on cross-track error inside atan
    k_soft: float = 0.1  # m/s softening speed
    k_delta: float = 1.2  # 1/s: yaw rate per rad of steering angle
    name: str = "stanley"

    def _forward(self, ctx):
        v = ctx.v_ref()
        # in the virtual frame ey keeps its sign; heading error is relative to the travel tangent
        _, _, _, kref = ctx.path.point_at(ctx.s)
        delta = -ctx.eyaw - math.atan2(self.k * ctx.ey, self.k_soft + abs(ctx.v_last))
        v *= max(0.0, math.cos(ctx.eyaw))
        return v, v * kref + self.k_delta * delta


@dataclass
class ProfileTracker(Controller):
    k_y: float = 4.0  # 1/m^2
    k_th: float = 2.5  # 1/m (scaled by speed)
    k_th0: float = 0.6  # 1/s (keeps heading correction alive at low speed)
    name: str = "profile_tracker"

    def _forward(self, ctx):
        v = ctx.v_ref()
        _, _, _, kref = ctx.path.point_at(ctx.s)
        ey, eth = ctx.ey, ctx.eyaw
        v = v * max(0.0, math.cos(eth)) / (1.0 + 2.0 * abs(ey))
        sinc = math.sin(eth) / eth if abs(eth) > 1e-6 else 1.0
        denom = 1.0 - kref * ey
        ff = v * kref * math.cos(eth) / denom if denom > 0.2 else v * kref
        w = ff - self.k_y * v * ey * sinc - (self.k_th * v + self.k_th0) * eth
        return v, w


@dataclass
class MPPI(Controller):
    """Sampling MPC (MPPI) over x, y, yaw, v, w with first-order lag (tau_v, tau_w) and the accel/rate limits.
    The cost tracks the reference built from the speed profile; controls are clipped to [0, vmax] x [-wz, wz]."""

    samples: int = 64
    horizon: int = 15
    dt_model: float = 0.1
    tau_v: float = 0.25  # s, the lag the model assumes (the real one is UNKNOWN until B4.3)
    tau_w: float = 0.2
    sigma_v: float = 0.05
    sigma_w: float = 0.2
    lam: float = 0.05  # temperature
    q_lat: float = 60.0  # cross-track error w.r.t. the reference point's tangent
    q_lon: float = 5.0  # along-track error (lagging the reference is cheap; cutting the curve is not)
    q_yaw: float = 2.0
    q_v: float = 20.0  # speed tracking toward the profile
    r_dv: float = 0.5
    r_dw: float = 0.5
    q_term: float = 3.0
    seed: int = 0
    name: str = "mppi"

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.U = [[0.0, 0.0] for _ in range(self.horizon)]

    def reset(self):
        self.rng = random.Random(self.seed)
        self.U = [[0.0, 0.0] for _ in range(self.horizon)]

    def _forward(self, ctx):
        L = ctx.limits
        vmax, wmax = L.vmax(ctx.path.direction), L.wz_max
        H, dt = self.horizon, self.dt_model
        # reference: integrate the profile forward from the projection
        ref, s = [], ctx.s
        for _ in range(H):
            s = min(ctx.path.length, s + max(ctx.v_ref(s), 0.03) * dt)
            x, y, h, _ = ctx.path.point_at(s)
            ref.append((x, y, h, ctx.v_ref(s)))
        vp = ctx.vpose()
        v0 = abs(ctx.v_last)
        w0 = ctx.w_last
        av, aw = 1.0 - math.exp(-dt / self.tau_v), 1.0 - math.exp(-dt / self.tau_w)
        dvmax, dwmax = L.a_max * dt, L.alpha_max * dt
        costs, seqs = [], []
        for k in range(self.samples):
            x, y, th, v, w = vp.x, vp.y, vp.yaw, v0, w0
            pu, pw = v0, w0
            J = 0.0
            seq = []
            for t in range(H):
                if k == 0:  # sample 0 = the warm-start sequence itself (noise-free)
                    uv, uw = self.U[t]
                else:
                    uv = self.U[t][0] + self.rng.gauss(0.0, self.sigma_v)
                    uw = self.U[t][1] + self.rng.gauss(0.0, self.sigma_w)
                uv = min(vmax, max(0.0, min(pu + dvmax, max(pu - L.d_max * dt, uv))))
                uw = min(wmax, max(-wmax, min(pw + dwmax, max(pw - dwmax, uw))))
                seq.append((uv, uw))
                J += self.r_dv * (uv - pu) ** 2 / dt + self.r_dw * (uw - pw) ** 2 / dt
                pu, pw = uv, uw
                v += av * (uv - v)
                w += aw * (uw - w)
                th += w * dt
                x += v * math.cos(th) * dt
                y += v * math.sin(th) * dt
                rx, ry, rh, rv = ref[t]
                ch, sh = math.cos(rh), math.sin(rh)
                e_lon = ch * (x - rx) + sh * (y - ry)
                e_lat = -sh * (x - rx) + ch * (y - ry)
                wgt = self.q_term if t == H - 1 else 1.0
                J += wgt * (self.q_lat * e_lat * e_lat + self.q_lon * e_lon * e_lon + self.q_yaw * (1.0 - math.cos(th - rh))
                            + self.q_v * (v - rv) ** 2) * dt
            costs.append(J)
            seqs.append(seq)
        jmin = min(costs)
        ws = [math.exp(-(c - jmin) / self.lam) for c in costs]
        tot = sum(ws)
        newU = []
        for t in range(H):
            uv = sum(w_ * s_[t][0] for w_, s_ in zip(ws, seqs)) / tot
            uw = sum(w_ * s_[t][1] for w_, s_ in zip(ws, seqs)) / tot
            newU.append([uv, uw])
        out = newU[0]
        self.U = newU[1:] + [newU[-1]]
        return out[0], out[1]


CONTROLLERS = {
    "pure_pursuit": PurePursuit,
    "regulated_pure_pursuit": RegulatedPurePursuit,
    "stanley": Stanley,
    "profile_tracker": ProfileTracker,
    "mppi": MPPI,
}
ALIASES = {"pp": "pure_pursuit", "rpp": "regulated_pure_pursuit", "tracker": "profile_tracker"}


def make_controller(name: str, **kw) -> Controller:
    name = ALIASES.get(name, name)
    if name not in CONTROLLERS:
        raise ValueError(f"unknown controller {name!r}; options: {sorted(CONTROLLERS)}")
    return CONTROLLERS[name](**kw)


__all__ = ["TrackContext", "Controller", "PurePursuit", "RegulatedPurePursuit", "Stanley", "ProfileTracker", "MPPI",
           "CONTROLLERS", "make_controller", "speed_profile"]
