"""CPU simulation harness: a walking-robot stand-in for comparing the controllers.

Plant (`WalkerPlant`): a unicycle whose actual (v, w) follow the command through
- a transport delay (0-100 ms, `delay_s`: command -> gait reacts),
- a first-order lag (`tau_v`, `tau_w`: the gait needs steps to change speed),
- an optional deadband (|v| < `deadband_v` or |w| < `deadband_w` -> the gait does not move; the AGILE policy stood
  still for small commands, the vendor gait's deadband is UNKNOWN),
- gait sway: a lateral velocity and yaw-rate oscillation at the step frequency while walking (a humanoid pelvis
  sways side to side every step; amplitude is a guess, not measured).
Odometry (`DriftOdometry`): integrates the TRUE body velocity with a scale error, a yaw-rate bias and white noise, so
the controller sees a drifting estimate while the metrics use the true pose.

None of these numbers are measurements of the G1: they are a test bench for relative ranking. B4.3 replaces them.
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass

from .controllers import make_controller
from .follower import FollowerConfig, PathFollower
from .limits import NavLimits
from .odometry import Odometry, TruthOdometry
from .path import PathPlan, Pose2D, wrap


@dataclass(frozen=True)
class PlantConfig:
    name: str = "nominal"
    dt: float = 0.005
    delay_s: float = 0.05
    tau_v: float = 0.25
    tau_w: float = 0.2
    deadband_v: float = 0.0
    deadband_w: float = 0.0
    sway_vy: float = 0.0  # m/s amplitude of lateral sway while walking
    sway_wz: float = 0.0  # rad/s amplitude of yaw sway while walking
    step_hz: float = 1.8
    odom_scale: float = 0.0  # relative speed error of the odometry (+0.05 = reads 5% long)
    odom_yaw_bias: float = 0.0  # rad/s
    odom_noise: float = 0.0  # std of white noise on the integrated velocity (m/s, rad/s)
    use_truth_odom: bool = True


CONDITIONS = {
    # truth odometry: these three rank the controllers (tracking under lag / delay / deadband / sway)
    "ideal": PlantConfig("ideal", delay_s=0.0, tau_v=0.05, tau_w=0.05),
    "nominal": PlantConfig("nominal", delay_s=0.05, tau_v=0.25, tau_w=0.2, sway_vy=0.03, sway_wz=0.05),
    "laggy": PlantConfig("laggy", delay_s=0.1, tau_v=0.4, tau_w=0.3, deadband_v=0.03, deadband_w=0.05,
                         sway_vy=0.05, sway_wz=0.1),
    # nominal plant, drifting dead-reckoning odometry: shows what stage 1 (no external localization) can achieve
    "drift": PlantConfig("drift", delay_s=0.05, tau_v=0.25, tau_w=0.2, sway_vy=0.03, sway_wz=0.05, odom_scale=0.05,
                         odom_yaw_bias=0.005, odom_noise=0.01, use_truth_odom=False),
}
RANKING_CONDITIONS = ("ideal", "nominal", "laggy")


class WalkerPlant:
    def __init__(self, cfg: PlantConfig, pose: Pose2D = Pose2D(0, 0, 0), seed: int = 0):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.x, self.y, self.yaw = pose.x, pose.y, pose.yaw
        self.v = self.w = self.vy = 0.0
        self.t = 0.0
        self.phase = self.rng.uniform(0, 2 * math.pi)
        n = int(round(cfg.delay_s / cfg.dt))
        self.queue = deque([(0.0, 0.0)] * n)
        self.body_vel = (0.0, 0.0, 0.0)

    def pose(self) -> Pose2D:
        return Pose2D(self.x, self.y, self.yaw)

    def step(self, cmd_v: float, cmd_w: float):
        c, dt = self.cfg, self.cfg.dt
        self.queue.append((cmd_v, cmd_w))
        uv, uw = self.queue.popleft()
        if abs(uv) < c.deadband_v:
            uv = 0.0
        if abs(uw) < c.deadband_w and abs(uv) < 1e-6:
            uw = 0.0
        self.v += (1 - math.exp(-dt / max(c.tau_v, 1e-6))) * (uv - self.v)
        self.w += (1 - math.exp(-dt / max(c.tau_w, 1e-6))) * (uw - self.w)
        walking = min(1.0, abs(self.v) / 0.05)
        ph = 2 * math.pi * c.step_hz * self.t + self.phase
        vy = c.sway_vy * walking * math.sin(ph)
        w = self.w + c.sway_wz * walking * math.cos(ph)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (cy * self.v - sy * vy) * dt
        self.y += (sy * self.v + cy * vy) * dt
        self.yaw = wrap(self.yaw + w * dt)
        self.body_vel = (self.v, vy, w)
        self.t += dt


class DriftOdometry(Odometry):
    """Integrates the plant's true body velocity with scale error, yaw bias and noise."""

    def __init__(self, plant: WalkerPlant, seed: int = 0):
        self.plant, self.rng = plant, random.Random(seed + 1000)
        p = plant.pose()
        self.x, self.y, self.yaw = p.x, p.y, p.yaw

    def integrate(self, dt: float):
        c = self.plant.cfg
        vx, vy, w = self.plant.body_vel
        k = 1.0 + c.odom_scale
        vx = k * vx + self.rng.gauss(0, c.odom_noise)
        vy = k * vy + self.rng.gauss(0, c.odom_noise)
        w = w + c.odom_yaw_bias + self.rng.gauss(0, c.odom_noise)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (cy * vx - sy * vy) * dt
        self.y += (sy * vx + cy * vy) * dt
        self.yaw = wrap(self.yaw + w * dt)

    def pose(self, now):
        return Pose2D(self.x, self.y, self.yaw)


def _rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else 0.0


def run_episode(plan: PathPlan, controller: str, cond: PlantConfig, limits: NavLimits = NavLimits(), seed: int = 0,
                control_hz: float = 20.0, controller_kw: dict | None = None, trace: bool = False) -> dict:
    """One episode; metrics are computed on the TRUE pose against the plan geometry."""
    plant = WalkerPlant(cond, seed=seed)
    odom = TruthOdometry(plant.pose) if cond.use_truth_odom else DriftOdometry(plant, seed)
    ctrl = make_controller(controller, **(controller_kw or {}))
    if hasattr(ctrl, "seed"):
        ctrl.seed = seed
        ctrl.reset()
    fol = PathFollower(odom, ctrl, limits, config=FollowerConfig())
    fol.follow(plan, 0.0)
    plan = fol.plan  # possibly split at corners
    period = 1.0 / control_hz
    steps_per = max(1, int(round(period / cond.dt)))
    cmd = (0.0, 0.0)
    t = 0.0
    cte, herr, cmds, comp, tr = [], [], [], [], []
    seg_progress = [0] * len(plan.segments)
    t_done = None
    max_past = -math.inf
    final = plan.segments[-1]
    fx, fy = final.end
    fh = final.heading[-1]
    k = 0
    while True:
        if k % steps_per == 0:
            c0 = time.perf_counter()
            bv = fol.step(t)
            comp.append(time.perf_counter() - c0)
            cmd = (bv.vx, bv.vy, bv.wz)
            cmds.append((cmd[0], cmd[2]))
            if fol.state == "SUCCEEDED" and t_done is None:
                t_done = t
            if fol.state in ("SUCCEEDED", "FAILED", "STOPPED") and fol.is_still and plant.v ** 2 < 1e-4:
                break
            if t > fol._timeout + 5:
                break
            # metrics on the true pose (segment the follower is on)
            tp = plant.pose()
            si = fol.seg_i
            seg = plan.segments[si]
            ds = seg.length / max(1, len(seg) - 1)
            ey, eyaw, s, i = seg.errors(tp, max(0, seg_progress[si] - 2), int(1.0 / ds) + 2)
            seg_progress[si] = max(seg_progress[si], i)
            if fol.state in ("TRACK", "ALIGN"):
                cte.append(abs(ey))
                if fol.state == "TRACK":
                    herr.append(abs(eyaw))
            if si == len(plan.segments) - 1 and s > seg.length - 0.5:
                max_past = max(max_past, (tp.x - fx) * math.cos(fh) + (tp.y - fy) * math.sin(fh))
            if trace:
                tr.append((t, tp.x, tp.y, tp.yaw, cmd[0], cmd[2], fol.state))
        plant.step(cmd[0], cmd[2])
        if isinstance(odom, DriftOdometry):
            odom.integrate(cond.dt)
        t += cond.dt
        k += 1
    tp = plant.pose()
    T = period
    acc = [(cmds[i][0] - cmds[i - 1][0]) / T for i in range(1, len(cmds))]
    jerk = [(acc[i] - acc[i - 1]) / T for i in range(1, len(acc))]
    wacc = [(cmds[i][1] - cmds[i - 1][1]) / T for i in range(1, len(cmds))]
    reversals = sum(1 for i in range(1, len(cmds)) if cmds[i][1] * cmds[i - 1][1] < 0 and abs(cmds[i][1]) > 0.02)
    out = {
        "controller": ctrl.name, "condition": cond.name, "seed": seed,
        "success": fol.state == "SUCCEEDED", "state": fol.state, "reason": fol.reason,
        "time_s": t_done if t_done is not None else t,
        "cte_rms": _rms(cte), "cte_max": max(cte) if cte else 0.0,
        "heading_rms": _rms(herr),
        "final_err": math.hypot(tp.x - fx, tp.y - fy),
        "overshoot": max(0.0, max_past) if max_past > -math.inf else 0.0,
        "jerk_rms": _rms(jerk), "acc_rms": _rms(acc), "wacc_rms": _rms(wacc), "w_reversals": reversals,
        "compute_ms_mean": 1000 * sum(comp) / len(comp), "compute_ms_max": 1000 * max(comp),
    }
    if trace:
        out["trace"] = tr
    return out


def describe_conditions() -> dict:
    return {k: asdict(v) for k, v in CONDITIONS.items()}
