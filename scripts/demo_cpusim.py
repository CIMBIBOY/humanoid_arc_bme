#!/usr/bin/env python3
"""Render the onboard stack running in the CPU simulator to a video (no Isaac, no GPU).

    python3 scripts/demo_cpusim.py --out /tmp/demo.mp4 [--seed 0] [--commands "pick up the red cup" "put it down on the left"]

Left: top-down map (table, cups, robot footprint and heading, wrist positions, the camera's view cone). Right: what the
down camera sees, and below it the live status: the sentence, the task phase, the safety level and why. The commands run
through exactly the code the robot runs: grammar -> perception -> task executor -> pipeline runner (guard) -> GR00T
planner messages -> arm IK. Kinematic only: grasp physics is checked in Isaac Lab.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation, patches  # noqa: E402

from humanoid_arc_bme.cpusim.session import make_session  # noqa: E402
from humanoid_arc_bme.cpusim.world import Box, Cylinder, World  # noqa: E402

PALETTE = {"red": (200, 30, 30), "green": (40, 170, 60), "blue": (30, 60, 200), "yellow": (230, 210, 40)}


def build_world(seed: int) -> tuple[World, list[str]]:
    rng = np.random.default_rng(seed)
    top = float(rng.uniform(0.68, 0.76))
    edge = 0.95 + float(rng.uniform(-0.03, 0.05))
    names = [str(n) for n in rng.choice(list(PALETTE), size=3, replace=False)]
    ys = [float(rng.uniform(0.0, 0.1)), float(rng.uniform(-0.12, -0.05)), float(rng.uniform(-0.28, -0.2))]
    cups = [Cylinder(edge + float(rng.uniform(0.07, 0.11)), y, 0.04, 0.12, PALETTE[n], "cup", base_z=top) for n, y in zip(names, ys)]
    return World(boxes=[Box(edge, edge + 0.7, -0.6, 0.6, top)], cylinders=cups), names


class Recorder:
    def __init__(self, s, names, out: str, fps: int = 10):
        self.s, self.names = s, names
        self.fig = plt.figure(figsize=(11, 5.6), dpi=80)
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[3, 1.1])
        self.ax = self.fig.add_subplot(gs[:, 0])
        self.cam = self.fig.add_subplot(gs[0, 1])
        self.txt = self.fig.add_subplot(gs[1, 1])
        self.writer = animation.FFMpegWriter(fps=fps, bitrate=2500)
        self.writer.setup(self.fig, out, dpi=80)
        self.sentence = ""
        self.trail: list[tuple[float, float]] = []

    def frame(self):
        s, r, w = self.s, self.s.robot, self.s.world
        ax = self.ax
        ax.clear()
        ax.set_xlim(-0.4, 1.9)
        ax.set_ylim(-0.9, 0.9)
        ax.set_aspect("equal")
        ax.set_title("top-down map (m)")
        for b in w.boxes:
            ax.add_patch(patches.Rectangle((b.x0, b.y0), b.x1 - b.x0, b.y1 - b.y0, color="#c9a37a", alpha=0.6))
        for c in w.cylinders:
            ax.add_patch(patches.Circle((c.x, c.y), c.radius, color=np.array(c.color) / 255, ec="k", zorder=5))
        self.trail.append((r.x, r.y))
        ax.plot(*zip(*self.trail), color="#888", lw=1)
        ax.add_patch(patches.Circle((r.x, r.y), 0.2, fc="#4a90d9", alpha=0.5, ec="k", zorder=6))
        ax.arrow(r.x, r.y, 0.25 * math.cos(r.yaw), 0.25 * math.sin(r.yaw), head_width=0.06, color="k", zorder=7)
        for side, col in (("left", "#d94a4a"), ("right", "#4ad97a")):
            t = r.tool_world(side)
            ax.plot(t[0], t[1], "o", color=col, ms=5, zorder=8)
        ax.grid(alpha=0.2)
        self.cam.clear()
        if s._rgb is not None:
            self.cam.imshow(s._rgb)
        self.cam.set_title("down camera")
        self.cam.axis("off")
        o = s.out
        self.txt.clear()
        self.txt.axis("off")
        if o is not None:
            seen = ", ".join(f"{x.color} {x.label}" for x in o.scene if x.label != "table") or "nothing"
            lines = [f"t = {s.t:5.1f} s    \"{self.sentence}\"", f"task: {o.status.command or '-'}  {o.status.state}  {o.status.phase}  {o.status.message}",
                     f"safety: {o.safety} {', '.join(o.reasons)}", f"sees: {seen}", f"holding: {'yes' if r.held is not None else 'no'}"]
            self.txt.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9)
        self.writer.grab_frame()

    def close(self):
        self.writer.finish()
        plt.close(self.fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="demo_cpusim.mp4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--commands", nargs="*", default=None, help="sentences; default: pick the first cup's colour, then put it down on the left")
    ap.add_argument("--every", type=int, default=2, help="render every Nth control tick (20 Hz)")
    ap.add_argument("--max-seconds", type=float, default=150.0)
    a = ap.parse_args()
    world, names = build_world(a.seed)
    s = make_session(world, pose_noise_m=0.005, seed=a.seed)
    cmds = a.commands or [f"pick up the {names[0]} cup", "put it down on the left"]
    rec = Recorder(s, names, a.out, fps=max(1, 20 // a.every))
    k = 0

    def pump(until=None, seconds=0.0):
        nonlocal k
        end = s.t + seconds
        while s.t < end:
            s.tick()
            k += 1
            if k % a.every == 0:
                rec.frame()
            if until is not None and until():
                break

    pump(seconds=1.0)
    for text in cmds:
        rec.sentence = text
        st = s.say(text)
        print(f"> {text!r}: {st.state} {st.message}")
        if st.state == "running":
            pump(until=lambda: not s.stack.executor.busy(), seconds=a.max_seconds)
            print(f"  -> {s.stack.executor.status.state}: {s.stack.executor.status.message}")
        pump(seconds=1.0)
    rec.close()
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
