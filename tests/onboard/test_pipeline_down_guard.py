"""Guard profile for a robot with only a down-tilted depth camera (no neck camera): `RunnerConfig(guard_camera="down")`.

SIM results (tests/onboard/_depth_render.py), not MEASURED. The mount (0.43 m high, 50 deg down, 0.06 m forward) is the
placeholder of the sim down camera; the floor look-ahead of that mount is ~1.3 m, so max_range is set to 1.0 m.
"""

from __future__ import annotations

import dataclasses

import pytest

from _depth_render import CFG, Box, Hole, noisy, render
from humanoid_arc_bme.onboard.pipeline.interfaces import DepthFrame, Intrinsics, LocalizerEstimate, yaw_pose
from humanoid_arc_bme.onboard.pipeline.perception.sensing import camera_pose_from_mount
from humanoid_arc_bme.onboard.pipeline.runner import OK, STOP, PipelineRunner, RunnerConfig, TickInput

DT = 0.1
VX = 0.2
DOWN = dataclasses.replace(CFG, cam_height=0.43, cam_pitch=0.873, cam_forward=0.06, min_range=0.05, max_range=1.0)
K = Intrinsics(DOWN.fx, DOWN.fy, DOWN.cx, DOWN.cy, DOWN.width, DOWN.height)
MOUNT = camera_pose_from_mount((DOWN.cam_forward, 0.0, DOWN.cam_height), DOWN.cam_pitch)
CFG_DOWN = RunnerConfig(guard_camera="down", max_range=1.0)


def down_frame(t, boxes=(), holes=(), seed=0, x=0.0):
    sb = [Box(b.x0 - x, b.x1 - x, b.y0, b.y1, b.z1) for b in boxes]
    sh = [Hole(h.x0 - x, h.x1 - x, h.y0, h.y1, h.depth) for h in holes]
    return DepthFrame(t, noisy(render(DOWN, sb, sh), seed), K, MOUNT, camera="down")


def tick(r, t, frame, x=0.0, cmd=(VX, 0.0, 0.0), forward=None):
    pose = LocalizerEstimate(t, yaw_pose(x, 0.0, 0.0, 0.0), (1e-4,) * 6)
    return r.tick(TickInput(t, forward, frame, pose, 0.0, 0.0, command=cmd))


def drive(boxes=(), holes=(), n=80, appear_tick=0):
    r = PipelineRunner(CFG_DOWN)
    x, outs = 0.0, []
    for i in range(n):
        t = i * DT
        on = i >= appear_tick
        out = tick(r, t, down_frame(t, boxes if on else (), holes if on else (), seed=i, x=x), x=x)
        x += out.command_out[0] * DT
        outs.append(out)
    return r, x, outs


def test_free_floor_with_only_the_down_camera_walks():
    _, x, outs = drive(n=40)
    assert all(o.command_out == (VX, 0.0, 0.0) for o in outs[1:]), [(o.command_out, o.safety.reasons) for o in outs[:4]]
    assert x > 0.7


def test_forward_frame_is_ignored_in_down_mode():
    r = PipelineRunner(CFG_DOWN)
    out = tick(r, 0.0, down_frame(0.0), forward=None)
    assert "no_forward_depth" not in out.safety.reasons


def test_missing_down_frame_stops_with_its_own_reason():
    r = PipelineRunner(CFG_DOWN)
    out = tick(r, 0.0, None)
    assert out.command_out == (0.0, 0.0, 0.0) and out.safety.level == STOP
    assert "no_down_depth" in out.safety.reasons


def test_forward_mode_still_reports_no_forward_depth():
    r = PipelineRunner()
    out = tick(r, 0.0, down_frame(0.0))
    assert "no_forward_depth" in out.safety.reasons and out.command_out == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("seed", range(5))
def test_box_ahead_stops_the_robot_before_it(seed):
    box = Box(0.9, 1.2, -0.3, 0.3, 0.3)
    _, x, outs = drive(boxes=(box,), n=120, appear_tick=0)
    assert abs(outs[-1].command_out[0]) < 0.05, outs[-1].command_out
    assert outs[-1].safety.level != OK
    assert x < box.x0 - 0.2, x


def test_drop_ahead_stops_the_robot():
    hole = Hole(0.9, 2.0, -0.6, 0.6, 0.4)
    _, x, outs = drive(holes=(hole,), n=120)
    assert outs[-1].command_out == (0.0, 0.0, 0.0) and outs[-1].safety.level == STOP
    assert x < hole.x0 - 0.2, x


def test_floor_the_camera_cannot_see_reads_as_a_drop_edge_not_as_clear():
    """max_range (3 m) beyond the camera's real floor look-ahead (~1.3 m): the unseen floor is reported as the end of
    verified ground, so the guard never vouches for it. At 0.2 m/s that edge is far enough not to slow the robot."""
    r = PipelineRunner(RunnerConfig(guard_camera="down", max_range=3.0))
    outs = [tick(r, i * DT, down_frame(i * DT, seed=i)) for i in range(10)]
    cl = outs[-1].clearance
    assert cl.ground_ok and 0.8 < cl.distance_negative < 1.6, cl
    assert outs[-1].command_out == (VX, 0.0, 0.0)
    # at a speed whose stopping distance exceeds the verified floor, the cap would bite
    assert r.guard.cap_for(cl.distance_negative) > VX


def test_bad_guard_camera_rejected():
    with pytest.raises(ValueError, match="guard_camera"):
        PipelineRunner(RunnerConfig(guard_camera="rear"))
