"""The shipped autonomy config builds the stack, and the builder wires fakes into a working loop (no hardware)."""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from humanoid_arc_bme.onboard.pipeline.interfaces import DepthFrame, Intrinsics, LocalizerEstimate, Pose, yaw_pose
from humanoid_arc_bme.onboard.pipeline.stubs import NullSemantics
from humanoid_arc_bme.onboard.runtime.autonomy_main import build_autonomy, build_stack, lidar_pose, load_config
from tests.onboard.test_operator_runtime import make

CONFIG = Path(__file__).resolve().parents[2] / "config" / "autonomy.yaml"
K = Intrinsics(100.0, 100.0, 9.5, 7.5, 20, 16)


class Cam:
    def latest(self):
        return np.zeros((16, 20, 3), np.uint8), DepthFrame(1.0, np.full((16, 20), 1.0, np.float32), K, Pose((0, 0, 1.2)))


class Loc:
    def estimate(self, stamp):
        return LocalizerEstimate(stamp, yaw_pose(0.0, 0.0, 0.0, 0.0), (1e-4,) * 6)


def test_shipped_config_parses_and_has_the_sections_the_builder_reads():
    cfg = load_config(CONFIG)
    for key in ("pelvis_height_m", "camera", "runner", "executor", "gateway", "groot", "detector"):
        assert key in cfg
    assert cfg["runner"]["guard_camera"] == "down" and cfg["gateway"]["host"] == "127.0.0.1"


def test_stack_is_built_with_the_down_camera_profile_and_the_configured_reach():
    st = build_stack(load_config(CONFIG), semantics=NullSemantics(), arm=None)
    assert st.runner.cfg.guard_camera == "down" and st.executor.cfg.reach_x_m == 0.25


def test_lidar_pose_rotates_about_y_by_the_down_pitch():
    p = lidar_pose((0.1, 0.0, 0.5), 60.0)
    assert p.position == (0.1, 0.0, 0.5)
    fwd = p.rotation_matrix() @ np.array([1.0, 0.0, 0.0])
    assert math.isclose(fwd[2], -math.sin(math.radians(60)), abs_tol=1e-9) and fwd[0] > 0      # +x pitched DOWN


def test_builder_wires_fakes_into_a_loop_that_reports_missing_hand_feedback(tmp_path):
    rt, clock, fake = make(tmp_path)
    loop = build_autonomy(rt, load_config(CONFIG), camera=Cam(), localizer=Loc(), hands=None, semantics=NullSemantics(), arm=SimpleNamespace())
    rt.tick()
    loop.clock = clock
    assert loop.hub.frame(1.0, rt.state) is not None
    assert loop.hub.frame(1.0, rt.state).joints is None                 # no hand feed: arm tasks will be refused with a reason
    loop.close()
