"""Build the autonomy stack of a real robot from config/autonomy.yaml and attach it to the operator runtime.

`build_autonomy` takes injectable sources so the wiring is testable without hardware; with none given it constructs the
real RealSense / Livox / DLIO sources (imported lazily: pyrealsense2, rclpy). Nothing here starts a thread or opens a port.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

from humanoid_arc_bme.onboard.perception.yolo_seg import YoloSegConfig, YoloSegDetector
from humanoid_arc_bme.onboard.pipeline.interfaces import Pose
from humanoid_arc_bme.onboard.pipeline.perception.sensing import camera_pose_from_mount
from humanoid_arc_bme.onboard.pipeline.runner import PipelineRunner, RunnerConfig
from humanoid_arc_bme.onboard.pipeline.manipulation.grasp_analytic import AnalyticGraspProposer
from humanoid_arc_bme.onboard.sensors.hub import SensorHub
from humanoid_arc_bme.onboard.stack import RobotStack
from humanoid_arc_bme.onboard.task.arm_task import ArmPickPlace
from humanoid_arc_bme.onboard.task.executor import ExecutorConfig

from .autonomy import AutonomyLoop


def lidar_pose(xyz, pitch_down_deg: float) -> Pose:
    """lidar -> body: a rotation about y by the down-pitch (positive = looking down) and the mount position."""
    a = math.radians(pitch_down_deg)
    return Pose(tuple(float(v) for v in xyz), (math.cos(a / 2.0), 0.0, math.sin(a / 2.0), 0.0))


def load_config(path) -> dict:
    import yaml

    return yaml.safe_load(Path(path).expanduser().read_text())


def build_arm() -> ArmPickPlace:
    from humanoid_arc_bme.onboard.manipulation.dex3 import Dex3Hand
    from humanoid_arc_bme.onboard.manipulation.kinematics import G1ArmKinematics

    return ArmPickPlace({s: G1ArmKinematics(s) for s in ("left", "right")}, {s: Dex3Hand(s) for s in ("left", "right")})


def build_stack(cfg: dict, semantics=None, arm=None) -> RobotStack:
    r = cfg.get("runner", {})
    runner_cfg = RunnerConfig(guard_camera=r.get("guard_camera", "down"), max_range=float(r.get("max_range", 1.0)),
                              require_lidar=bool(r.get("require_lidar", False)))
    ex = cfg.get("executor", {})
    exec_cfg = ExecutorConfig(pelvis_height_m=float(cfg.get("pelvis_height_m", 0.78)), reach_x_m=float(ex.get("reach_x_m", 0.25)),
                              reach_y_m=float(ex.get("reach_y_m", 0.05)))
    if semantics is None:
        d = cfg.get("detector", {})
        semantics = YoloSegDetector(str(Path(d["model"]).expanduser()),
                                    YoloSegConfig(conf=float(d.get("conf", 0.35)), classes=frozenset(d.get("classes", ())) or None))
    runner = PipelineRunner(runner_cfg, semantics=semantics, proposer=AnalyticGraspProposer(seed=0))
    return RobotStack(semantics=semantics, arm=arm if arm is not None else build_arm(), runner=runner, executor_cfg=exec_cfg)


def build_autonomy(runtime, cfg: dict, controller=None, camera=None, lidar=None, localizer=None, hands=None, semantics=None,
                   arm=None) -> AutonomyLoop:
    """Compose stack + hub + loop around `runtime`. Pass fakes for any source to test without hardware."""
    c = cfg["camera"]
    ground_z = float(cfg.get("pelvis_height_m", 0.78)) + float(c["mount_torso_xyz"][2])
    mount = camera_pose_from_mount((c["mount_torso_xyz"][0], c["mount_torso_xyz"][1], ground_z), math.radians(float(c["pitch_down_deg"])))
    if camera is None:
        from humanoid_arc_bme.onboard.sensors.realsense import RealSenseSource

        camera = RealSenseSource(mount, width=int(c.get("width", 640)), height=int(c.get("height", 480)), fps=int(c.get("fps", 15)))
    lc = cfg.get("lidar", {})
    if lidar is None and lc.get("enabled"):
        from humanoid_arc_bme.onboard.sensors.livox import LivoxSource

        lidar = LivoxSource(lidar_pose(lc["mount_xyz"], float(lc.get("pitch_down_deg", 0.0))), topic=lc.get("topic", "/livox/lidar"))
    if localizer is None:
        from humanoid_arc_bme.onboard.sensors.dlio import DlioSource

        localizer = DlioSource(topic=cfg.get("localizer", {}).get("topic", "/dlio/odom_node/odom"))
    stack = build_stack(cfg, semantics=semantics, arm=arm)
    hub = SensorHub(camera, lidar, localizer, hands)
    return AutonomyLoop(runtime, stack, hub, controller=controller)


def describe(cfg: dict) -> dict:
    """A JSON-friendly summary for the run log: the configuration this run used."""
    return dataclasses.asdict(RunnerConfig(guard_camera=cfg.get("runner", {}).get("guard_camera", "down"))) | {"config_keys": sorted(cfg)}
