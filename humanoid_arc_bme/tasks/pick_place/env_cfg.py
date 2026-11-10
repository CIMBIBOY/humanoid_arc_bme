"""Pick-and-place at one table, built on IsaacLab's native locomanipulation stack.

Upper body: Pink IK on both wrists (waist included) plus direct targets for
the 14 tri-hand joints. What carries it is a base plug-in chosen by
`robot_base` (robots/bases.py): `agile_legs` (default; the G1 standing on
NVIDIA's AGILE lower-body policy, the layout of all recorded datasets) or
`fixed` (fixed pelvis, upper body only). Everything the env
loads is local (robots/assets.py, setup.sh): nothing is fetched from Nucleus
at runtime.
"""

import os

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.manager_based_env_cfg import DefaultEventManagerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import G1_29DOF_CFG
from isaaclab_tasks.manager_based.locomanipulation.pick_place import mdp as locomanip_mdp
from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import (
    ActionsCfg,
    ObservationsCfg,
)

from humanoid_arc_bme.robots.assets import G1_29DOF_USD, G1_IK_URDF
from humanoid_arc_bme.robots.bases import get_base

from . import mdp as pick_place_mdp
from .settled_pose import JOINT_POS as SETTLED_JOINT_POS
from .settled_pose import ROOT_HEIGHT as SETTLED_ROOT_HEIGHT
from .settled_pose import ROOT_QUAT as SETTLED_ROOT_QUAT

TABLE_SIZE = (0.6, 0.3, 0.70)
TABLE_POS = (0.1, 0.45, TABLE_SIZE[2] / 2)
OBJECT_SIZE = (0.05, 0.05, 0.10)
OBJECT_POS = (0.18, 0.40, TABLE_SIZE[2] + OBJECT_SIZE[2] / 2 + 0.002)
# Where the object gets placed is computed by ScriptedPickPlace itself
# (DEFAULT_PLACE_OFFSET, relative to OBJECT_POS), not fixed here.

# Demo env (below): the object starts 6cm further right than OBJECT_POS. At
# OBJECT_POS the resting right hand's fingertips touch its near face, and as
# the robot settles after spawning the hand drags it 1-3cm and tilts it
# (every episode; measured on recorded demos), so the grasp starts from a
# stale position.
DEMO_OBJECT_POS = (OBJECT_POS[0] + 0.06, OBJECT_POS[1], OBJECT_POS[2])
# Fixed goal, forward of DEMO_OBJECT_POS (pelvis frame; the robot spawns
# facing +y, G1_29DOF_CFG's 90deg yaw). 0.08 rather than DEFAULT_PLACE_OFFSET's
# 0.10: from this further-right start, 0.10 forward landed 16-27mm short
# (reach limit, see skills/scripted_pick_place.py).
DEMO_PLACE_OFFSET = (0.08, 0.0, 0.0)
PLACE_GOAL = (DEMO_OBJECT_POS[0], DEMO_OBJECT_POS[1] + DEMO_PLACE_OFFSET[0], DEMO_OBJECT_POS[2] - 0.002)
# Object start randomization around DEMO_OBJECT_POS (env frame, m). Small:
# forward (+y) is near the right arm's reach limit, see skills/scripted_pick_place.py.
OBJECT_START_RANGE = {"x": (-0.03, 0.03), "y": (-0.02, 0.02)}
# Demo env robot spawn: already settled (settled_pose.py), 5cm closer to the
# table than where it used to settle after spawning at the origin (y -0.067).
# The shorter reach makes the lower body counter-lean less as the arm extends.
DEMO_ROBOT_POS = (0.0, -0.017, SETTLED_ROOT_HEIGHT)


@configclass
class G1PickPlaceSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = G1_29DOF_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=G1_29DOF_CFG.spawn.replace(usd_path=str(G1_29DOF_USD)),
    )

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
        spawn=sim_utils.CuboidCfg(
            size=(200.0, 200.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.25)),
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POS),
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.32, 0.2)),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_POS),
        spawn=sim_utils.CuboidCfg(
            size=OBJECT_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=locomanip_mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class G1PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    # Base plug-in name (robots/bases.py). Set it at construction, e.g.
    # G1PickPlaceEnvCfg(robot_base="fixed"): __post_init__ applies it.
    robot_base: str = "agile_legs"
    scene: G1PickPlaceSceneCfg = G1PickPlaceSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 17.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = 2

        self.actions.upper_body_ik.controller.urdf_path = str(G1_IK_URDF)
        self.actions.upper_body_ik.controller.show_ik_warnings = False
        get_base(self.robot_base).apply(self)

        self.viewer.eye = (1.9, 0.1, 1.5)
        self.viewer.lookat = (0.05, 0.35, 0.8)


@configclass
class DemoEventsCfg(DefaultEventManagerCfg):
    # Inherits reset_scene_to_default (robot back to its spawn pose); without
    # it, overriding `events` leaves the robot wherever the last episode ended.
    reset_object = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": OBJECT_START_RANGE,
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    # After reset_scene_to_default (inherited, so it runs first): start settled.
    settle_robot = EventTerm(
        func=pick_place_mdp.reset_joints_to, mode="reset", params={"joint_pos": SETTLED_JOINT_POS}
    )


@configclass
class DemoTerminationsCfg(TerminationsCfg):
    success = DoneTerm(func=pick_place_mdp.object_placed, params={"goal": PLACE_GOAL})


@configclass
class G1PickPlaceDemoEnvCfg(G1PickPlaceEnvCfg):
    """Demo-recording variant: the object starts at a random spot near
    DEMO_OBJECT_POS, must end at the fixed PLACE_GOAL, and a `success` term
    marks that (what IsaacLab's recorder and Mimic read)."""

    events: DemoEventsCfg = DemoEventsCfg()
    terminations: DemoTerminationsCfg = DemoTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.object.init_state.pos = DEMO_OBJECT_POS
        # Root pose only; the settled joints are written by DemoEventsCfg.settle_robot.
        self.scene.robot.init_state = self.scene.robot.init_state.replace(pos=DEMO_ROBOT_POS, rot=SETTLED_ROOT_QUAT)


# Domain-randomized variant (plan A5.5). Full ranges below are multiplied by the environment
# variable HUMANOID_ARC_RANDOM_SCALE (default 1.0) so the expert's usable range can be swept
# without new code. Every range is an unmeasured guess (see tasks/pick_place/randomization.py).
RANDOM_POSE_RANGE = {"x": (-0.10, 0.10), "y": (-0.05, 0.05), "z": (0.0, 0.01), "yaw": (-0.7, 0.7)}
RANDOM_MASS = (0.05, 0.40)
RANDOM_STATIC_FRICTION = (0.8, 1.5)


def _scaled(ranges, scale):
    return {k: (lo * scale, hi * scale) for k, (lo, hi) in ranges.items()}


@configclass
class RandomizedEventsCfg(DemoEventsCfg):
    randomize_object_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": RANDOM_MASS,
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    randomize_object_friction = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": RANDOM_STATIC_FRICTION,
            "dynamic_friction_range": RANDOM_STATIC_FRICTION,
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )


@configclass
class G1PickPlaceRandomizedEnvCfg(G1PickPlaceDemoEnvCfg):
    """Demo env with a wider object start (pose scaled by HUMANOID_ARC_RANDOM_SCALE), random object mass and friction."""

    events: RandomizedEventsCfg = RandomizedEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        scale = float(os.environ.get("HUMANOID_ARC_RANDOM_SCALE", "1.0"))
        self.events.reset_object.params["pose_range"] = _scaled(RANDOM_POSE_RANGE, scale)


# Robot-side randomization (plan A5.5): friction and PD gains of the arm and hand joints only; the legs
# stay untouched because the AGILE lower-body policy balances on them. All ranges are unmeasured guesses.
ARM_HAND_JOINTS = [".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*", ".*_hand_.*"]
HAND_JOINTS = [".*_hand_.*"]
RANDOM_JOINT_FRICTION = (0.0, 0.3)
RANDOM_ARM_GAIN_SCALE = (0.8, 1.2)
RANDOM_HAND_GAIN_SCALE = (0.7, 1.3)


@configclass
class RandomizedRobotEventsCfg(RandomizedEventsCfg):
    randomize_joint_friction = EventTerm(
        func=base_mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_HAND_JOINTS),
            "friction_distribution_params": RANDOM_JOINT_FRICTION,
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    randomize_arm_gains = EventTerm(
        func=base_mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_HAND_JOINTS),
            "stiffness_distribution_params": RANDOM_ARM_GAIN_SCALE,
            "damping_distribution_params": RANDOM_ARM_GAIN_SCALE,
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    randomize_hand_gains = EventTerm(
        func=base_mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=HAND_JOINTS),
            "stiffness_distribution_params": RANDOM_HAND_GAIN_SCALE,
            "damping_distribution_params": RANDOM_HAND_GAIN_SCALE,
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class G1PickPlaceRandomizedRobotEnvCfg(G1PickPlaceRandomizedEnvCfg):
    """The randomized env plus arm/hand joint friction and PD-gain randomization."""

    events: RandomizedRobotEventsCfg = RandomizedRobotEventsCfg()


@configclass
class G1PickPlaceFixedBaseDemoEnvCfg(G1PickPlaceDemoEnvCfg):
    """The demo env on the `fixed` base: 28-dim actions, no legs to balance."""

    robot_base: str = "fixed"
