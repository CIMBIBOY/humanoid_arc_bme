"""G1 lower-body training env: legs are learned, everything above the hips is scripted.

The waist, arms and hands follow recorded pick-place trajectories or random smooth motion
(mdp.ArmMotionAction) while random wrist payloads, pushes and friction disturb the legs. The
observation and action layout is the Agile policy's (4-value command + 79 proprioceptive
values in, 12 leg joint offsets out, scale 0.25 from the default pose), so a trained policy
drops into the pick-place and tour envs in Agile's place. Rewards are the IsaacLab G1
flat-walking set plus pelvis-height tracking and an upright-pelvis penalty.
"""

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_assets.robots.unitree import G1_29DOF_CFG
from isaaclab.managers import RewardTermCfg as RewTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp

from . import mdp
import os
from pathlib import Path

from humanoid_arc_core.robots.assets import G1_29DOF_USD

LEG_JOINTS = [".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"]
# The 29 body joints the Agile policy observes (hands excluded); order follows the articulation.
OBSERVED_JOINTS = [
    ".*_shoulder_.*_joint",
    ".*_elbow_joint",
    ".*_wrist_.*_joint",
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
    "waist_.*_joint",
]
# Recorded pick-place demos whose upper-body joint targets the arms replay
# (record them with the core's tools/record_demos.py --speed 1.5). Override
# with $HUMANOID_ARC_ARM_DEMOS (comma-separated paths).
_DEFAULT_DEMOS = Path(__file__).resolve().parents[3] / "datasets" / "g1_pick_place_scripted_v2.hdf5"
DEMO_FILES = os.environ.get("HUMANOID_ARC_ARM_DEMOS", str(_DEFAULT_DEMOS)).split(",")


@configclass
class LowerBodySceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    )
    robot: ArticulationCfg = G1_29DOF_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=G1_29DOF_CFG.spawn.replace(usd_path=str(G1_29DOF_USD), activate_contact_sensors=True),
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0)
    )


@configclass
class ActionsCfg:
    legs = base_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=LEG_JOINTS, scale=0.25, use_default_offset=True
    )
    arms = mdp.ArmMotionActionCfg(asset_name="robot", demo_files=DEMO_FILES)


@configclass
class CommandsCfg:
    base_command = mdp.VelocityHeightCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.3,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.VelocityHeightCommandCfg.Ranges(lin_vel_x=(-0.3, 0.8), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.8, 0.8)),
        height_range=(0.6, 0.78),
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        command = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "base_command"})
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=OBSERVED_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            scale=0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=OBSERVED_JOINTS)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=base_mdp.last_action, params={"action_name": "legs"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    physics_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    wrist_payload = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_wrist_yaw_link"),
            "mass_distribution_params": (0.0, 0.5),
            "operation": "add",
        },
    )
    reset_base = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-3.14, 3.14)},
            "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        },
    )
    reset_joints = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)},
    )
    push_robot = EventTerm(
        func=base_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(6.0, 12.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


FEET = SceneEntityCfg("robot", body_names=".*_ankle_roll_link")
LEGS = SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])


@configclass
class RewardsCfg:
    termination = RewTerm(func=base_mdp.is_terminated, weight=-200.0)
    track_lin_vel = RewTerm(
        func=vel_mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_command", "std": 0.5},
    )
    track_ang_vel = RewTerm(
        func=vel_mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_command", "std": 0.5}
    )
    track_height = RewTerm(func=mdp.track_pelvis_height_exp, weight=3.0, params={"command_name": "base_command", "std": 0.1})
    track_height_fine = RewTerm(
        func=mdp.track_pelvis_height_exp, weight=1.0, params={"command_name": "base_command", "std": 0.03}
    )
    upright = RewTerm(func=base_mdp.flat_orientation_l2, weight=-8.0)
    lin_vel_z = RewTerm(func=base_mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy = RewTerm(func=base_mdp.ang_vel_xy_l2, weight=-0.2)
    feet_air_time = RewTerm(
        func=vel_mdp.feet_air_time_positive_biped,
        weight=0.75,
        params={
            "command_name": "base_command",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=vel_mdp.feet_slide,
        weight=-0.3,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"), "asset_cfg": FEET},
    )
    ankle_limits = RewTerm(
        func=base_mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    hip_deviation = RewTerm(
        func=base_mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.005)
    joint_acc = RewTerm(func=base_mdp.joint_acc_l2, weight=-1.0e-7, params={"asset_cfg": LEGS})
    joint_torques = RewTerm(func=base_mdp.joint_torques_l2, weight=-2.0e-6, params={"asset_cfg": LEGS})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=base_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis", "torso_link"]), "threshold": 1.0},
    )


@configclass
class G1LowerBodyEnvCfg(ManagerBasedRLEnvCfg):
    scene: LowerBodySceneCfg = LowerBodySceneCfg(num_envs=4096, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
        self.viewer.eye = (3.0, 3.0, 1.8)
        self.viewer.lookat = (0.0, 0.0, 0.7)


@configclass
class G1LowerBodyEnvCfg_PLAY(G1LowerBodyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False


@configclass
class G1LowerBodyEasyEnvCfg(G1LowerBodyEnvCfg):
    """No payloads, pushes or arm motion: checks that the rewards teach height and walking at all."""

    def __post_init__(self):
        super().__post_init__()
        self.events.wrist_payload = None
        self.events.push_robot = None
        self.actions.arms.demo_prob = 0.0
        self.actions.arms.amplitude_range = (0.0, 0.0)
        self.actions.arms.walk_prob = 0.0
        self.actions.arms.held_prob = 0.0
