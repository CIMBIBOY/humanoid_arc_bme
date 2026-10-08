"""G1 walking a continuous circuit of tables, picking and placing at each.

Reuses everything from the core's pick-place task (same actions and
observations, the `agile_legs` base: AGILE lower body + Pink IK) and adds N
tables arranged around a square path. The robot walks the perimeter
(`policies/navigator.py`), and at each corner runs the core's scripted
pick-place skill (`humanoid_arc_core.skills.scripted_pick_place`),
orchestrated by `policies/table_tour.py`.

To add/replace a stop: edit STOPS below (or generate it -- `layout_square`
just derives positions/headings from a corner count and side length; a
different call, or a hand-written list of `Stop`, both work) and add a
correspondingly named object color. Everything downstream (scene, navigation,
grasp geometry) is derived from that list, not hand-tuned per table.
"""

import math
from dataclasses import dataclass

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from humanoid_arc_core.tasks.pick_place.env_cfg import OBJECT_SIZE, TABLE_SIZE, G1PickPlaceEnvCfg, G1PickPlaceSceneCfg
from humanoid_arc_core.tasks.pick_place.env_cfg import ObservationsCfg as G1PickPlaceObservationsCfg
from humanoid_arc_core.tasks.pick_place.env_cfg import TerminationsCfg as G1PickPlaceTerminationsCfg

# Forward/right offset of the object from the robot's pelvis, in the pelvis
# frame, reused unchanged from the validated single-table layout
# (the core task's OBJECT_POS is (0.18, 0.40, ...) relative to a pelvis
# at the origin facing +Y -- i.e. forward=0.40, right=0.18).
OBJECT_FORWARD, OBJECT_RIGHT = 0.40, 0.18
TABLE_FORWARD, TABLE_RIGHT = 0.45, 0.10

COLORS = {
    "red": (0.8, 0.1, 0.1),
    "green": (0.1, 0.7, 0.15),
    "blue": (0.15, 0.25, 0.85),
    "yellow": (0.85, 0.75, 0.1),
}


@dataclass
class Stop:
    name: str
    position: tuple[float, float]  # world XY the pelvis walks to
    heading: float  # radians; both the arrival heading and the table's facing
    color: tuple[float, float, float]


def layout_square(colors: list[str], side: float = 2.0) -> list[Stop]:
    """N stops around a square of the given side length, walked clockwise.

    Matches the robot's default spawn pose: heading 0 here means facing +Y
    (G1_29DOF_CFG's default `rot` is a 90-degree yaw), so the first stop's
    arrival heading equals the spawn heading and no initial turn is needed.
    """
    n = len(colors)
    headings = [i * (-2 * math.pi / n) for i in range(n)]  # clockwise turn each corner
    directions = [(math.cos(h), math.sin(h)) for h in headings]  # yaw -> world forward, see _offset()
    positions, pos = [], (0.0, 0.0)
    for dx, dy in directions:
        positions.append(pos)
        pos = (pos[0] + side * dx, pos[1] + side * dy)
    # Arrival heading at stop i is the direction of the edge that ends there.
    arrival = [headings[i - 1] for i in range(n)]
    return [Stop(c, positions[i], arrival[i], COLORS[c]) for i, c in enumerate(colors)]


STOPS = layout_square(["red", "green", "blue", "yellow"])


@configclass
class TourObservationsCfg(G1PickPlaceObservationsCfg):
    """Drops the single-table `ObservationsCfg`'s object-specific terms.

    `object_pos`/`object_rot` (SceneEntityCfg("object")) and `object`
    (`manip_mdp.object_obs`, which also hardcodes the scene entity name
    "object" internally) all assume the single-table scene's one `object`
    entity; this scene has one per stop (`object_<name>`) instead. Nothing
    here reads the observation buffer directly -- TableTourController reads
    `env.scene[...]` itself -- so these are just dropped rather than
    rewritten to take an object name (`lower_body_policy`, needed internally
    by the Agile action term, and the rest of `policy` are untouched)."""

    def __post_init__(self):
        super().__post_init__()
        self.policy.object_pos = None
        self.policy.object_rot = None
        self.policy.object = None


@configclass
class TourTerminationsCfg(G1PickPlaceTerminationsCfg):
    """Same as the single-table `TerminationsCfg`, but `object_dropping` (which
    hardcodes `SceneEntityCfg("object")`) becomes one term per stop's object,
    generated the same way the scene's tables/objects are."""

    object_dropping = None

    def __post_init__(self):
        for stop in STOPS:
            setattr(
                self,
                f"object_dropping_{stop.name}",
                DoneTerm(
                    func=base_mdp.root_height_below_minimum,
                    params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg(f"object_{stop.name}")},
                ),
            )


def _offset(stop: Stop, forward: float, right: float, z: float) -> tuple[float, float, float]:
    """A point `forward`/`right` of `stop`, in world XY, facing `stop.heading`.

    Matches ScriptedPickPlace's convention exactly (both ultimately feed
    isaaclab.utils.math.quat_apply/yaw_quat): for yaw angle theta, world
    forward is (cos theta, sin theta), right is forward rotated -90 degrees.
    `navigate()` also treats `stop.heading` as a plain yaw with no offset, so
    all three (navigation, object placement, grasp math) must agree here --
    this file used to add a spurious extra +pi/2, which sent the robot to
    the right position but grasping in a direction 90 degrees from where the
    object actually was (verified on-machine: the object never moved from
    its spawn pose across a full 2-lap run, `obj == home` to 3 decimals every
    time -- see docs/audit.md).
    """
    fwd = (math.cos(stop.heading), math.sin(stop.heading))
    rgt = (fwd[1], -fwd[0])
    x = stop.position[0] + forward * fwd[0] + right * rgt[0]
    y = stop.position[1] + forward * fwd[1] + right * rgt[1]
    return (x, y, z)


@configclass
class G1TableTourSceneCfg(G1PickPlaceSceneCfg):
    # `robot`, `ground`, `light` are inherited as-is from G1PickPlaceSceneCfg.
    # Overridden per-instance in __post_init__ below: the base class defines
    # a single `table`/`object` pair for the single-table demo, which this
    # scene replaces with `table_<name>`/`object_<name>` per stop.
    table = None
    object = None

    def __post_init__(self):
        for stop in STOPS:
            table_z = TABLE_SIZE[2] / 2
            setattr(
                self,
                f"table_{stop.name}",
                AssetBaseCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Table_{stop.name}",
                    init_state=AssetBaseCfg.InitialStateCfg(
                        pos=_offset(stop, TABLE_FORWARD, TABLE_RIGHT, table_z)
                    ),
                    spawn=sim_utils.CuboidCfg(
                        size=TABLE_SIZE,
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.32, 0.2)),
                    ),
                ),
            )
            object_z = TABLE_SIZE[2] + OBJECT_SIZE[2] / 2 + 0.002
            setattr(
                self,
                f"object_{stop.name}",
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Object_{stop.name}",
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=_offset(stop, OBJECT_FORWARD, OBJECT_RIGHT, object_z)
                    ),
                    spawn=sim_utils.CuboidCfg(
                        size=OBJECT_SIZE,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.5),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=stop.color),
                    ),
                ),
            )


@configclass
class G1TableTourEnvCfg(G1PickPlaceEnvCfg):
    scene: G1TableTourSceneCfg = G1TableTourSceneCfg(num_envs=1, env_spacing=8.0, replicate_physics=True)
    observations: TourObservationsCfg = TourObservationsCfg()
    terminations: TourTerminationsCfg = TourTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # No natural episode boundary for a continuous tour (unlike the
        # single-table demo, TerminationsCfg.time_out never should fire
        # mid-circuit); scripts/run_table_tour.py drives resets itself via
        # TableTourController when the object is dropped.
        self.episode_length_s = 1.0e6
        # Tables are ~2m apart; env_spacing must clear the full circuit
        # footprint so parallel envs (see --num_envs) don't overlap.
        span = max(abs(s.position[0]) for s in STOPS) + max(abs(s.position[1]) for s in STOPS)
        self.scene.env_spacing = span + 2.0
        # Look at the middle of the circuit from a corner, high enough to see all stops.
        cx = sum(s.position[0] for s in STOPS) / len(STOPS)
        cy = sum(s.position[1] for s in STOPS) / len(STOPS)
        self.viewer.eye = (cx + 3.2, cy - 3.6, 3.6)
        self.viewer.lookat = (cx, cy, 0.5)
