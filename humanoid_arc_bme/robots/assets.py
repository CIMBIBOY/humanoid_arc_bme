"""Where robot assets live.

Large or third-party files (robot USD, NVIDIA policies) are fetched by
`setup.sh` into an asset root, never committed. The root is
`$HUMANOID_ARC_ASSETS` if set, else `assets/` of this repository or of the parent repository that mounts it. Small kinematic files that the IK needs ship with the
package itself (`robots/g1/`).

Pure path logic: importing this needs no Isaac Sim.
"""

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent


def asset_root() -> Path:
    """`$HUMANOID_ARC_ASSETS`, else `assets/` of this repository (filled by setup.sh), else `assets/` of a parent
    repository that mounts this one as a submodule, else the first path (so error messages name it)."""
    env = os.environ.get("HUMANOID_ARC_ASSETS")
    if env:
        return Path(env).expanduser().resolve()
    candidates = [PACKAGE_DIR.parent / "assets", PACKAGE_DIR.parent.parent / "assets"]
    return next((c for c in candidates if c.is_dir()), candidates[0])


# Unitree G1, 29 DoF with the three-finger hands (NVIDIA's Isaac Sim 5.1 asset).
G1_29DOF_USD = asset_root() / "G1_29dof" / "g1.usd"
# Kinematics-only URDF for Pink IK (from IsaacLab's locomanipulation assets; no meshes).
G1_IK_URDF = Path(__file__).resolve().parent / "g1" / "g1_29dof_with_hand_only_kinematics.urdf"
# NVIDIA AGILE lower-body locomotion policy (TorchScript), used by the `agile_legs` base.
AGILE_LOCOMOTION_POLICY = asset_root() / "policies" / "agile" / "agile_locomotion.pt"
