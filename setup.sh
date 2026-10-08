#!/usr/bin/env bash
# One-time setup. Writes only inside this repo; the Isaac Sim / IsaacLab
# install is never modified. NVIDIA assets are fetched, not committed.
#
#   ./setup.sh [path/to/IsaacLab]      (default: $ISAACLAB_DIR or ../IsaacLab)
#
# 1. .deps/    numpy<2 overlay. IsaacLab's pinned pinocchio (Pink IK) is built
#              against numpy 1.x and segfaults under numpy 2.x;
#              scripts/launch.sh puts this overlay first on PYTHONPATH.
# 2. assets/   local copies of the NVIDIA assets the envs load (Isaac Sim 5.1
#              content): the G1 29-DoF tri-hand USD, the AGILE lower-body
#              policy, and IsaacLab's pretrained G1 flat-walking checkpoint
#              converted to the rsl-rl-lib >= 5 format.
#
# humanoid_arc_core (robots, pick-place task, skills) is not installed here;
# scripts/launch.sh finds it via $HUMANOID_ARC_CORE (default ../humanoid_arc).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_DIR="${1:-${ISAACLAB_DIR:-$REPO_ROOT/../IsaacLab}}"
BUCKET="https://omniverse-content-production.s3-us-west-2.amazonaws.com"
CDN="$BUCKET/Assets/Isaac/5.1/Isaac"
ASSETS="$REPO_ROOT/assets"

if [[ ! -f "$ISAACLAB_DIR/isaaclab.sh" ]]; then
  echo "[ERROR] isaaclab.sh not found under $ISAACLAB_DIR (pass the IsaacLab path as an argument)" >&2
  exit 1
fi

echo "[INFO] numpy<2 overlay -> .deps/"
"$ISAACLAB_DIR/isaaclab.sh" -p -m pip install -q --target "$REPO_ROOT/.deps" --no-deps --upgrade "numpy==1.26.4"

fetch() {  # fetch <cdn-relative-path> <local-path>
  [[ -s "$2" ]] && return 0
  mkdir -p "$(dirname "$2")"
  curl -sf -o "$2" "$CDN/$1" || { echo "[ERROR] failed to fetch $1" >&2; return 1; }
}

echo "[INFO] G1 29-DoF tri-hand robot -> assets/G1_29dof/ (~150MB)"
curl -sf "$BUCKET/?prefix=Assets/Isaac/5.1/Isaac/Robots/Unitree/G1/" \
  | grep -o '<Key>[^<]*</Key>' | sed 's#<Key>Assets/Isaac/5.1/Isaac/Robots/Unitree/G1/##; s#</Key>##' \
  | grep -vE '\.thumbs|\.png$|\.jpg$|/$' \
  | while read -r rel; do fetch "Robots/Unitree/G1/$rel" "$ASSETS/G1_29dof/$rel"; done

echo "[INFO] AGILE lower-body locomotion policy"
fetch "IsaacLab/Policies/Agile/agile_locomotion.pt" "$ASSETS/policies/agile/agile_locomotion.pt"

echo "[INFO] G1 flat velocity policy (IsaacLab pretrained, converted to rsl-rl-lib 5 format)"
VEL="$ASSETS/policies/g1_velocity_flat/checkpoint.pt"
if [[ ! -s "$VEL" ]]; then
  RAW="$(mktemp --suffix=.pt)"
  rm -f "$RAW"
  fetch "IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-G1-v0/checkpoint.pt" "$RAW"
  "$ISAACLAB_DIR/isaaclab.sh" -p "$REPO_ROOT/scripts/locomotion/convert_rsl_rl_checkpoint.py" "$RAW" "$VEL" | tail -1
  rm -f "$RAW"
fi

echo "[INFO] verifying pinocchio/pink import through the overlay"
PYTHONPATH="$REPO_ROOT/.deps" "$ISAACLAB_DIR/isaaclab.sh" -p -c \
  "import numpy, pinocchio, pink; assert numpy.__version__.startswith('1.'), numpy.__version__; print('[OK] numpy', numpy.__version__, '| pinocchio', pinocchio.__version__)"

echo "[OK] ready:  ISAACLAB_DIR=$ISAACLAB_DIR scripts/launch.sh scripts/run_table_tour.py --headless --laps 1"
