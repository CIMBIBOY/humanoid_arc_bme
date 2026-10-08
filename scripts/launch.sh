#!/usr/bin/env bash
# Run a script through IsaacLab with this repo, humanoid_arc_core and the
# numpy<2 overlay (.deps, from setup.sh) on PYTHONPATH. Nothing outside this
# repo is modified.
#
#   scripts/launch.sh <script.py | -m module> [args...]
#
# Environment:
#   ISAACLAB_DIR        IsaacLab checkout (default: ../IsaacLab)
#   HUMANOID_ARC_CORE   humanoid_arc checkout providing humanoid_arc_core (default: the parent repo when this
#                       repo is its bme/ submodule, else ../humanoid_arc)
#   HUMANOID_ARC_ASSETS asset root (default: ./assets, filled by setup.sh)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_DIR="${ISAACLAB_DIR:-$REPO_ROOT/../IsaacLab}"
if [[ -d "$REPO_ROOT/../humanoid_arc_core" ]]; then DEFAULT_CORE="$REPO_ROOT/.."; else DEFAULT_CORE="$REPO_ROOT/../humanoid_arc"; fi
CORE_DIR="${HUMANOID_ARC_CORE:-$DEFAULT_CORE}"

if [[ $# -lt 1 ]]; then
  sed -n '2,13p' "$0"; exit 1
fi
if [[ ! -d "$REPO_ROOT/.deps/numpy" ]]; then
  echo "[ERROR] .deps overlay missing; run ./setup.sh first" >&2; exit 1
fi
if [[ ! -d "$CORE_DIR/humanoid_arc_core" ]]; then
  echo "[ERROR] humanoid_arc_core not found in $CORE_DIR (set HUMANOID_ARC_CORE)" >&2; exit 1
fi

export HUMANOID_ARC_ASSETS="${HUMANOID_ARC_ASSETS:-$REPO_ROOT/assets}"
# Lets the core's tools/isaaclab_tool.py (rsl_rl train, ...) find this repo's gym ids.
export HUMANOID_ARC_TASK_MODULES="${HUMANOID_ARC_TASK_MODULES:-humanoid_arc_bme.envs.g1_lower_body}"
export PYTHONPATH="$REPO_ROOT/.deps:$REPO_ROOT:$CORE_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
exec "$ISAACLAB_DIR/isaaclab.sh" -p "$@"
