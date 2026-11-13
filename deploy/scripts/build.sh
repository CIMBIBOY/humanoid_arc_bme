#!/usr/bin/env bash
# Build the ROS 2 Humble package g1_runtime_ros (colcon). Only needed for the ROS mode; the plain-python mode needs no build.
#
#   bash deploy/scripts/build.sh            # -> <repo>/install_ros  (build_ros/ and log in build_ros/log)
#
# Uses --base-paths ros2, so colcon never looks at the rest of the repository (submodules, training code).
# Not --symlink-install: newer setuptools reject colcon's `develop --editable` (seen on this workstation); a normal build
# copies the core in, so REBUILD after every git pull. Sends nothing to the robot.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
G1_HOME="${G1_HOME:-$HOME/g1_deps}"
set +u
[ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash || echo "WARNING: /opt/ros/humble/setup.bash not found (ROS 2 Humble not on the host?)"
[ -f "$G1_HOME/env.sh" ] && source "$G1_HOME/env.sh" || echo "WARNING: $G1_HOME/env.sh missing: run deploy/scripts/install.sh first"
set -u
command -v colcon >/dev/null || { echo "colcon not found: sudo apt-get install -y python3-colcon-common-extensions (you run it)"; exit 1; }
cd "$REPO"
colcon --log-base build_ros/log build --base-paths ros2 --build-base build_ros --install-base install_ros "$@"
echo "built. ROS mode: bash deploy/scripts/run.sh --ros ...   (sources install_ros/setup.bash)"
