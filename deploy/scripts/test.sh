#!/usr/bin/env bash
# Run the runtime's tests with FAKE robot IO only (no DDS traffic, nothing sent to the robot). Safe on the robot computer.
#   bash deploy/scripts/test.sh [pytest args]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
G1_HOME="${G1_HOME:-$HOME/g1_deps}"
set +u; [ -f "$G1_HOME/env.sh" ] && source "$G1_HOME/env.sh"; set -u
cd "$REPO"
exec python3 -m pytest -q -p no:cacheprovider tests/runtime/test_operator.py tests/runtime/test_operator_runtime.py tests/runtime/test_controllers.py \
  tests/runtime/test_runlog.py tests/runtime/test_web.py tests/runtime/test_ros_wrapper.py tests/runtime/test_runtime.py \
  tests/runtime/test_runtime_failsafe.py tests/runtime/test_policies.py tests/runtime/test_obs_builders.py tests/runtime/test_robot_scripts.py \
  tests/core/test_safety.py tests/core/test_contract.py "$@"
