#!/usr/bin/env bash
# READ-ONLY checks before a session: python packages and versions, cyclonedds 0.10.2, Jetson (R36, TensorRT 10.7, MAXN),
# 192.168.123.x network, disk, clock/NTP, config, models (one ONNX inference on zeros). Never commands the robot.
#
#   bash deploy/scripts/preflight.sh [--iface eth0] [--listen 3] [--models DIR]
#
# --listen S subscribes to rt/lowstate for S seconds (a DDS subscriber: reads only) and prints the message rate.
# Exit code 1 when a required check is not OK.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
G1_HOME="${G1_HOME:-$HOME/g1_deps}"
set +u; [ -f "$G1_HOME/env.sh" ] && source "$G1_HOME/env.sh"; set -u
ARGS=(preflight --iface "${G1_IFACE:-eth0}" --models "${G1_MODELS:-$HOME/g1_models}")
cd "$REPO"
exec python3 -m humanoid_arc_bme.onboard.runtime "${ARGS[@]}" "$@"
