#!/usr/bin/env bash
# Start the operator runtime + web UI. It starts DISARMED: nothing moves until you arm in the UI with the typed code.
#
#   bash deploy/scripts/run.sh [--ros] [--io unitree|fake] [--iface eth0] [--models DIR] [--shadow DIR] [--port 8080]
#                             [--camera none|fake|realsense] [--allow-lowcmd] [--limits FILE] [--lan --lan-plain-http-ack]
#
# Defaults: plain python (no ROS), --io unitree, --iface $G1_IFACE (default eth0: check with `ip -4 addr`, the robot side
# is 192.168.123.x), --models $G1_MODELS (default ~/g1_models), UI on 127.0.0.1:8080 (laptop: ssh -L 8080:127.0.0.1:8080).
# Default and recommended: the UI on 127.0.0.1 through an SSH tunnel. --lan serves plain HTTP on 0.0.0.0 (the token can
# be sniffed, anyone on the path can drive) and is refused unless --lan-plain-http-ack is also given (wired only).
# --limits FILE: S1 + operator limits (first walk: config/g1_limits_firstwalk.yaml).
# --allow-lowcmd creates the gated low-level command path for the policy controller: HOLD STEP, see docs/plan/robot-deploy.md.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
G1_HOME="${G1_HOME:-$HOME/g1_deps}"
MODE=plain; IO=unitree; IFACE="${G1_IFACE:-eth0}"; MODELS="${G1_MODELS:-$HOME/g1_models}"; SHADOW=""; PORT=8080; LAN=0; LANACK=0; LOWCMD=0; CAMERA=none; LIMITS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --ros) MODE=ros ;; --plain) MODE=plain ;; --lan) LAN=1 ;; --lan-plain-http-ack) LANACK=1 ;; --allow-lowcmd) LOWCMD=1 ;;
    --limits) LIMITS="$2"; shift ;;
    --io) IO="$2"; shift ;; --iface) IFACE="$2"; shift ;; --models) MODELS="$2"; shift ;; --shadow) SHADOW="$2"; shift ;;
    --port) PORT="$2"; shift ;; --camera) CAMERA="$2"; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac; shift
done
set +u
if [ -f "$G1_HOME/env.sh" ]; then source "$G1_HOME/env.sh"; else echo "WARNING: $G1_HOME/env.sh missing (install.sh not run): using system python3"; fi
set -u
if [ "$MODE" = plain ]; then
  ARGS=(serve --io "$IO" --port "$PORT" --camera "$CAMERA")
  [ "$IO" = unitree ] && ARGS+=(--iface "$IFACE")
  [ -d "$MODELS" ] && ARGS+=(--models "$MODELS")
  [ -n "$SHADOW" ] && ARGS+=(--shadow "$SHADOW")
  [ "$LAN" = 1 ] && ARGS+=(--lan)
  [ "$LANACK" = 1 ] && ARGS+=(--lan-plain-http-ack)
  [ -n "$LIMITS" ] && ARGS+=(--limits "$LIMITS")
  [ "$LOWCMD" = 1 ] && ARGS+=(--allow-lowcmd)
  cd "$REPO"
  echo "+ python3 -m humanoid_arc_bme.onboard.runtime ${ARGS[*]}"
  exec python3 -m humanoid_arc_bme.onboard.runtime "${ARGS[@]}"
fi
set +u
source /opt/ros/humble/setup.bash
[ -f "$REPO/install_ros/setup.bash" ] || { echo "run deploy/scripts/build.sh first"; exit 1; }
source "$REPO/install_ros/setup.bash"
set -u
LA=(io:="$IO" port:="$PORT" camera:="$CAMERA" lan:="$([ "$LAN" = 1 ] && echo true || echo false)"
    allow_lowcmd:="$([ "$LOWCMD" = 1 ] && echo true || echo false)"
    lan_plain_http_ack:="$([ "$LANACK" = 1 ] && echo true || echo false)")
[ -n "$LIMITS" ] && LA+=(limits:="$LIMITS")
[ "$IO" = unitree ] && LA+=(iface:="$IFACE")
[ -d "$MODELS" ] && LA+=(models:="$MODELS")
[ -n "$SHADOW" ] && LA+=(shadow:="$SHADOW")
echo "+ ros2 launch g1_runtime_ros g1_runtime.launch.py ${LA[*]}"
exec ros2 launch g1_runtime_ros g1_runtime.launch.py "${LA[@]}"
