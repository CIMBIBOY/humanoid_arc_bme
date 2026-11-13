#!/usr/bin/env bash
# First-boot setup of the G1 dev computer after the JetPack 6.2 image (plan B2.5 step 4, B2.6).
# Dry run by default: prints what it would do. Pass --apply to run it.
# Commands are the ones on NVIDIA's G1 JetPack 6 page; TensorRT 10.7 is a manual TAR install.
set -u
APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1

step() {  # step "<description>" cmd...
  local d="$1"; shift
  echo "== $d"; echo "   $*"
  if [ "$APPLY" = 1 ]; then "$@" || { echo "   FAILED: $*"; exit 1; }; fi
}

[ -f /etc/nv_tegra_release ] || { echo "not a Jetson (no /etc/nv_tegra_release); stop."; exit 2; }
grep -q 'R36' /etc/nv_tegra_release || { echo "Jetson Linux is not R36 (JetPack 6.x); stop."; exit 2; }

step "performance mode"      sudo nvpmodel -m 0
step "fix clocks"            sudo jetson_clocks
step "DLA compiler packages" sudo apt-get install -y nvidia-l4t-dla-compiler libcudla-dev-12-6
echo "== TensorRT: install the 10.7 TAR package by hand into ~/TensorRT, add"
echo "   export TensorRT_ROOT=\$HOME/TensorRT   to ~/.bashrc, then run:"
echo "   python3 -m humanoid_arc_bme.onboard.deploy preflight"
[ "$APPLY" = 1 ] || echo "(dry run: nothing was changed; add --apply to run)"
