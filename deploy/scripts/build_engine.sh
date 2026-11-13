#!/usr/bin/env bash
# Build (or reuse) a TensorRT engine for a policy pack on THIS device, with TensorRT 10.7 exactly.
#
#   deploy/scripts/build_engine.sh <pack or policy dir> <fp32|fp16> [--force] [--time]
#
# Environment:
#   TensorRT_ROOT   the extracted TensorRT 10.7 TAR ("TensorRT 10.7 GA for L4T and CUDA 12.6 TAR Package");
#                   default $HOME/TensorRT. JetPack 6.2's own TensorRT (10.3, /usr/src/tensorrt) is NOT used.
#   ENGINE_CACHE    default $HOME/.cache/humanoid_arc/engines
#   TRTEXEC         override the trtexec path (still version-checked; used by the tests with a stub)
#
# Refuses (exit 2) unless: trtexec exists, the TensorRT library next to it is 10.7.x, trtexec resolves libnvinfer
# from $TensorRT_ROOT (not the system 10.3 with the same soname), and the build log banner says TensorRT v1007xx.
# Engines are device- and version-specific and are never portable: the cache key is (onnx sha256, precision,
# TensorRT version, device model). Prints the engine path on the last line of stdout. Writes <engine>.json (metadata,
# read by backends.py / check_parity.py / verify_policy_pack.py) and <engine>.build.log.
#
# Flags (TensorRT 10 trtexec): static shapes from the ONNX (batch 1, no --shapes: trtexec rejects --shapes on a static
# model); FP32 adds --noTF32 so Ampere (Orin) matmuls stay true FP32; FP16 adds --fp16 (I/O tensors stay FP32);
# --builderOptimizationLevel=3 (default, written out), --memPoolSize=workspace:256M, a timing cache next to the engines,
# --skipInference for the build, detailed profiling verbosity + per-layer export for the record.
set -euo pipefail

usage() { echo "usage: $0 <pack dir> <fp32|fp16> [--force] [--time]" >&2; exit 64; }
refuse() { echo "[build_engine] REFUSED: $*" >&2; exit 2; }

[[ $# -ge 2 ]] || usage
PACK=$1; PREC=$2; shift 2
FORCE=0; TIME=0
for a in "$@"; do case $a in --force) FORCE=1;; --time) TIME=1;; *) usage;; esac; done
[[ $PREC == fp32 || $PREC == fp16 ]] || usage
ONNX=$PACK/policy.onnx
[[ -f $ONNX ]] || refuse "no $ONNX"

REQ_MAJOR=10; REQ_MINOR=7
TensorRT_ROOT=${TensorRT_ROOT:-$HOME/TensorRT}
TRTEXEC=${TRTEXEC:-$TensorRT_ROOT/bin/trtexec}
[[ -x $TRTEXEC ]] || refuse "trtexec not found at $TRTEXEC (install the TensorRT 10.7 L4T TAR, export TensorRT_ROOT)"

# 1. library version on disk (the TAR ships libnvinfer.so.10.7.0)
shopt -s nullglob
LIBS=("$TensorRT_ROOT"/lib/libnvinfer.so.${REQ_MAJOR}.${REQ_MINOR}.*)
OTHER=("$TensorRT_ROOT"/lib/libnvinfer.so.${REQ_MAJOR}.*.*)
shopt -u nullglob
if [[ ${#LIBS[@]} -eq 0 ]]; then
  refuse "no libnvinfer.so.${REQ_MAJOR}.${REQ_MINOR}.* in $TensorRT_ROOT/lib (found: ${OTHER[*]:-none}); TensorRT ${REQ_MAJOR}.${REQ_MINOR} exactly is required"
fi
LIBVER=$(basename "${LIBS[0]}"); LIBVER=${LIBVER#libnvinfer.so.}
export LD_LIBRARY_PATH="$TensorRT_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 2. trtexec must load that library, not the JetPack one with the same soname
if LDD=$(ldd "$TRTEXEC" 2>/dev/null) && grep -q libnvinfer.so <<<"$LDD"; then
  RES=$(grep 'libnvinfer.so' <<<"$LDD" | head -1 | awk '{print $3}')
  [[ $(readlink -f "$RES") == "$(readlink -f "$TensorRT_ROOT")"/* ]] || refuse "trtexec resolves libnvinfer to $RES, not $TensorRT_ROOT/lib"
fi

# 3. cache key
SHA=$(sha256sum "$ONNX" | awk '{print $1}')
DEVICE=$(tr -d '\0' </proc/device-tree/model 2>/dev/null || uname -m)
DEVTAG=$(echo "$DEVICE" | tr -c 'A-Za-z0-9\n' '_' | tr -s '_' | cut -c1-40)
ENGINE_CACHE=${ENGINE_CACHE:-$HOME/.cache/humanoid_arc/engines}
DIR=$ENGINE_CACHE/${SHA:0:16}
ENGINE=$DIR/policy.$PREC.trt$LIBVER.$DEVTAG.engine
mkdir -p "$DIR"

if [[ $FORCE == 0 && -f $ENGINE && -f $ENGINE.json ]] && grep -q "\"onnx_sha256\": \"$SHA\"" "$ENGINE.json" \
   && grep -q "\"tensorrt\": \"$LIBVER\"" "$ENGINE.json"; then
  echo "[build_engine] cached (same onnx sha256, TensorRT $LIBVER, device $DEVICE)"
else
  FLAGS=(--onnx="$ONNX" --saveEngine="$ENGINE.tmp" --skipInference --builderOptimizationLevel=3
         --memPoolSize=workspace:256M --timingCacheFile="$DIR/timing.cache" --profilingVerbosity=detailed
         --exportLayerInfo="$ENGINE.layers.json")
  if [[ $PREC == fp16 ]]; then FLAGS+=(--fp16); else FLAGS+=(--noTF32); fi
  echo "[build_engine] $TRTEXEC ${FLAGS[*]}"
  if ! "$TRTEXEC" "${FLAGS[@]}" >"$ENGINE.build.log" 2>&1; then
    tail -20 "$ENGINE.build.log" >&2; rm -f "$ENGINE.tmp"; refuse "trtexec build failed (log $ENGINE.build.log)"
  fi
  # 4. the banner of the process that built it: "[TensorRT v100700]" = 10.7.0
  BANNER=$(grep -o 'TensorRT v[0-9]*' "$ENGINE.build.log" | head -1 | sed 's/TensorRT v//')
  if [[ -z $BANNER ]]; then rm -f "$ENGINE.tmp"; refuse "no 'TensorRT vNNNNNN' banner in the build log"; fi
  B_MAJ=$((10#$BANNER / 10000)); B_MIN=$(((10#$BANNER / 100) % 100))
  if [[ $B_MAJ != "$REQ_MAJOR" || $B_MIN != "$REQ_MINOR" ]]; then
    rm -f "$ENGINE.tmp"; refuse "built by TensorRT $B_MAJ.$B_MIN (banner v$BANNER); ${REQ_MAJOR}.${REQ_MINOR} exactly is required"
  fi
  [[ -s $ENGINE.tmp ]] || refuse "trtexec wrote no engine"
  mv "$ENGINE.tmp" "$ENGINE"
  ENGINE_SHA=$(sha256sum "$ENGINE" | awk '{print $1}')
  L4T=$(head -1 /etc/nv_tegra_release 2>/dev/null || echo none)
  python3 - "$ENGINE.json" <<EOF
import json, sys, time
json.dump({"onnx_sha256": "$SHA", "engine_sha256": "$ENGINE_SHA", "precision": "$PREC", "tensorrt": "$LIBVER",
           "trtexec_banner": "v$BANNER", "device": """$DEVICE""", "l4t_release": """$L4T""",
           "flags": """${FLAGS[*]}""".replace(".tmp", ""), "built": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
          open(sys.argv[1], "w"), indent=1)
EOF
  echo "[build_engine] built $PREC engine with TensorRT $LIBVER (log $ENGINE.build.log)"
fi

if [[ $TIME == 1 ]]; then
  # trtexec's own GPU-side view (no Python, no obs building): compare with bench_policy.py --engine
  "$TRTEXEC" --loadEngine="$ENGINE" --warmUp=1000 --duration=0 --iterations=10000 --percentile=50,95,99 \
    >"$ENGINE.time.log" 2>&1 || refuse "trtexec timing run failed (log $ENGINE.time.log)"
  grep -E 'Latency:|GPU Compute Time:|H2D Latency:|D2H Latency:' "$ENGINE.time.log" | head -4 || true
fi
echo "$ENGINE"
