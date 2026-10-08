#!/usr/bin/env bash
# Measure full PPO-loop throughput (env-steps/s) and VRAM vs num_envs on this
# GPU, using IsaacLab's stock G1 velocity task and rsl_rl train.py.
#
#   scripts/infra/benchmark_throughput.sh [num_envs ...]     (default: 1024 2048 4096 8192 16384)
#
# Env vars: TASK (default Isaac-Velocity-Flat-G1-v0), ITERS (default 30),
# SKIP (warmup iterations excluded from the average, default 5).
# Prints one CSV row per env count; per-run logs go to logs/benchmark/.
# VRAM is peak total GPU memory minus the idle baseline, so run on an idle GPU.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAACLAB_DIR="${ISAACLAB_DIR:-$REPO_ROOT/../IsaacLab}"
TASK="${TASK:-Isaac-Velocity-Flat-G1-v0}"
ITERS="${ITERS:-30}"
SKIP="${SKIP:-5}"
ENVS=("$@")
[[ $# -eq 0 ]] && ENVS=(1024 2048 4096 8192 16384)

OUT="$REPO_ROOT/logs/benchmark"
mkdir -p "$OUT"
gpu_mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

echo "[INFO] $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
echo "[INFO] other GPU processes: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
echo "num_envs,steps_per_s_mean,steps_per_s_min,steps_per_s_max,collection_s,learning_s,vram_mib,status"

for n in "${ENVS[@]}"; do
  log="$OUT/${TASK}_${n}.log"
  base=$(gpu_mem); peak=$base
  "$REPO_ROOT/scripts/launch.sh" "$ISAACLAB_DIR/scripts/reinforcement_learning/rsl_rl/train.py" \
    --task "$TASK" --num_envs "$n" --max_iterations "$ITERS" --headless >"$log" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    m=$(gpu_mem); (( m > peak )) && peak=$m
    sleep 1
  done
  wait "$pid"; rc=$?
  status=ok; [[ $rc -ne 0 ]] && status="exit$rc"
  # Not "CUDA error": Warp logs harmless driver-entry-point CUDA errors at startup.
  grep -qi "out of memory" "$log" && status=oom
  # rsl_rl prints one stats block per iteration; drop the first SKIP.
  awk -v skip="$SKIP" -v n="$n" -v vram=$((peak - base)) -v st="$status" '
    /Steps per second:/ { i++; if (i > skip) { s=$NF; sum+=s; c++; if (min==""||s<min) min=s; if (s>max) max=s } }
    /Collection time:/  { if (i > skip) { gsub("s","",$NF); col+=$NF } }
    /Learning time:/    { if (i > skip) { gsub("s","",$NF); lrn+=$NF } }
    END { if (c) printf "%d,%.0f,%d,%d,%.3f,%.3f,%d,%s\n", n, sum/c, min, max, col/c, lrn/c, vram, st
          else   printf "%d,,,,,,%d,%s\n", n, vram, (st=="ok"?"no_iters":st) }' "$log"
done
